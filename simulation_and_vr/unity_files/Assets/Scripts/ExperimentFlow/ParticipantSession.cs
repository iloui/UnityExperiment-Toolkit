using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Assets.Scripts.DataRecording;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityStandardAssets.Characters.FirstPerson;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

namespace Assets.Scripts.ExperimentFlow
{
    /// <summary>
    /// Runs one desktop human recording session from the bootstrap scene (Tools > Recording > Create Participant Session
    /// Scene): participant ID -> instructions -> untracked practice -> the layout scenes in the participant's
    /// Latin-square order, each ParticipantSessionConfig.repetitions times. Every scene is loaded on its own (Single mode);
    /// this object survives the loads, configures the scene's ParticipantRecorder in sceneLoaded (after Awake, before
    /// Start, so AutoStartForTesting never fires) and starts/stops recording itself. Each finished or aborted episode is
    /// appended to imitationagent_training/participant_sessions/session_log.csv. Nothing is set in the inspector per participant.
    /// A finished session requests one full retrain (RetrainPipeline, if config.retrainAfterSession); an aborted one only
    /// with config.retrainOnAbort.
    /// </summary>
    public class ParticipantSession : MonoBehaviour
    {
        public ParticipantSessionConfig config;

        private enum Stage { EnterId, Instructions, Loading, Ready, Running, TrialDone, Done, Error }

        private const float MinSecondsBeforeConfirm = 0.4f;
        private const string LogHeader =
            "logged_at,participant_id,status,dry_run,participant_number,session_id,latin_square_row,layout_order," +
            "trial_index,trial_count,order_position,layout_index,scene_name,layout_typology,layout_seed,repetition_index," +
            "episode_id,frame_count,duration_s,episode_dir";

        private Stage stage;
        private float stageShownAt;
        private bool confirmed;
        private string idInput = "";
        private string idError;
        private string errorMessage;
        private string trialDoneMessage;

        private string participantId;   // as written to the manifests (dry run: "dryrun_<ID>")
        private int participantNumber;
        private string sessionId;
        private int latinRow;
        private int[] layoutOrder;
        private List<SessionSchedule.Trial> trials;
        private int currentTrial = -1;  // -1 = practice
        private int completedEpisodes;
        private bool retrainHandled;    // at most one retrain request per session

        // Set while a scene loads; OnSceneLoaded configures that scene.
        private string pendingScenePath;
        private SessionSchedule.Trial? pendingTrial;
        private string sceneError;
        private ParticipantRecorder recorder;
        private FirstPersonController controller;
        private PoiProgressTracker practiceTracker;

        private GUIStyle boxStyle;
        private GUIStyle textStyle;
        private GUIStyle fieldStyle;
        private GUIStyle hudStyle;
        private GUIStyle labelStyle;

        private static string SimulationRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "../../"));
        private static string SessionLogDirectory => Path.Combine(SimulationRoot, "imitationagent_training", "participant_sessions");
        private static string AbortedEpisodesDirectory => Path.Combine(SimulationRoot, "imitationagent_training", "zarr_recordings_aborted");
        private string SessionLogPath => Path.Combine(SessionLogDirectory, config.dryRun ? "session_log_dryrun.csv" : "session_log.csv");

        private void Awake()
        {
            DontDestroyOnLoad(gameObject);
            SceneManager.sceneLoaded += OnSceneLoaded;
        }

        private void OnDestroy()
        {
            SceneManager.sceneLoaded -= OnSceneLoaded;
            // Play mode stopped mid-session.
            if (trials != null && stage != Stage.Done) RequestRetrainOnce(aborted: true);
        }

        private IEnumerator Start()
        {
            string configError = ValidateConfig();
            if (configError != null) { Fail(configError); yield break; }

            // 1. Participant ID (validated in OnGUI before 'confirmed' is set).
            SetStage(Stage.EnterId);
            yield return WaitForConfirm();

            string enteredId = idInput.Trim();
            SessionSchedule.TryParticipantNumber(enteredId, out participantNumber);
            participantId = config.dryRun ? "dryrun_" + enteredId : enteredId;
            sessionId = $"{participantId}_{DateTime.Now:yyyyMMdd_HHmmss}";
            trials = SessionSchedule.BuildTrials(participantNumber, config.LayoutScenePaths.Length, config.repetitions, out latinRow, out layoutOrder);
            Debug.Log($"[ParticipantSession] Session '{sessionId}': participant {participantId} (#{participantNumber}), Latin-square row {latinRow}, " +
                $"layout order {string.Join("-", layoutOrder)}, {trials.Count} trials{(config.dryRun ? " (DRY RUN)" : "")}.");

            // 2. Instructions.
            SetStage(Stage.Instructions);
            yield return WaitForConfirm();

            // 3. Untracked practice.
            currentTrial = -1;
            yield return LoadScene(config.PracticeScenePath, null);
            if (stage == Stage.Error) yield break;
            SetStage(Stage.Ready);
            yield return WaitForConfirm();
            yield return RunPractice();

            // 4. Recorded trials; an aborted trial is repeated with the same repetition index.
            for (currentTrial = 0; currentTrial < trials.Count;)
            {
                SessionSchedule.Trial trial = trials[currentTrial];
                yield return LoadScene(config.LayoutScenePaths[trial.layoutIndex], trial);
                if (stage == Stage.Error) yield break;
                SetStage(Stage.Ready);
                yield return WaitForConfirm();

                bool completed = false;
                yield return RunTrial(trial, result => completed = result);
                if (stage == Stage.Error) yield break;

                trialDoneMessage = completed
                    ? (currentTrial + 1 < trials.Count ? "Well done! Press Enter for the next run." : "Well done!")
                    : "The run was stopped and will be repeated. Press Enter to restart it.";
                if (completed)
                {
                    currentTrial++;
                    completedEpisodes++;
                }
                if (currentTrial < trials.Count)
                {
                    SetStage(Stage.TrialDone);
                    yield return WaitForConfirm();
                }
            }

            Debug.Log($"[ParticipantSession] Session '{sessionId}' complete: {trials.Count} episodes logged to {SessionLogPath}.");
            SetStage(Stage.Done);
            RequestRetrainOnce(aborted: false); // every episode and manifest is written by now
        }

        /// <summary>
        /// The session's one retrain request. Dry-run sessions trigger it too (their dryrun_ participants are never trained on),
        /// so the dry run also checks this step.
        /// </summary>
        private void RequestRetrainOnce(bool aborted)
        {
            if (retrainHandled) return;
            retrainHandled = true;
            if (aborted && (!config.retrainOnAbort || completedEpisodes == 0))
            {
                Debug.Log($"[ParticipantSession] Session '{sessionId}' ended early after {completedEpisodes} completed episodes - no retrain " +
                          $"(retrainOnAbort is {(config.retrainOnAbort ? "on, but nothing was recorded" : "off")}).");
                return;
            }
            if (!config.retrainAfterSession)
            {
                Debug.Log($"[ParticipantSession] retrainAfterSession is off - no retrain after session '{sessionId}'.");
                return;
            }
            RetrainPipeline.RequestFullRetrain($"session {participantId} {(aborted ? "aborted" : "finished")}");
        }

        // ------------------------------------------------------------------
        // Setup
        // ------------------------------------------------------------------

        private string ValidateConfig()
        {
            if (config == null) return "No ParticipantSessionConfig assigned to ParticipantSession.";
            string[] layouts = config.LayoutScenePaths;
            if (layouts == null || layouts.Length == 0) return "The session config has no layout scenes.";
            var seen = new HashSet<string>();
            foreach (string path in layouts)
            {
                if (string.IsNullOrEmpty(path)) return "The session config has an empty layout scene slot.";
                if (!seen.Add(path)) return $"Layout scene '{path}' is listed twice in the session config.";
            }
            if (string.IsNullOrEmpty(config.PracticeScenePath)) return "The session config has no practice scene.";
            if (seen.Contains(config.PracticeScenePath)) return "The practice scene must not be one of the recorded layout scenes.";
            if (config.repetitions < 1) return "repetitions must be at least 1.";
            return null;
        }

        private IEnumerator LoadScene(string path, SessionSchedule.Trial? trial)
        {
            SetStage(Stage.Loading);
            pendingScenePath = path;
            pendingTrial = trial;
            sceneError = null;
            recorder = null;
            controller = null;

            AsyncOperation op;
#if UNITY_EDITOR
            // Loads by path without the scene having to be in the build settings.
            op = EditorSceneManager.LoadSceneAsyncInPlayMode(path, new LoadSceneParameters(LoadSceneMode.Single));
#else
            op = SceneManager.LoadSceneAsync(path, LoadSceneMode.Single);
#endif
            if (op == null)
            {
                Fail($"Could not load scene '{path}'.");
                yield break;
            }
            yield return op;
            yield return null; // scene Start() calls (TaskScript etc.)
            pendingScenePath = null;

            if (sceneError == null && recorder == null) sceneError = $"Scene '{path}' loaded, but OnSceneLoaded never configured it.";
            if (sceneError != null) Fail(sceneError);
        }

        /// <summary>Runs after the scene's Awake/OnEnable and before its Start, so the recorder is configured before AutoStartForTesting could fire.</summary>
        private void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            if (pendingScenePath == null || scene.path != pendingScenePath) return;
            try
            {
                ConfigureScene(scene);
            }
            catch (Exception ex)
            {
                sceneError = ex.Message;
                Debug.LogException(ex);
            }
        }

        private void ConfigureScene(Scene scene)
        {
            var recorders = new List<ParticipantRecorder>();
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                recorders.AddRange(root.GetComponentsInChildren<ParticipantRecorder>(true));
                if (root.GetComponentInChildren<EvalHarness>(true) != null || root.GetComponentInChildren<RecordRandomWalksHarness>(true) != null ||
                    root.GetComponentInChildren<DaggerHarness>(true) != null)
                {
                    throw new InvalidOperationException($"Scene '{scene.name}' contains a harness (EvalHarness/RecordRandomWalksHarness/DaggerHarness). " +
                        "Use the plain layout scene for human sessions.");
                }
            }
            if (recorders.Count != 1)
            {
                throw new InvalidOperationException($"Scene '{scene.name}' must contain exactly one ParticipantRecorder (the FPSController's), found {recorders.Count}.");
            }

            recorder = recorders[0];
            controller = recorder.GetComponent<FirstPersonController>();
            if (controller == null)
            {
                throw new InvalidOperationException($"The ParticipantRecorder in '{scene.name}' is not on an FPSController (no FirstPersonController).");
            }
            if (recorder.task == null)
            {
                throw new InvalidOperationException($"The ParticipantRecorder in '{scene.name}' has no TaskScript. Run Tools > Thesis > Prepare Scene on it.");
            }

            recorder.AutoStartForTesting = false;   // the session starts recording itself
            controller.enabled = false;             // frozen until the participant presses Enter

            if (pendingTrial.HasValue)
            {
                recorder.OutputFormat = RecordingOutputFormat.Binary;
                recorder.source = ZarrDataContract.EpisodeSources.Human;
                recorder.participantId = participantId;
                recorder.sessionId = sessionId;
                recorder.repetitionIndex = pendingTrial.Value.repetitionIndex;
                recorder.inputMode = ZarrDataContract.InputModeDesktop;
            }
        }

        // ------------------------------------------------------------------
        // Practice and trials
        // ------------------------------------------------------------------

        private IEnumerator RunPractice()
        {
            practiceTracker = new PoiProgressTracker(recorder.task);
            BeginMoving();
            SetStage(Stage.Running);

            while (!practiceTracker.IsFinished)
            {
                if (Input.GetKeyDown(config.abortTrialKey))
                {
                    Debug.Log("[ParticipantSession] Practice skipped by the experimenter.");
                    break;
                }
                if (config.dryRun && Input.GetKeyDown(config.dryRunTeleportKey)) TeleportTo(practiceTracker.CurrentTarget);
                practiceTracker.Tick(recorder.transform.position); // same body position ParticipantRecorder ticks with
                yield return null;
            }

            StopMoving();
            practiceTracker = null;
            trialDoneMessage = "Practice finished. Press Enter to start the recorded runs.";
            SetStage(Stage.TrialDone);
            yield return WaitForConfirm();
        }

        private IEnumerator RunTrial(SessionSchedule.Trial trial, Action<bool> onFinished)
        {
            BeginMoving();
            try
            {
                recorder.StartRecording();
            }
            catch (Exception ex)
            {
                StopMoving();
                Fail($"Recording refused to start: {ex.Message}");
                yield break;
            }
            SetStage(Stage.Running);

            bool aborted = false;
            while (recorder.IsRecording) // ParticipantRecorder stops itself at the end marker
            {
                if (Input.GetKeyDown(config.abortTrialKey))
                {
                    aborted = true;
                    recorder.StopRecording();
                    break;
                }
                if (config.dryRun && Input.GetKeyDown(config.dryRunTeleportKey)) TeleportTo(recorder.CurrentTarget);
                yield return null;
            }
            StopMoving();

            string episodeDir = recorder.CurrentEpisodeDirectory;
            ZarrDataContract.EpisodeManifest manifest = ReadManifest(episodeDir);
            if (aborted && !string.IsNullOrEmpty(episodeDir) && Directory.Exists(episodeDir))
            {
                // Out of zarr_recordings, so training never sees it; kept for the record.
                Directory.CreateDirectory(AbortedEpisodesDirectory);
                string target = Path.Combine(AbortedEpisodesDirectory, Path.GetFileName(episodeDir));
                Directory.Move(episodeDir, target);
                episodeDir = target;
                Debug.Log($"[ParticipantSession] Trial {trial.trialIndex + 1} aborted by the experimenter; episode moved to {target}. Repeating the trial.");
            }

            AppendLog(trial, aborted ? "aborted" : "completed", manifest, episodeDir);
            onFinished(!aborted);
        }

        private void BeginMoving()
        {
            controller.enabled = true; // MouseLook locks the cursor again
        }

        private void StopMoving()
        {
            if (controller != null) controller.enabled = false;
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        /// <summary>Dry run only: puts the body over the target so the tracker's arrival check fires.</summary>
        private void TeleportTo(Transform target)
        {
            if (target == null) return;
            var body = recorder.GetComponent<CharacterController>();
            Vector3 p = recorder.transform.position;
            if (body != null) body.enabled = false;
            recorder.transform.position = new Vector3(target.position.x, p.y, target.position.z);
            if (body != null) body.enabled = true;
        }

        // ------------------------------------------------------------------
        // Session log
        // ------------------------------------------------------------------

        private static ZarrDataContract.EpisodeManifest ReadManifest(string episodeDir)
        {
            if (string.IsNullOrEmpty(episodeDir)) return null;
            string path = Path.Combine(episodeDir, ZarrDataContract.ManifestFileName);
            if (!File.Exists(path)) return null;
            return JsonUtility.FromJson<ZarrDataContract.EpisodeManifest>(File.ReadAllText(path));
        }

        private void AppendLog(SessionSchedule.Trial trial, string status, ZarrDataContract.EpisodeManifest manifest, string episodeDir)
        {
            Directory.CreateDirectory(SessionLogDirectory);
            bool writeHeader = !File.Exists(SessionLogPath);
            var inv = CultureInfo.InvariantCulture;
            string[] fields =
            {
                DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss", inv),
                participantId,
                status,
                config.dryRun ? "1" : "0",
                participantNumber.ToString(inv),
                sessionId,
                latinRow.ToString(inv),
                string.Join("-", layoutOrder),
                trial.trialIndex.ToString(inv),
                trials.Count.ToString(inv),
                trial.orderPosition.ToString(inv),
                trial.layoutIndex.ToString(inv),
                manifest?.scene_name ?? Path.GetFileNameWithoutExtension(config.LayoutScenePaths[trial.layoutIndex]),
                manifest?.layout_typology ?? "",
                manifest != null ? manifest.layout_seed.ToString(inv) : "",
                (manifest?.repetition_index ?? trial.repetitionIndex).ToString(inv),
                manifest?.episode_id ?? "",
                manifest != null ? manifest.frame_count.ToString(inv) : "",
                manifest != null ? (manifest.end_time - manifest.start_time).ToString("F2", inv) : "",
                episodeDir ?? ""
            };

            var line = new StringBuilder();
            for (int i = 0; i < fields.Length; i++)
            {
                if (i > 0) line.Append(',');
                line.Append(CsvField(fields[i]));
            }
            File.AppendAllText(SessionLogPath, (writeHeader ? LogHeader + "\n" : "") + line + "\n");

            if (manifest == null)
            {
                Debug.LogError($"[ParticipantSession] Trial {trial.trialIndex + 1}: no manifest in '{episodeDir}' - logged without episode metadata.");
            }
            else if (manifest.participant_id != participantId || manifest.repetition_index != trial.repetitionIndex || manifest.source != ZarrDataContract.EpisodeSources.Human)
            {
                Debug.LogError($"[ParticipantSession] Trial {trial.trialIndex + 1}: manifest has participant '{manifest.participant_id}', repetition " +
                    $"{manifest.repetition_index}, source '{manifest.source}' - expected '{participantId}', {trial.repetitionIndex}, 'human'.");
            }
        }

        private static string CsvField(string value)
        {
            if (value.IndexOfAny(new[] { ',', '"', '\n' }) < 0) return value;
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        /// <summary>Completed episodes already logged for this ID (columns 1 and 2 are participant_id and status and never quoted).</summary>
        private int CountCompletedEpisodes(string id)
        {
            if (!File.Exists(SessionLogPath)) return 0;
            int count = 0;
            foreach (string row in File.ReadAllLines(SessionLogPath))
            {
                string[] cols = row.Split(',');
                if (cols.Length > 2 && cols[1] == id && cols[2] == "completed") count++;
            }
            return count;
        }

        // ------------------------------------------------------------------
        // UI (IMGUI: drawn on screen only, never into the recorder's capture camera)
        // ------------------------------------------------------------------

        private void SetStage(Stage next)
        {
            stage = next;
            stageShownAt = Time.unscaledTime;
            confirmed = false;
        }

        private IEnumerator WaitForConfirm()
        {
            confirmed = false;
            while (!confirmed) yield return null;
        }

        private void Fail(string message)
        {
            errorMessage = message;
            Debug.LogError($"[ParticipantSession] {message}");
            SetStage(Stage.Error);
            if (trials != null) RequestRetrainOnce(aborted: true);
        }

        private void OnGUI()
        {
            EnsureStyles();

            Event e = Event.current;
            bool enterPressed = e.type == EventType.KeyDown && (e.keyCode == KeyCode.Return || e.keyCode == KeyCode.KeypadEnter);
            bool canConfirm = Time.unscaledTime - stageShownAt > MinSecondsBeforeConfirm;

            switch (stage)
            {
                case Stage.EnterId:
                    DrawEnterId(enterPressed && canConfirm);
                    break;
                case Stage.Instructions:
                    DrawPanel("Instructions", config.instructions + "\n\nPress Enter to start the practice run.", enterPressed && canConfirm);
                    break;
                case Stage.Loading:
                    DrawPanel(null, "Loading...", false);
                    break;
                case Stage.Ready:
                    DrawPanel(currentTrial < 0 ? "Practice run (not recorded)" : $"Run {currentTrial + 1} of {trials.Count}",
                        "Walk to the POI sphere, then to the EXIT sphere.\n\nPress Enter to start.", enterPressed && canConfirm);
                    break;
                case Stage.Running:
                    DrawHud();
                    break;
                case Stage.TrialDone:
                    DrawPanel(null, trialDoneMessage, enterPressed && canConfirm);
                    break;
                case Stage.Done:
                    DrawPanel("Session complete", "Thank you for taking part!", false);
                    break;
                case Stage.Error:
                    DrawPanel("Session stopped", errorMessage + "\n\nFix the problem and restart Play mode.", false);
                    break;
            }
        }

        private void DrawEnterId(bool enterPressed)
        {
            string text = (config.dryRun ? "DRY RUN - episodes are recorded as dryrun_<ID>.\n\n" : "") + "Participant ID (e.g. P07):";
            if (!string.IsNullOrEmpty(idError)) text += "\n\n" + idError;
            Rect panel = PanelRect();
            GUI.Box(panel, GUIContent.none, boxStyle);
            GUILayout.BeginArea(Inset(panel));
            GUILayout.Label("Experimenter setup", textStyle);
            GUILayout.Space(10);
            GUILayout.Label(text, textStyle);
            GUI.SetNextControlName("participantId");
            idInput = GUILayout.TextField(idInput, 16, fieldStyle, GUILayout.Height(36));
            GUI.FocusControl("participantId");
            GUILayout.Space(10);
            bool clicked = GUILayout.Button("Start session", GUILayout.Height(36));
            GUILayout.EndArea();

            if (!(enterPressed || clicked)) return;
            string id = idInput.Trim();
            idError = SessionSchedule.ValidateParticipantId(id, config.participantIdPattern);
            if (idError == null && !config.allowExistingParticipant)
            {
                int existing = CountCompletedEpisodes(config.dryRun ? "dryrun_" + id : id);
                if (existing > 0)
                {
                    idError = $"{id} already has {existing} completed episodes in {Path.GetFileName(SessionLogPath)}. " +
                              "Use a new ID, or enable 'allowExistingParticipant' in the session config.";
                }
            }
            if (idError == null) confirmed = true;
        }

        private void DrawPanel(string title, string body, bool enterPressed)
        {
            Rect panel = PanelRect();
            GUI.Box(panel, GUIContent.none, boxStyle);
            GUILayout.BeginArea(Inset(panel));
            if (!string.IsNullOrEmpty(title))
            {
                GUILayout.Label(title, textStyle);
                GUILayout.Space(10);
            }
            GUILayout.Label(body, textStyle);
            GUILayout.EndArea();
            if (enterPressed) confirmed = true;
        }

        private void DrawHud()
        {
            string run = currentTrial < 0 ? "Practice (not recorded)" : $"Run {currentTrial + 1} of {trials.Count}";
            Transform target = currentTrial < 0 ? practiceTracker?.CurrentTarget : recorder?.CurrentTarget;
            string targetName = TargetName(target);
            string keys = $"[{config.abortTrialKey}] abort" + (config.dryRun ? $"  [{config.dryRunTeleportKey}] teleport" : "");
            GUI.Label(new Rect(16, 12, 600, 30), $"{run}  -  go to: {targetName}", hudStyle);
            GUI.Label(new Rect(16, 40, 600, 30), keys, hudStyle);

            // Label the current target sphere (POI and EXIT spheres look the same). The recorded goal vector gives the
            // model the same direction information, so the label does not give the human anything the agent lacks.
            Camera cam = Camera.main;
            if (target == null || cam == null) return;
            Vector3 sp = cam.WorldToScreenPoint(target.position);
            if (sp.z > 0f && sp.x >= 0f && sp.x <= Screen.width && sp.y >= 0f && sp.y <= Screen.height)
            {
                GUI.Label(new Rect(sp.x - 60f, Screen.height - sp.y - 50f, 120f, 30f), targetName, labelStyle);
            }
            else
            {
                Vector3 local = cam.transform.InverseTransformPoint(target.position);
                bool left = local.x < 0f;
                GUI.Label(new Rect(left ? 16f : Screen.width - 136f, Screen.height * 0.5f - 15f, 120f, 30f),
                    left ? "< " + targetName : targetName + " >", labelStyle);
            }
        }

        private string TargetName(Transform target)
        {
            if (target == null || recorder == null || recorder.task == null) return "-";
            foreach (GameObject end in recorder.task.end)
            {
                if (end != null && end.transform == target) return "EXIT";
            }
            return "POI";
        }

        private static Rect PanelRect()
        {
            float w = Mathf.Min(760f, Screen.width - 40f);
            float h = Mathf.Min(520f, Screen.height - 40f);
            return new Rect((Screen.width - w) * 0.5f, (Screen.height - h) * 0.5f, w, h);
        }

        private static Rect Inset(Rect r) => new Rect(r.x + 24f, r.y + 24f, r.width - 48f, r.height - 48f);

        private void EnsureStyles()
        {
            if (textStyle != null) return;
            boxStyle = new GUIStyle(GUI.skin.box);
            var bg = new Texture2D(1, 1);
            bg.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.85f));
            bg.Apply();
            boxStyle.normal.background = bg;
            textStyle = new GUIStyle(GUI.skin.label) { fontSize = 22, wordWrap = true };
            textStyle.normal.textColor = Color.white;
            fieldStyle = new GUIStyle(GUI.skin.textField) { fontSize = 22 };
            hudStyle = new GUIStyle(GUI.skin.label) { fontSize = 18 };
            hudStyle.normal.textColor = Color.white;
            labelStyle = new GUIStyle(GUI.skin.label) { fontSize = 20, fontStyle = FontStyle.Bold, alignment = TextAnchor.MiddleCenter };
            labelStyle.normal.textColor = Color.yellow;
        }
    }
}
