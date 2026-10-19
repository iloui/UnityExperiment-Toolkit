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
    /// Runs one desktop human recording session from the bootstrap scene (Tools > Imitation Learning Agent > 2 · Record Human Walks > Create Participant Session
    /// Scene): participant ID -> instructions -> untracked practice -> the layout scenes in the participant's
    /// Latin-square order. In each layout the participant walks the scene's HumanTaskSet training tasks in a counterbalanced
    /// order and then the test task last, each once (methodology 3.8.2). Every trial loads its scene on its own (Single mode)
    /// and moves the scene's Start/POI_0/End markers and the rig onto the trial's task;
    /// this object survives the loads, configures the scene's ParticipantRecorder in sceneLoaded (after Awake, before
    /// Start, so AutoStartForTesting never fires) and starts/stops recording itself. Each finished or aborted episode is
    /// appended to imitationagent_training/participant_sessions/session_log.csv. Nothing is set in the inspector per participant.
    /// The session never retrains the model: training on the human data is a separate step after all sessions.
    /// </summary>
    public class ParticipantSession : MonoBehaviour
    {
        public ParticipantSessionConfig config;

        private enum Stage { EnterId, Instructions, Loading, Ready, Running, TrialDone, Done, Error }

        private const float MinSecondsBeforeConfirm = 0.4f;
        private const string LogHeader =
            "logged_at,participant_id,status,dry_run,participant_number,session_id,latin_square_row,layout_order," +
            "trial_index,trial_count,order_position,layout_index,scene_name,layout_typology,layout_seed,repetition_index," +
            "episode_id,frame_count,duration_s,episode_dir," +
            "fold,task_id,task_role,task_order_position,task_order_latin_row,attempt_index,c_excluded,task_set_seed";

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
        private int fold;
        private readonly Dictionary<int, int> abortedAttempts = new Dictionary<int, int>(); // trial index -> aborted walks so far
        private HumanTaskSet taskSet;   // of the loaded layout scene

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
            fold = SessionSchedule.FoldOf(participantNumber);
            trials = SessionSchedule.BuildTrials(participantNumber, config.LayoutScenePaths.Length, config.trainingTasksPerLayout, out latinRow, out layoutOrder);
            Debug.Log($"[ParticipantSession] Session '{sessionId}': participant {participantId} (#{participantNumber}, fold {fold}), Latin-square row {latinRow}, " +
                $"layout order {string.Join("-", layoutOrder)}, {trials.Count} trials{(config.dryRun ? " (DRY RUN)" : "")}:\n" + DescribeTrials());

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

            // 4. Recorded trials; an aborted trial is repeated with the next attempt_index.
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
                else
                {
                    abortedAttempts[trial.trialIndex] = AttemptIndex(trial) + 1;
                }
                if (currentTrial < trials.Count)
                {
                    SetStage(Stage.TrialDone);
                    yield return WaitForConfirm();
                }
            }

            Debug.Log($"[ParticipantSession] Session '{sessionId}' complete: {trials.Count} episodes logged to {SessionLogPath}.");
            SetStage(Stage.Done);
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
            if (config.trainingTasksPerLayout < 1) return "trainingTasksPerLayout must be at least 1.";
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
            taskSet = null;

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
                // Disabled harnesses (component unticked or GameObject inactive) are fine: they never run.
                foreach (MonoBehaviour harness in root.GetComponentsInChildren<MonoBehaviour>(true))
                {
                    if (!(harness is EvalHarness || harness is RecordRandomWalksHarness || harness is DaggerHarness)) continue;
                    if (!harness.isActiveAndEnabled) continue;
                    throw new InvalidOperationException($"Scene '{scene.name}' has an enabled {harness.GetType().Name} on '{harness.name}'. " +
                        "Untick it (or deactivate its GameObject) and save the scene before a human session.");
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
                throw new InvalidOperationException($"The ParticipantRecorder in '{scene.name}' has no TaskScript. Run Tools > Scene Preparation > Prepare Scene on it.");
            }

            recorder.AutoStartForTesting = false;   // the session starts recording itself
            controller.enabled = false;             // frozen until the participant presses Enter

            if (pendingTrial.HasValue)
            {
                recorder.OutputFormat = RecordingOutputFormat.Binary;
                recorder.source = ZarrDataContract.EpisodeSources.Human;
                recorder.participantId = participantId;
                recorder.sessionId = sessionId;
                recorder.inputMode = ZarrDataContract.InputModeDesktop;
                ApplyTrialTask(scene, pendingTrial.Value);
            }
        }

        /// <summary>
        /// Moves the scene's single Start/POI_0/End onto the trial's task and the rig onto its start, and labels the episode.
        /// Refuses a scene without a HumanTaskSet, with another number of training tasks, or whose test task moved since the
        /// set was generated.
        /// </summary>
        private void ApplyTrialTask(Scene scene, SessionSchedule.Trial trial)
        {
            var sets = new List<HumanTaskSet>();
            foreach (GameObject root in scene.GetRootGameObjects()) sets.AddRange(root.GetComponentsInChildren<HumanTaskSet>(true));
            if (sets.Count != 1)
            {
                throw new InvalidOperationException($"Scene '{scene.name}' must contain exactly one HumanTaskSet, found {sets.Count}. " +
                    "Run Tools > Imitation Learning Agent > 2 · Record Human Walks > Generate Missing Human Tasks For Session Layouts.");
            }
            taskSet = sets[0];
            if (taskSet.TrainingCount != config.trainingTasksPerLayout)
            {
                throw new InvalidOperationException($"Scene '{scene.name}' has {taskSet.TrainingCount} human training tasks, the session expects {config.trainingTasksPerLayout}.");
            }
            TaskScript task = recorder.task;
            string stale = taskSet.CheckTestTaskUnchanged(task);
            if (stale != null)
            {
                throw new InvalidOperationException($"Scene '{scene.name}': {stale}. Regenerate the human tasks only if no participant has walked them yet.");
            }
            foreach (GameObject marker in new[] { task.start[0], task.pointsOfInterest[0], task.end[0] })
            {
                if (marker.isStatic) throw new InvalidOperationException($"Task marker '{marker.name}' in '{scene.name}' is static and can't be moved. Clear its Static flag.");
            }

            HumanTaskSet.TaskDef def = trial.IsTest ? taskSet.test : taskSet.training[trial.trainingTaskIndex];
            HumanTaskSet.ApplyToTask(task, def);
            taskSet.PlaceRig(recorder.transform, def);

            recorder.repetitionIndex = trial.taskOrderPosition; // walks of this layout before this one
            recorder.trialInfo = new ZarrDataContract.TrialInfo
            {
                task_id = def.id,
                task_role = trial.IsTest ? HumanTaskSet.RoleTest : HumanTaskSet.RoleTrain,
                fold = fold,
                c_excluded = !trial.IsTest && trial.trainingTaskIndex == taskSet.cExcludedIndex,
                task_order_position = trial.taskOrderPosition,
                layout_order_position = trial.orderPosition,
                trial_index = trial.trialIndex,
                attempt_index = AttemptIndex(trial),
                latin_square_row = latinRow,
                task_order_latin_row = trial.taskOrderLatinRow,
                task_set_version = taskSet.generatorVersion,
                task_set_seed = taskSet.generatorSeed
            };
            Debug.Log($"[ParticipantSession] Trial {trial.trialIndex + 1}/{trials.Count}: {scene.name} task '{def.id}' " +
                $"(position {trial.taskOrderPosition}, attempt {recorder.trialInfo.attempt_index}{(recorder.trialInfo.c_excluded ? ", C-excluded" : "")}).");
        }

        private int AttemptIndex(SessionSchedule.Trial trial)
        {
            return abortedAttempts.TryGetValue(trial.trialIndex, out int n) ? n : 0;
        }

        /// <summary>The session plan for the console: layout and task per trial.</summary>
        private string DescribeTrials()
        {
            var sb = new StringBuilder();
            foreach (SessionSchedule.Trial t in trials)
            {
                sb.Append($"  {t.trialIndex + 1,2}. {Path.GetFileNameWithoutExtension(config.LayoutScenePaths[t.layoutIndex])} " +
                          $"{(t.IsTest ? HumanTaskSet.TestTaskId : HumanTaskSet.TrainTaskId(t.trainingTaskIndex))}\n");
            }
            return sb.ToString();
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
            RotateLogWithOldColumns();
            bool writeHeader = !File.Exists(SessionLogPath);
            ZarrDataContract.TrialInfo info = recorder != null ? recorder.trialInfo : null;
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
                (manifest?.repetition_index ?? trial.taskOrderPosition).ToString(inv),
                manifest?.episode_id ?? "",
                manifest != null ? manifest.frame_count.ToString(inv) : "",
                manifest != null ? (manifest.end_time - manifest.start_time).ToString("F2", inv) : "",
                episodeDir ?? "",
                fold.ToString(inv),
                info?.task_id ?? "",
                info?.task_role ?? "",
                trial.taskOrderPosition.ToString(inv),
                trial.taskOrderLatinRow.ToString(inv),
                AttemptIndex(trial).ToString(inv),
                info != null && info.c_excluded ? "1" : "0",
                info != null ? info.task_set_seed.ToString(inv) : ""
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
            else if (manifest.participant_id != participantId || manifest.repetition_index != trial.taskOrderPosition ||
                     manifest.source != ZarrDataContract.EpisodeSources.Human || manifest.trial == null || info == null || manifest.trial.task_id != info.task_id)
            {
                Debug.LogError($"[ParticipantSession] Trial {trial.trialIndex + 1}: manifest has participant '{manifest.participant_id}', repetition " +
                    $"{manifest.repetition_index}, source '{manifest.source}', task '{manifest.trial?.task_id}' - expected '{participantId}', " +
                    $"{trial.taskOrderPosition}, 'human', '{info?.task_id}'.");
            }
        }

        /// <summary>A log written with another column set (older protocol) is renamed, so every row of the active log has the same columns.</summary>
        private void RotateLogWithOldColumns()
        {
            if (!File.Exists(SessionLogPath)) return;
            string firstLine;
            using (var reader = new StreamReader(SessionLogPath)) firstLine = reader.ReadLine();
            if (firstLine == LogHeader) return;
            string rotated = Path.Combine(SessionLogDirectory, $"{Path.GetFileNameWithoutExtension(SessionLogPath)}_oldcolumns_{DateTime.Now:yyyyMMdd_HHmmss}.csv");
            File.Move(SessionLogPath, rotated);
            Debug.LogWarning($"[ParticipantSession] {SessionLogPath} had another column set; moved to {rotated}.");
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
