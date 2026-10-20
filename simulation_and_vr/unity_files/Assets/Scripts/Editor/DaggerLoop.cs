using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Assets.Scripts.DataRecording;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Settings and progress of one DAgger loop run, kept in imitationagent_training/dagger/&lt;run&gt;/state.json. The file
    /// is the only state: entering Play mode reloads all scripts, so DaggerLoop re-reads it on every tick.
    /// </summary>
    [Serializable]
    internal class DaggerLoopState
    {
        // --- settings ---
        public string run_name = "dagger_v5";
        public string start_model = "";            // asset path of the round-1 policy
        public List<string> base_sources = new List<string> { "synthetic_rrw_v5" };
        public List<string> scenes = new List<string>();  // scene asset paths
        public int episodes_per_scene = 25;
        public int rounds = 3;
        public int evaluation_episodes = 10;
        public float hidden_goal_share = 0.6f;
        public float expert_segment_probability = 0.2f;
        public bool expert_walking_turns = true;
        public float expert_side_commit_ratio = 1.25f;
        public float dagger_weight = 0.3f;          // share of the synthetic batch drawn from this run's DAgger episodes
        public int epochs = 20;
        public int seed = 1;
        public float target_stop_share = 0.09f;
        public bool warm_start = false;
        public bool pause_after_round = true;
        public bool final_evaluation = true;

        // --- progress ---
        public string status = DaggerLoop.Idle;
        public string resume_status = "";
        public bool pause_requested;
        public int round = 1;
        public int scene_index;
        public bool final_eval_phase;
        public string current_model = "";
        public string current_checkpoint = "";
        public string request_id = "";
        public string requested_utc = "";
        public string message = "";
        public List<DaggerRoundRecord> history = new List<DaggerRoundRecord>();

        public DaggerLoopState CloneSettings()
        {
            var copy = JsonUtility.FromJson<DaggerLoopState>(JsonUtility.ToJson(this));
            copy.status = DaggerLoop.Idle;
            copy.resume_status = "";
            copy.pause_requested = false;
            copy.round = 1;
            copy.scene_index = 0;
            copy.final_eval_phase = false;
            copy.current_model = copy.current_checkpoint = copy.request_id = copy.requested_utc = copy.message = "";
            copy.history = new List<DaggerRoundRecord>();
            return copy;
        }
    }

    [Serializable]
    internal class DaggerRoundRecord
    {
        public int round;               // rounds + 1 = final evaluation
        public bool final_evaluation;
        public string model = "";       // the policy that drove this round
        public string trained_model = ""; // the model trained from this round's data
        public List<DaggerSceneRecord> scenes = new List<DaggerSceneRecord>();
    }

    [Serializable]
    internal class DaggerSceneRecord
    {
        public string scene;
        public int recorded, reached_goal, hidden_recorded, hidden_reached, stall_rescues, expert_side_commits;
        public float expert_executed_share;
        public int eval_episodes, eval_reached, eval_hidden, eval_hidden_reached;
        public bool aborted;
    }

    /// <summary>The run summary DaggerHarness writes (same field names).</summary>
    [Serializable]
    internal class DaggerHarnessSummary
    {
        public string scene;
        public int episodes_recorded, episodes_reached_goal, hidden_goal_recorded, hidden_goal_reached, stall_rescues, expert_side_commits;
        public float expert_executed_share;
        public int eval_episodes, eval_reached, eval_hidden, eval_hidden_reached;
        public bool aborted;
    }

    /// <summary>
    /// The DAgger loop (Ross et al. 2011) as an Editor state machine, controlled from DaggerLoopWindow
    /// (Tools > Imitation Learning Agent > 3 · Train Model > Improve Model with DAgger…). One round:
    ///   for every selected scene: open it, enter Play mode with a DaggerRunBootstrap request (the current policy drives,
    ///   the NavMesh expert labels; first evaluation_episodes policy-only episodes on a fixed task sequence), wait for the
    ///   harness's summary;
    ///   then train condition A (synthetic only) on base_sources + every DAgger round of this run so far, through
    ///   RetrainPipeline.RequestTraining, and continue with the new model.
    /// After the last round an optional evaluation-only pass measures the final model. Only synthetic data and condition A
    /// are used, so the DAgger models carry no participant's data into any fold.
    ///
    /// Each step is written to state.json before the next starts, so the loop survives domain reloads (every Play session),
    /// leaving Play mode by hand and Editor restarts: Resume continues at the last unfinished step.
    /// </summary>
    [InitializeOnLoad]
    internal static class DaggerLoop
    {
        public const string Idle = "idle";
        public const string CollectScene = "collect_scene";
        public const string EnteringPlay = "entering_play";
        public const string Playing = "playing";
        public const string Train = "train";
        public const string Training = "training";
        public const string Paused = "paused";
        public const string Done = "done";
        public const string Failed = "failed";
        public const string Aborted = "aborted";

        private const double TickSeconds = 0.5;
        private const double PlayStartTimeoutSeconds = 90;
        private const double TrainingLostGraceSeconds = 30;
        private const string LogTag = "[DAgger loop]";

        public static string TrainingDir => Path.GetFullPath(Path.Combine(Application.dataPath, "../../imitationagent_training"));
        public static string DaggerRoot => Path.Combine(TrainingDir, "dagger");
        private static string ActivePointerPath => Path.Combine(DaggerRoot, "active_run.txt");
        public static string RunDir(string run) => Path.Combine(DaggerRoot, run);
        private static string StatePath(string run) => Path.Combine(RunDir(run), "state.json");

        private static DateTime nextTick = DateTime.MinValue;
        private static DateTime trainingLastSeenRunning = DateTime.MinValue;

        static DaggerLoop()
        {
            EditorApplication.update -= Tick;
            EditorApplication.update += Tick;
        }

        // ------------------------------------------------------------------
        // State file
        // ------------------------------------------------------------------

        public static bool IsActive(DaggerLoopState s) =>
            s != null && (s.status == CollectScene || s.status == EnteringPlay || s.status == Playing || s.status == Train || s.status == Training);

        public static DaggerLoopState LoadActive()
        {
            try
            {
                if (!File.Exists(ActivePointerPath)) return null;
                string run = File.ReadAllText(ActivePointerPath).Trim();
                return run.Length > 0 && File.Exists(StatePath(run)) ? JsonUtility.FromJson<DaggerLoopState>(File.ReadAllText(StatePath(run))) : null;
            }
            catch (Exception)
            {
                return null;
            }
        }

        public static void Save(DaggerLoopState s)
        {
            Directory.CreateDirectory(RunDir(s.run_name));
            string path = StatePath(s.run_name);
            File.WriteAllText(path + ".tmp", JsonUtility.ToJson(s, true));
            if (File.Exists(path)) File.Delete(path);
            File.Move(path + ".tmp", path);
        }

        /// <summary>Starts a new run (the window has validated the settings).</summary>
        public static void Start(DaggerLoopState settings)
        {
            DaggerLoopState s = settings.CloneSettings();
            s.current_model = s.start_model;
            s.current_checkpoint = CheckpointFor(s.start_model) ?? "";
            s.status = CollectScene;
            s.message = "Starting round 1.";
            Save(s);
            Directory.CreateDirectory(DaggerRoot);
            File.WriteAllText(ActivePointerPath, s.run_name);
            Debug.Log($"{LogTag} Run '{s.run_name}' started: {s.rounds} rounds x {s.scenes.Count} scenes, start model {s.start_model}.");
        }

        /// <summary>Forgets the active run (its folder stays), so a new one can be configured.</summary>
        public static void Close()
        {
            DaggerLoopState s = LoadActive();
            if (IsActive(s)) return;
            if (File.Exists(ActivePointerPath)) File.Delete(ActivePointerPath);
        }

        public static void RequestPause()
        {
            DaggerLoopState s = LoadActive();
            if (!IsActive(s)) return;
            s.pause_requested = true;
            s.message = "Pausing after the current step...";
            Save(s);
        }

        public static void Resume()
        {
            DaggerLoopState s = LoadActive();
            if (s == null || (s.status != Paused && s.status != Failed)) return;
            s.status = string.IsNullOrEmpty(s.resume_status) ? CollectScene : s.resume_status;
            s.resume_status = "";
            s.pause_requested = false;
            s.message = "Resumed.";
            Save(s);
        }

        public static void Abort()
        {
            DaggerLoopState s = LoadActive();
            if (s == null) return;
            bool wasTraining = s.status == Training;
            s.status = Aborted;
            s.message = wasTraining
                ? "Aborted. The training process keeps running in the background and finishes on its own (Show Training Status)."
                : "Aborted.";
            Save(s);
            DaggerRunBootstrap.ClearRequest();
            if (EditorApplication.isPlaying) EditorApplication.isPlaying = false;
            Debug.Log($"{LogTag} Run '{s.run_name}' aborted.");
        }

        // ------------------------------------------------------------------
        // Tick
        // ------------------------------------------------------------------

        private static void Tick()
        {
            if (DateTime.UtcNow < nextTick) return;
            nextTick = DateTime.UtcNow.AddSeconds(TickSeconds);
            if (EditorApplication.isCompiling || EditorApplication.isUpdating) return;

            DaggerLoopState s = LoadActive();
            if (!IsActive(s)) return;
            try
            {
                switch (s.status)
                {
                    case CollectScene: BeginScene(s); break;
                    case EnteringPlay:
                    case Playing: WatchPlay(s); break;
                    case Train: BeginTraining(s); break;
                    case Training: WatchTraining(s); break;
                }
            }
            catch (Exception ex)
            {
                Fail(s, $"Unexpected error: {ex.Message}", s.status == Training ? Training : s.status == Playing || s.status == EnteringPlay ? CollectScene : s.status);
                Debug.LogException(ex);
            }
        }

        private static bool PauseIfRequested(DaggerLoopState s, string resumeAt)
        {
            if (!s.pause_requested) return false;
            s.status = Paused;
            s.resume_status = resumeAt;
            s.pause_requested = false;
            s.message = "Paused. Resume continues with " + Describe(resumeAt, s) + ".";
            Save(s);
            return true;
        }

        private static void Fail(DaggerLoopState s, string why, string resumeAt)
        {
            s.status = Failed;
            s.resume_status = resumeAt;
            s.message = why;
            Save(s);
            Debug.LogError($"{LogTag} {why}");
        }

        // --- collection -----------------------------------------------------

        private static void BeginScene(DaggerLoopState s)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode) return;
            if (PauseIfRequested(s, CollectScene)) return;

            string scenePath = s.scenes[s.scene_index];
            if (EditorSceneManager.GetActiveScene().path != scenePath)
            {
                if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                {
                    s.status = Paused;
                    s.resume_status = CollectScene;
                    s.message = "Paused: the open scene has unsaved changes. Save or discard them, then Resume.";
                    Save(s);
                    return;
                }
                EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
            }

            string sceneName = Path.GetFileNameWithoutExtension(scenePath);
            DaggerRoundRecord record = CurrentRecord(s);
            if (string.IsNullOrEmpty(record.model)) record.model = s.current_model;
            string summary = SummaryPath(s, sceneName);
            if (File.Exists(summary)) File.Delete(summary);

            var request = new DaggerRunBootstrap.PlayRequest
            {
                request_id = Guid.NewGuid().ToString("N"),
                scene_path = scenePath,
                model_asset = s.current_model,
                participant_id = ParticipantId(s, s.round),
                episodes = s.final_eval_phase ? 0 : s.episodes_per_scene,
                evaluation_episodes = s.evaluation_episodes,
                evaluation_seed = StableSeed($"{s.run_name}|eval|{sceneName}"), // same tasks every round
                seed = StableSeed($"{s.run_name}|{s.seed}|r{s.round}|{sceneName}"),
                hidden_goal_share = s.hidden_goal_share,
                expert_segment_probability = s.expert_segment_probability,
                expert_walking_turns = s.expert_walking_turns,
                expert_side_commit_ratio = s.expert_side_commit_ratio,
                summary_path = summary,
            };
            DaggerRunBootstrap.WriteRequest(request);
            s.request_id = request.request_id;
            s.requested_utc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
            s.status = EnteringPlay;
            s.message = $"{(s.final_eval_phase ? "Final evaluation" : $"Round {s.round}/{s.rounds}")}: {sceneName} " +
                        $"({s.scene_index + 1}/{s.scenes.Count}) - Play mode running.";
            Save(s);
            EditorApplication.EnterPlaymode();
        }

        private static void WatchPlay(DaggerLoopState s)
        {
            DaggerRunBootstrap.PlayRequest consumed = DaggerRunBootstrap.ReadConsumed();
            bool ours = consumed != null && consumed.request_id == s.request_id;
            if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
            {
                if (ours && s.status == EnteringPlay)
                {
                    s.status = Playing;
                    Save(s);
                }
                return;
            }

            if (!ours)
            {
                DaggerRunBootstrap.PlayRequest pending = DaggerRunBootstrap.ReadPending();
                double waited = (DateTime.UtcNow - ParseUtc(s.requested_utc)).TotalSeconds;
                if (pending != null && pending.request_id == s.request_id && waited < PlayStartTimeoutSeconds) return;
                DaggerRunBootstrap.ClearRequest();
                Fail(s, "Play mode did not start, or the run request was not picked up (compile errors in the Console?). Resume tries this scene again.", CollectScene);
                return;
            }

            // Play mode has ended after our harness started.
            string sceneName = Path.GetFileNameWithoutExtension(s.scenes[s.scene_index]);
            string summaryPath = SummaryPath(s, sceneName);
            DaggerRunBootstrap.ClearRequest();
            DaggerHarnessSummary summary = null;
            try { if (File.Exists(summaryPath)) summary = JsonUtility.FromJson<DaggerHarnessSummary>(File.ReadAllText(summaryPath)); }
            catch (Exception) { summary = null; }
            if (summary == null)
            {
                s.status = Paused;
                s.resume_status = CollectScene;
                s.message = $"Play mode ended before the DAgger run in {sceneName} finished (stopped by hand?). Resume repeats this scene; " +
                            $"episodes already recorded as '{ParticipantId(s, s.round)}' stay in zarr_recordings.";
                Save(s);
                return;
            }

            DaggerRoundRecord record = CurrentRecord(s);
            record.scenes.RemoveAll(r => r.scene == sceneName);
            record.scenes.Add(new DaggerSceneRecord
            {
                scene = sceneName,
                recorded = summary.episodes_recorded,
                reached_goal = summary.episodes_reached_goal,
                hidden_recorded = summary.hidden_goal_recorded,
                hidden_reached = summary.hidden_goal_reached,
                stall_rescues = summary.stall_rescues,
                expert_side_commits = summary.expert_side_commits,
                expert_executed_share = summary.expert_executed_share,
                eval_episodes = summary.eval_episodes,
                eval_reached = summary.eval_reached,
                eval_hidden = summary.eval_hidden,
                eval_hidden_reached = summary.eval_hidden_reached,
                aborted = summary.aborted,
            });
            if (summary.aborted)
            {
                Fail(s, $"The DaggerHarness in {sceneName} aborted (see its error in the Console). Resume repeats this scene.", CollectScene);
                return;
            }

            s.scene_index++;
            if (s.scene_index < s.scenes.Count)
            {
                s.status = CollectScene;
            }
            else
            {
                s.scene_index = 0;
                if (s.final_eval_phase)
                {
                    s.status = Done;
                    s.message = "Done. The final model is " + s.current_model + ".";
                    Debug.Log($"{LogTag} Run '{s.run_name}' done: {s.current_model}.");
                }
                else
                {
                    s.status = Train;
                }
            }
            Save(s);
        }

        // --- training -------------------------------------------------------

        private static void BeginTraining(DaggerLoopState s)
        {
            if (PauseIfRequested(s, Train)) return;
            if (RetrainPipeline.IsRunning)
            {
                if (!s.message.StartsWith("Waiting")) { s.message = "Waiting for another training run to finish..."; Save(s); }
                return;
            }

            string modelName = $"ImitationAgent_A_{s.run_name}_r{s.round}";
            string outputOnnx = Path.Combine(TrainingDir, "imitation_agent_model", "dagger", s.run_name, modelName + ".onnx");
            string resultPath = TrainResultPath(s, s.round);
            if (File.Exists(resultPath)) File.Delete(resultPath);

            string daggerPattern = $"synthetic_dagger_{s.run_name}_r*"; // every round of this run so far
            var args = new List<string> { "--condition", "A", "--recordings_dir", "../zarr_recordings", "--synthetic_sources" };
            args.AddRange(s.base_sources);
            args.Add(daggerPattern);
            args.Add("--source_weights");
            float baseShare = (1f - s.dagger_weight) / Mathf.Max(1, s.base_sources.Count);
            foreach (string src in s.base_sources) args.Add($"{src}={baseShare.ToString("0.####", CultureInfo.InvariantCulture)}");
            args.Add($"{daggerPattern}={s.dagger_weight.ToString("0.####", CultureInfo.InvariantCulture)}");
            args.AddRange(new[]
            {
                "--epochs", s.epochs.ToString(CultureInfo.InvariantCulture),
                "--seed", s.seed.ToString(CultureInfo.InvariantCulture),
                "--target_stop_share", s.target_stop_share.ToString("0.####", CultureInfo.InvariantCulture),
                "--output_onnx", outputOnnx,
                "--reason", $"DAgger {s.run_name} round {s.round}",
            });
            if (s.warm_start && !string.IsNullOrEmpty(s.current_checkpoint) && File.Exists(s.current_checkpoint))
            {
                args.Add("--init_from");
                args.Add(s.current_checkpoint);
            }

            var job = new RetrainPipeline.TrainingJob
            {
                Arguments = args,
                OutputOnnx = outputOnnx,
                Reason = $"DAgger {s.run_name} round {s.round}",
                AssetsSubdir = "Dagger/" + s.run_name,
                InstallAsLatest = false,
                ResultPath = resultPath,
            };
            if (!RetrainPipeline.RequestTraining(job))
            {
                Fail(s, "Training could not be started (see the [Retrain] message in the Console). Resume tries again.", Train);
                return;
            }
            trainingLastSeenRunning = DateTime.UtcNow;
            s.status = Training;
            s.message = $"Round {s.round}/{s.rounds}: training {modelName} (log in the Console with the [Retrain] prefix).";
            Save(s);
        }

        private static void WatchTraining(DaggerLoopState s)
        {
            string resultPath = TrainResultPath(s, s.round);
            if (!File.Exists(resultPath))
            {
                if (RetrainPipeline.IsRunning) { trainingLastSeenRunning = DateTime.UtcNow; return; }
                if (trainingLastSeenRunning == DateTime.MinValue) trainingLastSeenRunning = DateTime.UtcNow; // after a reload
                if ((DateTime.UtcNow - trainingLastSeenRunning).TotalSeconds < TrainingLostGraceSeconds) return;
                Fail(s, "The training process ended without a result (killed?). Resume starts this round's training again.", Train);
                return;
            }

            RetrainPipeline.TrainingResult result;
            try { result = JsonUtility.FromJson<RetrainPipeline.TrainingResult>(File.ReadAllText(resultPath)); }
            catch (Exception) { return; }
            if (result == null || !result.ok)
            {
                Fail(s, $"Training round {s.round} failed (exit code {result?.exit_code}); log: {result?.log_path}. Resume trains again.", Train);
                return;
            }

            CurrentRecord(s).trained_model = result.model_asset;
            s.current_model = result.model_asset;
            s.current_checkpoint = Path.ChangeExtension(result.output_onnx, ".pt");
            Debug.Log($"{LogTag} Round {s.round} trained: {result.model_asset}.");
            int finished = s.round;
            s.round++;
            s.scene_index = 0;
            if (s.round > s.rounds)
            {
                if (s.final_evaluation && s.evaluation_episodes > 0)
                {
                    s.final_eval_phase = true;
                    s.status = CollectScene;
                    s.message = "All rounds trained - final evaluation.";
                }
                else
                {
                    s.status = Done;
                    s.message = "Done. The final model is " + s.current_model + ".";
                }
            }
            else
            {
                s.status = CollectScene;
                s.message = $"Round {finished} done.";
            }
            if (s.status == CollectScene && s.pause_after_round && !s.final_eval_phase)
            {
                s.status = Paused;
                s.resume_status = CollectScene;
                s.message = $"Round {finished} done ({result.model_asset}). Check the table, then Resume for round {s.round}.";
            }
            Save(s);
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        public static string ParticipantId(DaggerLoopState s, int round) => $"synthetic_dagger_{s.run_name}_r{round}";

        private static string SummaryPath(DaggerLoopState s, string sceneName) =>
            Path.Combine(RunDir(s.run_name), s.final_eval_phase ? $"final_{sceneName}.json" : $"r{s.round}_{sceneName}.json");

        private static string TrainResultPath(DaggerLoopState s, int round) => Path.Combine(RunDir(s.run_name), $"train_r{round}.json");

        private static DaggerRoundRecord CurrentRecord(DaggerLoopState s)
        {
            DaggerRoundRecord r = s.history.FirstOrDefault(h => h.round == s.round);
            if (r == null)
            {
                r = new DaggerRoundRecord { round = s.round, final_evaluation = s.final_eval_phase };
                s.history.Add(r);
            }
            return r;
        }

        /// <summary>The .pt checkpoint written next to a model by train_2DCNN_discrete.py, if one exists in imitation_agent_model.</summary>
        public static string CheckpointFor(string modelAssetPath)
        {
            if (string.IsNullOrEmpty(modelAssetPath)) return null;
            string name = Path.GetFileNameWithoutExtension(modelAssetPath) + ".pt";
            string root = Path.Combine(TrainingDir, "imitation_agent_model");
            if (!Directory.Exists(root)) return null;
            return Directory.GetFiles(root, name, SearchOption.AllDirectories).FirstOrDefault();
        }

        private static string Describe(string status, DaggerLoopState s) =>
            status == Train ? $"training round {s.round}" : $"{(s.final_eval_phase ? "the final evaluation" : $"round {s.round}")}, scene {s.scene_index + 1}";

        /// <summary>FNV-1a: the same seed for the same text in every Editor session (string.GetHashCode is not guaranteed to be).</summary>
        private static int StableSeed(string text)
        {
            unchecked
            {
                uint h = 2166136261;
                foreach (char c in text) { h ^= c; h *= 16777619; }
                return (int)(h & 0x7fffffff) | 1;
            }
        }

        private static DateTime ParseUtc(string iso) =>
            DateTime.TryParse(iso, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out DateTime t) ? t.ToUniversalTime() : DateTime.MinValue;
    }
}
