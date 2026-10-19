using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using UnityEngine;
using UnityEngine.AI;
using Unity.Sentis;
using Agent_Based_Modelling;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Closed-loop evaluation of the open layout scene (methodology 3.8 / 3.9). One Play session runs a queue of entries -
    /// the shortest-path reference and every imitation model - one after another on the scene's own hand-placed
    /// Start -> POI -> End task. The markers are never moved except for the start jitter below (training used randomized
    /// tasks, evaluation uses the one fixed test task per layout). randomizeMarkersPerConfiguration still exists for other
    /// uses, but is off and not part of the thesis evaluation.
    ///
    /// Per entry, configurationCount episodes are run. Configuration i gets the same seeded start jitter (position within
    /// startJitterPositionMeters, heading within +-startJitterHeadingDegrees) for every entry, so all models and the
    /// shortest-path agent walk the same set of starts. The spawn heading is along the NavMesh path to the first target
    /// (PoiProgressTracker.StartYawAlongPath), as every participant started, plus the heading jitter.
    ///
    /// Output: SimulationLogs/EvalQueue_<scene>_<queueName>/EvalRun_<entry>/ with run_config.json, episodes.csv and one
    /// trajectory file per episode (ZarrDataContract.Trajectory, 0.1 s sim-time grid), plus run_complete.json once the entry
    /// is done. Pressing Play again with the same queueName skips completed entries and renames an unfinished one to
    /// *_incomplete_<time>, so an interrupted queue resumes. queue_status.json in the queue folder shows progress and an
    /// estimate of the time left, and every finished episode prints one console line.
    ///
    /// Condition, fold, seed and held-out participants of a model come from its training sidecar (.json next to the .onnx),
    /// never from typing. All metrics are computed downstream from the trajectory files (eval_tools); the per-episode
    /// columns here (path length, stalls, collisions, ...) are diagnostics.
    ///
    /// Runs unattended in the background: the harness turns on Application.runInBackground and turns off vSync. Episode
    /// time is simulated time (sum of Time.deltaTime, frame step capped at the 0.1 s tick), so a throttled or slow editor
    /// only takes longer; it does not change results.
    /// </summary>
    public class EvalHarness : MonoBehaviour
    {
        public enum AgentKind { Imitation, ShortestPath }

        [Serializable]
        public class QueueEntry
        {
            public AgentKind agentKind = AgentKind.Imitation;
            [Tooltip("Imitation only: the .onnx model.")]
            public ModelAsset modelAsset;
            [Tooltip("Imitation only: the training sidecar (<model>.json, written next to the .onnx by the training script). " +
                     "Condition, fold, seed and held-out participants are read from it.")]
            public TextAsset sidecar;

            public string Label => agentKind == AgentKind.ShortestPath ? "ShortestPath" : (modelAsset != null ? modelAsset.name : "<missing model>");
        }

        /// <summary>The fields of the training sidecar the harness uses (train_2DCNN_discrete.py writes many more).</summary>
        [Serializable]
        private class SidecarInfo
        {
            public string condition;
            public int seed;
            public int fold;
            public string[] heldout_participants;
            public string splits_design;
        }

        [Header("Task")]
        [Tooltip("The scene's TaskScript (Environment_Root/Task) with the hand-placed Start/POI_0/End markers.")]
        public TaskScript task;

        [Header("Queue")]
        [Tooltip("Run in this order. Tools > Imitation Learning Agent > 5 · Evaluate Design > Set Up Evaluation Harness fills it: ShortestPath, then every " +
                 "ImitationAgent_<A|B|C>_proprio2* model in Assets/ImitationModel with its sidecar.")]
        public List<QueueEntry> queue = new List<QueueEntry>();
        [Tooltip("Names the queue folder (EvalQueue_<scene>_<queueName>). Keep it the same to resume an interrupted queue; " +
                 "change it to start a fresh one.")]
        public string queueName = "final";
        [Tooltip("Seeds the per-configuration start jitter and the stochastic walk. Fixed, so every entry and every rerun gets " +
                 "the same starts.")]
        public int baseSeed = 20261005;
        [Tooltip("Leave Play mode when the queue is done (Editor only).")]
        public bool exitPlayModeWhenDone = true;

        [Header("Shortest-Path Reference (agentKind = ShortestPath)")]
        [Tooltip("m/s. Default = the recording body's walk speed (humans, bot and ImitationAgent all walk at 1.3 m/s).")]
        public float shortestPathSpeedMps = ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond;
        [Tooltip("Meters. Default = the recording body's radius. Wall clearance comes from the NavMesh bake radius (logged as navmesh_bake_agent_radius), not from this.")]
        public float shortestPathRadiusM = ZarrDataContract.RecordingBody.Radius;
        [Tooltip("m/s^2. High, so the reference is at full speed almost immediately.")]
        public float shortestPathAccelerationMps2 = 50f;
        [Tooltip("deg/s. 120 = the largest discrete yaw bin (12 deg per 0.1 s sample).")]
        public float shortestPathAngularSpeedDegPerS = 120f;

        [Header("Episodes (identical for every entry)")]
        [Tooltip("Episodes per entry (methodology 3.8.1: 20).")]
        [Min(1)] public int configurationCount = 20;
        [Tooltip("Time.timeScale during the run. The frame step is capped at the 0.1 s tick, so values above what the frame rate " +
                 "allows only stop speeding the run up. Keep it the same for all thesis runs: 3.")]
        [Min(0.01f)] public float simulationSpeed = 3f;
        [Tooltip("SIMULATED seconds. An episode that has not reached the End by then is a timeout. Canonical: 90 (about 2.5x the " +
                 "human median of 36 s on the test tasks).")]
        public float episodeTimeoutSeconds = 90f;
        [Tooltip("Meters. Seeded start-position jitter radius (NavMesh-valid, with a complete path to the End). Canonical: 0.5.")]
        public float startJitterPositionMeters = 0.5f;
        [Tooltip("Degrees. Seeded heading jitter (+-) added to the spawn heading along the NavMesh path. Canonical: 15.")]
        public float startJitterHeadingDegrees = 15f;
        [Tooltip("If true (default), ImitationAgent's deadlock nudge is off, so a stuck agent times out instead of being teleported.")]
        public bool disableDeadlockNudge = true;
        [Tooltip("Imitation only: ImitationAgent.stochasticWalk (walk key drawn with probability P(walk), seeded per configuration).")]
        public bool imitationStochasticWalk = true;
        [Tooltip("Imitation only: ImitationAgent.minWalkHoldSeconds, the shortest walk-key press (sim s).")]
        [Min(0f)] public float imitationMinWalkHoldSeconds = 1.5f;

        [Header("Random Markers (NOT used for the thesis evaluation)")]
        [Tooltip("Off (default and thesis setting): every episode uses the scene's hand-placed Start/POI/End, with only the start " +
                 "jitter above. On: every configuration moves start, POI(s) and end to seeded random NavMesh positions instead " +
                 "(start jitter is then ignored) - a generalisation test, not the methodology's evaluation.")]
        public bool randomizeMarkersPerConfiguration = false;
        [Tooltip("Random markers only: minimum spacing (m) between all randomized marker positions.")]
        public float minPositionSpacingMeters = 3f;
        [Tooltip("Random markers only: radius (m) around a random candidate to search for the NavMesh.")]
        public float navMeshSampleRadius = 2f;
        [Tooltip("Random markers only: candidates tried per marker before the configuration is skipped.")]
        public int maxPlacementAttempts = 30;

        [Header("Episode Validity / Diagnostics")]
        [Tooltip("m/s. Mean speed or fastest 1-sim-second window above this marks the episode invalid_speed (teleport or broken timing).")]
        public float maxPlausibleSpeedMps = 2.5f;
        [Tooltip("Simulated seconds per window used for stall detection and the fastest-window speed check.")]
        public float metricWindowSeconds = 1f;
        [Tooltip("Meters. Moving less than this over one metric window counts towards stall_seconds.")]
        public float stallDisplacementMeters = 0.1f;
        [Tooltip("Sim seconds. A stall must last at least this long to count as one stall event.")]
        public float stallEventMinSeconds = 2f;
        [Tooltip("Sim seconds. collision_events counts a new event only after this long without a raw collision increment.")]
        public float collisionDebounceSeconds = 1f;
        [Tooltip("REAL seconds to wait for a spawned agent's first Update (model loading) before logging agent_never_started. " +
                 "Generous, because a backgrounded editor can run at a few frames per second.")]
        public float agentStartTimeoutRealSeconds = 120f;

        [Header("Output")]
        [Tooltip("Leave empty for <project root>/SimulationLogs.")]
        public string outputRootFolder = "";
        [Tooltip("Imitation only: also write modeldebug_config<C>_agent0.csv (model inputs/outputs per tick) for stall analysis.")]
        public bool logModelDebugTicks = true;
        [Tooltip("Save AnalyticsManager's heatmap PNG every Nth episode; 0 = never (default). It is a slow main-thread render; " +
                 "the thesis figures come from eval_tools instead.")]
        public int heatmapEveryNthEpisode = 0;
        [Tooltip("Trajectory points are subsampled to at most this many before the heatmap export (its cost is O(n^2)).")]
        public int maxHeatmapTrajectoryPoints = 300;
        [Tooltip("Color of the spawned agent's capsule (visible in the Scene/Game view only; the agent's own camera sits inside it).")]
        public Color agentColor = new Color(1f, 0.549f, 0f);

        private const string HarnessVersion = "7";
        private const string SpawnHeadingRule = "navmesh_path_first_corner_plus_jitter";
        private static readonly string TrajectoryCsvHeader = string.Join(ZarrDataContract.Trajectory.CsvDelimiter, ZarrDataContract.Trajectory.Columns);
        private static readonly string ModelDebugCsvHeader = string.Join(ZarrDataContract.Trajectory.CsvDelimiter, new[]
        {
            "t", "goal_x", "goal_y", "goal_z", "goal_dist_m", "blocked",
            "raw_fwd", "raw_yaw_deg_per_sample", "raw_pitch", "applied_fwd_mps", "applied_yaw_deg_per_s",
            "proprio_speed", "proprio_prev_speed", "proprio_yaw", "proprio_turn", "proprio_available"
        });

        // Queue state
        private string queueFolderPath;
        private string sceneName;
        private int entryIndex = -1;
        private QueueEntry entry;
        private SidecarInfo entrySidecar;
        private int[] configSeeds;
        private bool harnessFinished = false;
        private string queueStartedAt;
        private float queueStartRealtime;
        private int episodesThisSession = 0;
        private float realSecondsThisSession = 0f;
        private readonly List<string> entryResults = new List<string>();

        // Current entry (run) state
        private string runFolderPath;
        private StreamWriter csvWriter;
        private int episodeIndex = 0;
        private int entrySuccesses = 0;
        private int entryTimeouts = 0;
        private int entryOther = 0;
        private float entryStartRealtime;
        private string modelSha256 = "";
        private string modelFileMtime = "";

        // Saved engine settings
        private float previousMaximumDeltaTime = 0f;
        private bool previousRunInBackground;
        private int previousVSyncCount;
        private int previousTargetFrameRate;
        private bool engineSettingsChanged = false;

        // Random markers (off for the thesis evaluation)
        private Vector3[] navMeshVertices;
        private int consecutivePlacementFailures = 0;
        private const int MaxConsecutivePlacementFailures = 10;

        // Current configuration
        private Vector3[] baseStartPositions;
        private int currentConfigId = -1;
        private int currentConfigSeed = 0;
        private int currentConfigAgentsSpawned = 0;
        private int currentConfigAgentsCompleted = 0;
        private float currentConfigStartOffsetM = 0f;
        private float currentConfigHeadingJitterDeg = 0f;
        private List<Vector3> currentConfigPoiPositions;
        private Vector3 currentConfigEndPos;

        private class ActiveEpisode
        {
            public GameObject agentGO;
            public IEvalAgent agent;
            public ImitationAgent imitation; // null for ShortestPath
            public float startRealtime;
            public int configId;
            public int agentId;
            public int seed;
            public List<Vector3> poiPositions;
            public Vector3 endPos;
            public StreamWriter trajWriter;
            public StreamWriter modelDebugWriter;
            public string trajFileName;

            // Harness-side clock and diagnostics, all from the agent's own trajectory and Time.deltaTime (simulated time).
            public bool started;
            public int startFrame;
            public float simElapsed;
            public int processedPoints;
            public float pathLengthM;
            public float maxStepM;
            public float windowStartSim;
            public float windowStartPath;
            public Vector3 windowStartPos;
            public float maxWindowSpeedMps;
            public float stallSeconds;
            public bool firstCollisionLogged;
            public float simAtFirstCollision;

            public readonly List<float> poiTimes = new List<float>();
            public int lastRawCollisions;
            public float lastCollisionSim = -1000f;
            public int collisionEvents;
            public float stallRunSeconds;
            public Vector3 stallRunPos;
            public bool stallRunCounted;
            public int stallEvents;
            public float longestStallSeconds;
            public Vector3 longestStallPos;
            public float startOffsetM;
            public float headingJitterDeg;
            public float spawnYawDeg;

            public float nextTickSim;
            public float prevTickSim;
            public Vector3 prevTickEyePos;
            public float prevTickHeading;
        }
        private readonly List<ActiveEpisode> activeEpisodes = new List<ActiveEpisode>();

        // OnEnable, not Awake: Awake also runs on a disabled harness, which would switch off the recorder of a human
        // session in the same scene. OnEnable still runs before every Start in the scene.
        void OnEnable()
        {
            // A layout scene's human FPSController has AutoStartForTesting on - without this it would record (and
            // train on) a motionless episode during every eval run.
            RecordingRig.DisableSceneRecorders("EvalHarness");
        }

        void Start()
        {
            sceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            if (!ValidateScene() || !ValidateQueue())
            {
                enabled = false;
                return;
            }

            baseStartPositions = new Vector3[task.start.Length];
            for (int i = 0; i < baseStartPositions.Length; i++) baseStartPositions[i] = task.start[i].transform.position;

            // One seed per configuration, from baseSeed only: configuration i is the same for every entry and every rerun.
            var seedRng = new System.Random(baseSeed);
            configSeeds = new int[configurationCount];
            for (int i = 0; i < configSeeds.Length; i++) configSeeds[i] = seedRng.Next();

            ApplyEngineSettings();
            SetupQueueFolder();
            queueStartedAt = DateTime.Now.ToString("o", CultureInfo.InvariantCulture);
            queueStartRealtime = Time.realtimeSinceStartup;
            StartNextEntry();
        }

        // ------------------------------------------------------------------
        // Validation
        // ------------------------------------------------------------------

        /// <summary>Scene task: one start, the POI(s), one end, all on the NavMesh and connected. Nothing is moved.</summary>
        private bool ValidateScene()
        {
            if (task == null)
            {
                Debug.LogError("[EvalHarness] 'task' is not assigned. Run Tools > Imitation Learning Agent > 5 · Evaluate Design > Set Up Evaluation Harness. Refusing to start.");
                return false;
            }
            navMeshVertices = NavMesh.CalculateTriangulation().vertices;
            if (navMeshVertices == null || navMeshVertices.Length == 0)
            {
                Debug.LogError("[EvalHarness] The scene has no baked NavMesh. Refusing to start.");
                return false;
            }
            if (!HasMarkers(task.start) || !HasMarkers(task.end))
            {
                Debug.LogError("[EvalHarness] task.start / task.end are empty or contain null markers. Refusing to start.");
                return false;
            }
            if (randomizeMarkersPerConfiguration)
            {
                Debug.LogWarning("[EvalHarness] randomizeMarkersPerConfiguration is ON: markers are re-placed randomly every configuration. " +
                    "This is NOT the thesis evaluation (fixed hand-placed test task).");
                if (task.numberOfNeeds > 0 && !HasMarkers(task.pointsOfInterest))
                {
                    Debug.LogError("[EvalHarness] task.numberOfNeeds > 0 but the task has no POI markers. Refusing to start.");
                    return false;
                }
                return true;
            }
            if (task.start.Length != 1 || task.end.Length != 1)
            {
                Debug.LogError($"[EvalHarness] The evaluation task needs exactly one start and one end marker (found {task.start.Length} and {task.end.Length}). Refusing to start.");
                return false;
            }
            if (task.numberOfNeeds > 0 && (!HasMarkers(task.pointsOfInterest) || task.numberOfNeeds > task.pointsOfInterest.Length))
            {
                Debug.LogError($"[EvalHarness] task.numberOfNeeds={task.numberOfNeeds} but the task has too few POI markers. Refusing to start.");
                return false;
            }
            if (task.numberOfNeeds == 0) Debug.LogWarning("[EvalHarness] task.numberOfNeeds is 0 - this task has no POI (start -> end only).");

            GameObject startMarker = task.start[0];
            if (!NavMesh.SamplePosition(startMarker.transform.position, out NavMeshHit startHit, 2f, NavMesh.AllAreas))
            {
                Debug.LogError($"[EvalHarness] Start marker at {startMarker.transform.position} is not within 2 m of the NavMesh. Refusing to start.");
                return false;
            }
            var targets = new List<GameObject>();
            if (task.numberOfNeeds > 0) targets.AddRange(task.pointsOfInterest);
            targets.AddRange(task.end);
            foreach (var target in targets)
            {
                if (!HasCompletePath(startHit.position, target.transform.position))
                {
                    Debug.LogError($"[EvalHarness] No complete NavMesh path from the start to '{target.name}' at {target.transform.position}. Refusing to start.");
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Every entry is usable before anything runs: imitation entries have a model and the matching sidecar, the sidecar's
        /// condition is A/B/C, B and C name their fold and held-out participants, and labels are unique (one folder each).
        /// </summary>
        private bool ValidateQueue()
        {
            if (queue == null || queue.Count == 0)
            {
                Debug.LogError("[EvalHarness] The queue is empty. Run Tools > Imitation Learning Agent > 5 · Evaluate Design > Set Up Evaluation Harness. Refusing to start.");
                return false;
            }
            if (string.IsNullOrWhiteSpace(queueName) || queueName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                Debug.LogError($"[EvalHarness] queueName '{queueName}' is empty or not a valid folder name. Refusing to start.");
                return false;
            }

            var labels = new HashSet<string>();
            var problems = new List<string>();
            for (int i = 0; i < queue.Count; i++)
            {
                QueueEntry e = queue[i];
                if (e == null) { problems.Add($"entry {i} is empty"); continue; }
                if (!labels.Add(e.Label)) problems.Add($"entry {i} '{e.Label}' appears twice");
                if (e.agentKind != AgentKind.Imitation) continue;

                if (e.modelAsset == null) { problems.Add($"entry {i} has no model"); continue; }
                if (e.sidecar == null) { problems.Add($"'{e.Label}' has no sidecar (.json)"); continue; }
                if (e.sidecar.name != e.modelAsset.name) problems.Add($"'{e.Label}' is paired with sidecar '{e.sidecar.name}'");
                if (!TryReadSidecar(e, out SidecarInfo info, out string error)) { problems.Add($"'{e.Label}': {error}"); continue; }
                if (info.condition != "A" && info.condition != "B" && info.condition != "C")
                    problems.Add($"'{e.Label}': sidecar condition '{info.condition}' is not A, B or C");
                if ((info.condition == "B" || info.condition == "C") &&
                    (info.fold < 1 || info.heldout_participants == null || info.heldout_participants.Length == 0))
                    problems.Add($"'{e.Label}': condition {info.condition} sidecar has no fold / held-out participants");
            }
            if (problems.Count > 0)
            {
                Debug.LogError("[EvalHarness] Queue problems - refusing to start:\n  " + string.Join("\n  ", problems));
                return false;
            }
            return true;
        }

        private static bool TryReadSidecar(QueueEntry e, out SidecarInfo info, out string error)
        {
            info = null;
            error = null;
            try
            {
                info = JsonUtility.FromJson<SidecarInfo>(e.sidecar.text);
            }
            catch (Exception ex)
            {
                error = $"sidecar is not valid JSON ({ex.Message})";
                return false;
            }
            if (info == null || string.IsNullOrEmpty(info.condition))
            {
                error = "sidecar has no 'condition'";
                return false;
            }
            info.condition = info.condition.Trim();
            return true;
        }

        private bool HasCompletePath(Vector3 from, Vector3 to)
        {
            if (!NavMesh.SamplePosition(to, out NavMeshHit toHit, 2f, NavMesh.AllAreas)) return false;
            var path = new NavMeshPath();
            return NavMesh.CalculatePath(from, toHit.position, NavMesh.AllAreas, path) && path.status == NavMeshPathStatus.PathComplete;
        }

        private static bool HasMarkers(GameObject[] markers)
        {
            if (markers == null || markers.Length == 0) return false;
            foreach (var m in markers) if (m == null) return false;
            return true;
        }

        // ------------------------------------------------------------------
        // Engine settings (background running)
        // ------------------------------------------------------------------

        private void ApplyEngineSettings()
        {
            previousMaximumDeltaTime = Time.maximumDeltaTime;
            previousRunInBackground = Application.runInBackground;
            previousVSyncCount = QualitySettings.vSyncCount;
            previousTargetFrameRate = Application.targetFrameRate;
            engineSettingsChanged = true;

            // Keep running when another app has focus, and don't wait for the display between frames.
            Application.runInBackground = true;
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
            // Caps one frame's scaled deltaTime at the sampling interval so ImitationAgent's capture gate never skips a tick;
            // if frames get slow (background throttling) the simulation slows down instead of changing.
            Time.maximumDeltaTime = ZarrDataContract.SamplingIntervalSeconds;
            Time.timeScale = simulationSpeed;
        }

        private void RestoreEngineSettings()
        {
            if (!engineSettingsChanged) return;
            engineSettingsChanged = false;
            Time.timeScale = 1f;
            if (previousMaximumDeltaTime > 0f) Time.maximumDeltaTime = previousMaximumDeltaTime;
            Application.runInBackground = previousRunInBackground;
            QualitySettings.vSyncCount = previousVSyncCount;
            Application.targetFrameRate = previousTargetFrameRate;
        }

        // ------------------------------------------------------------------
        // Queue lifecycle
        // ------------------------------------------------------------------

        private void SetupQueueFolder()
        {
            string root = string.IsNullOrWhiteSpace(outputRootFolder)
                ? Path.Combine(Application.dataPath, "..", "SimulationLogs")
                : outputRootFolder;
            queueFolderPath = Path.GetFullPath(Path.Combine(root, $"EvalQueue_{sceneName}_{queueName.Trim()}"));
            Directory.CreateDirectory(queueFolderPath);
            Debug.Log($"[EvalHarness] Queue '{queueName}' on '{sceneName}': {queue.Count} entries x {configurationCount} episodes -> '{queueFolderPath}'.");
        }

        private static string RunFolderName(QueueEntry e) => $"EvalRun_{e.Label}";

        private bool IsEntryComplete(QueueEntry e) => File.Exists(Path.Combine(queueFolderPath, RunFolderName(e), "run_complete.json"));

        private void StartNextEntry()
        {
            while (true)
            {
                entryIndex++;
                if (entryIndex >= queue.Count)
                {
                    FinishQueue();
                    return;
                }
                entry = queue[entryIndex];
                if (IsEntryComplete(entry))
                {
                    entryResults.Add($"{entry.Label}: already complete (skipped)");
                    Debug.Log($"[EvalHarness] {entryIndex + 1}/{queue.Count} {entry.Label}: already complete in this queue folder - skipped.");
                    continue;
                }
                break;
            }

            entrySidecar = null;
            if (entry.agentKind == AgentKind.Imitation) TryReadSidecar(entry, out entrySidecar, out _);

            runFolderPath = Path.Combine(queueFolderPath, RunFolderName(entry));
            if (Directory.Exists(runFolderPath))
            {
                // Unfinished run from an interrupted session: keep it for inspection, never mix it into the new run.
                string aside = $"{runFolderPath}_incomplete_{DateTime.Now:yyyyMMdd_HHmmss}";
                Directory.Move(runFolderPath, aside);
                Debug.LogWarning($"[EvalHarness] {entry.Label}: found an unfinished run folder - moved to '{Path.GetFileName(aside)}' and starting over.");
            }
            Directory.CreateDirectory(runFolderPath);
            csvWriter = new StreamWriter(Path.Combine(runFolderPath, "episodes.csv"), append: false);
            csvWriter.WriteLine(string.Join(";", CsvColumns));
            csvWriter.Flush();

            episodeIndex = 0;
            entrySuccesses = entryTimeouts = entryOther = 0;
            entryStartRealtime = Time.realtimeSinceStartup;
            ComputeModelIdentity();
            WriteRunConfig();
            Debug.Log($"[EvalHarness] {entryIndex + 1}/{queue.Count} {entry.Label}: starting {configurationCount} episodes" +
                (entrySidecar != null ? $" (condition {entrySidecar.condition}, fold {entrySidecar.fold}, seed {entrySidecar.seed})." : "."));

            currentConfigId = -1;
            AdvanceToNextConfiguration();
        }

        private void FinishEntry()
        {
            if (csvWriter != null)
            {
                csvWriter.Flush();
                csvWriter.Close();
                csvWriter = null;
            }
            float realSeconds = Time.realtimeSinceStartup - entryStartRealtime;
            var done = new RunCompleteDump
            {
                finished_at = DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                episodes = episodeIndex,
                successes = entrySuccesses,
                timeouts = entryTimeouts,
                other = entryOther,
                real_seconds = realSeconds
            };
            File.WriteAllText(Path.Combine(runFolderPath, "run_complete.json"), JsonUtility.ToJson(done, true));
            string summary = $"{entry.Label}: {entrySuccesses}/{episodeIndex} successful, {entryTimeouts} timeouts, {entryOther} other ({realSeconds / 60f:F0} min)";
            entryResults.Add(summary);
            Debug.Log($"[EvalHarness] {entryIndex + 1}/{queue.Count} done - {summary}.");
        }

        private void FinishQueue()
        {
            harnessFinished = true;
            RestoreEngineSettings();
            WriteStatus("done");
            Debug.Log($"[EvalHarness] Queue '{queueName}' on '{sceneName}' complete ({(Time.realtimeSinceStartup - queueStartRealtime) / 60f:F0} min):\n  " +
                string.Join("\n  ", entryResults) + $"\nResults in '{queueFolderPath}'.");
#if UNITY_EDITOR
            if (exitPlayModeWhenDone) UnityEditor.EditorApplication.isPlaying = false;
#endif
        }

        // ------------------------------------------------------------------
        // Run config, model identity, status
        // ------------------------------------------------------------------

        /// <summary>SHA-256 (first 12 hex chars) and modification time of the entry's .onnx file.</summary>
        private void ComputeModelIdentity()
        {
            modelSha256 = "";
            modelFileMtime = "";
            if (entry.agentKind != AgentKind.Imitation) return;
#if UNITY_EDITOR
            try
            {
                string assetPath = UnityEditor.AssetDatabase.GetAssetPath(entry.modelAsset);
                string file = string.IsNullOrEmpty(assetPath) ? null : Path.GetFullPath(Path.Combine(Application.dataPath, "..", assetPath));
                if (file != null && File.Exists(file))
                {
                    using (var sha = SHA256.Create())
                    using (var fs = File.OpenRead(file))
                    {
                        modelSha256 = BitConverter.ToString(sha.ComputeHash(fs)).Replace("-", "").ToLowerInvariant().Substring(0, 12);
                    }
                    modelFileMtime = File.GetLastWriteTimeUtc(file).ToString("o", CultureInfo.InvariantCulture);
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[EvalHarness] Could not hash model file: {ex.Message}");
            }
#endif
        }

        [Serializable]
        private class RunConfigDump
        {
            public string harness_version;
            public string agent_kind;
            public string condition;
            public string scene;
            public string started_at;
            public string unity_version;
            public string queue_name;
            public int queue_entry_index;
            public int queue_entries;
            public int base_seed;
            public string model_asset;
            public string model_sha256;
            public string model_disk_mtime_utc;
            public int model_fold;
            public int model_seed;
            public string model_splits_design;
            public List<string> heldout_participants;
            public int configuration_count;
            public int agents_per_configuration;
            public bool randomize_markers;
            public float simulation_speed;
            public float episode_timeout_sim_s;
            public bool nudge_enabled;
            public float start_jitter_position_m;
            public float start_jitter_heading_deg;
            public string spawn_heading_rule;
            public float stall_event_min_s;
            public float collision_debounce_s;
            public float max_plausible_speed_mps;
            public float metric_window_s;
            public float stall_displacement_m;
            public int task_number_of_needs;
            public int task_start_markers;
            public int task_poi_markers;
            public int task_end_markers;
            // Hand-placed marker positions (the start before jitter); the per-episode spawn is in episodes.csv.
            public List<Vector3> task_start_positions;
            public List<Vector3> task_poi_positions;
            public List<Vector3> task_end_positions;
            // What actually walks: the recording body (Imitation) or the NavMeshAgent (ShortestPath).
            public float body_speed_mps;
            public float body_radius_m;
            public float shortest_path_acceleration_mps2;
            public float shortest_path_angular_speed_deg_per_s;
            public float navmesh_bake_agent_radius;
            public float arrival_radius_m;
            public float tick_interval_s;
            public bool imitation_stochastic_walk;
            public float imitation_min_walk_hold_s;
        }

        [Serializable]
        private class RunCompleteDump
        {
            public string finished_at;
            public int episodes;
            public int successes;
            public int timeouts;
            public int other;
            public float real_seconds;
        }

        private void WriteRunConfig()
        {
            bool imitation = entry.agentKind == AgentKind.Imitation;
            var startPositions = new List<Vector3>(baseStartPositions);
            var dump = new RunConfigDump
            {
                harness_version = HarnessVersion,
                agent_kind = entry.agentKind.ToString(),
                condition = ConditionName(),
                scene = sceneName,
                started_at = DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                unity_version = Application.unityVersion,
                queue_name = queueName.Trim(),
                queue_entry_index = entryIndex,
                queue_entries = queue.Count,
                base_seed = baseSeed,
                model_asset = imitation ? entry.modelAsset.name : "",
                model_sha256 = modelSha256,
                model_disk_mtime_utc = modelFileMtime,
                model_fold = entrySidecar != null ? entrySidecar.fold : 0,
                model_seed = entrySidecar != null ? entrySidecar.seed : 0,
                model_splits_design = entrySidecar != null ? entrySidecar.splits_design ?? "" : "",
                heldout_participants = entrySidecar != null && entrySidecar.heldout_participants != null
                    ? new List<string>(entrySidecar.heldout_participants) : new List<string>(),
                configuration_count = configurationCount,
                agents_per_configuration = 1,
                randomize_markers = randomizeMarkersPerConfiguration,
                simulation_speed = simulationSpeed,
                episode_timeout_sim_s = episodeTimeoutSeconds,
                nudge_enabled = imitation && !disableDeadlockNudge,
                start_jitter_position_m = startJitterPositionMeters,
                start_jitter_heading_deg = startJitterHeadingDegrees,
                spawn_heading_rule = SpawnHeadingRule,
                stall_event_min_s = stallEventMinSeconds,
                collision_debounce_s = collisionDebounceSeconds,
                max_plausible_speed_mps = maxPlausibleSpeedMps,
                metric_window_s = metricWindowSeconds,
                stall_displacement_m = stallDisplacementMeters,
                task_number_of_needs = task.numberOfNeeds,
                task_start_markers = task.start.Length,
                task_poi_markers = task.pointsOfInterest != null ? task.pointsOfInterest.Length : 0,
                task_end_markers = task.end.Length,
                task_start_positions = startPositions,
                task_poi_positions = MarkerPositions(task.pointsOfInterest),
                task_end_positions = MarkerPositions(task.end),
                body_speed_mps = imitation ? ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond : shortestPathSpeedMps,
                body_radius_m = imitation ? ZarrDataContract.RecordingBody.Radius : shortestPathRadiusM,
                shortest_path_acceleration_mps2 = imitation ? 0f : shortestPathAccelerationMps2,
                shortest_path_angular_speed_deg_per_s = imitation ? 0f : shortestPathAngularSpeedDegPerS,
                navmesh_bake_agent_radius = NavMesh.GetSettingsByID(0).agentRadius,
                arrival_radius_m = ZarrDataContract.ArrivalRadiusMeters,
                tick_interval_s = ZarrDataContract.SamplingIntervalSeconds,
                imitation_stochastic_walk = imitation && imitationStochasticWalk,
                imitation_min_walk_hold_s = imitation ? imitationMinWalkHoldSeconds : 0f
            };
            File.WriteAllText(Path.Combine(runFolderPath, "run_config.json"), JsonUtility.ToJson(dump, true));
        }

        private string ConditionName() => entry != null && entry.agentKind == AgentKind.Imitation && entrySidecar != null ? entrySidecar.condition : "";

        private static List<Vector3> MarkerPositions(GameObject[] markers)
        {
            var list = new List<Vector3>();
            if (markers == null) return list;
            foreach (var m in markers) if (m != null) list.Add(m.transform.position);
            return list;
        }

        [Serializable]
        private class StatusDump
        {
            public string state;
            public string queue_name;
            public string scene;
            public string started_at;
            public string updated_at;
            public int entry;
            public int entries;
            public string current;
            public int episodes_done_in_current;
            public int episodes_per_entry;
            public float elapsed_min;
            public float estimated_remaining_min;
            public float sim_speed_vs_real;
            public List<string> results;
        }

        /// <summary>queue_status.json: progress readable without switching to Unity. The estimate uses the mean real time per
        /// episode of this Play session.</summary>
        private void WriteStatus(string state)
        {
            if (string.IsNullOrEmpty(queueFolderPath)) return;
            int remaining = configurationCount - episodeIndex;
            for (int i = entryIndex + 1; i < queue.Count; i++) if (!IsEntryComplete(queue[i])) remaining += configurationCount;
            if (state == "done") remaining = 0;
            float perEpisode = episodesThisSession > 0 ? realSecondsThisSession / episodesThisSession : 0f;
            var status = new StatusDump
            {
                state = state,
                queue_name = queueName,
                scene = sceneName,
                started_at = queueStartedAt,
                updated_at = DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                entry = Mathf.Min(entryIndex + 1, queue.Count),
                entries = queue.Count,
                current = state == "done" || entry == null ? "" : entry.Label,
                episodes_done_in_current = state == "done" ? 0 : episodeIndex,
                episodes_per_entry = configurationCount,
                elapsed_min = (Time.realtimeSinceStartup - queueStartRealtime) / 60f,
                estimated_remaining_min = perEpisode * remaining / 60f,
                sim_speed_vs_real = sessionSimSeconds > 0f && realSecondsThisSession > 0f ? sessionSimSeconds / realSecondsThisSession : 0f,
                results = new List<string>(entryResults)
            };
            try
            {
                File.WriteAllText(Path.Combine(queueFolderPath, "queue_status.json"), JsonUtility.ToJson(status, true));
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[EvalHarness] Could not write queue_status.json: {ex.Message}");
            }
        }
        private float sessionSimSeconds = 0f;

        // ------------------------------------------------------------------
        // Main loop
        // ------------------------------------------------------------------

        void Update()
        {
            if (harnessFinished || entry == null) return;

            for (int i = activeEpisodes.Count - 1; i >= 0; i--)
            {
                ActiveEpisode ep = activeEpisodes[i];

                if (ep.agentGO == null)
                {
                    // Destroyed some other way than FinishEpisode() - drop it without a row rather than crash.
                    CloseTickWriters(ep);
                    activeEpisodes.RemoveAt(i);
                    currentConfigAgentsCompleted++;
                    continue;
                }

                TickEpisodeClock(ep);

                if (!ep.started)
                {
                    // The agent's Start() is still initialising (model loading), or it threw.
                    if (Time.realtimeSinceStartup - ep.startRealtime >= agentStartTimeoutRealSeconds)
                    {
                        FinishEpisode(ep, "other", "agent_never_started");
                        activeEpisodes.RemoveAt(i);
                        currentConfigAgentsCompleted++;
                    }
                    continue;
                }

                if (ep.agent.DestroyRequest)
                {
                    FinishEpisode(ep, "success", null);
                    activeEpisodes.RemoveAt(i);
                    currentConfigAgentsCompleted++;
                }
                else if (ep.simElapsed >= episodeTimeoutSeconds)
                {
                    FinishEpisode(ep, "timeout", null);
                    activeEpisodes.RemoveAt(i);
                    currentConfigAgentsCompleted++;
                }
            }

            if (currentConfigAgentsSpawned < 1)
            {
                SpawnAgentForEpisode();
                currentConfigAgentsSpawned++;
            }
            else if (currentConfigAgentsCompleted >= 1)
            {
                AdvanceToNextConfiguration();
            }
        }

        // ------------------------------------------------------------------
        // Configurations
        // ------------------------------------------------------------------

        private void AdvanceToNextConfiguration()
        {
            currentConfigId++;
            if (currentConfigId >= configurationCount)
            {
                FinishEntry();
                StartNextEntry();
                return;
            }

            currentConfigSeed = configSeeds[currentConfigId];
            if (randomizeMarkersPerConfiguration)
            {
                UnityEngine.Random.InitState(currentConfigSeed);
                currentConfigStartOffsetM = 0f;
                currentConfigHeadingJitterDeg = 0f;
                if (!RandomizeTaskMarkers())
                {
                    consecutivePlacementFailures++;
                    Debug.LogError($"[EvalHarness] Configuration {currentConfigId}: no valid spaced NavMesh positions for all markers - logged as outcome=other and skipped.");
                    WriteEpisodeRow(new EpisodeRecord
                    {
                        configId = currentConfigId, agentId = -1, seed = currentConfigSeed,
                        outcome = "other", rawOutcome = "other", secondsToFirstCollision = -1f, error = "navmesh_placement_failed"
                    });
                    episodeIndex++;
                    entryOther++;
                    if (consecutivePlacementFailures >= MaxConsecutivePlacementFailures)
                    {
                        Debug.LogError($"[EvalHarness] {MaxConsecutivePlacementFailures} consecutive placement failures - ending this entry early.");
                        currentConfigId = configurationCount;
                    }
                    AdvanceToNextConfiguration();
                    return;
                }
                consecutivePlacementFailures = 0;
            }
            else
            {
                SnapshotFixedTaskMarkers();
                ApplyStartJitter(new System.Random(currentConfigSeed));
            }
            // The stochastic walk (and PoiProgressTracker's draws) use UnityEngine.Random: seed it per configuration too.
            UnityEngine.Random.InitState(currentConfigSeed);

            currentConfigAgentsSpawned = 0;
            currentConfigAgentsCompleted = 0;
        }

        private void SnapshotFixedTaskMarkers()
        {
            currentConfigPoiPositions = new List<Vector3>();
            if (task.pointsOfInterest != null && task.numberOfNeeds > 0)
            {
                foreach (var poi in task.pointsOfInterest)
                {
                    if (poi != null) currentConfigPoiPositions.Add(poi.transform.position);
                }
            }
            currentConfigEndPos = task.end[0].transform.position;
        }

        /// <summary>
        /// Moves the start marker to base + a seeded offset (uniform in a disc of startJitterPositionMeters) if that point is on
        /// the NavMesh with a complete path to the End, else keeps the base position; draws the heading jitter. Uses its own
        /// System.Random(configuration seed), so the draw is identical for every entry.
        /// </summary>
        private void ApplyStartJitter(System.Random rng)
        {
            double angle = rng.NextDouble() * 2.0 * Math.PI;
            double radius = startJitterPositionMeters * Math.Sqrt(rng.NextDouble());
            float heading = (float)((rng.NextDouble() * 2.0 - 1.0) * startJitterHeadingDegrees);

            Vector3 target = baseStartPositions[0];
            if (radius > 1e-4)
            {
                Vector3 candidate = target + new Vector3((float)(radius * Math.Cos(angle)), 0f, (float)(radius * Math.Sin(angle)));
                if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, 1f, NavMesh.AllAreas) && HasCompletePath(hit.position, currentConfigEndPos))
                {
                    target = hit.position;
                }
            }
            task.start[0].transform.position = target;

            currentConfigStartOffsetM = Vector3.Distance(target, baseStartPositions[0]);
            currentConfigHeadingJitterDeg = heading;
        }

        /// <summary>Random markers only: start(s), POI(s) and end(s) to seeded, mutually spaced NavMesh positions.</summary>
        private bool RandomizeTaskMarkers()
        {
            var chosen = new List<Vector3>();
            if (!RepositionArray(task.start, chosen)) return false;

            var poiChosen = new List<Vector3>();
            if (!RepositionArray(task.pointsOfInterest, chosen, poiChosen)) return false;
            currentConfigPoiPositions = poiChosen;

            var endChosen = new List<Vector3>();
            if (!RepositionArray(task.end, chosen, endChosen)) return false;
            currentConfigEndPos = endChosen.Count > 0 ? endChosen[0] : Vector3.zero;
            return true;
        }

        private bool RepositionArray(GameObject[] markers, List<Vector3> chosen, List<Vector3> outPositions = null)
        {
            if (markers == null) return true;
            foreach (var marker in markers)
            {
                if (marker == null) continue;
                if (!TryFindSpacedNavMeshPosition(chosen, out Vector3 pos)) return false;
                marker.transform.position = pos;
                chosen.Add(pos);
                outPositions?.Add(pos);
            }
            return true;
        }

        private bool TryFindSpacedNavMeshPosition(List<Vector3> alreadyChosen, out Vector3 result)
        {
            for (int attempt = 0; attempt < maxPlacementAttempts; attempt++)
            {
                Vector3 baseVertex = navMeshVertices[UnityEngine.Random.Range(0, navMeshVertices.Length)];
                Vector2 jitter = UnityEngine.Random.insideUnitCircle * navMeshSampleRadius;
                Vector3 candidate = baseVertex + new Vector3(jitter.x, 0f, jitter.y);
                if (!NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleRadius, NavMesh.AllAreas)) continue;

                bool farEnough = true;
                foreach (Vector3 other in alreadyChosen)
                {
                    if (Vector3.Distance(hit.position, other) < minPositionSpacingMeters) { farEnough = false; break; }
                }
                if (farEnough)
                {
                    result = hit.position;
                    return true;
                }
            }
            result = Vector3.zero;
            return false;
        }

        private void SpawnAgentForEpisode()
        {
            GameObject agentGO = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            agentGO.name = $"EvalHarness_{entry.Label}_Config{currentConfigId}";

            Collider capCollider = agentGO.GetComponent<Collider>();
            if (capCollider != null) Destroy(capCollider);

            MeshRenderer agentRenderer = agentGO.GetComponent<MeshRenderer>();
            if (agentRenderer != null)
            {
                agentRenderer.material = CreateColorMaterial(agentColor);
                // The human rig has no body; a shadow on the floor would be in the policy's RGB but never in the training data.
                agentRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            }

            // Fields are set right after AddComponent, before the agent's Start() runs on its first frame.
            IEvalAgent agent;
            ImitationAgent imitation = null;
            if (entry.agentKind == AgentKind.Imitation)
            {
                imitation = agentGO.AddComponent<ImitationAgent>();
                imitation.task = task;
                imitation.ArchitectureLayerMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);
                imitation.BaselineModelAsset = entry.modelAsset;
                imitation.enableDeadlockNudge = !disableDeadlockNudge;
                imitation.hotReloadModel = false; // the run evaluates exactly this model (its sha256 is logged)
                imitation.stochasticWalk = imitationStochasticWalk;
                imitation.minWalkHoldSeconds = imitationMinWalkHoldSeconds;
                imitation.spawnHeadingOffsetDeg = currentConfigHeadingJitterDeg;
                agent = imitation;
            }
            else
            {
                ShortestPathAgent shortestPath = agentGO.AddComponent<ShortestPathAgent>();
                shortestPath.task = task;
                shortestPath.speedMetersPerSecond = shortestPathSpeedMps;
                shortestPath.radiusMeters = shortestPathRadiusM;
                shortestPath.accelerationMetersPerSecond2 = shortestPathAccelerationMps2;
                shortestPath.angularSpeedDegreesPerSecond = shortestPathAngularSpeedDegPerS;
                shortestPath.spawnHeadingOffsetDeg = currentConfigHeadingJitterDeg;
                agent = shortestPath;
            }

            // AutoFlush: both files survive stopping Play mid-episode.
            string episodeKey = $"config{currentConfigId}_agent0";
            string trajFileName = $"traj_{episodeKey}.csv";
            var trajWriter = new StreamWriter(Path.Combine(runFolderPath, trajFileName), false) { AutoFlush = true };
            trajWriter.WriteLine(TrajectoryCsvHeader);
            StreamWriter modelDebugWriter = null;
            if (logModelDebugTicks && imitation != null)
            {
                modelDebugWriter = new StreamWriter(Path.Combine(runFolderPath, $"modeldebug_{episodeKey}.csv"), false) { AutoFlush = true };
                modelDebugWriter.WriteLine(ModelDebugCsvHeader);
            }

            activeEpisodes.Add(new ActiveEpisode
            {
                trajWriter = trajWriter,
                modelDebugWriter = modelDebugWriter,
                trajFileName = trajFileName,
                agentGO = agentGO,
                agent = agent,
                imitation = imitation,
                startRealtime = Time.realtimeSinceStartup,
                configId = currentConfigId,
                agentId = 0,
                seed = currentConfigSeed,
                poiPositions = currentConfigPoiPositions,
                endPos = currentConfigEndPos,
                startOffsetM = currentConfigStartOffsetM,
                headingJitterDeg = currentConfigHeadingJitterDeg
            });
        }

        // ------------------------------------------------------------------
        // Per-frame episode clock, trajectory ticks, diagnostics
        // ------------------------------------------------------------------

        /// <summary>
        /// Advances one episode's clock and diagnostics by one frame. All numbers come from the agent's own trajectory and
        /// Time.deltaTime (simulated time), never from realtime.
        /// </summary>
        private void TickEpisodeClock(ActiveEpisode ep)
        {
            List<Vector3> traj = ep.agent.Trajectory;
            if (traj == null || traj.Count == 0) return;

            if (!ep.started)
            {
                // First Update of the agent: it has spawned and already faces its start heading (path + jitter).
                ep.started = true;
                ep.startFrame = Time.frameCount;
                ep.processedPoints = 1;
                ep.windowStartPos = traj[0];
                ep.prevTickEyePos = Eye(ep).position;
                ZarrDataContract.HeadingPitch(Eye(ep), out ep.prevTickHeading, out _);
                ep.spawnYawDeg = ep.prevTickHeading;
                WriteTick(ep);
                return;
            }

            ep.simElapsed += Time.deltaTime;
            ProcessNewTrajectoryPoints(ep);
            if (ep.simElapsed >= ep.nextTickSim - 1e-4f) WriteTick(ep);

            if (ep.imitation != null)
            {
                if (!ep.firstCollisionLogged && ep.imitation.HasCollided)
                {
                    ep.firstCollisionLogged = true;
                    ep.simAtFirstCollision = ep.simElapsed;
                }

                int rawCollisions = ep.imitation.CollisionCount;
                if (rawCollisions > ep.lastRawCollisions)
                {
                    if (ep.collisionEvents == 0 || ep.simElapsed - ep.lastCollisionSim > collisionDebounceSeconds) ep.collisionEvents++;
                    ep.lastCollisionSim = ep.simElapsed;
                    ep.lastRawCollisions = rawCollisions;
                }
            }

            int fulfilled = ep.agent.NeedsFulfilledCount;
            while (ep.poiTimes.Count < fulfilled) ep.poiTimes.Add(ep.simElapsed);

            float windowDur = ep.simElapsed - ep.windowStartSim;
            if (windowDur >= metricWindowSeconds)
            {
                Vector3 cur = traj[traj.Count - 1];
                float windowSpeed = (ep.pathLengthM - ep.windowStartPath) / windowDur;
                if (windowSpeed > ep.maxWindowSpeedMps) ep.maxWindowSpeedMps = windowSpeed;

                if (Vector3.Distance(cur, ep.windowStartPos) < stallDisplacementMeters)
                {
                    ep.stallSeconds += windowDur;
                    if (ep.stallRunSeconds <= 0f) { ep.stallRunPos = ep.windowStartPos; ep.stallRunCounted = false; }
                    ep.stallRunSeconds += windowDur;
                    if (!ep.stallRunCounted && ep.stallRunSeconds >= stallEventMinSeconds) { ep.stallRunCounted = true; ep.stallEvents++; }
                    if (ep.stallRunSeconds > ep.longestStallSeconds) { ep.longestStallSeconds = ep.stallRunSeconds; ep.longestStallPos = ep.stallRunPos; }
                }
                else
                {
                    ep.stallRunSeconds = 0f;
                }

                ep.windowStartSim = ep.simElapsed;
                ep.windowStartPath = ep.pathLengthM;
                ep.windowStartPos = cur;
            }
        }

        private static Transform Eye(ActiveEpisode ep) => ep.agent.EyeAnchor != null ? ep.agent.EyeAnchor : ep.agentGO.transform;

        /// <summary>
        /// One row of the trajectory file (ZarrDataContract.Trajectory) and, if enabled, of the model-debug file, on the 0.1 s
        /// sim-time grid (t = 0 on the agent's first Update; t is the sim time of the frame that crossed the grid point, and
        /// eval_tools/trajectories.py resamples onto the exact grid). Position, heading and pitch are the eye's, as the
        /// recordings' pose is the camera's.
        /// </summary>
        private void WriteTick(ActiveEpisode ep)
        {
            float interval = ZarrDataContract.SamplingIntervalSeconds;
            while (ep.nextTickSim <= ep.simElapsed + 1e-4f) ep.nextTickSim += interval;
            if (ep.trajWriter == null) return;

            Transform eye = Eye(ep);
            Vector3 pos = eye.position;
            ZarrDataContract.HeadingPitch(eye, out float heading, out float pitch);

            float dt = ep.simElapsed - ep.prevTickSim;
            float speed = 0f, yawRate = 0f;
            if (dt > 1e-5f)
            {
                Vector3 moved = pos - ep.prevTickEyePos;
                moved.y = 0f;
                speed = moved.magnitude / dt;
                yawRate = Mathf.DeltaAngle(ep.prevTickHeading, heading) / dt;
            }
            ep.prevTickSim = ep.simElapsed;
            ep.prevTickEyePos = pos;
            ep.prevTickHeading = heading;

            int poisReached = ep.agent.NeedsFulfilledCount;
            int poiCount = ep.agent.PlannedPois != null ? ep.agent.PlannedPois.Count : ep.poiPositions.Count;
            int phase = poisReached >= poiCount ? ZarrDataContract.Trajectory.PhaseToEnd : ZarrDataContract.Trajectory.PhaseToPoi;

            string d = ZarrDataContract.Trajectory.CsvDelimiter;
            ep.trajWriter.WriteLine(string.Join(d,
                F(ep.simElapsed), F(pos.x), F(pos.y), F(pos.z), F(heading), F(pitch), F(speed), F(yawRate), I(phase), I(poisReached)));

            ImitationAgent im = ep.imitation;
            if (ep.modelDebugWriter == null || im == null) return;
            Vector3 goalOffset = ep.agent.CurrentGoalPosition - pos;
            Vector3 goalLocal = eye.InverseTransformDirection(goalOffset.normalized);
            ep.modelDebugWriter.WriteLine(string.Join(d,
                F(ep.simElapsed), F(goalLocal.x), F(goalLocal.y), F(goalLocal.z), F(goalOffset.magnitude),
                B(im.IsWallBlocked),
                FOrEmpty(im.LastRawForward), FOrEmpty(im.LastRawYawPerSample), FOrEmpty(im.LastRawPitch),
                FOrEmpty(im.LastCommandedForwardMps), FOrEmpty(im.LastCommandedYawDegPerSecond),
                string.Join(d, Array.ConvertAll(im.LastProprioV2(), FOrEmpty))));
        }

        private static void ProcessNewTrajectoryPoints(ActiveEpisode ep)
        {
            List<Vector3> traj = ep.agent.Trajectory;
            if (traj == null) return;
            for (; ep.processedPoints < traj.Count; ep.processedPoints++)
            {
                float step = Vector3.Distance(traj[ep.processedPoints - 1], traj[ep.processedPoints]);
                ep.pathLengthM += step;
                if (step > ep.maxStepM) ep.maxStepM = step;
            }
        }

        private void FinishEpisode(ActiveEpisode ep, string outcome, string errorMessage)
        {
            ProcessNewTrajectoryPoints(ep);

            List<Vector3> trajectory = ep.agent.Trajectory ?? new List<Vector3>();
            Vector3 spawnPos = trajectory.Count > 0 ? trajectory[0] : ep.agentGO.transform.position;
            Vector3 finalPos = trajectory.Count > 0 ? trajectory[trajectory.Count - 1] : ep.agentGO.transform.position;
            Vector3 finalGoalPos = ep.agent.CurrentGoalPosition;

            float simSeconds = ep.simElapsed;
            float realSeconds = Time.realtimeSinceStartup - ep.startRealtime;
            int frames = ep.started ? Time.frameCount - ep.startFrame : 0;
            int steps = Mathf.RoundToInt(simSeconds / ZarrDataContract.SamplingIntervalSeconds);
            float pathLengthM = ep.pathLengthM;

            // NavMesh shortest route from this episode's actual spawn via the planned POI(s) to the end (methodology 3.9.2).
            List<Vector3> routePois = ep.poiPositions;
            Vector3 routeEnd = ep.endPos;
            if (ep.agent.PlannedPois != null && ep.agent.PlannedEnd != null)
            {
                routePois = new List<Vector3>();
                foreach (Transform poi in ep.agent.PlannedPois) routePois.Add(poi.position);
                routeEnd = ep.agent.PlannedEnd.position;
            }
            var optimalSegmentsM = new List<float>();
            float optimalPathLengthM = 0f;
            Vector3 segmentFrom = spawnPos;
            foreach (Vector3 waypoint in routePois)
            {
                float segment = ComputeNavMeshPathLength(segmentFrom, waypoint);
                optimalSegmentsM.Add(segment);
                optimalPathLengthM += segment;
                segmentFrom = waypoint;
            }
            float lastSegment = ComputeNavMeshPathLength(segmentFrom, routeEnd);
            optimalSegmentsM.Add(lastSegment);
            optimalPathLengthM += lastSegment;

            float meanSpeed = pathLengthM / Mathf.Max(simSeconds, 0.1f);
            bool speedViolation = meanSpeed > maxPlausibleSpeedMps || ep.maxWindowSpeedMps > maxPlausibleSpeedMps;
            string rawOutcome = outcome;
            if (speedViolation) outcome = "invalid_speed";

            int stuckCount = ep.imitation != null ? ep.imitation.StuckCount : 0;
            bool successNoRescue = outcome == "success" && stuckCount == 0;
            float spl = successNoRescue ? Mathf.Clamp01(optimalPathLengthM / Mathf.Max(pathLengthM, optimalPathLengthM)) : 0f;

            float finalDistToGoal = Vector2.Distance(new Vector2(finalPos.x, finalPos.z), new Vector2(finalGoalPos.x, finalGoalPos.z));

            string pngFilename = "";
            if (heatmapEveryNthEpisode > 0 && episodeIndex % heatmapEveryNthEpisode == 0)
            {
                pngFilename = $"episode_{episodeIndex:D4}.png";
                SaveEpisodeHeatmap(trajectory, pngFilename);
            }

            WriteEpisodeRow(new EpisodeRecord
            {
                configId = ep.configId,
                agentId = ep.agentId,
                seed = ep.seed,
                startPos = spawnPos,
                startOffsetM = ep.startOffsetM,
                startHeadingJitterDeg = ep.headingJitterDeg,
                spawnYawDeg = ep.spawnYawDeg,
                poiPositions = routePois,
                endPos = routeEnd,
                outcome = outcome,
                rawOutcome = rawOutcome,
                successNoRescue = successNoRescue,
                speedViolation = speedViolation,
                poisReached = ep.agent.NeedsFulfilledCount,
                steps = steps,
                simSeconds = simSeconds,
                successTimeS = outcome == "success" ? simSeconds : -1f,
                poiTimes = ep.poiTimes,
                realSeconds = realSeconds,
                frames = frames,
                pathLengthM = pathLengthM,
                optimalPathLengthM = optimalPathLengthM,
                optimalSegmentsM = optimalSegmentsM,
                spl = spl,
                meanSpeedMps = meanSpeed,
                maxWindowSpeedMps = ep.maxWindowSpeedMps,
                maxStepM = ep.maxStepM,
                finalDistToGoalM = finalDistToGoal,
                collisions = ep.imitation != null ? ep.imitation.CollisionCount : 0,
                collisionEvents = ep.collisionEvents,
                secondsToFirstCollision = ep.firstCollisionLogged ? ep.simAtFirstCollision : -1f,
                stallSeconds = ep.stallSeconds,
                stallEvents = ep.stallEvents,
                longestStallS = ep.longestStallSeconds,
                longestStallPos = ep.longestStallSeconds > 0f ? (Vector3?)ep.longestStallPos : null,
                nudgeCount = stuckCount,
                png = pngFilename,
                trajectoryFile = ep.trajFileName,
                error = errorMessage
            });

            CloseTickWriters(ep);
            Destroy(ep.agentGO);
            episodeIndex++;

            if (outcome == "success") entrySuccesses++;
            else if (outcome == "timeout") entryTimeouts++;
            else entryOther++;
            episodesThisSession++;
            realSecondsThisSession += realSeconds;
            sessionSimSeconds += simSeconds;
            WriteStatus("running");

            float perEpisode = realSecondsThisSession / episodesThisSession;
            int remaining = configurationCount - episodeIndex;
            for (int i = entryIndex + 1; i < queue.Count; i++) if (!IsEntryComplete(queue[i])) remaining += configurationCount;
            Debug.Log($"[EvalHarness] {entryIndex + 1}/{queue.Count} {entry.Label} episode {episodeIndex}/{configurationCount}: " +
                $"{outcome} after {simSeconds:F1} s sim ({simSeconds / Mathf.Max(realSeconds, 1e-3f):F1}x real) | " +
                $"{entrySuccesses}/{episodeIndex} successful | about {perEpisode * remaining / 60f:F0} min left in the queue");
        }

        private static void CloseTickWriters(ActiveEpisode ep)
        {
            ep.trajWriter?.Close();
            ep.trajWriter = null;
            ep.modelDebugWriter?.Close();
            ep.modelDebugWriter = null;
        }

        private float ComputeNavMeshPathLength(Vector3 from, Vector3 to)
        {
            // Snap both ends onto the NavMesh first: a marker floating above the floor (e.g. end at y=1) makes
            // CalculatePath fail, which used to silently fall back to the straight-line distance.
            if (NavMesh.SamplePosition(from, out NavMeshHit fromHit, 2f, NavMesh.AllAreas)) from = fromHit.position;
            if (NavMesh.SamplePosition(to, out NavMeshHit toHit, 2f, NavMesh.AllAreas)) to = toHit.position;

            NavMeshPath path = new NavMeshPath();
            if (!NavMesh.CalculatePath(from, to, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete)
            {
                return Vector3.Distance(from, to);
            }

            float length = 0f;
            for (int i = 1; i < path.corners.Length; i++) length += Vector3.Distance(path.corners[i - 1], path.corners[i]);
            return length;
        }

        // ------------------------------------------------------------------
        // Output: heatmap PNG, episodes.csv
        // ------------------------------------------------------------------

        private static Material CreateColorMaterial(Color color)
        {
            Shader shader = Shader.Find("Diffuse");
            Material material = new Material(shader != null ? shader : Shader.Find("Standard"));
            material.color = color;
            return material;
        }

        private void SaveEpisodeHeatmap(List<Vector3> trajectory, string pngFilename)
        {
            try
            {
                var trajectoryCopy = SubsampleTrajectory(trajectory, maxHeatmapTrajectoryPoints);
                var allAgentPositions = new List<List<Vector3>>[] { new List<List<Vector3>> { trajectoryCopy } };

                var pois = new List<GameObject>();
                pois.AddRange(task.start);
                if (task.pointsOfInterest != null) pois.AddRange(task.pointsOfInterest);
                pois.AddRange(task.end);

                AnalyticsManager.ExportHeatMap(allAgentPositions, pois, Path.Combine(runFolderPath, pngFilename));
            }
            catch (Exception ex)
            {
                Debug.LogError($"[EvalHarness] Episode {episodeIndex}: heatmap export failed: {ex.Message}");
            }
        }

        private static List<Vector3> SubsampleTrajectory(List<Vector3> trajectory, int maxPoints)
        {
            if (maxPoints <= 0 || trajectory.Count <= maxPoints) return new List<Vector3>(trajectory);

            var result = new List<Vector3>(maxPoints);
            float stride = (float)(trajectory.Count - 1) / (maxPoints - 1);
            for (int i = 0; i < maxPoints; i++)
            {
                int idx = Mathf.Clamp(Mathf.RoundToInt(i * stride), 0, trajectory.Count - 1);
                result.Add(trajectory[idx]);
            }
            return result;
        }

        private static string FormatVector(Vector3 v)
        {
            return $"{v.x.ToString(CultureInfo.InvariantCulture)},{v.y.ToString(CultureInfo.InvariantCulture)},{v.z.ToString(CultureInfo.InvariantCulture)}";
        }

        private static string FormatVectorList(List<Vector3> vectors)
        {
            if (vectors == null || vectors.Count == 0) return "";
            var parts = new string[vectors.Count];
            for (int i = 0; i < vectors.Count; i++) parts[i] = FormatVector(vectors[i]);
            return string.Join("|", parts);
        }

        private static readonly string[] CsvColumns =
        {
            "episode_id", "config_id", "agent_id", "timestamp", "harness_version", "agent_kind", "condition", "scene",
            "model_checkpoint", "model_sha256", "model_disk_mtime_utc", "random_seed",
            "layout_mode", "sim_speed", "timeout_s", "nudge_enabled",
            "start_pos", "start_offset_m", "start_heading_jitter_deg", "poi_positions", "end_pos",
            "outcome", "raw_outcome", "success_no_rescue", "speed_violation",
            "num_pois_total", "num_pois_reached",
            "episode_length_steps", "episode_length_seconds", "success_time_s", "time_to_first_poi_s", "poi_times_s",
            "real_seconds", "frames",
            "path_length_m", "optimal_path_length_m", "optimal_segments_m", "spl",
            "mean_speed_mps", "max_window_speed_mps", "max_step_m", "final_dist_to_goal_m",
            "collision_count", "collision_events", "steps_to_first_collision", "seconds_to_first_collision",
            "stall_seconds", "stall_events", "longest_stall_s", "longest_stall_pos",
            "nudge_count", "png_filename", "trajectory_file", "error_message",
            "spawn_yaw_deg"
        };

        private class EpisodeRecord
        {
            public int configId;
            public int agentId;
            public int seed;
            public Vector3 startPos;
            public List<Vector3> poiPositions;
            public Vector3 endPos;
            public string outcome;
            public string rawOutcome;
            public bool successNoRescue;
            public bool speedViolation;
            public int poisReached;
            public int steps;
            public float simSeconds;
            public float successTimeS = -1f;
            public List<float> poiTimes;
            public float startOffsetM;
            public float startHeadingJitterDeg;
            public float spawnYawDeg;
            public int collisionEvents;
            public int stallEvents;
            public float longestStallS;
            public Vector3? longestStallPos;
            public float realSeconds;
            public int frames;
            public float pathLengthM;
            public float optimalPathLengthM;
            public List<float> optimalSegmentsM;
            public float spl;
            public float meanSpeedMps;
            public float maxWindowSpeedMps;
            public float maxStepM;
            public float finalDistToGoalM;
            public int collisions;
            public float secondsToFirstCollision = -1f;
            public float stallSeconds;
            public int nudgeCount;
            public string png = "";
            public string trajectoryFile = "";
            public string error;
        }

        private static string F(float v) => v.ToString("F3", CultureInfo.InvariantCulture);
        private static string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        private static string B(bool v) => v ? "1" : "0";
        private static string FOrEmpty(float v) => float.IsNaN(v) ? "" : F(v);

        private void WriteEpisodeRow(EpisodeRecord r)
        {
            int stepsToFirstCollision = r.secondsToFirstCollision >= 0f
                ? Mathf.RoundToInt(r.secondsToFirstCollision / ZarrDataContract.SamplingIntervalSeconds)
                : -1;

            string poiTimesJoined = "";
            if (r.poiTimes != null && r.poiTimes.Count > 0)
            {
                var parts = new string[r.poiTimes.Count];
                for (int i = 0; i < parts.Length; i++) parts[i] = F(r.poiTimes[i]);
                poiTimesJoined = string.Join("|", parts);
            }

            // One NavMesh length per route leg (spawn -> POI_0, ..., POI_last -> end); they sum to optimal_path_length_m.
            string optimalSegmentsJoined = "";
            if (r.optimalSegmentsM != null && r.optimalSegmentsM.Count > 0)
            {
                var parts = new string[r.optimalSegmentsM.Count];
                for (int i = 0; i < parts.Length; i++) parts[i] = F(r.optimalSegmentsM[i]);
                optimalSegmentsJoined = string.Join("|", parts);
            }

            // Model identity, nudge and collision columns only exist for ImitationAgent (a NavMeshAgent cannot touch a wall).
            bool imitation = entry.agentKind == AgentKind.Imitation;
            string[] cells =
            {
                I(episodeIndex),
                I(r.configId),
                I(r.agentId),
                DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                HarnessVersion,
                entry.agentKind.ToString(),
                ConditionName(),
                sceneName,
                imitation ? entry.modelAsset.name : "",
                modelSha256,
                modelFileMtime,
                I(r.seed),
                randomizeMarkersPerConfiguration ? "randomized" : "fixed",
                F(simulationSpeed),
                F(episodeTimeoutSeconds),
                imitation ? B(!disableDeadlockNudge) : "",
                FormatVector(r.startPos),
                F(r.startOffsetM),
                F(r.startHeadingJitterDeg),
                FormatVectorList(r.poiPositions),
                FormatVector(r.endPos),
                r.outcome ?? "",
                r.rawOutcome ?? "",
                B(r.successNoRescue),
                B(r.speedViolation),
                I(task.numberOfNeeds),
                I(r.poisReached),
                I(r.steps),
                F(r.simSeconds),
                r.successTimeS >= 0f ? F(r.successTimeS) : "",
                (r.poiTimes != null && r.poiTimes.Count > 0) ? F(r.poiTimes[0]) : "",
                poiTimesJoined,
                F(r.realSeconds),
                I(r.frames),
                F(r.pathLengthM),
                F(r.optimalPathLengthM),
                optimalSegmentsJoined,
                F(r.spl),
                F(r.meanSpeedMps),
                F(r.maxWindowSpeedMps),
                F(r.maxStepM),
                F(r.finalDistToGoalM),
                imitation ? I(r.collisions) : "",
                imitation ? I(r.collisionEvents) : "",
                stepsToFirstCollision >= 0 ? I(stepsToFirstCollision) : "",
                r.secondsToFirstCollision >= 0f ? F(r.secondsToFirstCollision) : "",
                F(r.stallSeconds),
                I(r.stallEvents),
                F(r.longestStallS),
                r.longestStallPos.HasValue ? FormatVector(r.longestStallPos.Value) : "",
                imitation ? I(r.nudgeCount) : "",
                r.png ?? "",
                r.trajectoryFile ?? "",
                r.error ?? "",
                F(r.spawnYawDeg)
            };

            csvWriter.WriteLine(string.Join(";", cells));
            csvWriter.Flush();
        }

        void OnDestroy()
        {
            RestoreEngineSettings();
            foreach (ActiveEpisode ep in activeEpisodes) CloseTickWriters(ep);
            if (csvWriter != null)
            {
                csvWriter.Flush();
                csvWriter.Close();
                csvWriter = null;
            }
            // Play mode reverts the moved start marker on exit; outside the Editor, put it back explicitly.
            if (task != null && baseStartPositions != null && HasMarkers(task.start))
                task.start[0].transform.position = baseStartPositions[0];
        }
    }
}
