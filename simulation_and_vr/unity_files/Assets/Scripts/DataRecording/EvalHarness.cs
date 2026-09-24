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
    /// Closed-loop evaluation harness for ImitationAgent and for the NavMesh shortest-path baseline (ShortestPathAgent),
    /// picked by agentKind; one run evaluates one kind. Runs configurationCount scene
    /// configurations back-to-back in a single Editor Play session (no manual replay, no
    /// headless/batchmode). Every layout is evaluated on its own TaskScript (start -> POI(s) -> end, as
    /// placed in the scene). For each configuration: if
    /// randomizeMarkersPerConfiguration is true, randomizes the scene's start/POI/end marker positions (NavMesh-valid,
    /// mutually spaced) - if false, leaves them exactly where they already are in the scene, so
    /// every configuration reuses the same start/POI/end (useful for isolating a layout's own
    /// effect from goal-placement variance, e.g. when comparing the same fixed task across
    /// different scenes/layouts). Either way, agentsPerConfiguration agents are then spawned
    /// onto that configuration, staggered spawnIntervalSeconds apart (so several can be
    /// active at once, exactly like EngineScript's existing spawnInterval pattern) - each
    /// spawned the way EngineScript.SpawnAgent() does. Once every agent on the current
    /// configuration has either succeeded (poiTracker.IsFinished) or timed out, the harness
    /// advances to the next configuration and the cycle repeats. One scalar metrics row + one
    /// top-down trajectory PNG is appended per agent, immediately on that agent's completion,
    /// so an unattended multi-hour run survives a mid-run Editor stop.
    ///
    /// Both kinds get the same configurations, start jitter, timeout, metrics, episodes.csv columns and trajectory file
    /// (traj_config<C>_agent<A>.csv in the unified ZarrDataContract.Trajectory schema, sampled by the harness itself on a
    /// 0.1 s sim-time grid - the same schema eval_tools/trajectories.py derives for human episodes); columns that only
    /// exist for the imitation model (model identity, collisions, nudges) are left empty for ShortestPath, and its per-tick
    /// model outputs go to a separate optional modeldebug_config<C>_agent<A>.csv. AgentScript/
    /// EngineScript are not used - see ShortestPathAgent for why.
    /// Aggregating these rows into Goal Completion Rate / Collision Frequency / MTTC / SPL is a
    /// deliberately separate downstream (pandas) step, not this harness's job.
    /// </summary>
    public class EvalHarness : MonoBehaviour
    {
        public enum AgentKind { Imitation, ShortestPath }

        [Header("Task & Agent")]
        [Tooltip("The TaskScript whose start/end/pointsOfInterest marker GameObjects get repositioned each configuration. Must be the same TaskScript ImitationAgent/ParticipantRecorder use for this scene.")]
        public TaskScript task;
        [Tooltip("Imitation = ImitationAgent running modelAsset. ShortestPath = ShortestPathAgent (NavMeshAgent following the NavMesh " +
                 "shortest path through the same start -> POI -> end task). Written to every row (agent_kind) and to run_config.json.")]
        public AgentKind agentKind = AgentKind.Imitation;
        [Tooltip("The .onnx model every spawned agent runs. Only used with agentKind = Imitation.")]
        public ModelAsset modelAsset;
        [Tooltip("Training condition of modelAsset: A = synthetic only, B = synthetic + some training participants, C = synthetic + " +
                 "all training participants. Required for Imitation, ignored for ShortestPath. Written to run_config.json and every " +
                 "episodes.csv row, and read by eval_tools/trajectories.py.")]
        public string condition = "";

        [Header("Shortest-Path Baseline (agentKind = ShortestPath)")]
        [Tooltip("m/s. Default = the recording body's walk speed (humans, bot and ImitationAgent all walk at 1.3 m/s).")]
        public float shortestPathSpeedMps = ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond;
        [Tooltip("Meters. Default = the recording body's radius. Wall clearance comes from the NavMesh bake radius (logged as navmesh_bake_agent_radius), not from this.")]
        public float shortestPathRadiusM = ZarrDataContract.RecordingBody.Radius;
        [Tooltip("m/s^2. High, so the baseline is at full speed almost immediately and its time is the path length / speed.")]
        public float shortestPathAccelerationMps2 = 50f;
        [Tooltip("deg/s. 120 = the largest discrete yaw bin (12 deg per 0.1 s sample).")]
        public float shortestPathAngularSpeedDegPerS = 120f;

        [Header("Run Configuration")]
        [Tooltip("How many distinct randomized start/POI/end layouts to test.")]
        [Min(1)] public int configurationCount = 50;
        [Tooltip("How many agents to run on each configuration before advancing to the next one.")]
        [Min(1)] public int agentsPerConfiguration = 1;
        [Tooltip("If true, a fresh random start/POI/end placement is generated every configuration. " +
                 "If false (default), start/POI/end are left exactly where they are in the scene - unchanged - for every configuration, " +
                 "so all configurationCount runs reuse the same task.")]
        public bool randomizeMarkersPerConfiguration = false;
        [Tooltip("Sets UnityEngine.Time.timeScale for this run to shrink wall-clock time. The harness also caps Time.maximumDeltaTime " +
                 "at ZarrDataContract.SamplingIntervalSeconds (0.1s) so ImitationAgent's observation gate never skips ticks. The " +
                 "catch: once (real frame time x simulationSpeed) exceeds 0.1s the sim stops getting faster, so values above " +
                 "~5-6x at 60fps buy nothing. Canonical value for all eval runs: 3.")]
        [Min(0.01f)] public float simulationSpeed = 3f;
        [Tooltip("If true (default), ImitationAgent's deadlock nudge is disabled for every eval agent. That nudge fires roughly " +
                 "every 10 real seconds regardless of whether the agent is actually stuck, and moves it to the nearest NavMesh " +
                 "point without a wall check - so it inflates completion rate and distorts path length/SPL. With it off, a genuinely " +
                 "stuck agent times out. The value is written to every CSV row (nudge_enabled); a row is only counted in " +
                 "success_no_rescue if nudge_count == 0 (with the nudge on this is only true for episodes shorter than ~10 real seconds, " +
                 "because it fires on a clock, not on real stuck-ness; use stall_seconds / longest_stall_s to judge stuck-ness).")]
        public bool disableDeadlockNudge = true;
        [Tooltip("Seconds between spawning each successive agent within a configuration (0 = spawn all at once). Mirrors EngineScript's existing spawnInterval.")]
        public float spawnIntervalSeconds = 2f;
        [Tooltip("SIMULATED seconds (sum of Time.deltaTime, so independent of simulationSpeed and editor stalls). An agent that " +
                 "hasn't finished by this point is logged as outcome=timeout. Canonical value: 90 (~3x the median human duration of " +
                 "~31.4s). The time of every success is logged (success_time_s), so a stricter cutoff such as 45s can be applied " +
                 "afterwards in summarize_eval.py (--ref 45) without re-running.")]
        public float episodeTimeoutSeconds = 90f;
        [Tooltip("Sim seconds. Only written to run_config.json as a reference: the human-comparable cutoff the summary script uses by default.")]
        public float humanReferenceSeconds = 45f;

        [Header("Episode Validity")]
        [Tooltip("m/s. If an episode's mean speed OR its fastest 1-simulated-second window exceeds this, the episode is flagged " +
                 "speed_violation=1 and its outcome is rewritten to invalid_speed (original kept in raw_outcome). ImitationAgent " +
                 "clamps the model to 2.0 m/s and the fastest recorded human walk is ~1.56 m/s, so anything above ~2.5 " +
                 "means the agent was teleported or timing is broken.")]
        public float maxPlausibleSpeedMps = 2.5f;
        [Tooltip("Simulated seconds per window used for stall detection and the fastest-window speed check.")]
        public float metricWindowSeconds = 1f;
        [Tooltip("Meters. If the agent moved less than this over one metric window, that window counts towards stall_seconds " +
                 "(a harness-side stall measure that works with the nudge off, unlike stuck_count).")]
        public float stallDisplacementMeters = 0.1f;
        [Tooltip("REAL seconds to wait for a freshly spawned agent to run its first Update before giving up (agent_never_started).")]
        public float agentStartTimeoutRealSeconds = 30f;
        [Tooltip("Sim seconds. A stall (moved less than stallDisplacementMeters per window) must last at least this long to count as one stall event.")]
        public float stallEventMinSeconds = 2f;
        [Tooltip("Sim seconds. Raw CollisionCount goes up every time the agent re-touches a wall (rotation jitter while pinned re-triggers it hundreds of times). " +
                 "collision_events only counts a new event after this many sim seconds without a raw increment.")]
        public float collisionDebounceSeconds = 1f;

        [Header("Start Jitter (off by default - keeps the task identical)")]
        [Tooltip("Meters. If > 0, each configuration shifts the start position by a seeded random offset within this radius (NavMesh-valid, " +
                 "and a complete path to the end must exist). Makes agentsPerConfiguration x configurationCount runs independent samples " +
                 "instead of near-copies of one deterministic episode. Logged per row (start_offset_m). Ignored with randomized markers.")]
        public float startJitterPositionMeters = 0f;
        [Tooltip("Degrees. If > 0, each configuration rotates the agent by a seeded random angle within +-this value right after spawn " +
                 "(ImitationAgent normally spawns facing the goal). Logged per row (start_heading_jitter_deg).")]
        public float startJitterHeadingDegrees = 0f;

        [Header("Random Placement (only used when randomizeMarkersPerConfiguration is on)")]
        [Tooltip("Minimum spacing (meters) enforced between every pair of randomized start/POI/end positions each configuration, so goals don't spawn on top of each other.")]
        public float minPositionSpacingMeters = 3f;
        [Tooltip("Radius (meters) around a random candidate point to search for a valid NavMesh surface.")]
        public float navMeshSampleRadius = 2f;
        [Tooltip("How many random candidates to try per marker before giving up on that configuration's placement.")]
        public int maxPlacementAttempts = 30;

        [Header("Output")]
        [Tooltip("Leave empty to default to <project root>/SimulationLogs, matching EngineScript's existing convention.")]
        public string outputRootFolder = "";
        [Tooltip("Save a heatmap PNG every Nth episode (1 = every episode). AnalyticsManager.ExportHeatMap does a " +
                 "synchronous ~2048x2048 render+readback on the main thread every time it's called, so this is a real " +
                 "per-call cost - raise this if episodes are still stalling after maxHeatmapTrajectoryPoints. 0 or less disables PNGs entirely.")]
        public int heatmapEveryNthEpisode = 1;
        [Tooltip("Trajectory points are uniformly subsampled to at most this many before being handed to the heatmap " +
                 "exporter. ExportHeatMap's dwell-time coloring is O(n^2) in trajectory length - an uncapped ~95s " +
                 "episode at 60fps accumulates thousands of points (trajectory.Add() runs every Update(), not just " +
                 "every sampling tick), which is the main cause of the per-episode stall/lag.")]
        public int maxHeatmapTrajectoryPoints = 300;
        [Tooltip("Imitation only: also write modeldebug_config<C>_agent<A>.csv, one row per trajectory row (same t) with the goal " +
                 "vector/distance the model sees, the wall-blocked flag and the latest raw and commanded model outputs. The " +
                 "trajectory file (traj_*.csv) is always written. Use analyze_stall.py to inspect where and why an agent stalls.")]
        public bool logModelDebugTicks = true;

        [Header("Visualization")]
        [Tooltip("Color applied to every spawned agent's capsule.")]
        public Color agentColor = new Color(1f, 0.549f, 0f); // orange
        [Tooltip("Recolor the task markers with the colors below. Off by default: the markers are part of what the policy sees, " +
                 "and in every recording (human and synthetic) they keep the scene's own look (white default material).")]
        public bool colorizeTaskMarkers = false;
        [Tooltip("Color applied to the start marker(s), if they have a Renderer to recolor.")]
        public Color startMarkerColor = Color.green;
        [Tooltip("Color applied to the POI marker(s), if they have a Renderer to recolor.")]
        public Color poiMarkerColor = Color.cyan;
        [Tooltip("Color applied to the end marker(s), if they have a Renderer to recolor.")]
        public Color endMarkerColor = Color.red;

        private string runFolderPath;
        private StreamWriter csvWriter;
        private Vector3[] navMeshVertices;

        private int episodeIndex = 0; // global row/PNG counter, unique across the whole run
        private bool harnessFinished = false;
        private float previousMaximumDeltaTime = 0f;
        private int consecutivePlacementFailures = 0;
        private const int MaxConsecutivePlacementFailures = 10;

        private const string HarnessVersion = "6";
        private static readonly string TrajectoryCsvHeader = string.Join(ZarrDataContract.Trajectory.CsvDelimiter, ZarrDataContract.Trajectory.Columns);
        private static readonly string ModelDebugCsvHeader = string.Join(ZarrDataContract.Trajectory.CsvDelimiter, new[]
        {
            "t", "goal_x", "goal_y", "goal_z", "goal_dist_m", "blocked",
            "raw_fwd", "raw_yaw_deg_per_sample", "raw_pitch", "applied_fwd_mps", "applied_yaw_deg_per_s"
        });
        private string modelSha256 = "";
        private string diskModelPath;
        private string diskModelMtimeAtStart = "";
        private bool warnedDiskModelChanged = false;

        private Vector3[] baseStartPositions;
        private float currentConfigStartOffsetM = 0f;
        private float currentConfigHeadingJitterDeg = 0f;

        // Current configuration (fixed layout) state
        private int currentConfigId = -1;
        private int currentConfigSeed = 0;
        private int currentConfigAgentsSpawned = 0;
        private int currentConfigAgentsCompleted = 0;
        private float nextSpawnTime = 0f;
        private List<Vector3> currentConfigPoiPositions;
        private Vector3 currentConfigEndPos;

        // One entry per currently-running agent. Positions are snapshotted per-episode rather
        // than read from shared/mutable harness fields, since several agents can now be
        // in-flight at once (staggered spawnIntervalSeconds) and finish in any order.
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

            // Harness-side episode clock/metrics. Everything is derived from the agent's own trajectory
            // (one point per agent Update) and Time.deltaTime, so it is independent of realtime stalls
            // (heatmap export, editor hitches) and of simulationSpeed. The clock only starts once the agent
            // has run its first Update (i.e. after ImitationAgent.Start() finished its slow setup).
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

            // Trajectory sampling (sim time).
            public float nextTickSim;
            public float prevTickSim;
            public Vector3 prevTickEyePos;
            public float prevTickHeading;
        }
        private readonly List<ActiveEpisode> activeEpisodes = new List<ActiveEpisode>();

        void Awake()
        {
            // A layout scene's human FPSController has AutoStartForTesting on - without this it would record (and
            // train on) a motionless episode during every eval run.
            RecordingRig.DisableSceneRecorders("EvalHarness");
        }

        void Start()
        {
            if (task == null)
            {
                Debug.LogError("[EvalHarness] 'task' is not assigned. Refusing to start.");
                enabled = false;
                return;
            }
            if (agentKind == AgentKind.Imitation && modelAsset == null)
            {
                Debug.LogError("[EvalHarness] agentKind is Imitation but 'modelAsset' is not assigned. Refusing to start.");
                enabled = false;
                return;
            }
            if (agentKind == AgentKind.Imitation && string.IsNullOrWhiteSpace(condition))
            {
                Debug.LogError("[EvalHarness] agentKind is Imitation but 'condition' (A/B/C: what modelAsset was trained on) is empty. Refusing to start.");
                enabled = false;
                return;
            }

            var triangulation = NavMesh.CalculateTriangulation();
            navMeshVertices = triangulation.vertices;
            if (navMeshVertices == null || navMeshVertices.Length == 0)
            {
                Debug.LogError("[EvalHarness] NavMesh triangulation has no vertices - is the scene's NavMesh baked? Refusing to start.");
                enabled = false;
                return;
            }

            if (!ValidateSceneTask())
            {
                enabled = false;
                return;
            }
            if (agentKind == AgentKind.Imitation && !disableDeadlockNudge)
            {
                Debug.LogWarning("[EvalHarness] disableDeadlockNudge is OFF - the deadlock nudge will move stuck agents. " +
                    "Rows with stuck_count > 0 are excluded from success_no_rescue, but the run is not a clean baseline.");
            }

            CaptureBaseStartPositions();
            if (agentKind == AgentKind.Imitation) ComputeModelIdentity();

            previousMaximumDeltaTime = Time.maximumDeltaTime;
            // Caps one frame's scaled deltaTime at the sampling interval so ImitationAgent's single-`if` capture gate
            // can never skip ticks; at high simulationSpeed the sim just stops getting faster instead of corrupting
            // the observation cadence.
            Time.maximumDeltaTime = ZarrDataContract.SamplingIntervalSeconds;
            Time.timeScale = simulationSpeed;

            if (colorizeTaskMarkers) ColorizeTaskMarkers();
            SetupRunFolder();
            WriteRunConfig();
            AdvanceToNextConfiguration();
        }

        // ------------------------------------------------------------------
        // Scene task validation, model identity, run config
        // ------------------------------------------------------------------

        /// <summary>
        /// Checks the scene's own TaskScript before the run starts: start/end markers exist, there are enough POI
        /// markers for numberOfNeeds, and (with fixed markers) every POI and end marker is reachable over the NavMesh
        /// from every start marker. Returns false (harness refuses to run) otherwise. Nothing is moved.
        /// </summary>
        private bool ValidateSceneTask()
        {
            if (!HasMarkers(task.start) || !HasMarkers(task.end))
            {
                Debug.LogError("[EvalHarness] task.start / task.end are empty or contain null markers. Refusing to start.");
                return false;
            }
            if (task.numberOfNeeds > 0)
            {
                if (!HasMarkers(task.pointsOfInterest))
                {
                    Debug.LogError($"[EvalHarness] task.numberOfNeeds={task.numberOfNeeds} but task.pointsOfInterest is empty or contains null markers. Refusing to start.");
                    return false;
                }
                if (!task.revisit && task.numberOfNeeds > task.pointsOfInterest.Length)
                {
                    Debug.LogError($"[EvalHarness] task.numberOfNeeds={task.numberOfNeeds} exceeds the {task.pointsOfInterest.Length} POI marker(s) and revisit is off. Refusing to start.");
                    return false;
                }
            }
            else
            {
                Debug.LogWarning("[EvalHarness] task.numberOfNeeds is 0 - this scene's task has no POI (start -> end only).");
            }

            // Randomized markers are re-placed (NavMesh-valid) every configuration, so only fixed markers are checked here.
            if (randomizeMarkersPerConfiguration) return true;

            var targets = new List<GameObject>();
            if (task.numberOfNeeds > 0) targets.AddRange(task.pointsOfInterest);
            targets.AddRange(task.end);
            foreach (var startMarker in task.start)
            {
                if (!NavMesh.SamplePosition(startMarker.transform.position, out NavMeshHit startHit, navMeshSampleRadius, NavMesh.AllAreas))
                {
                    Debug.LogError($"[EvalHarness] Start marker '{startMarker.name}' at {startMarker.transform.position} is not within {navMeshSampleRadius}m of the NavMesh. Refusing to start.");
                    return false;
                }
                foreach (var target in targets)
                {
                    if (!HasCompletePath(startHit.position, target.transform.position))
                    {
                        Debug.LogError($"[EvalHarness] No complete NavMesh path from start marker '{startMarker.name}' to '{target.name}' at {target.transform.position}. Refusing to start.");
                        return false;
                    }
                }
            }

            Debug.Log($"[EvalHarness] Scene task: {task.start.Length} start(s), {task.numberOfNeeds} need(s) over " +
                $"{(task.pointsOfInterest != null ? task.pointsOfInterest.Length : 0)} POI marker(s), {task.end.Length} end(s).");
            return true;
        }

        private void CaptureBaseStartPositions()
        {
            baseStartPositions = null;
            if (!HasMarkers(task.start)) return;
            baseStartPositions = new Vector3[task.start.Length];
            for (int i = 0; i < baseStartPositions.Length; i++) baseStartPositions[i] = task.start[i].transform.position;
        }

        /// <summary>
        /// Seeded (via the per-configuration Random state) start jitter. Position is applied by moving the start
        /// markers; heading is applied to the agent on its first Update (see TickEpisodeClock).
        /// </summary>
        private void ApplyStartJitter()
        {
            currentConfigStartOffsetM = 0f;
            currentConfigHeadingJitterDeg = 0f;
            if (baseStartPositions == null) return;

            Vector2 offset = startJitterPositionMeters > 0f ? UnityEngine.Random.insideUnitCircle * startJitterPositionMeters : Vector2.zero;
            float heading = startJitterHeadingDegrees > 0f ? UnityEngine.Random.Range(-startJitterHeadingDegrees, startJitterHeadingDegrees) : 0f;

            for (int i = 0; i < baseStartPositions.Length; i++)
            {
                Vector3 target = baseStartPositions[i];
                if (offset != Vector2.zero)
                {
                    Vector3 candidate = target + new Vector3(offset.x, 0f, offset.y);
                    if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, 1f, NavMesh.AllAreas) && HasCompletePath(hit.position, currentConfigEndPos))
                    {
                        target = hit.position;
                    }
                }
                task.start[i].transform.position = target;
            }

            currentConfigStartOffsetM = Vector3.Distance(task.start[0].transform.position, baseStartPositions[0]);
            currentConfigHeadingJitterDeg = heading;
        }

        private bool HasCompletePath(Vector3 from, Vector3 to)
        {
            if (!NavMesh.SamplePosition(to, out NavMeshHit toHit, navMeshSampleRadius, NavMesh.AllAreas)) return false;
            var path = new NavMeshPath();
            return NavMesh.CalculatePath(from, toHit.position, NavMesh.AllAreas, path) && path.status == NavMeshPathStatus.PathComplete;
        }

        private static bool HasMarkers(GameObject[] markers)
        {
            if (markers == null || markers.Length == 0) return false;
            foreach (var m in markers) if (m == null) return false;
            return true;
        }

        /// <summary>SHA-256 (first 12 hex chars) of the actual .onnx file, so a CSV row identifies the exact weights, not just the file name.</summary>
        private void ComputeModelIdentity()
        {
            diskModelPath = Path.Combine(Application.dataPath, "ImitationModel", "ImitationAgentModel.onnx");
            diskModelMtimeAtStart = ReadDiskModelMtime();
            try
            {
                string assetFile = null;
#if UNITY_EDITOR
                string assetPath = UnityEditor.AssetDatabase.GetAssetPath(modelAsset);
                if (!string.IsNullOrEmpty(assetPath)) assetFile = Path.GetFullPath(Path.Combine(Application.dataPath, "..", assetPath));
#endif
                if (assetFile != null && File.Exists(assetFile))
                {
                    using (var sha = SHA256.Create())
                    using (var fs = File.OpenRead(assetFile))
                    {
                        byte[] hash = sha.ComputeHash(fs);
                        modelSha256 = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant().Substring(0, 12);
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[EvalHarness] Could not hash model file: {ex.Message}");
            }
        }

        private string ReadDiskModelMtime()
        {
            if (string.IsNullOrEmpty(diskModelPath)) return "";
            try
            {
                return File.Exists(diskModelPath) ? File.GetLastWriteTimeUtc(diskModelPath).ToString("o", CultureInfo.InvariantCulture) : "";
            }
            catch { return ""; }
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
            public string model_asset;
            public string model_sha256;
            public string model_disk_mtime_utc;
            public int configuration_count;
            public int agents_per_configuration;
            public bool randomize_markers;
            public float simulation_speed;
            public float episode_timeout_sim_s;
            public bool nudge_enabled;
            public float human_reference_s;
            public float start_jitter_position_m;
            public float start_jitter_heading_deg;
            public float stall_event_min_s;
            public float collision_debounce_s;
            public float max_plausible_speed_mps;
            public float metric_window_s;
            public float stall_displacement_m;
            public int task_number_of_needs;
            public int task_start_markers;
            public int task_poi_markers;
            public int task_end_markers;
            public bool task_revisit;
            public bool task_choose_non_deterministically;
            // Marker positions at run start, in the TaskScript's array order. With randomize_markers they are
            // re-placed every configuration - the per-episode route is in episodes.csv.
            public List<Vector3> task_start_positions;
            public List<Vector3> task_poi_positions;
            public List<Vector3> task_end_positions;
            // TaskScript's agent fields; neither agent kind moves at task_agent_speed - see body_*.
            public float task_agent_speed;
            public float task_agent_size;
            public float task_agent_radius;
            // What actually walks: the recording body (Imitation) or the NavMeshAgent (ShortestPath).
            public float body_speed_mps;
            public float body_radius_m;
            public float shortest_path_acceleration_mps2;
            public float shortest_path_angular_speed_deg_per_s;
            public float navmesh_bake_agent_radius;
            public float arrival_radius_m;
            public float tick_interval_s;
        }

        private void WriteRunConfig()
        {
            try
            {
                var dump = new RunConfigDump
                {
                    harness_version = HarnessVersion,
                    agent_kind = agentKind.ToString(),
                    condition = ConditionName(),
                    scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                    started_at = DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                    unity_version = Application.unityVersion,
                    model_asset = ModelName(),
                    model_sha256 = modelSha256,
                    model_disk_mtime_utc = diskModelMtimeAtStart,
                    configuration_count = configurationCount,
                    agents_per_configuration = agentsPerConfiguration,
                    randomize_markers = randomizeMarkersPerConfiguration,
                    simulation_speed = simulationSpeed,
                    episode_timeout_sim_s = episodeTimeoutSeconds,
                    nudge_enabled = agentKind == AgentKind.Imitation && !disableDeadlockNudge,
                    human_reference_s = humanReferenceSeconds,
                    start_jitter_position_m = startJitterPositionMeters,
                    start_jitter_heading_deg = startJitterHeadingDegrees,
                    stall_event_min_s = stallEventMinSeconds,
                    collision_debounce_s = collisionDebounceSeconds,
                    max_plausible_speed_mps = maxPlausibleSpeedMps,
                    metric_window_s = metricWindowSeconds,
                    stall_displacement_m = stallDisplacementMeters,
                    task_number_of_needs = task.numberOfNeeds,
                    task_start_markers = task.start != null ? task.start.Length : 0,
                    task_poi_markers = task.pointsOfInterest != null ? task.pointsOfInterest.Length : 0,
                    task_end_markers = task.end != null ? task.end.Length : 0,
                    task_revisit = task.revisit,
                    task_choose_non_deterministically = task.chooseNonDeterministically,
                    task_start_positions = MarkerPositions(task.start),
                    task_poi_positions = MarkerPositions(task.pointsOfInterest),
                    task_end_positions = MarkerPositions(task.end),
                    task_agent_speed = task.agentSpeed,
                    task_agent_size = task.agentSize,
                    task_agent_radius = task.agentRadius,
                    body_speed_mps = agentKind == AgentKind.Imitation ? ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond : shortestPathSpeedMps,
                    body_radius_m = agentKind == AgentKind.Imitation ? ZarrDataContract.RecordingBody.Radius : shortestPathRadiusM,
                    shortest_path_acceleration_mps2 = agentKind == AgentKind.ShortestPath ? shortestPathAccelerationMps2 : 0f,
                    shortest_path_angular_speed_deg_per_s = agentKind == AgentKind.ShortestPath ? shortestPathAngularSpeedDegPerS : 0f,
                    navmesh_bake_agent_radius = NavMesh.GetSettingsByID(0).agentRadius,
                    arrival_radius_m = ZarrDataContract.ArrivalRadiusMeters,
                    tick_interval_s = ZarrDataContract.SamplingIntervalSeconds
                };
                File.WriteAllText(Path.Combine(runFolderPath, "run_config.json"), JsonUtility.ToJson(dump, true));
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[EvalHarness] Could not write run_config.json: {ex.Message}");
            }
        }

        private string ModelName() => agentKind == AgentKind.Imitation && modelAsset != null ? modelAsset.name : "";
        private string ConditionName() => agentKind == AgentKind.Imitation ? condition.Trim() : "";

        private static List<Vector3> MarkerPositions(GameObject[] markers)
        {
            var list = new List<Vector3>();
            if (markers == null) return list;
            foreach (var m in markers) if (m != null) list.Add(m.transform.position);
            return list;
        }

        void Update()
        {
            if (harnessFinished) return;

            // 1. Check every in-flight agent for success/timeout.
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
                    // Agent hasn't run its first Update yet (ImitationAgent.Start() still initialising, or it threw).
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

            // 2. Spawn the current configuration's next agent, if it's due.
            if (currentConfigAgentsSpawned < agentsPerConfiguration && Time.realtimeSinceStartup >= nextSpawnTime)
            {
                SpawnAgentForEpisode();
                currentConfigAgentsSpawned++;
                nextSpawnTime = Time.realtimeSinceStartup + spawnIntervalSeconds;
            }

            // 3. Once every agent on this layout has been spawned AND has finished, move on.
            if (currentConfigAgentsSpawned >= agentsPerConfiguration && currentConfigAgentsCompleted >= agentsPerConfiguration)
            {
                AdvanceToNextConfiguration();
            }
        }

        // ------------------------------------------------------------------
        // Configuration (layout) lifecycle
        // ------------------------------------------------------------------

        private void AdvanceToNextConfiguration()
        {
            currentConfigId++;
            if (currentConfigId >= configurationCount)
            {
                FinishRun();
                return;
            }

            currentConfigSeed = UnityEngine.Random.Range(int.MinValue, int.MaxValue);
            UnityEngine.Random.InitState(currentConfigSeed);

            if (randomizeMarkersPerConfiguration)
            {
                if (!RandomizeTaskMarkers())
                {
                    consecutivePlacementFailures++;
                    Debug.LogError($"[EvalHarness] Configuration {currentConfigId}: could not find valid spaced NavMesh positions for all markers after {maxPlacementAttempts} attempts each. Logging as outcome=other and skipping.");
                    WriteEpisodeRow(new EpisodeRecord
                    {
                        configId = currentConfigId,
                        agentId = -1,
                        seed = currentConfigSeed,
                        outcome = "other",
                        rawOutcome = "other",
                        secondsToFirstCollision = -1f,
                        error = "navmesh_placement_failed"
                    });
                    episodeIndex++;

                    if (consecutivePlacementFailures >= MaxConsecutivePlacementFailures)
                    {
                        Debug.LogError($"[EvalHarness] {MaxConsecutivePlacementFailures} consecutive placement failures - aborting run early. Check the scene's NavMesh bake and minPositionSpacingMeters.");
                        FinishRun();
                        return;
                    }

                    AdvanceToNextConfiguration();
                    return;
                }
            }
            else
            {
                SnapshotFixedTaskMarkers();
                ApplyStartJitter();
            }
            consecutivePlacementFailures = 0;

            currentConfigAgentsSpawned = 0;
            currentConfigAgentsCompleted = 0;
            nextSpawnTime = Time.realtimeSinceStartup;
        }

        private void SpawnAgentForEpisode()
        {
            GameObject agentGO = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            agentGO.name = $"EvalHarness_{agentKind}_Config{currentConfigId}_Agent{currentConfigAgentsSpawned}";

            Collider capCollider = agentGO.GetComponent<Collider>();
            if (capCollider != null) Destroy(capCollider);

            MeshRenderer agentRenderer = agentGO.GetComponent<MeshRenderer>();
            if (agentRenderer != null)
            {
                agentRenderer.material = CreateColorMaterial(agentColor);
                // The human rig has no body; a shadow on the floor would be in the policy's RGB but never in the training data.
                agentRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            }

            IEvalAgent agent;
            ImitationAgent imitation = null;
            if (agentKind == AgentKind.Imitation)
            {
                imitation = agentGO.AddComponent<ImitationAgent>();
                imitation.task = task;
                imitation.ArchitectureLayerMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);
                imitation.BaselineModelAsset = modelAsset;
                imitation.enableDeadlockNudge = !disableDeadlockNudge;
                imitation.hotReloadModel = false; // the run evaluates exactly modelAsset (its sha256 is logged)
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
                agent = shortestPath;
            }

            // AutoFlush: both files survive stopping Play mid-episode.
            string episodeKey = $"config{currentConfigId}_agent{currentConfigAgentsSpawned}";
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
                agentId = currentConfigAgentsSpawned,
                seed = currentConfigSeed,
                poiPositions = currentConfigPoiPositions,
                endPos = currentConfigEndPos,
                startOffsetM = currentConfigStartOffsetM,
                headingJitterDeg = currentConfigHeadingJitterDeg
            });
        }

        /// <summary>
        /// Advances one episode's harness-side clock and metrics by one frame. All numbers come from the agent's own
        /// trajectory and Time.deltaTime (simulated time), never from realtime.
        /// </summary>
        private void TickEpisodeClock(ActiveEpisode ep)
        {
            List<Vector3> traj = ep.agent.Trajectory;
            if (traj == null || traj.Count == 0) return;

            if (!ep.started)
            {
                ep.started = true;
                ep.startFrame = Time.frameCount;
                ep.processedPoints = 1;
                ep.windowStartPos = traj[0];
                // The agent's Start() has just spawned it facing its goal; apply the (seeded) heading jitter now,
                // before the model's temporal window has filled and produced its first action.
                if (Mathf.Abs(ep.headingJitterDeg) > 0.001f) ep.agentGO.transform.Rotate(0f, ep.headingJitterDeg, 0f, Space.World);
                ep.prevTickEyePos = Eye(ep).position;
                ZarrDataContract.HeadingPitch(Eye(ep), out ep.prevTickHeading, out _);
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
        /// One row of the trajectory file (ZarrDataContract.Trajectory) and, if enabled, of the model-debug file, at the current
        /// episode sim time. Called on a ZarrDataContract.SamplingIntervalSeconds grid (t = 0 on the agent's first Update); t is
        /// the actual sim time of the frame that crossed the grid point (eval_tools/trajectories.py resamples onto the exact grid).
        /// Position, heading and pitch are the eye's, as ParticipantRecorder's pose is the camera's; speed and yaw rate come from
        /// the eye's change since the previous row, so they mean the same thing for both agent kinds and for recorded episodes.
        /// Model-debug: the goal vector is in the eye's frame, as in the model input; model columns are empty until the first call.
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
                FOrEmpty(im.LastCommandedForwardMps), FOrEmpty(im.LastCommandedYawDegPerSecond)));
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

            List<Vector3> trajectory = (ep.agent.Trajectory != null) ? ep.agent.Trajectory : new List<Vector3>();
            Vector3 spawnPos = trajectory.Count > 0 ? trajectory[0] : ep.agentGO.transform.position;
            Vector3 finalPos = trajectory.Count > 0 ? trajectory[trajectory.Count - 1] : ep.agentGO.transform.position;
            Vector3 finalGoalPos = ep.agent.CurrentGoalPosition;

            float simSeconds = ep.simElapsed;
            float realSeconds = Time.realtimeSinceStartup - ep.startRealtime;
            int frames = ep.started ? Time.frameCount - ep.startFrame : 0;
            int steps = Mathf.RoundToInt(simSeconds / ZarrDataContract.SamplingIntervalSeconds);

            float pathLengthM = ep.pathLengthM;

            // Optimal path: sum of NavMesh-shortest segments spawn -> POI_0 -> ... -> end, in the order this agent's
            // PoiProgressTracker planned (and therefore had to visit) them.
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

            var goalXZ = new Vector2(finalGoalPos.x, finalGoalPos.z);
            float finalDistToGoal = Vector2.Distance(new Vector2(finalPos.x, finalPos.z), goalXZ);

            string note = errorMessage;
            if (agentKind == AgentKind.Imitation && !warnedDiskModelChanged && ReadDiskModelMtime() != diskModelMtimeAtStart)
            {
                warnedDiskModelChanged = true;
                Debug.LogWarning("[EvalHarness] The model file on disk changed during this run - ImitationAgent may have hot-swapped it. Rows from here on are tagged disk_model_changed.");
            }
            if (warnedDiskModelChanged) note = string.IsNullOrEmpty(note) ? "disk_model_changed" : note + "|disk_model_changed";

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
                error = note
            });

            CloseTickWriters(ep);
            Destroy(ep.agentGO);
            episodeIndex++;
        }

        private static void CloseTickWriters(ActiveEpisode ep)
        {
            ep.trajWriter?.Close();
            ep.trajWriter = null;
            ep.modelDebugWriter?.Close();
            ep.modelDebugWriter = null;
        }

        private void FinishRun()
        {
            harnessFinished = true;
            RestoreTimeSettings();
            if (csvWriter != null)
            {
                csvWriter.Flush();
                csvWriter.Close();
                csvWriter = null;
            }
            Debug.Log($"[EvalHarness] Run complete. {episodeIndex} row(s) across {currentConfigId} configuration(s) written to '{runFolderPath}'. Editor stays in Play mode - stop it manually whenever you're done watching.");
        }

        // ------------------------------------------------------------------
        // Visualization
        // ------------------------------------------------------------------

        /// <summary>
        /// Recolors each marker's own Renderer (if it has one) once, at harness startup - marker
        /// GameObjects persist across configurations (only their position changes), so this
        /// doesn't need to re-run per configuration. If a marker has no Renderer of its own (e.g.
        /// it's only visualized via POIMarkerScript's separately-generated cylinders), this logs
        /// a warning instead of fabricating new geometry, since that would need a different, more
        /// invasive change (and risks showing up in the agent's own RGB camera differently than
        /// training data did).
        /// </summary>
        private void ColorizeTaskMarkers()
        {
            ColorizeMarkerArray(task.start, startMarkerColor, "start");
            ColorizeMarkerArray(task.pointsOfInterest, poiMarkerColor, "POI");
            ColorizeMarkerArray(task.end, endMarkerColor, "end");
        }

        private void ColorizeMarkerArray(GameObject[] markers, Color color, string roleName)
        {
            if (markers == null) return;

            foreach (var marker in markers)
            {
                if (marker == null) continue;

                Renderer r = marker.GetComponent<Renderer>();
                if (r == null) r = marker.GetComponentInChildren<Renderer>();

                if (r != null)
                {
                    r.material = CreateColorMaterial(color);
                }
                else
                {
                    Debug.LogWarning($"[EvalHarness] '{marker.name}' ({roleName} marker) has no Renderer to recolor - " +
                        "if it's visualized some other way (e.g. POIMarkerScript's generated cylinders), that needs a separate change.");
                }
            }
        }

        private static Material CreateColorMaterial(Color color)
        {
            Shader shader = Shader.Find("Diffuse");
            Material material = new Material(shader != null ? shader : Shader.Find("Standard"));
            material.color = color;
            return material;
        }

        // ------------------------------------------------------------------
        // NavMesh-valid, mutually-spaced random placement
        // ------------------------------------------------------------------

        private bool RandomizeTaskMarkers()
        {
            var chosen = new List<Vector3>();

            // Start positions aren't snapshotted here: PoiProgressTracker.ChooseSpawnPosition()
            // picks randomly among task.start's *repositioned* candidates inside each agent's own
            // Start(), so which one a given agent actually used is only known once it has spawned
            // - FinishEpisode() reads it back from that agent's own recorded trajectory instead.
            if (!RepositionArray(task.start, chosen)) return false;

            var poiChosen = new List<Vector3>();
            if (!RepositionArray(task.pointsOfInterest, chosen, poiChosen)) return false;
            currentConfigPoiPositions = poiChosen;

            var endChosen = new List<Vector3>();
            if (!RepositionArray(task.end, chosen, endChosen)) return false;
            currentConfigEndPos = endChosen.Count > 0 ? endChosen[0] : Vector3.zero;

            return true;
        }

        /// <summary>
        /// Used instead of RandomizeTaskMarkers() when randomizeMarkersPerConfiguration is false:
        /// reads the start/POI/end markers' CURRENT positions (whatever they already are in the
        /// scene - untouched) into currentConfigPoiPositions/currentConfigEndPos, purely so
        /// FinishEpisode()/WriteEpisodeRow() have something to log. Mirrors RandomizeTaskMarkers()'s
        /// contract but never moves anything and cannot fail placement.
        /// </summary>
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

            currentConfigEndPos = (task.end != null && task.end.Length > 0 && task.end[0] != null)
                ? task.end[0].transform.position
                : Vector3.zero;
        }

        private bool RepositionArray(GameObject[] markers, List<Vector3> chosen, List<Vector3> outPositions = null)
        {
            if (markers == null) return true;

            foreach (var marker in markers)
            {
                if (marker == null) continue;

                if (!TryFindSpacedNavMeshPosition(chosen, out Vector3 pos))
                {
                    return false;
                }

                marker.transform.position = pos;
                chosen.Add(pos);
                if (outPositions != null) outPositions.Add(pos);
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

                if (NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleRadius, NavMesh.AllAreas))
                {
                    bool farEnough = true;
                    for (int i = 0; i < alreadyChosen.Count; i++)
                    {
                        if (Vector3.Distance(hit.position, alreadyChosen[i]) < minPositionSpacingMeters)
                        {
                            farEnough = false;
                            break;
                        }
                    }

                    if (farEnough)
                    {
                        result = hit.position;
                        return true;
                    }
                }
            }

            result = Vector3.zero;
            return false;
        }

        private float ComputeNavMeshPathLength(Vector3 from, Vector3 to)
        {
            // Snap both ends onto the NavMesh first: a marker floating above the floor (e.g. end at y=1) makes
            // CalculatePath fail, which used to silently fall back to the straight-line distance.
            if (NavMesh.SamplePosition(from, out NavMeshHit fromHit, navMeshSampleRadius, NavMesh.AllAreas)) from = fromHit.position;
            if (NavMesh.SamplePosition(to, out NavMeshHit toHit, navMeshSampleRadius, NavMesh.AllAreas)) to = toHit.position;

            NavMeshPath path = new NavMeshPath();
            if (!NavMesh.CalculatePath(from, to, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete)
            {
                return Vector3.Distance(from, to);
            }

            float length = 0f;
            for (int i = 1; i < path.corners.Length; i++)
            {
                length += Vector3.Distance(path.corners[i - 1], path.corners[i]);
            }
            return length;
        }

        // ------------------------------------------------------------------
        // Output: run folder, per-episode PNG, per-episode CSV row
        // ------------------------------------------------------------------

        private void SetupRunFolder()
        {
            string root = string.IsNullOrWhiteSpace(outputRootFolder)
                ? Path.Combine(Application.dataPath, "..", "SimulationLogs")
                : outputRootFolder;

            string sceneName = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            string baseName = $"EvalRun_{sceneName}_{agentKind}_{DateTime.Now:yyyyMMdd_HHmmss}";
            string candidate = Path.Combine(root, baseName);

            int suffix = 0;
            while (Directory.Exists(candidate))
            {
                suffix++;
                candidate = Path.Combine(root, $"{baseName}_{suffix}");
            }

            Directory.CreateDirectory(candidate);
            runFolderPath = candidate;

            string csvPath = Path.Combine(runFolderPath, "episodes.csv");
            csvWriter = new StreamWriter(csvPath, append: false);
            csvWriter.WriteLine(string.Join(";", CsvColumns));
            csvWriter.Flush();

            Debug.Log($"[EvalHarness] Run folder created at '{runFolderPath}'. Testing {agentKind} on {configurationCount} configuration(s) x {agentsPerConfiguration} agent(s) each.");
        }

        private void SaveEpisodeHeatmap(List<Vector3> trajectory, string pngFilename)
        {
            try
            {
                var trajectoryCopy = SubsampleTrajectory(trajectory, maxHeatmapTrajectoryPoints);
                var allAgentPositions = new List<List<Vector3>>[] { new List<List<Vector3>> { trajectoryCopy } };

                var pois = new List<GameObject>();
                if (task.start != null) pois.AddRange(task.start);
                if (task.pointsOfInterest != null) pois.AddRange(task.pointsOfInterest);
                if (task.end != null) pois.AddRange(task.end);

                string pngPath = Path.Combine(runFolderPath, pngFilename);
                AnalyticsManager.ExportHeatMap(allAgentPositions, pois, pngPath);
            }
            catch (Exception ex)
            {
                Debug.LogError($"[EvalHarness] Episode {episodeIndex}: heatmap export failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Uniformly subsamples a trajectory down to at most maxPoints, keeping the first and
        /// last point. Exists because AnalyticsManager.ExportHeatMap's dwell-time coloring is
        /// O(n^2) in trajectory length, and trajectory.Add() in ImitationAgent.Update() runs
        /// every frame (not just every observation-sampling tick) - an uncapped near-timeout
        /// episode can have thousands of points, turning one heatmap export into a multi-second
        /// main-thread stall.
        /// </summary>
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
            "nudge_count", "png_filename", "trajectory_file", "error_message"
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

            // One NavMesh length per route leg (start -> POI_0, ..., POI_last -> end); they sum to optimal_path_length_m.
            string optimalSegmentsJoined = "";
            if (r.optimalSegmentsM != null && r.optimalSegmentsM.Count > 0)
            {
                var parts = new string[r.optimalSegmentsM.Count];
                for (int i = 0; i < parts.Length; i++) parts[i] = F(r.optimalSegmentsM[i]);
                optimalSegmentsJoined = string.Join("|", parts);
            }

            // Model identity, nudge and collision columns only exist for ImitationAgent (a NavMeshAgent cannot touch a wall).
            bool imitation = agentKind == AgentKind.Imitation;
            string[] cells =
            {
                I(episodeIndex),
                I(r.configId),
                I(r.agentId),
                DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                HarnessVersion,
                agentKind.ToString(),
                ConditionName(),
                UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                ModelName(),
                modelSha256,
                ReadDiskModelMtime(),
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
                r.error ?? ""
            };

            csvWriter.WriteLine(string.Join(";", cells));
            csvWriter.Flush();
        }

        private void RestoreTimeSettings()
        {
            Time.timeScale = 1f;
            if (previousMaximumDeltaTime > 0f) Time.maximumDeltaTime = previousMaximumDeltaTime;
        }

        void OnDestroy()
        {
            RestoreTimeSettings();
            foreach (ActiveEpisode ep in activeEpisodes) CloseTickWriters(ep);
            if (csvWriter != null)
            {
                csvWriter.Flush();
                csvWriter.Close();
                csvWriter = null;
            }
        }
    }
}
