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
    /// Closed-loop evaluation harness for ImitationAgent. Runs configurationCount scene
    /// configurations back-to-back in a single Editor Play session (no manual replay, no
    /// headless/batchmode). By default (enforceCanonicalTask) every layout gets the same fixed
    /// start -> end task with no POIs. Otherwise, for each configuration: if
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
    /// Does not touch AgentScript/EngineScript at all - this only drives ImitationAgent.
    /// Aggregating these rows into Goal Completion Rate / Collision Frequency / MTTC / SPL is a
    /// deliberately separate downstream (pandas) step, not this harness's job.
    /// </summary>
    public class EvalHarness : MonoBehaviour
    {
        [Header("Task & Model")]
        [Tooltip("The TaskScript whose start/end/pointsOfInterest marker GameObjects get repositioned each configuration. Must be the same TaskScript ImitationAgent/ParticipantRecorder use for this scene.")]
        public TaskScript task;
        [Tooltip("The .onnx model every spawned agent runs.")]
        public ModelAsset modelAsset;

        [Header("Run Configuration")]
        [Tooltip("How many distinct randomized start/POI/end layouts to test.")]
        [Min(1)] public int configurationCount = 50;
        [Tooltip("How many agents to run on each configuration before advancing to the next one.")]
        [Min(1)] public int agentsPerConfiguration = 1;
        [Tooltip("If true, a fresh random start/POI/end placement is generated every configuration. " +
                 "If false (default), start/POI/end are left exactly where they are in the scene - unchanged - for every configuration, " +
                 "so all configurationCount runs reuse the same task. Forced to false while enforceCanonicalTask is on.")]
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

        [Header("Canonical Task (identical across every layout)")]
        [Tooltip("If true (default), at Start() every task.start marker is moved to canonicalStart, every task.end marker to " +
                 "canonicalEnd and task.numberOfNeeds is set to 0 (no POIs), IN MEMORY ONLY - nothing is saved to the scene. " +
                 "This guarantees every layout is evaluated on exactly the same task. Also forces randomizeMarkersPerConfiguration=false. " +
                 "Refuses to run if the start isn't on the NavMesh or no complete path exists.")]
        public bool enforceCanonicalTask = true;
        public Vector3 canonicalStart = new Vector3(-11.25f, 0.08333334f, 0f);
        public Vector3 canonicalEnd = new Vector3(11.25f, 1f, 0f);

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
        [Tooltip("If true, writes ticks_config<C>_agent<A>.csv per agent into the run folder: one row per model call with raw " +
                 "(unclamped) and applied forward/yaw/pitch, pose, goal vector/distance and wall-blocked flag. Use " +
                 "analyze_stall.py to inspect where and why an agent stalls.")]
        public bool logPerTickOutputs = true;

        [Header("Visualization")]
        [Tooltip("Color applied to every spawned agent's capsule.")]
        public Color agentColor = new Color(1f, 0.549f, 0f); // orange
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

        private const string HarnessVersion = "3";
        private string modelSha256 = "unknown";
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
            public ImitationAgent agent;
            public float startRealtime;
            public int configId;
            public int agentId;
            public int seed;
            public List<Vector3> poiPositions;
            public Vector3 endPos;
            public StreamWriter tickWriter;

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
        }
        private readonly List<ActiveEpisode> activeEpisodes = new List<ActiveEpisode>();

        void Start()
        {
            if (task == null)
            {
                Debug.LogError("[EvalHarness] 'task' is not assigned. Refusing to start.");
                enabled = false;
                return;
            }
            if (modelAsset == null)
            {
                Debug.LogError("[EvalHarness] 'modelAsset' is not assigned. Refusing to start.");
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

            if (!ApplyCanonicalTask())
            {
                enabled = false;
                return;
            }
            if (!disableDeadlockNudge)
            {
                Debug.LogWarning("[EvalHarness] disableDeadlockNudge is OFF - the deadlock nudge will move stuck agents. " +
                    "Rows with stuck_count > 0 are excluded from success_no_rescue, but the run is not a clean baseline.");
            }

            CaptureBaseStartPositions();
            ComputeModelIdentity();

            previousMaximumDeltaTime = Time.maximumDeltaTime;
            // Caps one frame's scaled deltaTime at the sampling interval so ImitationAgent's single-`if` capture gate
            // can never skip ticks; at high simulationSpeed the sim just stops getting faster instead of corrupting
            // the observation cadence.
            Time.maximumDeltaTime = ZarrDataContract.SamplingIntervalSeconds;
            Time.timeScale = simulationSpeed;

            ColorizeTaskMarkers();
            SetupRunFolder();
            WriteRunConfig();
            AdvanceToNextConfiguration();
        }

        // ------------------------------------------------------------------
        // Canonical task, model identity, run config
        // ------------------------------------------------------------------

        /// <summary>
        /// Forces the identical start -> end task (no POIs) onto whatever scene this harness is in, in memory
        /// only. Returns false (harness refuses to run) if the task can't be made valid on this layout.
        /// </summary>
        private bool ApplyCanonicalTask()
        {
            if (!enforceCanonicalTask) return true;

            if (randomizeMarkersPerConfiguration)
            {
                Debug.LogWarning("[EvalHarness] enforceCanonicalTask is on - forcing randomizeMarkersPerConfiguration=false.");
                randomizeMarkersPerConfiguration = false;
            }

            if (!HasMarkers(task.start) || !HasMarkers(task.end))
            {
                Debug.LogError("[EvalHarness] enforceCanonicalTask: task.start / task.end are empty or contain null markers. Refusing to start.");
                return false;
            }
            if (task.pointsOfInterest == null) task.pointsOfInterest = new GameObject[0];

            foreach (var m in task.start) m.transform.position = canonicalStart;
            foreach (var m in task.end) m.transform.position = canonicalEnd;
            task.numberOfNeeds = 0;

            if (!NavMesh.SamplePosition(canonicalStart, out NavMeshHit startHit, navMeshSampleRadius, NavMesh.AllAreas))
            {
                Debug.LogError($"[EvalHarness] Canonical start {canonicalStart} is not within {navMeshSampleRadius}m of the NavMesh on this layout. Refusing to start.");
                return false;
            }
            if (!NavMesh.SamplePosition(canonicalEnd, out NavMeshHit endHit, navMeshSampleRadius, NavMesh.AllAreas))
            {
                Debug.LogError($"[EvalHarness] Canonical end {canonicalEnd} is not within {navMeshSampleRadius}m of the NavMesh on this layout. Refusing to start.");
                return false;
            }
            var path = new NavMeshPath();
            if (!NavMesh.CalculatePath(startHit.position, endHit.position, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete)
            {
                Debug.LogError("[EvalHarness] No complete NavMesh path from canonical start to canonical end on this layout. Refusing to start.");
                return false;
            }

            Debug.Log($"[EvalHarness] Canonical task enforced (in memory): start={canonicalStart}, end={canonicalEnd}, numberOfNeeds=0.");
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
            public string scene;
            public string started_at;
            public string unity_version;
            public string model_asset;
            public string model_sha256;
            public string model_disk_mtime_utc;
            public int configuration_count;
            public int agents_per_configuration;
            public bool randomize_markers;
            public bool enforce_canonical_task;
            public Vector3 canonical_start;
            public Vector3 canonical_end;
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
            public float agent_speed;
            public float agent_size;
            public float agent_radius;
            public float arrival_radius_m;
        }

        private void WriteRunConfig()
        {
            try
            {
                var dump = new RunConfigDump
                {
                    harness_version = HarnessVersion,
                    scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                    started_at = DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                    unity_version = Application.unityVersion,
                    model_asset = modelAsset != null ? modelAsset.name : "",
                    model_sha256 = modelSha256,
                    model_disk_mtime_utc = diskModelMtimeAtStart,
                    configuration_count = configurationCount,
                    agents_per_configuration = agentsPerConfiguration,
                    randomize_markers = randomizeMarkersPerConfiguration,
                    enforce_canonical_task = enforceCanonicalTask,
                    canonical_start = canonicalStart,
                    canonical_end = canonicalEnd,
                    simulation_speed = simulationSpeed,
                    episode_timeout_sim_s = episodeTimeoutSeconds,
                    nudge_enabled = !disableDeadlockNudge,
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
                    agent_speed = task.agentSpeed,
                    agent_size = task.agentSize,
                    agent_radius = task.agentRadius,
                    arrival_radius_m = ZarrDataContract.ArrivalRadiusMeters
                };
                File.WriteAllText(Path.Combine(runFolderPath, "run_config.json"), JsonUtility.ToJson(dump, true));
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[EvalHarness] Could not write run_config.json: {ex.Message}");
            }
        }

        void Update()
        {
            if (harnessFinished) return;

            // 1. Check every in-flight agent for success/timeout.
            for (int i = activeEpisodes.Count - 1; i >= 0; i--)
            {
                ActiveEpisode ep = activeEpisodes[i];

                if (ep.agent == null)
                {
                    // Destroyed some other way than FinishEpisode() - drop it without a row rather than crash.
                    CloseTickWriter(ep);
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

                if (ep.agent.destroyRequest)
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
            agentGO.name = $"EvalHarness_Config{currentConfigId}_Agent{currentConfigAgentsSpawned}";

            Collider capCollider = agentGO.GetComponent<Collider>();
            if (capCollider != null) Destroy(capCollider);

            MeshRenderer agentRenderer = agentGO.GetComponent<MeshRenderer>();
            if (agentRenderer != null) agentRenderer.material = CreateColorMaterial(agentColor);

            ImitationAgent agent = agentGO.AddComponent<ImitationAgent>();
            agent.task = task;
            agent.ArchitectureLayerMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);
            agent.BaselineModelAsset = modelAsset;
            agent.enableDeadlockNudge = !disableDeadlockNudge;

            StreamWriter tickWriter = null;
            if (logPerTickOutputs)
            {
                string tickPath = Path.Combine(runFolderPath, $"ticks_config{currentConfigId}_agent{currentConfigAgentsSpawned}.csv");
                tickWriter = new StreamWriter(tickPath, false) { AutoFlush = true }; // survives stopping Play mid-episode
                tickWriter.WriteLine("t;x;z;heading_deg;goal_x;goal_y;goal_z;goal_dist_m;raw_fwd;raw_yaw_deg_per_sample;raw_pitch;applied_fwd_mps;applied_yaw_deg_per_s;blocked;pois_reached");
                StreamWriter w = tickWriter;
                agent.TickLogSink = row => w.WriteLine(row);
            }

            activeEpisodes.Add(new ActiveEpisode
            {
                tickWriter = tickWriter,
                agentGO = agentGO,
                agent = agent,
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
            List<Vector3> traj = ep.agent.trajectory;
            if (traj == null || traj.Count == 0) return;

            if (!ep.started)
            {
                ep.started = true;
                ep.startFrame = Time.frameCount;
                ep.processedPoints = 1;
                ep.windowStartPos = traj[0];
                // ImitationAgent.Start() has just spawned the agent facing its goal; apply the (seeded) heading jitter now,
                // before the model's temporal window has filled and produced its first action.
                if (Mathf.Abs(ep.headingJitterDeg) > 0.001f) ep.agent.transform.Rotate(0f, ep.headingJitterDeg, 0f, Space.World);
                return;
            }

            ep.simElapsed += Time.deltaTime;
            ProcessNewTrajectoryPoints(ep);

            if (!ep.firstCollisionLogged && ep.agent.HasCollided)
            {
                ep.firstCollisionLogged = true;
                ep.simAtFirstCollision = ep.simElapsed;
            }

            int rawCollisions = ep.agent.CollisionCount;
            if (rawCollisions > ep.lastRawCollisions)
            {
                if (ep.collisionEvents == 0 || ep.simElapsed - ep.lastCollisionSim > collisionDebounceSeconds) ep.collisionEvents++;
                ep.lastCollisionSim = ep.simElapsed;
                ep.lastRawCollisions = rawCollisions;
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

        private static void ProcessNewTrajectoryPoints(ActiveEpisode ep)
        {
            List<Vector3> traj = ep.agent.trajectory;
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

            List<Vector3> trajectory = (ep.agent.trajectory != null) ? ep.agent.trajectory : new List<Vector3>();
            Vector3 spawnPos = trajectory.Count > 0 ? trajectory[0] : ep.agent.transform.position;
            Vector3 finalPos = trajectory.Count > 0 ? trajectory[trajectory.Count - 1] : ep.agent.transform.position;
            Vector3 finalGoalPos = ep.agent.CurrentGoalPosition;

            float simSeconds = ep.simElapsed;
            float realSeconds = Time.realtimeSinceStartup - ep.startRealtime;
            int frames = ep.started ? Time.frameCount - ep.startFrame : 0;
            int steps = Mathf.RoundToInt(simSeconds / ZarrDataContract.SamplingIntervalSeconds);

            float pathLengthM = ep.pathLengthM;

            // Simplification: optimal path is the NavMesh-shortest route from spawn straight to the episode's end
            // target. With the canonical task (0 POIs) that is exactly the whole task.
            float optimalPathLengthM = ComputeNavMeshPathLength(spawnPos, ep.endPos);

            float meanSpeed = pathLengthM / Mathf.Max(simSeconds, 0.1f);
            bool speedViolation = meanSpeed > maxPlausibleSpeedMps || ep.maxWindowSpeedMps > maxPlausibleSpeedMps;
            string rawOutcome = outcome;
            if (speedViolation) outcome = "invalid_speed";

            int stuckCount = ep.agent.StuckCount;
            bool successNoRescue = outcome == "success" && stuckCount == 0;
            float spl = successNoRescue ? Mathf.Clamp01(optimalPathLengthM / Mathf.Max(pathLengthM, optimalPathLengthM)) : 0f;

            var goalXZ = new Vector2(finalGoalPos.x, finalGoalPos.z);
            float finalDistToGoal = Vector2.Distance(new Vector2(finalPos.x, finalPos.z), goalXZ);

            string note = errorMessage;
            if (!warnedDiskModelChanged && ReadDiskModelMtime() != diskModelMtimeAtStart)
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
                poiPositions = ep.poiPositions,
                endPos = ep.endPos,
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
                spl = spl,
                meanSpeedMps = meanSpeed,
                maxWindowSpeedMps = ep.maxWindowSpeedMps,
                maxStepM = ep.maxStepM,
                finalDistToGoalM = finalDistToGoal,
                collisions = ep.agent.CollisionCount,
                collisionEvents = ep.collisionEvents,
                secondsToFirstCollision = ep.firstCollisionLogged ? ep.simAtFirstCollision : -1f,
                stallSeconds = ep.stallSeconds,
                stallEvents = ep.stallEvents,
                longestStallS = ep.longestStallSeconds,
                longestStallPos = ep.longestStallSeconds > 0f ? (Vector3?)ep.longestStallPos : null,
                nudgeCount = stuckCount,
                png = pngFilename,
                error = note
            });

            CloseTickWriter(ep);
            Destroy(ep.agentGO);
            episodeIndex++;
        }

        private static void CloseTickWriter(ActiveEpisode ep)
        {
            if (ep.tickWriter == null) return;
            if (ep.agent != null) ep.agent.TickLogSink = null;
            ep.tickWriter.Flush();
            ep.tickWriter.Close();
            ep.tickWriter = null;
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
            string baseName = $"EvalRun_{sceneName}_{DateTime.Now:yyyyMMdd_HHmmss}";
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

            Debug.Log($"[EvalHarness] Run folder created at '{runFolderPath}'. Testing {configurationCount} configuration(s) x {agentsPerConfiguration} agent(s) each.");
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
            "episode_id", "config_id", "agent_id", "timestamp", "harness_version", "scene",
            "model_checkpoint", "model_sha256", "model_disk_mtime_utc", "random_seed",
            "layout_mode", "sim_speed", "timeout_s", "nudge_enabled",
            "start_pos", "start_offset_m", "start_heading_jitter_deg", "poi_positions", "end_pos",
            "outcome", "raw_outcome", "success_no_rescue", "speed_violation",
            "num_pois_total", "num_pois_reached",
            "episode_length_steps", "episode_length_seconds", "success_time_s", "time_to_first_poi_s", "poi_times_s",
            "real_seconds", "frames",
            "path_length_m", "optimal_path_length_m", "spl",
            "mean_speed_mps", "max_window_speed_mps", "max_step_m", "final_dist_to_goal_m",
            "collision_count", "collision_events", "steps_to_first_collision", "seconds_to_first_collision",
            "stall_seconds", "stall_events", "longest_stall_s", "longest_stall_pos",
            "nudge_count", "png_filename", "error_message"
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
            public string error;
        }

        private static string F(float v) => v.ToString("F3", CultureInfo.InvariantCulture);
        private static string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        private static string B(bool v) => v ? "1" : "0";

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

            string[] cells =
            {
                I(episodeIndex),
                I(r.configId),
                I(r.agentId),
                DateTime.Now.ToString("o", CultureInfo.InvariantCulture),
                HarnessVersion,
                UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                modelAsset != null ? modelAsset.name : "",
                modelSha256,
                ReadDiskModelMtime(),
                I(r.seed),
                randomizeMarkersPerConfiguration ? "randomized" : "fixed",
                F(simulationSpeed),
                F(episodeTimeoutSeconds),
                B(!disableDeadlockNudge),
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
                F(r.spl),
                F(r.meanSpeedMps),
                F(r.maxWindowSpeedMps),
                F(r.maxStepM),
                F(r.finalDistToGoalM),
                I(r.collisions),
                I(r.collisionEvents),
                stepsToFirstCollision >= 0 ? I(stepsToFirstCollision) : "",
                r.secondsToFirstCollision >= 0f ? F(r.secondsToFirstCollision) : "",
                F(r.stallSeconds),
                I(r.stallEvents),
                F(r.longestStallS),
                r.longestStallPos.HasValue ? FormatVector(r.longestStallPos.Value) : "",
                I(r.nudgeCount),
                r.png ?? "",
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
            foreach (ActiveEpisode ep in activeEpisodes) CloseTickWriter(ep);
            if (csvWriter != null)
            {
                csvWriter.Flush();
                csvWriter.Close();
                csvWriter = null;
            }
        }
    }
}
