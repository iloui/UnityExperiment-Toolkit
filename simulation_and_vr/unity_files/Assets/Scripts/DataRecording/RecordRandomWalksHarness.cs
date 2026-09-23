using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;
using UnityStandardAssets.CrossPlatformInput;
using UnityStandardAssets.Characters.FirstPerson;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Drop-in Editor Play-mode harness that generates synthetic episodes on whatever layout (baked NavMesh +
    /// Architecture_Obstacle geometry) the current scene contains - recorded through exactly the same pipeline as a
    /// human participant: the scene's own FPSController (or a fresh FPSController prefab instance if the scene has none),
    /// its FirstPersonCharacter camera and the ParticipantRecorder on it. The only thing replaced is the person at the
    /// keyboard: RecordRandomWalksAgent feeds the walk key and mouse through CrossPlatformInputManager's virtual axes.
    ///
    /// Matched to the human recordings on purpose:
    ///   - rig/kinematics: FirstPersonController (1.3 m/s, 50 Hz FixedUpdate, 0.25 m CharacterController), MouseLook
    ///   - camera: the rig's own camera and optics; eye height set to ZarrDataContract.CameraHeightAboveNavMeshMeters
    ///   - frame timing: a fixed simulatedFrameSeconds step (Time.captureDeltaTime) instead of Time.timeScale, so the
    ///     recorder's 0.1 s samples are as evenly spaced as in the human sessions (0.100-0.101 s) and the per-sample
    ///     action labels (displacement / 0.1 s) are not inflated by frame-quantised sampling
    ///   - markers: copies of this scene's own task markers (mesh, material, size, height above the floor); the scene's
    ///     originals are hidden, and a start marker is shown only if the scene's start is a visible marker
    ///   - no visible body (the human rig has none, so nothing casts a shadow into the RGB)
    ///
    /// For each of `pairCount` (k) random start/end pairs, `agentsPerPair` (m) walks are recorded one after another.
    /// With probability waypointProbability a walk goes start -> random waypoint -> end, the waypoint registered as the
    /// task's single POI (exactly the human 1-POI format); otherwise start -> end directly. Episodes are written to
    /// imitationagent_training/zarr_recordings with participant_id "synthetic_rrw" and session_id = scene name.
    ///
    /// An episode is discarded and re-sampled when: no path exists, the route is longer than maxPathLengthFactor x the
    /// shortest path, the end is not reached before the timeout, or it lasted less than minEpisodeSeconds. After
    /// maxResampleAttempts failures for one slot the pair is skipped with a warning.
    ///
    /// Owns its own runtime TaskScript + markers; the scene's authored task is never modified (its markers are only
    /// hidden during Play). The scene rig's own auto-recording is switched off for the run.
    /// </summary>
    public class RecordRandomWalksHarness : MonoBehaviour
    {
        [Header("Run Configuration")]
        [Tooltip("k: number of random start/end pairs on this layout.")]
        [Min(1)] public int pairCount = 25;
        [Tooltip("m: number of walks (recordings) per start/end pair. Total episodes = k x m.")]
        [Min(1)] public int agentsPerPair = 1;
        [Tooltip("Seed for all sampling. 0 = time-based (the used seed is logged).")]
        public int seed = 0;
        [Tooltip("Max attempts per slot (pair sampling, and each recording slot) before the pair is skipped.")]
        [Min(1)] public int maxResampleAttempts = 15;
        [Tooltip("Leave Play mode automatically when the run has finished.")]
        public bool stopPlayModeWhenDone = true;

        [Header("Rig")]
        [Tooltip("Instantiated only if the scene has no FirstPersonController. Normally Assets/Scenes/FPSController.prefab " +
                 "(assigned by Tools > Random Walks > Add RecordRandomWalksHarness).")]
        public GameObject fpsControllerPrefab;

        [Header("Frame timing")]
        [Tooltip("Simulated seconds per frame (Time.captureDeltaTime). The human sessions ran at ~1 ms frames; 2 ms keeps the " +
                 "recorder's sample spacing within 0.100-0.102 s. The run is as fast as the machine renders frames - smaller is " +
                 "more exact but slower.")]
        [Range(0.0005f, 0.02f)] public float simulatedFrameSeconds = 0.002f;
        [Tooltip("Stop the rig camera from drawing the Game view every frame (the recorder renders its own 64x64 capture " +
                 "on each sample regardless). Much faster; the recorded data is identical.")]
        public bool disableGameViewRendering = true;

        [Header("Route")]
        [Tooltip("Max total route length (start -> waypoint -> end) as a multiple of the shortest start -> end path.")]
        public float maxPathLengthFactor = 3f;
        [Tooltip("Probability that an episode gets a random intermediate waypoint (registered as the task's POI). " +
                 "The rest walk start -> end directly with no POI. 0.5 covers both eval task types (CentralPillar: 1 POI, TJunction: none).")]
        [Range(0f, 1f)] public float waypointProbability = 0.5f;
        [Tooltip("A waypoint must be at least this far (straight line) from both start and end, so it is neither 'reached' on the first tick nor a no-op.")]
        public float waypointMinDistanceMeters = 3f;
        [Tooltip("Start/end pairs whose shortest path is shorter than this are re-sampled (they can't produce a minEpisodeSeconds-long walk).")]
        public float minPairPathLengthMeters = 6f;
        [Tooltip("Radius (meters) to snap a random candidate point onto the NavMesh.")]
        public float navMeshSampleRadius = 2f;

        [Header("Timeout")]
        [Tooltip("Timeout = max(minTimeoutSeconds, timeoutFactor x plannedLength / walk speed + behaviourTimeBudgetSeconds), simulated seconds.")]
        public float minTimeoutSeconds = 30f;
        public float timeoutFactor = 3f;
        [Tooltip("Extra seconds for the start turn, look-around pauses and wall excursions.")]
        public float behaviourTimeBudgetSeconds = 15f;

        [Header("Episode filter")]
        [Tooltip("Episodes shorter than this (simulated seconds) are discarded and re-sampled.")]
        public float minEpisodeSeconds = 5f;

        [Header("Synthetic participant")]
        public RandomWalkBotSettings bot = new RandomWalkBotSettings();

        private System.Random rng;
        private Vector3[] triVerts;
        private int[] triIndices;
        private float[] triCumulativeArea;
        private float previousCaptureDeltaTime;
        private bool timeSettingsApplied;
        private bool inputSwitched;

        private TaskScript task;
        private Transform startMarker;
        private Transform endMarker;
        private Transform waypointMarker;
        private RecordingRig.MarkerTemplate startTemplate;
        private RecordingRig.MarkerTemplate poiTemplate;
        private RecordingRig.MarkerTemplate endTemplate;

        private FirstPersonController rig;
        private CharacterController rigController;
        private Transform rigCamera;
        private ParticipantRecorder recorder;
        private RecordRandomWalksAgent agent;
        private LayerMask architectureMask;
        private bool fatal;
        private bool lastEpisodeOk;

        private int episodesRecorded;
        private int episodesDiscarded;
        private int pairsSkipped;

        void Awake()
        {
            // Must run before the scene rig's ParticipantRecorder.Start (AutoStartForTesting is on in the layout scenes).
            RecordingRig.DisableSceneRecorders("RecordRandomWalks");
        }

        IEnumerator Start()
        {
            if (!Validate()) yield break;

            int usedSeed = seed != 0 ? seed : Environment.TickCount;
            rng = new System.Random(usedSeed);

            if (!BuildRuntimeObjects()) yield break;
            yield return null; // let TaskScript.Start(), FirstPersonController.Start() (Camera.main) etc. run first
            if (disableGameViewRendering) rigCamera.GetComponent<Camera>().enabled = false;

            CrossPlatformInputManager.SwitchActiveInputMethod(CrossPlatformInputManager.ActiveInputMethod.Touch);
            inputSwitched = true;
            previousCaptureDeltaTime = Time.captureDeltaTime;
            Time.captureDeltaTime = simulatedFrameSeconds;
            Time.timeScale = 1f;
            timeSettingsApplied = true;

            Debug.Log($"[RecordRandomWalks] Starting: k={pairCount} pairs x m={agentsPerPair} walks, seed={usedSeed}, rig='{rig.name}', " +
                $"walk speed={rig.WalkSpeed:F2} m/s, frame={simulatedFrameSeconds * 1000f:F1} ms, waypointProbability={waypointProbability:F2}, " +
                $"look-arounds={bot.lookAroundRatePerMinute:F1}/min, wall excursions={bot.wallExcursionRatePerMinute:F1}/min, minEpisode={minEpisodeSeconds:F1}s.");

            for (int pair = 0; pair < pairCount && !fatal; pair++)
            {
                if (!TrySamplePair(out Vector3 start, out Vector3 end, out float directLength))
                {
                    Debug.LogWarning($"[RecordRandomWalks] Pair {pair + 1}/{pairCount}: no valid start/end after {maxResampleAttempts} attempts - skipped.");
                    pairsSkipped++;
                    continue;
                }
                Debug.Log($"[RecordRandomWalks] Pair {pair + 1}/{pairCount}: shortest path {directLength:F1} m.");

                for (int round = 0; round < agentsPerPair && !fatal; round++)
                {
                    bool ok = false;
                    for (int attempt = 1; attempt <= maxResampleAttempts && !fatal; attempt++)
                    {
                        yield return RunEpisode(start, end, directLength);
                        if (lastEpisodeOk) { ok = true; break; }
                        episodesDiscarded++;
                    }

                    if (!ok && !fatal)
                    {
                        Debug.LogWarning($"[RecordRandomWalks] Pair {pair + 1}/{pairCount}, round {round + 1}/{agentsPerPair}: " +
                            $"{maxResampleAttempts} failed attempts - skipping the rest of this pair.");
                        pairsSkipped++;
                        break;
                    }
                    Debug.Log($"[RecordRandomWalks] Pair {pair + 1}/{pairCount}, round {round + 1}/{agentsPerPair} recorded ({episodesRecorded} total).");
                }
            }

            Debug.Log($"[RecordRandomWalks] Done. Recorded {episodesRecorded} episodes (target {pairCount * agentsPerPair}), " +
                $"discarded {episodesDiscarded} attempts, skipped {pairsSkipped} pair(s){(fatal ? " - ABORTED by an error, see log" : "")}.");
            RestoreSettings();
#if UNITY_EDITOR
            if (stopPlayModeWhenDone) UnityEditor.EditorApplication.isPlaying = false;
#endif
        }

        // ------------------------------------------------------------------
        // Setup
        // ------------------------------------------------------------------

        private bool Validate()
        {
            int layer = LayerMask.NameToLayer(ZarrDataContract.ArchitectureObstacleLayerName);
            if (layer < 0)
            {
                Debug.LogError($"[RecordRandomWalks] Layer '{ZarrDataContract.ArchitectureObstacleLayerName}' does not exist. Refusing to start.");
                enabled = false;
                return false;
            }
            architectureMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);

            bool anyArchitecture = false;
            foreach (Collider c in FindObjectsOfType<Collider>())
            {
                if (c.gameObject.layer == layer) { anyArchitecture = true; break; }
            }
            if (!anyArchitecture)
            {
                Debug.LogError($"[RecordRandomWalks] No collider on layer '{ZarrDataContract.ArchitectureObstacleLayerName}' in this scene - " +
                    "every depth value would be 'no hit'. Refusing to start.");
                enabled = false;
                return false;
            }

            NavMeshTriangulation tri = NavMesh.CalculateTriangulation();
            if (tri.vertices == null || tri.vertices.Length == 0 || tri.indices == null || tri.indices.Length < 3)
            {
                Debug.LogError("[RecordRandomWalks] NavMesh triangulation is empty - is the scene's NavMesh baked? Refusing to start.");
                enabled = false;
                return false;
            }
            triVerts = tri.vertices;
            triIndices = tri.indices;
            triCumulativeArea = new float[triIndices.Length / 3];
            float total = 0f;
            for (int t = 0; t < triCumulativeArea.Length; t++)
            {
                Vector3 a = triVerts[triIndices[3 * t]], b = triVerts[triIndices[3 * t + 1]], c = triVerts[triIndices[3 * t + 2]];
                total += 0.5f * Vector3.Cross(b - a, c - a).magnitude;
                triCumulativeArea[t] = total;
            }
            if (total <= 0f)
            {
                Debug.LogError("[RecordRandomWalks] NavMesh has zero area. Refusing to start.");
                enabled = false;
                return false;
            }

            if (NavMeshLooksStale(layer, out float staleRatio))
            {
                Debug.LogError($"[RecordRandomWalks] {staleRatio:P0} of sampled NavMesh triangles overlap '{ZarrDataContract.ArchitectureObstacleLayerName}' " +
                    "geometry - the baked NavMesh does not match the current walls/obstacles (a stale bake, e.g. after this layout's geometry was " +
                    "generated/edited without rebaking). Re-bake via Window > AI > Navigation > Bake, then press Play again. Refusing to start, since " +
                    "running now would record agents walking through walls.");
                enabled = false;
                return false;
            }
            return true;
        }

        /// <summary>
        /// Samples NavMesh triangle centroids and checks each for solid Architecture_Obstacle geometry
        /// directly above it - a triangle whose centroid sits inside a wall/pillar footprint means the
        /// NavMesh was baked before/without that geometry (or geometry moved after the last bake).
        ///
        /// A correctly-baked NavMesh legitimately has thin, boundary-hugging triangles running right
        /// along every obstacle edge (Recast's contouring near constrained boundaries, especially at
        /// doorways) - their centroids sit close enough to a wall collider that a naive check flags a
        /// real, non-stale bake as "stale" purely from this edge-grazing (confirmed empirically: ~7% on
        /// a visually clean, correctly-masked doorway-heavy layout). A genuinely stale bake - one that
        /// ignored obstacles entirely - has centroids landing throughout each wall's *interior*, not
        /// just a thin sliver along edges, so it shows up far above that noise floor. The threshold
        /// below is set with headroom over the observed edge-grazing rate.
        /// </summary>
        private bool NavMeshLooksStale(int architectureLayer, out float staleRatio)
        {
            int triCount = triCumulativeArea.Length;
            int sampleCount = Mathf.Min(60, triCount);
            int badHits = 0;
            for (int i = 0; i < sampleCount; i++)
            {
                int t = triCount == 1 ? 0 : (i * (triCount - 1)) / Mathf.Max(1, sampleCount - 1);
                Vector3 a = triVerts[triIndices[3 * t]], b = triVerts[triIndices[3 * t + 1]], c = triVerts[triIndices[3 * t + 2]];
                Vector3 centroid = (a + b + c) / 3f;
                if (Physics.CheckBox(centroid + Vector3.up * 0.15f, new Vector3(0.02f, 0.1f, 0.02f), Quaternion.identity, 1 << architectureLayer))
                {
                    badHits++;
                }
            }
            staleRatio = sampleCount > 0 ? (float)badHits / sampleCount : 0f;
            return staleRatio > 0.25f;
        }

        private bool BuildRuntimeObjects()
        {
            // --- the human rig: the scene's own FPSController, else a fresh prefab instance ---
            FirstPersonController[] rigs = FindObjectsOfType<FirstPersonController>();
            if (rigs.Length > 0)
            {
                rig = rigs[0];
                for (int i = 1; i < rigs.Length; i++)
                {
                    Debug.LogWarning($"[RecordRandomWalks] Disabling extra FirstPersonController '{rigs[i].name}' - only one rig may record.");
                    rigs[i].gameObject.SetActive(false);
                }
            }
            else if (fpsControllerPrefab != null)
            {
                rig = Instantiate(fpsControllerPrefab).GetComponent<FirstPersonController>();
                if (rig != null) rig.name = "FPSController (RecordRandomWalks)";
            }
            if (rig == null)
            {
                Debug.LogError("[RecordRandomWalks] The scene has no FirstPersonController and 'fpsControllerPrefab' is not assigned (or has none). " +
                    "Assign Assets/Scenes/FPSController.prefab. Refusing to start.");
                fatal = true;
                return false;
            }
            rigController = rig.GetComponent<CharacterController>();
            Transform camTransform = rig.transform.Find("FirstPersonCharacter");
            if (camTransform == null) camTransform = rig.GetComponentInChildren<Camera>(true)?.transform;
            if (camTransform == null || rigController == null)
            {
                Debug.LogError($"[RecordRandomWalks] Rig '{rig.name}' has no FirstPersonCharacter camera or CharacterController. Refusing to start.");
                fatal = true;
                return false;
            }
            rigCamera = camTransform;

            // Exactly one CanonicalRecordingCamera: the rig's own camera, like in a human recording.
            foreach (CanonicalRecordingCamera existing in FindObjectsOfType<CanonicalRecordingCamera>())
            {
                if (existing.transform != rigCamera) Destroy(existing);
            }
            if (rigCamera.GetComponent<CanonicalRecordingCamera>() == null) rigCamera.gameObject.AddComponent<CanonicalRecordingCamera>();

            // --- markers: copies of what a human sees in this scene ---
            RecordingRig.FindSceneMarkerTemplates(out startTemplate, out poiTemplate, out endTemplate);
            endMarker = RecordingRig.CreateMarker(endTemplate, "RRW_EndMarker", transform);
            waypointMarker = RecordingRig.CreateMarker(poiTemplate, "RRW_WaypointMarker", transform);
            if (startTemplate != null)
            {
                startMarker = RecordingRig.CreateMarker(startTemplate, "RRW_StartMarker", transform);
            }
            else
            {
                startMarker = new GameObject("RRW_StartPoint").transform; // task.start needs an object; nothing visible, as in the scene
                startMarker.SetParent(transform, false);
            }
            RecordingRig.HideSceneTaskMarkers("RecordRandomWalks");

            var taskGO = new GameObject("RRW_Task");
            taskGO.transform.SetParent(transform, false);
            task = taskGO.AddComponent<TaskScript>();
            task.taskName = "RecordRandomWalks";
            task.agentType = "RecordRandomWalks";
            task.start = new[] { startMarker.gameObject };
            task.end = new[] { endMarker.gameObject };
            task.pointsOfInterest = Array.Empty<GameObject>();
            task.numberOfNeeds = 0;
            task.revisit = false;
            task.chooseNonDeterministically = false;

            // --- the recorder on the rig, configured as for a human except for the source tag ---
            recorder = rig.GetComponent<ParticipantRecorder>();
            if (recorder == null) recorder = rig.gameObject.AddComponent<ParticipantRecorder>();
            recorder.task = task;
            recorder.ArchitectureLayerMask = architectureMask;
            recorder.OutputFormat = RecordingOutputFormat.Binary;
            recorder.EnableDebugRays = false;
            recorder.AutoStartForTesting = false;
            recorder.suppressPythonPipeline = true;
            recorder.participantId = "synthetic_rrw";
            recorder.sessionId = SceneManager.GetActiveScene().name;

            agent = rig.gameObject.GetComponent<RecordRandomWalksAgent>();
            if (agent == null) agent = rig.gameObject.AddComponent<RecordRandomWalksAgent>();
            agent.settings = bot;
            agent.Configure(rig, rigCamera, recorder, architectureMask);
            return true;
        }

        // ------------------------------------------------------------------
        // Sampling
        // ------------------------------------------------------------------

        private bool TrySamplePair(out Vector3 start, out Vector3 end, out float directLength)
        {
            start = end = Vector3.zero;
            directLength = 0f;
            var path = new NavMeshPath();
            for (int attempt = 0; attempt < maxResampleAttempts; attempt++)
            {
                if (!TrySampleNavMeshPoint(out start) || !TrySampleNavMeshPoint(out end)) continue;
                // Start inside the arrival radius would end the episode on the first tick.
                if (Vector3.Distance(start, end) <= ZarrDataContract.ArrivalRadiusMeters) continue;
                if (!NavMesh.CalculatePath(start, end, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete) continue;
                directLength = PathLength(path.corners);
                if (directLength < minPairPathLengthMeters) continue;
                return true;
            }
            return false;
        }

        private bool TrySampleNavMeshPoint(out Vector3 point)
        {
            point = Vector3.zero;
            float total = triCumulativeArea[triCumulativeArea.Length - 1];
            float pick = (float)(rng.NextDouble() * total);
            int lo = 0, hi = triCumulativeArea.Length - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (triCumulativeArea[mid] < pick) lo = mid + 1; else hi = mid;
            }

            Vector3 a = triVerts[triIndices[3 * lo]], b = triVerts[triIndices[3 * lo + 1]], c = triVerts[triIndices[3 * lo + 2]];
            float r1 = Mathf.Sqrt((float)rng.NextDouble());
            float r2 = (float)rng.NextDouble();
            Vector3 candidate = (1f - r1) * a + r1 * (1f - r2) * b + r1 * r2 * c;

            if (!NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleRadius, NavMesh.AllAreas)) return false;
            point = hit.position;
            return true;
        }

        private static float PathLength(Vector3[] corners)
        {
            float len = 0f;
            for (int i = 1; i < corners.Length; i++) len += Vector3.Distance(corners[i - 1], corners[i]);
            return len;
        }

        // ------------------------------------------------------------------
        // One episode
        // ------------------------------------------------------------------

        private IEnumerator RunEpisode(Vector3 start, Vector3 end, float directLength)
        {
            lastEpisodeOk = false;

            // --- plan: start -> (random waypoint, registered as the task's POI) -> end ---
            bool useWaypoint = rng.NextDouble() < waypointProbability;
            Vector3 waypoint = end;
            float planned = directLength;
            if (useWaypoint)
            {
                if (!TrySampleNavMeshPoint(out waypoint)) yield break;
                if (Vector3.Distance(waypoint, start) < waypointMinDistanceMeters || Vector3.Distance(waypoint, end) < waypointMinDistanceMeters) yield break;
                var leg1 = new NavMeshPath();
                var leg2 = new NavMeshPath();
                if (!NavMesh.CalculatePath(start, waypoint, NavMesh.AllAreas, leg1) || leg1.status != NavMeshPathStatus.PathComplete) yield break;
                if (!NavMesh.CalculatePath(waypoint, end, NavMesh.AllAreas, leg2) || leg2.status != NavMeshPathStatus.PathComplete) yield break;

                planned = PathLength(leg1.corners) + PathLength(leg2.corners);
                if (planned > maxPathLengthFactor * directLength) yield break;
            }

            // The recorder builds its PoiProgressTracker from these in StartRecording(), so set them first.
            RecordingRig.PlaceMarker(startMarker, startTemplate, start);
            RecordingRig.PlaceMarker(endMarker, endTemplate, end);
            RecordingRig.PlaceMarker(waypointMarker, poiTemplate, waypoint);
            waypointMarker.gameObject.SetActive(useWaypoint);
            task.pointsOfInterest = useWaypoint ? new[] { waypointMarker.gameObject } : Array.Empty<GameObject>();
            task.numberOfNeeds = useWaypoint ? 1 : 0;

            // --- place the rig on the floor at the start, facing the first target +- the start jitter ---
            agent.Stop();
            Vector3 toFirst = (useWaypoint ? waypoint : end) - start;
            float yaw = (toFirst.x * toFirst.x + toFirst.z * toFirst.z) > 1e-6f ? Mathf.Atan2(toFirst.x, toFirst.z) * Mathf.Rad2Deg : 0f;
            yaw += (float)(rng.NextDouble() * 2.0 - 1.0) * bot.startHeadingJitterDegrees;
            Vector3 spawn = start + Vector3.up;
            if (RecordingRig.TryGroundedPivotY(rigController, spawn, out float pivotY)) spawn.y = pivotY;
            rig.gameObject.PlaceObject(spawn, Quaternion.Euler(0f, yaw, 0f));
            if (!RecordingRig.ApplyContractCameraHeight(rig.transform, rigCamera, out string heightMessage))
            {
                Debug.LogError($"[RecordRandomWalks] Camera height: {heightMessage} Aborting the run.");
                fatal = true;
                yield break;
            }

            // Let the CharacterController settle onto the floor (a few FixedUpdates) before recording.
            float settleUntil = Time.time + 0.3f;
            while (Time.time < settleUntil || (!rigController.isGrounded && Time.time < settleUntil + 2f))
            {
                yield return null;
            }

            float timeout = Mathf.Max(minTimeoutSeconds, timeoutFactor * planned / Mathf.Max(0.1f, rig.WalkSpeed) + behaviourTimeBudgetSeconds);

            // --- record ---
            if (!TryStartRecording()) yield break;
            agent.Begin(rng);
            string episodeDir = recorder.CurrentEpisodeDirectory;
            float t0 = Time.time;

            while (recorder.IsRecording)
            {
                if (Time.time - t0 > timeout)
                {
                    Debug.LogWarning($"[RecordRandomWalks] Timeout after {timeout:F0}s (planned {planned:F1} m) - discarding episode.");
                    agent.Stop();
                    recorder.StopRecording();
                    DiscardEpisode(episodeDir);
                    yield break;
                }
                yield return null;
            }

            agent.Stop();
            float duration = Time.time - t0;
            if (duration < minEpisodeSeconds)
            {
                Debug.LogWarning($"[RecordRandomWalks] Episode lasted only {duration:F1}s (< minEpisodeSeconds {minEpisodeSeconds:F1}) - discarding.");
                DiscardEpisode(episodeDir);
                yield break;
            }

            Debug.Log($"[RecordRandomWalks] Episode ok: {duration:F1}s, {(useWaypoint ? "via waypoint (POI)" : "direct")}, " +
                $"turns-in-place={agent.TurnsInPlace}, look-arounds={agent.LookArounds}, wall excursions={agent.WallExcursions}, stall recoveries={agent.StallRecoveries}.");
            episodesRecorded++;
            lastEpisodeOk = true;
        }

        private bool TryStartRecording()
        {
            try
            {
                recorder.StartRecording();
                return recorder.IsRecording;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[RecordRandomWalks] ParticipantRecorder refused to start: {ex.Message}. Aborting the run.");
                fatal = true;
                return false;
            }
        }

        /// <summary>Deletes a just-recorded, failed episode directory. Refuses anything that is not an episode_* dir inside zarr_recordings.</summary>
        internal static void DiscardEpisode(string episodeDir)
        {
            if (string.IsNullOrEmpty(episodeDir) || !Directory.Exists(episodeDir)) return;
            string name = Path.GetFileName(episodeDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            string parent = Path.GetFileName(Path.GetDirectoryName(episodeDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)) ?? "");
            if (!name.StartsWith("episode_") || parent != "zarr_recordings")
            {
                Debug.LogWarning($"[RecordRandomWalks] Refusing to delete unexpected directory '{episodeDir}'.");
                return;
            }
            try { Directory.Delete(episodeDir, true); }
            catch (Exception ex) { Debug.LogWarning($"[RecordRandomWalks] Could not delete discarded episode '{episodeDir}': {ex.Message}"); }
        }

        private void RestoreSettings()
        {
            if (inputSwitched)
            {
                if (agent != null) agent.Stop();
                CrossPlatformInputManager.SwitchActiveInputMethod(CrossPlatformInputManager.ActiveInputMethod.Hardware);
                inputSwitched = false;
            }
            if (!timeSettingsApplied) return;
            Time.captureDeltaTime = previousCaptureDeltaTime;
            timeSettingsApplied = false;
        }

        void OnDestroy()
        {
            RestoreSettings();
        }
    }
}
