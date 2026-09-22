using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.AI;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Drop-in Editor Play-mode harness that generates synthetic "expert" episodes on whatever layout
    /// (baked NavMesh + Architecture_Obstacle geometry) the current scene contains.
    ///
    /// For each of `pairCount` (k) random start/end pairs it sends `agentsPerPair` (m) RecordRandomWalksAgents
    /// one after another (never in parallel, so agents never appear in each other's RGB/depth) from start
    /// to end. Every agent takes start -> one random intermediate waypoint -> end (total length capped at
    /// maxPathLengthFactor x the shortest start->end path) and follows it with an Ornstein-Uhlenbeck lateral
    /// deviation (see RecordRandomWalksAgent). Each walk is recorded by a ParticipantRecorder, producing one
    /// episode_* directory in imitationagent_training/zarr_recordings - the same format as the human episodes.
    /// Result: k x m episodes per layout.
    ///
    /// Within a pair the start and end are reused; only the waypoint and the deviation noise are re-sampled.
    /// An episode is discarded (its directory deleted) and re-sampled when: no path exists, the route is longer
    /// than maxPathLengthFactor x shortest, or the end is not reached within max(minTimeoutSeconds,
    /// timeoutFactor x plannedLength / speed). After maxResampleAttempts failures for one slot the pair is
    /// skipped with a warning.
    ///
    /// Does not touch AgentScript/EngineScript/EvalHarness. Owns its own runtime TaskScript + markers, so the
    /// scene's authored task is never modified.
    /// </summary>
    public class RecordRandomWalksHarness : MonoBehaviour
    {
        [Header("Run Configuration")]
        [Tooltip("k: number of random start/end pairs on this layout.")]
        [Min(1)] public int pairCount = 5;
        [Tooltip("m: number of agents (recordings) per start/end pair. Total episodes = k x m.")]
        [Min(1)] public int agentsPerPair = 5;
        [Tooltip("Seed for all sampling. 0 = time-based (the used seed is logged).")]
        public int seed = 0;
        [Tooltip("Max attempts per slot (pair sampling, and each recording slot) before the pair is skipped.")]
        [Min(1)] public int maxResampleAttempts = 15;
        [Tooltip("Sets Time.timeScale for this run. Time.maximumDeltaTime is capped at the sampling interval so the " +
                 "recorder's 0.1s cadence is never skipped; values above ~5x at 60fps buy nothing.")]
        [Min(0.01f)] public float simulationSpeed = 3f;
        [Tooltip("Leave Play mode automatically when the run has finished.")]
        public bool stopPlayModeWhenDone = true;

        [Header("Route")]
        [Tooltip("Max total route length (start -> waypoint -> end) as a multiple of the shortest start -> end path.")]
        public float maxPathLengthFactor = 3f;
        [Tooltip("Radius (meters) to snap a random candidate point onto the NavMesh.")]
        public float navMeshSampleRadius = 2f;

        [Header("Speed (human plateau ~1.3 m/s)")]
        public float speedMeanMps = 1.3f;
        public float speedStdMps = 0.05f;

        [Header("Timeout")]
        [Tooltip("Timeout = max(minTimeoutSeconds, timeoutFactor x plannedLength / speed), in simulated seconds.")]
        public float minTimeoutSeconds = 20f;
        public float timeoutFactor = 3f;

        [Header("Agent (copied onto every RecordRandomWalksAgent)")]
        public float headHeightMeters = 1.78f;
        public float lateralSigmaMeters = 0.3f;
        public float lateralTauSeconds = 1.5f;
        public float edgeClearanceMeters = 0.3f;
        public float lookAheadMeters = 3f;
        public float headYawRateDegPerSec = 180f;
        public float bodyTurnRateDegPerSec = 90f;
        public float pitchSigmaDegrees = 5f;
        public float pitchTauSeconds = 1f;
        public float pitchMaxDegrees = 15f;
        [Tooltip("Passed to NavMeshAgent.radius (and task.agentRadius), matching this project's own convention " +
                 "(AgentScript.cs: navMeshAgent.radius = agentRadius). Purely a clearance/avoidance setting - it " +
                 "does NOT change which NavMesh polygons exist, since those are fixed by the last bake.")]
        public float agentRadius = 0.25f;

        [Header("Visualization (same scheme as EvalHarness)")]
        [Tooltip("Color of the visible synthetic-agent capsule (visual only - a plain primitive with no collider, never affects the NavMesh, recording or the deviation model).")]
        public Color agentColor = new Color(1f, 0.549f, 0f); // orange, matches EvalHarness.agentColor
        public Color startMarkerColor = Color.green;
        [Tooltip("Color of the intermediate waypoint marker - shown in the 'POI' role/color slot since it's this run's routing analogue of a POI.")]
        public Color waypointMarkerColor = Color.cyan;
        public Color endMarkerColor = Color.red;
        [Tooltip("Visual-only marker sphere radius, meters.")]
        public float markerRadius = 0.25f;

        private System.Random rng;
        private Vector3[] triVerts;
        private int[] triIndices;
        private float[] triCumulativeArea;
        private float previousMaximumDeltaTime;
        private bool timeSettingsApplied;

        private TaskScript task;
        private Transform startMarker;
        private Transform endMarker;
        private Transform waypointMarker;
        private RecordRandomWalksAgent agent;
        private LayerMask architectureMask;
        private bool fatal;
        private bool lastEpisodeOk;

        private int episodesRecorded;
        private int episodesDiscarded;
        private int pairsSkipped;

        IEnumerator Start()
        {
            if (!Validate()) yield break;

            int usedSeed = seed != 0 ? seed : Environment.TickCount;
            rng = new System.Random(usedSeed);
            Debug.Log($"[RecordRandomWalks] Starting: k={pairCount} pairs x m={agentsPerPair} agents, seed={usedSeed}.");

            previousMaximumDeltaTime = Time.maximumDeltaTime;
            Time.maximumDeltaTime = ZarrDataContract.SamplingIntervalSeconds;
            Time.timeScale = simulationSpeed;
            timeSettingsApplied = true;

            BuildRuntimeObjects();
            yield return null; // let TaskScript.Start() and the new components initialise

            for (int pair = 0; pair < pairCount && !fatal; pair++)
            {
                if (!TrySamplePair(out Vector3 start, out Vector3 end, out float directLength))
                {
                    Debug.LogWarning($"[RecordRandomWalks] Pair {pair + 1}/{pairCount}: no valid start/end after {maxResampleAttempts} attempts - skipped.");
                    pairsSkipped++;
                    continue;
                }

                startMarker.position = start;
                endMarker.position = end;
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
            RestoreTimeSettings();
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

        private void BuildRuntimeObjects()
        {
            // Exactly one CanonicalRecordingCamera may exist (ParticipantRecorder hard-fails otherwise). Strip just the
            // marker component from any the scene already carries (e.g. an FPSController head) - NOT the GameObject or
            // its Camera - so other scripts that depend on that object staying alive (e.g. Camera.main-reliant
            // FirstPersonController) are unaffected. The stripped camera simply stops being a recording-camera candidate.
            foreach (CanonicalRecordingCamera existing in FindObjectsOfType<CanonicalRecordingCamera>())
            {
                Debug.LogWarning($"[RecordRandomWalks] Removing existing CanonicalRecordingCamera marker from '{existing.name}' for this run " +
                    "(the synthetic agent's head camera is the only recording camera; the object itself and its Camera are left untouched).");
                Destroy(existing);
            }

            startMarker = CreateMarkerVisual("RRW_StartMarker", startMarkerColor).transform;
            endMarker = CreateMarkerVisual("RRW_EndMarker", endMarkerColor).transform;
            waypointMarker = CreateMarkerVisual("RRW_WaypointMarker", waypointMarkerColor).transform;
            startMarker.SetParent(transform, false);
            endMarker.SetParent(transform, false);
            waypointMarker.SetParent(transform, false);

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
            task.agentRadius = agentRadius;

            // Created on the NavMesh so the NavMeshAgent can initialise without a "not close enough to NavMesh" error.
            Vector3 spawn = triVerts[0];
            if (NavMesh.SamplePosition(spawn, out NavMeshHit spawnHit, navMeshSampleRadius, NavMesh.AllAreas)) spawn = spawnHit.position;
            var agentGO = new GameObject("RRW_Agent");
            agentGO.transform.SetParent(transform, false);
            agentGO.transform.position = spawn;
            agentGO.AddComponent<NavMeshAgent>();
            agent = agentGO.AddComponent<RecordRandomWalksAgent>();
            agent.headHeightMeters = headHeightMeters;
            agent.lateralSigmaMeters = lateralSigmaMeters;
            agent.lateralTauSeconds = lateralTauSeconds;
            agent.edgeClearanceMeters = edgeClearanceMeters;
            agent.lookAheadMeters = lookAheadMeters;
            agent.headYawRateDegPerSec = headYawRateDegPerSec;
            agent.bodyTurnRateDegPerSec = bodyTurnRateDegPerSec;
            agent.pitchSigmaDegrees = pitchSigmaDegrees;
            agent.pitchTauSeconds = pitchTauSeconds;
            agent.pitchMaxDegrees = pitchMaxDegrees;

            agent.BuildRig(architectureMask, task);

            // Visual-only capsule tracking the recorded Body (i.e. the path position + lateral offset the
            // recorder actually sees) so wall/NavMesh-clearance behaviour can be checked by eye. No collider,
            // no layer assignment beyond Default - it cannot affect NavMesh queries, the depth raycasts
            // (which only hit ArchitectureLayerMask), or the deviation model.
            GameObject agentVisual = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            agentVisual.name = "RRW_AgentVisual";
            Collider avCollider = agentVisual.GetComponent<Collider>();
            if (avCollider != null) Destroy(avCollider);
            MeshRenderer avRenderer = agentVisual.GetComponent<MeshRenderer>();
            if (avRenderer != null) avRenderer.material = CreateColorMaterial(agentColor);
            agentVisual.transform.SetParent(agent.Body, false);
            agentVisual.transform.localScale = new Vector3(0.5f, 0.9f, 0.5f);
            agentVisual.transform.localPosition = Vector3.up * 1f; // default capsule height 2 -> sits on the floor
        }

        private GameObject CreateMarkerVisual(string name, Color color)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            Collider col = go.GetComponent<Collider>();
            if (col != null) Destroy(col);
            go.transform.localScale = Vector3.one * (markerRadius * 2f);
            MeshRenderer renderer = go.GetComponent<MeshRenderer>();
            if (renderer != null) renderer.material = CreateColorMaterial(color);
            return go;
        }

        private static Material CreateColorMaterial(Color color)
        {
            Shader shader = Shader.Find("Diffuse");
            Material material = new Material(shader != null ? shader : Shader.Find("Standard"));
            material.color = color;
            return material;
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

        private float SampleSpeed()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            float g = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
            return Mathf.Clamp(speedMeanMps + g * speedStdMps, speedMeanMps - 3f * speedStdMps, speedMeanMps + 3f * speedStdMps);
        }

        // ------------------------------------------------------------------
        // One episode
        // ------------------------------------------------------------------

        private IEnumerator RunEpisode(Vector3 start, Vector3 end, float directLength)
        {
            lastEpisodeOk = false;

            // --- plan: start -> random waypoint -> end ---
            if (!TrySampleNavMeshPoint(out Vector3 waypoint)) yield break;
            var leg1 = new NavMeshPath();
            var leg2 = new NavMeshPath();
            if (!NavMesh.CalculatePath(start, waypoint, NavMesh.AllAreas, leg1) || leg1.status != NavMeshPathStatus.PathComplete) yield break;
            if (!NavMesh.CalculatePath(waypoint, end, NavMesh.AllAreas, leg2) || leg2.status != NavMeshPathStatus.PathComplete) yield break;

            float planned = PathLength(leg1.corners) + PathLength(leg2.corners);
            if (planned > maxPathLengthFactor * directLength) yield break;

            waypointMarker.position = waypoint;

            float speed = SampleSpeed();
            float timeout = Mathf.Max(minTimeoutSeconds, timeoutFactor * planned / speed);

            agent.BeginWalk(start, waypoint, end, speed, rng);

            int waitFrames = 0;
            while (agent.Nav.pathPending && waitFrames++ < 60) yield return null;
            yield return null;

            // --- record ---
            if (!TryStartRecording()) yield break;
            string episodeDir = agent.Recorder.CurrentEpisodeDirectory;
            float t0 = Time.time;

            while (agent.Recorder.IsRecording)
            {
                if (Time.time - t0 > timeout)
                {
                    Debug.LogWarning($"[RecordRandomWalks] Timeout after {timeout:F0}s (planned {planned:F1} m) - discarding episode.");
                    agent.Recorder.StopRecording();
                    agent.StopWalk();
                    DiscardEpisode(episodeDir);
                    yield break;
                }
                yield return null;
            }

            agent.StopWalk();
            episodesRecorded++;
            lastEpisodeOk = true;
        }

        private bool TryStartRecording()
        {
            try
            {
                agent.Recorder.StartRecording();
                return agent.Recorder.IsRecording;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[RecordRandomWalks] ParticipantRecorder refused to start: {ex.Message}. Aborting the run.");
                fatal = true;
                return false;
            }
        }

        /// <summary>Deletes a just-recorded, failed episode directory. Refuses anything that is not an episode_* dir inside zarr_recordings.</summary>
        private static void DiscardEpisode(string episodeDir)
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

        private void RestoreTimeSettings()
        {
            if (!timeSettingsApplied) return;
            Time.timeScale = 1f;
            if (previousMaximumDeltaTime > 0f) Time.maximumDeltaTime = previousMaximumDeltaTime;
            timeSettingsApplied = false;
        }

        void OnDestroy()
        {
            RestoreTimeSettings();
        }
    }
}
