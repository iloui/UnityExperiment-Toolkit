using System;
using System.Collections.Generic;
using Assets.Scripts.DataRecording;
using Unity.AI.Navigation;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;
using UnityStandardAssets.Characters.FirstPerson;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Brings a generated layout scene into the state every pipeline step expects: Architecture_Obstacle layer, baked NavMesh,
    /// one TaskScript with a Start -> POI_0 -> End task, FPSController at the start with the contract eye height, and the
    /// occupancy grid. LayoutGeneratorWindow runs it after building a layout; Tools > Thesis > Prepare Scene re-runs it on
    /// the open scene. Idempotent: markers that are still valid are kept, everything else is reset to the values below.
    /// </summary>
    internal static class ScenePreparation
    {
        private const string EnvironmentRootName = "Environment_Root";
        private const string TaskObjectName = "Task";
        private const string StartName = "Start";
        private const string PoiName = "POI_0";
        private const string EndName = "End";

        // Task behaviour (read by ImitationAgent / AgentScript). Speed and radius are the recording body's, so the agent and
        // the NavMesh baseline move like the FPSController the humans and the bot recorded with.
        private const float PoiTimeSeconds = 1f;
        private const float AgentSize = 1f; // ImitationAgent puts its head at agentSize * 1.5 m = the contract eye height

        // Marker placement.
        private const int TaskRngSalt = 0x7A5C0001;          // task RNG = layout seed + salt, independent of LayoutBuilder's draws
        private const float MinMarkerSpacingMeters = 3f;       // between every pair of Start / POI_0 / End
        private const float MinEdgeClearanceMeters = 0.5f;     // from the NavMesh border (which is already the bake radius off the walls)
        private const float MaxHeightAboveFloorMeters = 0.3f;  // rejects NavMesh on top of walls / the ceiling
        private const float MinStartEndPathMeters = 12f;       // first start/end pair with a path at least this long wins
        private const float PreferredRouteOffsetMeters = 2f;   // POI at least this far from the direct start -> end path
        private const int StartEndAttempts = 200;
        private const int PoiAttempts = 300;
        private const float MarkerHeightAboveNavMesh = 1f;     // centre of the visible POI/End spheres
        private const float MarkerDiameter = 0.4f;
        private const float ControllerPivotAboveNavMesh = 1.05f;

        [MenuItem("Tools/Thesis/Prepare Scene")]
        public static void PrepareActiveSceneMenu()
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            try
            {
                string summary = Prepare(scene);
                EditorUtility.DisplayDialog("Prepare Scene", summary + "\n\nSave the scene to keep the changes.", "OK");
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorUtility.DisplayDialog("Prepare Scene", $"Preparation failed:\n{ex.Message}", "OK");
            }
        }

        /// <summary>Runs every preparation step on `scene` and marks it dirty. Throws if the scene can't get a valid task.</summary>
        public static string Prepare(Scene scene)
        {
            if (!scene.IsValid()) throw new InvalidOperationException("No valid scene is open.");

            SceneExporter.EnsureLayer("Walkable_Floor");
            SceneExporter.EnsureLayer(ZarrDataContract.ArchitectureObstacleLayerName);

            GameObject root = FindRoot(scene, EnvironmentRootName);
            if (root == null)
                throw new InvalidOperationException($"'{scene.name}' has no '{EnvironmentRootName}' - generate it with Tools > Generate Layout.");
            LayoutInfo layoutInfo = root.GetComponent<LayoutInfo>();
            if (layoutInfo == null)
                throw new InvalidOperationException($"'{EnvironmentRootName}' has no LayoutInfo - generate the scene with Tools > Generate Layout.");

            SceneExporter.ConfigureAndBakeNavMesh(root);
            Physics.SyncTransforms();

            TaskScript task = EnsureTask(scene, root.transform, layoutInfo.seed, out string taskMessage);
            int rigs = PlaceAndWireControllers(scene, task);
            Physics.SyncTransforms();
            int cameras = RecordingRigTools.ApplyInScene(scene);
            string gridPath = OccupancyGridExporter.Export(scene, task, layoutInfo, root.GetComponent<NavMeshSurface>());

            EditorSceneManager.MarkSceneDirty(scene);
            string summary = $"'{scene.name}': NavMesh baked; {taskMessage}; {rigs} FPSController(s) moved to Start and wired; " +
                $"camera height applied to {cameras}; occupancy grid written to {gridPath}.";
            Debug.Log("[ScenePreparation] " + summary);
            return summary;
        }

        /// <summary>The scene's single task (the TaskScript on Environment_Root/Task), or null.</summary>
        public static TaskScript FindTask(Scene scene)
        {
            GameObject root = FindRoot(scene, EnvironmentRootName);
            Transform taskTransform = root != null ? root.transform.Find(TaskObjectName) : null;
            return taskTransform != null ? taskTransform.GetComponent<TaskScript>() : null;
        }

        // ------------------------------------------------------------------
        // Task
        // ------------------------------------------------------------------

        private static TaskScript EnsureTask(Scene scene, Transform root, int layoutSeed, out string message)
        {
            Transform taskTransform = root.Find(TaskObjectName);
            if (taskTransform == null)
            {
                taskTransform = new GameObject(TaskObjectName).transform;
                taskTransform.SetParent(root, false);
            }
            GameObject taskGo = taskTransform.gameObject;

            RemoveStaleTaskObjects(scene, taskGo);

            Transform start = taskTransform.Find(StartName);
            Transform poi = taskTransform.Find(PoiName);
            Transform end = taskTransform.Find(EndName);

            string invalidReason = "markers missing";
            if (start != null && poi != null && end != null && ValidateMarkers(start.position, poi.position, end.position, out invalidReason))
            {
                message = "existing Start/POI_0/End kept";
            }
            else
            {
                Debug.Log($"[ScenePreparation] Placing new Start/POI_0/End: {invalidReason}.");
                PlaceMarkers(layoutSeed, out Vector3 s, out Vector3 p, out Vector3 e);
                start = EnsureMarker(taskTransform, StartName, s, visible: false);
                poi = EnsureMarker(taskTransform, PoiName, p + Vector3.up * MarkerHeightAboveNavMesh, visible: true);
                end = EnsureMarker(taskTransform, EndName, e + Vector3.up * MarkerHeightAboveNavMesh, visible: true);
                message = $"Start/POI_0/End placed (seed {layoutSeed})";
            }

            TaskScript task = taskGo.GetComponent<TaskScript>();
            if (task == null) task = taskGo.AddComponent<TaskScript>();
            task.taskName = "Task_" + scene.name;
            task.start = new[] { start.gameObject };
            task.pointsOfInterest = new[] { poi.gameObject };
            task.end = new[] { end.gameObject };
            task.numberOfNeeds = 1;
            task.revisit = false;
            task.chooseNonDeterministically = false;
            ApplyBehaviour(task);
            EditorUtility.SetDirty(task);
            return task;
        }

        /// <summary>Behaviour values every scene task gets, from the recording body contract.</summary>
        private static void ApplyBehaviour(TaskScript task)
        {
            task.numberOfAgents = 1;
            task.spawnInterval = 0f;
            task.taskColor = Color.red;
            task.agentType = "ImitationAgent";
            task.agentSpeed = ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond;
            task.agentSize = AgentSize;
            task.agentRadius = ZarrDataContract.RecordingBody.Radius;
            task.privacyRadius = ZarrDataContract.RecordingBody.Radius;
            task.poiTime = PoiTimeSeconds;
        }

        /// <summary>The scene has exactly one task: any other TaskScript is removed.</summary>
        private static void RemoveStaleTaskObjects(Scene scene, GameObject taskGo)
        {
            foreach (GameObject rootGo in scene.GetRootGameObjects())
            {
                foreach (TaskScript other in rootGo.GetComponentsInChildren<TaskScript>(true))
                {
                    if (other.gameObject == taskGo) continue;
                    Debug.LogWarning($"[ScenePreparation] Removing extra TaskScript on '{other.name}' - the scene's task lives on '{TaskObjectName}'.");
                    UnityEngine.Object.DestroyImmediate(other);
                }
            }
        }

        private static Transform EnsureMarker(Transform parent, string name, Vector3 position, bool visible)
        {
            Transform existing = parent.Find(name);
            if (existing != null) UnityEngine.Object.DestroyImmediate(existing.gameObject);

            GameObject go;
            if (visible)
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
                go.transform.localScale = Vector3.one * MarkerDiameter;
            }
            else
            {
                go = new GameObject();
            }
            go.name = name;
            go.transform.SetParent(parent, true);
            go.transform.position = position;
            return go.transform;
        }

        // ------------------------------------------------------------------
        // Marker placement
        // ------------------------------------------------------------------

        private sealed class NavMeshSampler
        {
            private readonly Vector3[] vertices;
            private readonly int[] indices;
            private readonly float[] cumulativeArea;
            public readonly float FloorY;

            public NavMeshSampler()
            {
                NavMeshTriangulation tri = NavMesh.CalculateTriangulation();
                vertices = tri.vertices;
                indices = tri.indices;
                if (vertices == null || vertices.Length == 0)
                    throw new InvalidOperationException("The NavMesh is empty after baking - does the layout have a Walkable_Floor?");

                FloorY = float.MaxValue;
                foreach (Vector3 v in vertices) FloorY = Mathf.Min(FloorY, v.y);

                cumulativeArea = new float[indices.Length / 3];
                float total = 0f;
                for (int t = 0; t < cumulativeArea.Length; t++)
                {
                    Vector3 a = vertices[indices[3 * t]], b = vertices[indices[3 * t + 1]], c = vertices[indices[3 * t + 2]];
                    // Only floor-level triangles count; NavMesh on wall tops or the ceiling gets no probability mass.
                    bool floorLevel = Mathf.Max(a.y, Mathf.Max(b.y, c.y)) <= FloorY + MaxHeightAboveFloorMeters;
                    if (floorLevel) total += 0.5f * Vector3.Cross(b - a, c - a).magnitude;
                    cumulativeArea[t] = total;
                }
                if (total <= 0f) throw new InvalidOperationException("The NavMesh has no floor-level area.");
            }

            /// <summary>An area-uniform point on the floor-level NavMesh with enough clearance from its border, or false.</summary>
            public bool TrySample(System.Random rng, out Vector3 point)
            {
                float r = (float)rng.NextDouble() * cumulativeArea[cumulativeArea.Length - 1];
                int t = Array.BinarySearch(cumulativeArea, r);
                if (t < 0) t = ~t;
                t = Mathf.Clamp(t, 0, cumulativeArea.Length - 1);

                Vector3 a = vertices[indices[3 * t]], b = vertices[indices[3 * t + 1]], c = vertices[indices[3 * t + 2]];
                float u = (float)rng.NextDouble(), v = (float)rng.NextDouble();
                if (u + v > 1f) { u = 1f - u; v = 1f - v; }
                Vector3 candidate = a + u * (b - a) + v * (c - a);

                point = candidate;
                return OnFloorNavMesh(candidate, 0.2f, FloorY, out point);
            }
        }

        private static bool OnFloorNavMesh(Vector3 position, float searchRadius, float floorY, out Vector3 snapped)
        {
            snapped = position;
            if (!NavMesh.SamplePosition(position, out NavMeshHit hit, searchRadius, NavMesh.AllAreas)) return false;
            if (hit.position.y > floorY + MaxHeightAboveFloorMeters) return false;
            if (!NavMesh.FindClosestEdge(hit.position, out NavMeshHit edge, NavMesh.AllAreas) || edge.distance < MinEdgeClearanceMeters) return false;
            snapped = hit.position;
            return true;
        }

        private static void PlaceMarkers(int layoutSeed, out Vector3 start, out Vector3 poi, out Vector3 end)
        {
            var rng = new System.Random(unchecked(layoutSeed + TaskRngSalt));
            var sampler = new NavMeshSampler();

            // Start and end: the first valid pair whose path is long enough, else the longest valid pair seen.
            bool havePair = false;
            float bestLength = -1f;
            start = end = Vector3.zero;
            for (int i = 0; i < StartEndAttempts; i++)
            {
                if (!sampler.TrySample(rng, out Vector3 s) || !sampler.TrySample(rng, out Vector3 e)) continue;
                if (Vector3.Distance(s, e) < MinMarkerSpacingMeters) continue;
                if (!TryPath(s, e, out NavMeshPath path)) continue;
                float length = PathLength(path);
                if (length > bestLength)
                {
                    bestLength = length;
                    start = s;
                    end = e;
                    havePair = true;
                }
                if (length >= MinStartEndPathMeters) break;
            }
            if (!havePair) throw new InvalidOperationException("Could not find a start and end with a complete NavMesh path between them.");

            TryPath(start, end, out NavMeshPath direct);
            Vector3[] directRoute = direct.corners;

            // POI: the first valid candidate off the direct route, else the valid candidate furthest from it.
            bool havePoi = false;
            float bestOffset = -1f;
            poi = Vector3.zero;
            for (int i = 0; i < PoiAttempts; i++)
            {
                if (!sampler.TrySample(rng, out Vector3 p)) continue;
                if (Vector3.Distance(p, start) < MinMarkerSpacingMeters || Vector3.Distance(p, end) < MinMarkerSpacingMeters) continue;
                if (!TryPath(start, p, out _) || !TryPath(p, end, out _)) continue;
                float offset = DistanceToPolyline(p, directRoute);
                if (offset > bestOffset)
                {
                    bestOffset = offset;
                    poi = p;
                    havePoi = true;
                }
                if (offset >= PreferredRouteOffsetMeters) break;
            }
            if (!havePoi) throw new InvalidOperationException("Could not find a POI with complete paths from the start and to the end.");

            if (bestLength < MinStartEndPathMeters)
                Debug.LogWarning($"[ScenePreparation] Start -> End path is only {bestLength:F1} m (wanted >= {MinStartEndPathMeters} m).");
            if (bestOffset < PreferredRouteOffsetMeters)
                Debug.LogWarning($"[ScenePreparation] POI_0 is only {bestOffset:F1} m off the direct Start -> End path (wanted >= {PreferredRouteOffsetMeters} m).");
        }

        /// <summary>Marker positions (the visible spheres float above the NavMesh) still make a valid task on the current NavMesh.</summary>
        private static bool ValidateMarkers(Vector3 startPos, Vector3 poiPos, Vector3 endPos, out string reason)
        {
            reason = null;
            float floorY;
            try { floorY = new NavMeshSampler().FloorY; }
            catch (InvalidOperationException ex) { reason = ex.Message; return false; }

            float radius = MarkerHeightAboveNavMesh + 0.5f;
            if (!OnFloorNavMesh(startPos, radius, floorY, out Vector3 s)) { reason = "Start is off the NavMesh"; return false; }
            if (!OnFloorNavMesh(poiPos, radius, floorY, out Vector3 p)) { reason = "POI_0 is off the NavMesh"; return false; }
            if (!OnFloorNavMesh(endPos, radius, floorY, out Vector3 e)) { reason = "End is off the NavMesh"; return false; }
            if (Vector3.Distance(s, p) < MinMarkerSpacingMeters || Vector3.Distance(p, e) < MinMarkerSpacingMeters ||
                Vector3.Distance(s, e) < MinMarkerSpacingMeters) { reason = "markers are closer than 3 m"; return false; }
            if (!TryPath(s, p, out _) || !TryPath(p, e, out _)) { reason = "no complete path Start -> POI_0 -> End"; return false; }
            return true;
        }

        private static bool TryPath(Vector3 from, Vector3 to, out NavMeshPath path)
        {
            path = new NavMeshPath();
            return NavMesh.CalculatePath(from, to, NavMesh.AllAreas, path) && path.status == NavMeshPathStatus.PathComplete;
        }

        private static float PathLength(NavMeshPath path)
        {
            float length = 0f;
            for (int i = 1; i < path.corners.Length; i++) length += Vector3.Distance(path.corners[i - 1], path.corners[i]);
            return length;
        }

        private static float DistanceToPolyline(Vector3 p, IReadOnlyList<Vector3> polyline)
        {
            if (polyline.Count == 1) return Vector3.Distance(p, polyline[0]);
            float best = float.MaxValue;
            for (int i = 1; i < polyline.Count; i++)
            {
                Vector3 a = polyline[i - 1], ab = polyline[i] - a;
                float t = ab.sqrMagnitude > 1e-6f ? Mathf.Clamp01(Vector3.Dot(p - a, ab) / ab.sqrMagnitude) : 0f;
                best = Mathf.Min(best, Vector3.Distance(p, a + t * ab));
            }
            return best;
        }

        // ------------------------------------------------------------------
        // FPSController
        // ------------------------------------------------------------------

        /// <summary>Puts every FPSController on the start marker, facing along the path to the POI, and points its recorder at the task.</summary>
        private static int PlaceAndWireControllers(Scene scene, TaskScript task)
        {
            Vector3 start = task.start[0].transform.position;
            Vector3 firstTarget = task.pointsOfInterest[0].transform.position;
            NavMesh.SamplePosition(firstTarget, out NavMeshHit poiHit, MarkerHeightAboveNavMesh + 0.5f, NavMesh.AllAreas);
            Vector3 facing = TryPath(start, poiHit.position, out NavMeshPath path) && path.corners.Length > 1
                ? path.corners[1] - start
                : firstTarget - start;
            facing.y = 0f;
            Quaternion rotation = facing.sqrMagnitude > 1e-6f ? Quaternion.LookRotation(facing) : Quaternion.identity;

            int count = 0;
            foreach (GameObject rootGo in scene.GetRootGameObjects())
            {
                foreach (FirstPersonController fps in rootGo.GetComponentsInChildren<FirstPersonController>(true))
                {
                    Undo.RecordObject(fps.transform, "Place FPSController at task start");
                    fps.transform.SetPositionAndRotation(start + Vector3.up * ControllerPivotAboveNavMesh, rotation);
                    var recorder = fps.GetComponent<ParticipantRecorder>();
                    if (recorder != null)
                    {
                        recorder.task = task;
                        EditorUtility.SetDirty(recorder);
                    }
                    count++;
                }
                foreach (EvalHarness harness in rootGo.GetComponentsInChildren<EvalHarness>(true))
                {
                    harness.task = task;
                    EditorUtility.SetDirty(harness);
                }
            }
            return count;
        }

        private static GameObject FindRoot(Scene scene, string name)
        {
            foreach (GameObject go in scene.GetRootGameObjects())
            {
                if (go.name == name) return go;
            }
            return null;
        }
    }
}
