using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Assets.Scripts.DataRecording;
using Assets.Scripts.ExperimentFlow;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Places the three human training tasks of a layout scene around its fixed test task (the scene's Start -> POI_0 -> End)
    /// under the rules of methodology 3.8.5 and stores them in a HumanTaskSet on Environment_Root:
    /// - every marker at least MinWallClearance from any obstacle (which also keeps it out of 2.2 m doorways),
    /// - every marker outside dead-end pockets: at least MinThroughTraffic of the reference routes between random NavMesh points
    ///   (both ends further than TrafficEndpointExclusion away) pass within TrafficPassRadius,
    /// - every marker at least MinTestMarkerDistance from every test marker (so no training task repeats the test task),
    /// - shortest route Start -> POI -> End between MinRouteLength and MaxRouteLength, markers 3 m apart, POI 2 m off the direct path.
    /// Among the valid candidates it picks the triple that covers most of the test route (share of test-route samples within
    /// CoverageRadius of a training route); near-ties go to the triple whose routes span the most floor cells. The training
    /// task with the smallest test-route overlap is marked as the one condition C leaves out. Seeded by the layout seed, so the
    /// result is reproducible; an existing set is never overwritten without confirmation, since the tasks must not change once
    /// participants have walked them.
    /// </summary>
    internal static class HumanTaskGenerator
    {
        private const string Version = "human_tasks_v1";
        private const int TrainingTaskCount = 3;
        private const int SeedSalt = 0x48554D31; // "HUM1"

        private const float MinWallClearance = 1.8f;
        private const float MinTestMarkerDistance = 4f;
        private const float MinRouteLength = 30f;
        private const float MaxRouteLength = 50f;
        private const float MinThroughTraffic = 0.02f;
        private const float TrafficPassRadius = 2f;
        private const float TrafficEndpointExclusion = 4f;
        private const float ReferenceRouteMinLength = 10f;
        private const int ReferenceRouteCount = 400;
        private const float CoverageRadius = 2f;
        private const float RouteSampleSpacing = 0.5f;
        private const float CellSize = 2f;
        private const float CoverageTieTolerance = 0.02f;
        private const float ClearanceProbeHeight = 1f;
        private const float ClearanceProbeRange = 10f;
        private const int ClearanceRays = 72;

        private const int PoolSamples = 6000;
        private const int PoolMax = 500;
        private const int CandidateAttempts = 40000;
        private const int CandidateMax = 150;

        [MenuItem(ToolsMenu.HumanWalks + "Generate Human Tasks (Active Scene)", priority = ToolsMenu.HumanWalksPriority)]
        private static void GenerateActiveMenu() => RunOnActiveScene(reroll: false);

        [MenuItem(ToolsMenu.HumanWalks + "Re-roll Human Tasks (Active Scene)", priority = ToolsMenu.HumanWalksPriority + 1)]
        private static void RerollActiveMenu() => RunOnActiveScene(reroll: true);

        [MenuItem(ToolsMenu.HumanWalks + "Generate Missing Human Tasks For Session Layouts", priority = ToolsMenu.HumanWalksPriority + 2)]
        private static void GenerateSessionLayoutsMenu()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            string previous = EditorSceneManager.GetActiveScene().path;
            string report;
            try { report = GenerateMissingForSessionLayouts(); }
            catch (Exception ex) { report = ex.Message; }
            if (!string.IsNullOrEmpty(previous)) EditorSceneManager.OpenScene(previous, OpenSceneMode.Single);
            EditorUtility.DisplayDialog("Human Tasks", report, "OK");
        }

        /// <summary>Batch mode: Unity -batchmode -quit -executeMethod Assets.Scripts.Editor.HumanTaskGenerator.BatchGenerateMissing</summary>
        public static void BatchGenerateMissing()
        {
            string report = GenerateMissingForSessionLayouts();
            if (report.Contains("FAILED")) EditorApplication.Exit(1);
        }

        /// <summary>Generates (and saves) the task set of every session layout scene that has none or a stale one; keeps valid ones.</summary>
        private static string GenerateMissingForSessionLayouts()
        {
            string[] guids = AssetDatabase.FindAssets("t:ParticipantSessionConfig");
            if (guids.Length != 1) throw new InvalidOperationException($"Expected exactly one ParticipantSessionConfig asset, found {guids.Length}.");
            var config = AssetDatabase.LoadAssetAtPath<ParticipantSessionConfig>(AssetDatabase.GUIDToAssetPath(guids[0]));

            var report = new StringBuilder();
            foreach (string path in config.LayoutScenePaths)
            {
                Scene scene = EditorSceneManager.OpenScene(path, OpenSceneMode.Single);
                try
                {
                    HumanTaskSet existing = FindSet(scene);
                    string stale = existing != null ? existing.CheckTestTaskUnchanged(ScenePreparation.FindTask(scene)) : "no task set";
                    if (stale == null && existing.TrainingCount == TrainingTaskCount)
                    {
                        ExportJson(scene, existing);
                        report.AppendLine($"{scene.name}: kept (coverage {existing.testRouteCoverage:P0}, C leaves out {existing.training[existing.cExcludedIndex].id}).");
                        continue;
                    }
                    HumanTaskSet set = Generate(scene, existing != null ? existing.reroll : 0);
                    EditorSceneManager.SaveScene(scene);
                    ExportJson(scene, set);
                    report.AppendLine($"{scene.name}: generated ({stale}) - coverage {set.testRouteCoverage:P0}, C leaves out {set.training[set.cExcludedIndex].id}.");
                }
                catch (Exception ex)
                {
                    Debug.LogException(ex);
                    report.AppendLine($"{scene.name}: FAILED - {ex.Message}");
                }
            }
            Debug.Log("[HumanTaskGenerator] " + report);
            return report.ToString();
        }

        private static void RunOnActiveScene(bool reroll)
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            HumanTaskSet existing = FindSet(scene);
            if (existing != null && existing.TrainingCount > 0 &&
                !EditorUtility.DisplayDialog("Human Tasks",
                    $"'{scene.name}' already has human training tasks. Replacing them after participants have walked them breaks the " +
                    "comparability of the sessions. Replace?", reroll ? "Re-roll" : "Regenerate", "Cancel"))
            {
                return;
            }
            try
            {
                int rerollIndex = (existing != null ? existing.reroll : 0) + (reroll ? 1 : 0);
                HumanTaskSet set = Generate(scene, rerollIndex);
                ExportJson(scene, set);
                Selection.activeObject = set.gameObject;
                EditorUtility.DisplayDialog("Human Tasks", set.generationReport + "\n\nSave the scene to keep them.", "OK");
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorUtility.DisplayDialog("Human Tasks", $"Generation failed:\n{ex.Message}", "OK");
            }
        }

        private static HumanTaskSet FindSet(Scene scene)
        {
            GameObject root = ScenePreparation.FindRoot(scene, ScenePreparation.EnvironmentRootName);
            return root != null ? root.GetComponent<HumanTaskSet>() : null;
        }

        // ------------------------------------------------------------------
        // Generation
        // ------------------------------------------------------------------

        private sealed class Candidate
        {
            public HumanTaskSet.TaskDef def;
            public Vector3[] route;
            public ulong[] coverBits;
            public int[] cells;
            public Vector3 navStart, navPoi, navEnd;
        }

        private sealed class ReferenceRoute
        {
            public Vector3 a, b;
            public Vector3[] corners;
        }

        public static HumanTaskSet Generate(Scene scene, int reroll)
        {
            GameObject root = ScenePreparation.FindRoot(scene, ScenePreparation.EnvironmentRootName);
            if (root == null) throw new InvalidOperationException($"'{scene.name}' has no {ScenePreparation.EnvironmentRootName}.");
            LayoutInfo layoutInfo = root.GetComponent<LayoutInfo>();
            TaskScript task = ScenePreparation.FindTask(scene);
            if (task == null || task.start == null || task.start.Length != 1 || task.pointsOfInterest == null || task.pointsOfInterest.Length != 1 ||
                task.end == null || task.end.Length != 1)
            {
                throw new InvalidOperationException($"'{scene.name}' has no single Start -> POI_0 -> End task. Run Tools > Scene Preparation > Prepare Scene first.");
            }
            if (NavMesh.CalculateTriangulation().vertices.Length == 0)
                throw new InvalidOperationException($"'{scene.name}' has no baked NavMesh. Run Tools > Scene Preparation > Prepare Scene first.");

            int layoutSeed = layoutInfo != null ? layoutInfo.seed : scene.name.GetHashCode();
            int seed = unchecked(layoutSeed + SeedSalt + reroll * 7919);
            var rng = new System.Random(seed);
            int obstacleMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);
            var sampler = new ScenePreparation.NavMeshSampler();
            float floorY = sampler.FloorY;

            // Test task: the scene's markers, snapped to the NavMesh for routing.
            Vector3 testStartMarker = task.start[0].transform.position;
            Vector3 testPoiMarker = task.pointsOfInterest[0].transform.position;
            Vector3 testEndMarker = task.end[0].transform.position;
            Vector3 tS = Snap(testStartMarker), tP = Snap(testPoiMarker), tE = Snap(testEndMarker);
            if (!TryRoute(tS, tP, tE, out Vector3[] testRoute, out float testLength))
                throw new InvalidOperationException("The test task has no complete NavMesh route Start -> POI_0 -> End.");
            Vector3[] testSamples = Resample(testRoute, RouteSampleSpacing);
            Vector3[] testMarkers = { tS, tP, tE };

            List<ReferenceRoute> references = BuildReferenceRoutes(sampler, rng);

            var test = new HumanTaskSet.TaskDef
            {
                id = HumanTaskSet.TestTaskId,
                start = testStartMarker,
                poi = testPoiMarker,
                end = testEndMarker,
                startYawDeg = StartYaw(tS, tP),
                routeLengthM = testLength,
                testRouteOverlap = 1f,
                minWallClearanceM = Mathf.Min(WallClearance(tS, obstacleMask), Mathf.Min(WallClearance(tP, obstacleMask), WallClearance(tE, obstacleMask))),
                minThroughTraffic = Mathf.Min(ThroughTraffic(tS, references), Mathf.Min(ThroughTraffic(tP, references), ThroughTraffic(tE, references)))
            };

            // Pool of valid marker positions.
            var pool = new List<Vector3>();
            for (int i = 0; i < PoolSamples && pool.Count < PoolMax; i++)
            {
                if (!sampler.TrySample(rng, out Vector3 p)) continue;
                if (p.y > floorY + ScenePreparation.MaxHeightAboveFloorMeters) continue;
                bool nearTest = false;
                foreach (Vector3 m in testMarkers) if (HorizontalDistance(p, m) < MinTestMarkerDistance) { nearTest = true; break; }
                if (nearTest) continue;
                if (WallClearance(p, obstacleMask) < MinWallClearance) continue;
                if (ThroughTraffic(p, references) < MinThroughTraffic) continue;
                pool.Add(p);
            }
            if (pool.Count < 9)
                throw new InvalidOperationException($"Only {pool.Count} marker positions satisfy the rules (>= {MinWallClearance} m from walls, " +
                    $">= {MinTestMarkerDistance} m from test markers, through-traffic >= {MinThroughTraffic:P0}). The layout is too tight for these rules.");

            // Candidate tasks from the pool.
            var cellIndex = new Dictionary<long, int>();
            var candidates = new List<Candidate>();
            for (int i = 0; i < CandidateAttempts && candidates.Count < CandidateMax; i++)
            {
                Vector3 s = pool[rng.Next(pool.Count)], p = pool[rng.Next(pool.Count)], e = pool[rng.Next(pool.Count)];
                if (HorizontalDistance(s, p) < ScenePreparation.MinMarkerSpacingMeters || HorizontalDistance(p, e) < ScenePreparation.MinMarkerSpacingMeters ||
                    HorizontalDistance(s, e) < ScenePreparation.MinMarkerSpacingMeters) continue;
                if (!TryRoute(s, p, e, out Vector3[] route, out float length)) continue;
                if (length < MinRouteLength || length > MaxRouteLength) continue;
                if (!ScenePreparation.TryPath(s, e, out NavMeshPath direct)) continue;
                if (ScenePreparation.DistanceToPolyline(p, direct.corners) < ScenePreparation.PreferredRouteOffsetMeters) continue;
                if (IsNearDuplicate(s, p, e, candidates)) continue;

                Vector3[] samples = Resample(route, RouteSampleSpacing);
                ulong[] bits = CoverBits(testSamples, route);
                candidates.Add(new Candidate
                {
                    navStart = s, navPoi = p, navEnd = e,
                    route = route,
                    coverBits = bits,
                    cells = Cells(samples, cellIndex),
                    def = new HumanTaskSet.TaskDef
                    {
                        start = s,
                        poi = p + Vector3.up * ScenePreparation.MarkerHeightAboveNavMesh,
                        end = e + Vector3.up * ScenePreparation.MarkerHeightAboveNavMesh,
                        startYawDeg = StartYaw(s, p),
                        routeLengthM = length,
                        testRouteOverlap = PopCount(bits) / (float)testSamples.Length,
                        minWallClearanceM = Mathf.Min(WallClearance(s, obstacleMask), Mathf.Min(WallClearance(p, obstacleMask), WallClearance(e, obstacleMask))),
                        minThroughTraffic = Mathf.Min(ThroughTraffic(s, references), Mathf.Min(ThroughTraffic(p, references), ThroughTraffic(e, references)))
                    }
                });
            }
            if (candidates.Count < TrainingTaskCount)
                throw new InvalidOperationException($"Only {candidates.Count} candidate tasks with a {MinRouteLength}-{MaxRouteLength} m route were found " +
                    $"from {pool.Count} valid marker positions.");

            Candidate[] chosen = SelectTriple(candidates, testSamples.Length, cellIndex.Count, out float coverage);

            // Stable order: by route length, so train_0..2 don't depend on the candidate order.
            Array.Sort(chosen, (a, b) => a.def.routeLengthM.CompareTo(b.def.routeLengthM));
            var training = new HumanTaskSet.TaskDef[chosen.Length];
            int cExcluded = 0;
            for (int i = 0; i < chosen.Length; i++)
            {
                training[i] = chosen[i].def;
                training[i].id = HumanTaskSet.TrainTaskId(i);
                if (training[i].testRouteOverlap < training[cExcluded].testRouteOverlap) cExcluded = i;
            }

            HumanTaskSet set = root.GetComponent<HumanTaskSet>();
            if (set == null) set = Undo.AddComponent<HumanTaskSet>(root);
            else Undo.RecordObject(set, "Generate human tasks");
            set.test = test;
            set.training = training;
            set.cExcludedIndex = cExcluded;
            set.testRouteCoverage = coverage;
            set.controllerPivotAboveStart = ScenePreparation.ControllerPivotAboveNavMesh;
            set.reroll = reroll;
            set.generatorSeed = seed;
            set.generatorVersion = Version;
            set.generationReport = Report(scene, set, pool.Count, candidates.Count);
            EditorUtility.SetDirty(set);
            EditorSceneManager.MarkSceneDirty(scene);
            Debug.Log("[HumanTaskGenerator] " + set.generationReport);
            return set;
        }

        private static Candidate[] SelectTriple(List<Candidate> candidates, int testSampleCount, int cellCount, out float coverage)
        {
            int n = candidates.Count, words = candidates[0].coverBits.Length;
            var union = new ulong[words];
            int best = -1;
            var scores = new List<(int a, int b, int c, int covered)>();
            for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++)
            for (int c = b + 1; c < n; c++)
            {
                int covered = 0;
                for (int w = 0; w < words; w++)
                {
                    union[w] = candidates[a].coverBits[w] | candidates[b].coverBits[w] | candidates[c].coverBits[w];
                    covered += PopCount(union[w]);
                }
                if (covered > best) best = covered;
                scores.Add((a, b, c, covered));
            }

            // Near-ties: the triple whose routes span the most floor cells (the most varied routes).
            int threshold = best - Mathf.FloorToInt(CoverageTieTolerance * testSampleCount);
            var stamp = new int[cellCount];
            int stampId = 0, bestCells = -1;
            (int a, int b, int c) pick = (0, 1, 2);
            foreach (var s in scores)
            {
                if (s.covered < threshold) continue;
                stampId++;
                int cells = 0;
                foreach (int idx in new[] { s.a, s.b, s.c })
                    foreach (int cell in candidates[idx].cells)
                        if (stamp[cell] != stampId) { stamp[cell] = stampId; cells++; }
                if (cells > bestCells) { bestCells = cells; pick = (s.a, s.b, s.c); }
            }

            var chosen = new[] { candidates[pick.a], candidates[pick.b], candidates[pick.c] };
            int finalCovered = 0;
            for (int w = 0; w < words; w++) finalCovered += PopCount(chosen[0].coverBits[w] | chosen[1].coverBits[w] | chosen[2].coverBits[w]);
            coverage = finalCovered / (float)testSampleCount;
            return chosen;
        }

        // ------------------------------------------------------------------
        // Export
        // ------------------------------------------------------------------

        [Serializable] private sealed class JsonTask
        {
            public string id, role;
            public bool c_excluded;
            public Vector3 start, poi, end;
            public float start_yaw_deg, route_length_m, test_route_overlap, min_wall_clearance_m, min_through_traffic;
            public Vector3[] route;
        }

        [Serializable] private sealed class JsonTaskSet
        {
            public string scene_name, generator_version;
            public int generator_seed, reroll;
            public float test_route_coverage;
            public JsonTask[] tasks;
        }

        /// <summary>
        /// Writes eval_tools/layouts/&lt;scene&gt;_human_tasks.json (next to the occupancy grid): every task with its markers and
        /// NavMesh route, so Python can plot the tasks and select episodes by task without opening the scene.
        /// </summary>
        private static void ExportJson(Scene scene, HumanTaskSet set)
        {
            var tasks = new List<JsonTask> { ToJson(set.test, HumanTaskSet.RoleTest, false) };
            for (int i = 0; i < set.training.Length; i++) tasks.Add(ToJson(set.training[i], HumanTaskSet.RoleTrain, i == set.cExcludedIndex));
            var json = new JsonTaskSet
            {
                scene_name = scene.name,
                generator_version = set.generatorVersion,
                generator_seed = set.generatorSeed,
                reroll = set.reroll,
                test_route_coverage = set.testRouteCoverage,
                tasks = tasks.ToArray()
            };
            string dir = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "..", "..", "eval_tools", "layouts"));
            System.IO.Directory.CreateDirectory(dir);
            string path = System.IO.Path.Combine(dir, scene.name + "_human_tasks.json");
            System.IO.File.WriteAllText(path, JsonUtility.ToJson(json, true));
            Debug.Log($"[HumanTaskGenerator] Wrote {path}");
        }

        private static JsonTask ToJson(HumanTaskSet.TaskDef t, string role, bool cExcluded)
        {
            TryRoute(Snap(t.start), Snap(t.poi), Snap(t.end), out Vector3[] route, out _);
            return new JsonTask
            {
                id = t.id, role = role, c_excluded = cExcluded,
                start = t.start, poi = t.poi, end = t.end,
                start_yaw_deg = t.startYawDeg, route_length_m = t.routeLengthM, test_route_overlap = t.testRouteOverlap,
                min_wall_clearance_m = t.minWallClearanceM, min_through_traffic = t.minThroughTraffic,
                route = route ?? new Vector3[0]
            };
        }

        // ------------------------------------------------------------------
        // Measures
        // ------------------------------------------------------------------

        private static List<ReferenceRoute> BuildReferenceRoutes(ScenePreparation.NavMeshSampler sampler, System.Random rng)
        {
            var routes = new List<ReferenceRoute>();
            for (int i = 0; i < ReferenceRouteCount * 20 && routes.Count < ReferenceRouteCount; i++)
            {
                if (!sampler.TrySample(rng, out Vector3 a) || !sampler.TrySample(rng, out Vector3 b)) continue;
                if (!ScenePreparation.TryPath(a, b, out NavMeshPath path)) continue;
                if (ScenePreparation.PathLength(path) < ReferenceRouteMinLength) continue;
                routes.Add(new ReferenceRoute { a = a, b = b, corners = path.corners });
            }
            if (routes.Count < ReferenceRouteCount / 4)
                throw new InvalidOperationException($"Only {routes.Count} reference routes of >= {ReferenceRouteMinLength} m found on the NavMesh.");
            return routes;
        }

        /// <summary>Share of reference routes that pass within TrafficPassRadius of p without starting or ending near it. ~0 in dead-end pockets.</summary>
        private static float ThroughTraffic(Vector3 p, List<ReferenceRoute> routes)
        {
            int passing = 0;
            foreach (ReferenceRoute r in routes)
            {
                if (HorizontalDistance(p, r.a) < TrafficEndpointExclusion || HorizontalDistance(p, r.b) < TrafficEndpointExclusion) continue;
                if (ScenePreparation.DistanceToPolyline(p, r.corners) <= TrafficPassRadius) passing++;
            }
            return passing / (float)routes.Count;
        }

        /// <summary>Smallest horizontal distance to an Architecture_Obstacle collider (fan of rays at 1 m height, capped at ClearanceProbeRange).</summary>
        private static float WallClearance(Vector3 navPoint, int mask)
        {
            Vector3 origin = navPoint + Vector3.up * ClearanceProbeHeight;
            float min = ClearanceProbeRange;
            for (int i = 0; i < ClearanceRays; i++)
            {
                float angle = i * Mathf.PI * 2f / ClearanceRays;
                var dir = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));
                if (Physics.Raycast(origin, dir, out RaycastHit hit, min, mask, QueryTriggerInteraction.Ignore)) min = hit.distance;
            }
            return min;
        }

        private static ulong[] CoverBits(Vector3[] testSamples, Vector3[] route)
        {
            var bits = new ulong[(testSamples.Length + 63) / 64];
            for (int i = 0; i < testSamples.Length; i++)
            {
                if (ScenePreparation.DistanceToPolyline(testSamples[i], route) <= CoverageRadius) bits[i >> 6] |= 1UL << (i & 63);
            }
            return bits;
        }

        private static int[] Cells(Vector3[] samples, Dictionary<long, int> cellIndex)
        {
            var cells = new HashSet<int>();
            foreach (Vector3 s in samples)
            {
                long key = ((long)Mathf.FloorToInt(s.x / CellSize) << 32) ^ (uint)Mathf.FloorToInt(s.z / CellSize);
                if (!cellIndex.TryGetValue(key, out int idx)) { idx = cellIndex.Count; cellIndex[key] = idx; }
                cells.Add(idx);
            }
            var result = new int[cells.Count];
            cells.CopyTo(result);
            return result;
        }

        /// <summary>Rejects a task whose three markers all lie within MinTestMarkerDistance of an already accepted candidate's.</summary>
        private static bool IsNearDuplicate(Vector3 s, Vector3 p, Vector3 e, List<Candidate> accepted)
        {
            foreach (Candidate c in accepted)
            {
                if (HorizontalDistance(s, c.navStart) < MinTestMarkerDistance && HorizontalDistance(p, c.navPoi) < MinTestMarkerDistance &&
                    HorizontalDistance(e, c.navEnd) < MinTestMarkerDistance) return true;
            }
            return false;
        }

        // ------------------------------------------------------------------
        // Geometry
        // ------------------------------------------------------------------

        private static Vector3 Snap(Vector3 marker)
        {
            return NavMesh.SamplePosition(marker, out NavMeshHit hit, ScenePreparation.MarkerHeightAboveNavMesh + 0.5f, NavMesh.AllAreas) ? hit.position : marker;
        }

        private static bool TryRoute(Vector3 s, Vector3 p, Vector3 e, out Vector3[] route, out float length)
        {
            route = null;
            length = 0f;
            if (!ScenePreparation.TryPath(s, p, out NavMeshPath first) || !ScenePreparation.TryPath(p, e, out NavMeshPath second)) return false;
            var corners = new List<Vector3>(first.corners);
            for (int i = 1; i < second.corners.Length; i++) corners.Add(second.corners[i]);
            route = corners.ToArray();
            length = ScenePreparation.PathLength(first) + ScenePreparation.PathLength(second);
            return true;
        }

        private static float StartYaw(Vector3 start, Vector3 poi)
        {
            Vector3 facing = ScenePreparation.TryPath(start, poi, out NavMeshPath path) && path.corners.Length > 1 ? path.corners[1] - start : poi - start;
            facing.y = 0f;
            return facing.sqrMagnitude > 1e-6f ? Quaternion.LookRotation(facing).eulerAngles.y : 0f;
        }

        private static Vector3[] Resample(Vector3[] polyline, float spacing)
        {
            var samples = new List<Vector3> { polyline[0] };
            float carry = 0f;
            for (int i = 1; i < polyline.Length; i++)
            {
                Vector3 a = polyline[i - 1], b = polyline[i];
                float segment = Vector3.Distance(a, b);
                float t = spacing - carry;
                while (t <= segment)
                {
                    samples.Add(Vector3.Lerp(a, b, t / segment));
                    t += spacing;
                }
                carry = segment - (t - spacing);
            }
            samples.Add(polyline[polyline.Length - 1]);
            return samples.ToArray();
        }

        private static float HorizontalDistance(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x, dz = a.z - b.z;
            return Mathf.Sqrt(dx * dx + dz * dz);
        }

        private static int PopCount(ulong[] words)
        {
            int count = 0;
            foreach (ulong w in words) count += PopCount(w);
            return count;
        }

        private static int PopCount(ulong x)
        {
            int count = 0;
            while (x != 0) { x &= x - 1; count++; }
            return count;
        }

        private static string Report(Scene scene, HumanTaskSet set, int poolCount, int candidateCount)
        {
            var inv = CultureInfo.InvariantCulture;
            var sb = new StringBuilder();
            sb.AppendLine($"'{scene.name}' ({set.generatorVersion}, seed {set.generatorSeed}, reroll {set.reroll}): {poolCount} valid marker positions, " +
                          $"{candidateCount} candidate tasks.");
            sb.AppendLine($"Test-route coverage by the training routes: {set.testRouteCoverage.ToString("P0", inv)}.");
            void Line(HumanTaskSet.TaskDef t, string note) =>
                sb.AppendLine($"  {t.id,-8} route {t.routeLengthM.ToString("F1", inv)} m, test overlap {t.testRouteOverlap.ToString("P0", inv)}, " +
                              $"wall clearance {t.minWallClearanceM.ToString("F2", inv)} m, through-traffic {t.minThroughTraffic.ToString("P1", inv)}{note}");
            Line(set.test, "");
            for (int i = 0; i < set.training.Length; i++) Line(set.training[i], i == set.cExcludedIndex ? "  <- left out in C" : "");
            if (set.test.minWallClearanceM < MinWallClearance || set.test.minThroughTraffic < MinThroughTraffic)
            {
                sb.AppendLine($"Note: the (kept) test task itself misses the placement rules (clearance >= {MinWallClearance} m, " +
                              $"through-traffic >= {MinThroughTraffic.ToString("P0", inv)}).");
            }
            return sb.ToString();
        }
    }
}
