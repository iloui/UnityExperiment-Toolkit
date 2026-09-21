using System;
using Assets.Scripts.DataRecording;
using Unity.Sentis;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// One-click setup for running EvalHarness (zero-shot / in-distribution generalization
    /// diagnostic, see EvalHarness.cs) against whichever scene is currently open - so the exact
    /// same command produces an apples-to-apples EvalHarness setup (same configurationCount,
    /// same agentsPerConfiguration, same model) whether that's CentralPillar_Code_927712 (the
    /// in-distribution baseline) or one of the procedurally-generated, never-trained-on
    /// Assets/Scenes/TestLayouts/* scenes.
    ///
    /// Two cases, detected automatically:
    ///  1. The scene already has a fully-configured TaskScript (e.g. CentralPillar_Code_927712's
    ///     SimulationManager, authored by hand with its own POIs/behavioural values) - reuse it
    ///     exactly as-is, untouched, and just point a fresh/existing EvalHarness at it.
    ///  2. The scene has no TaskScript yet (the TestLayouts scenes: LayoutBuilder.Build() only
    ///     produces geometry + a baked NavMesh + a single "Goal_POI" marker - the task-wiring
    ///     step AgentLinker.SpawnAndWireParticipants() is deliberately not run on them, since
    ///     they're meant to stay held out as unseen evaluation layouts, not become more training
    ///     data) - bootstrap a minimal TaskScript for it: a new Start_Point marker plus the
    ///     existing Goal_POI as the sole "end", following the same pointsOfInterest=[] /
    ///     numberOfNeeds=0 convention AgentLinker already uses for these generated layouts.
    ///     The one thing that *is* pulled from CentralPillar_Code_927712 in this case is the
    ///     behavioural block (agentSpeed/agentSize/agentRadius/privacyRadius/poiTime/...) -
    ///     ImitationAgent.Start() reads those straight off TaskScript, so leaving them at the
    ///     AddComponent<TaskScript>() default of zero would silently give every spawned eval
    ///     agent zero movement speed. That reference scene is loaded additively, read, and closed
    ///     again without saving - it is never modified.
    /// </summary>
    internal static class SetupEvalScene
    {
        private const string ReferenceScenePath = "Assets/Scenes/TestLayouts/CentralPillar_Code_927712.unity";
        private const string ModelAssetPath = "Assets/ImitationModel/ImitationAgentModel.onnx";
        private const float NavMeshSnapRadius = 5f;
        private const int DefaultConfigurationCount = 20;

        private struct TaskBehaviourSnapshot
        {
            public int numberOfAgents;
            public float spawnInterval;
            public Color taskColor;
            public string agentType;
            public float agentSpeed;
            public float agentSize;
            public float agentRadius;
            public float privacyRadius;
            public float poiTime;
        }

        [MenuItem("Tools/Eval Harness/Wire Up Current Scene For Eval")]
        public static void WireUpCurrentScene()
        {
            Scene targetScene = EditorSceneManager.GetActiveScene();
            if (!targetScene.IsValid())
            {
                Debug.LogError("[SetupEvalScene] No valid active scene - open a scene first.");
                return;
            }

            TaskScript task = FindExistingUsableTask(targetScene);
            bool reusedExisting = task != null;

            if (reusedExisting)
            {
                if (task.agentSpeed > 0f)
                {
                    Debug.Log($"[SetupEvalScene] '{targetScene.name}' already has a fully-configured TaskScript ('{task.name}', agentSpeed={task.agentSpeed:F2}) - reusing it untouched.");
                }
                else
                {
                    // Marker wiring (start/end/POIs) is correct - e.g. AgentLinker.SpawnAndWireParticipants()
                    // already ran on this scene - but its behavioural fields (agentSpeed/agentSize/agentRadius/...)
                    // were left at TaskScript's C# defaults because that wiring is for ParticipantRecorder (a
                    // human), which never reads them. ImitationAgent.Start() DOES read them, so patch them in
                    // from the reference scene without touching the (already correct) start/end/POI wiring.
                    Debug.LogWarning($"[SetupEvalScene] '{targetScene.name}''s existing TaskScript ('{task.name}') has agentSpeed=0 " +
                        "(likely wired for ParticipantRecorder, not ImitationAgent) - patching in behavioural values from the reference scene.");
                    if (!TryReadReferenceTaskBehaviour(targetScene, out TaskBehaviourSnapshot behaviour)) return;
                    ApplyBehaviour(task, behaviour);
                    EditorUtility.SetDirty(task);
                }
            }
            else
            {
                task = BootstrapMinimalTask(targetScene);
                if (task == null) return; // error already logged
            }

            GameObject harnessGO = GameObject.Find("EvalHarness") ?? new GameObject("EvalHarness");
            EvalHarness harness = harnessGO.GetComponent<EvalHarness>();
            if (harness == null) harness = harnessGO.AddComponent<EvalHarness>();

            harness.task = task;
            harness.modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelAssetPath);
            if (harness.modelAsset == null)
            {
                Debug.LogWarning($"[SetupEvalScene] Could not load model asset at '{ModelAssetPath}' - assign EvalHarness.modelAsset manually before pressing Play.");
            }
            harness.configurationCount = DefaultConfigurationCount;
            harness.agentsPerConfiguration = 1;

            // Canonical eval settings - identical for every scene so runs are apples-to-apples. The harness enforces
            // the canonical start -> end task (no POIs) in memory at Play time; nothing about the scene's own
            // TaskScript is saved. Re-running this menu item resets a previously wired harness to these values.
            harness.enforceCanonicalTask = true;
            harness.canonicalStart = new Vector3(-11.25f, 0.08333334f, 0f);
            harness.canonicalEnd = new Vector3(11.25f, 1f, 0f);
            harness.randomizeMarkersPerConfiguration = false;
            harness.disableDeadlockNudge = true;
            harness.simulationSpeed = 3f;
            harness.episodeTimeoutSeconds = 90f; // simulated seconds; a 45s human-comparable cutoff is applied afterwards by summarize_eval.py
            harness.humanReferenceSeconds = 45f;
            harness.startJitterPositionMeters = 0f;
            harness.startJitterHeadingDegrees = 0f;
            harness.maxPlausibleSpeedMps = 2.5f;
            EditorUtility.SetDirty(harness);

            EditorSceneManager.MarkSceneDirty(targetScene);

            string modelName = harness.modelAsset != null ? harness.modelAsset.name : "MISSING - assign manually";
            Debug.Log($"[SetupEvalScene] '{targetScene.name}' wired up: task='{task.name}' ({(reusedExisting ? "existing" : "bootstrapped")}), " +
                $"configurationCount={harness.configurationCount}, canonical task (fixed start->end, 0 POIs), simulationSpeed=3, timeout=90s (sim), nudge off, model='{modelName}'. Save the scene, then press Play.");
            EditorUtility.DisplayDialog("Eval scene wired up",
                $"'{targetScene.name}' is ready: EvalHarness -> task '{task.name}' ({(reusedExisting ? "reused existing" : "bootstrapped")}), " +
                $"{harness.configurationCount} configurations x 1 agent, canonical task, 3x speed, 90s sim timeout, nudge off, model = '{modelName}'." +
                "\n\nSave the scene (Ctrl/Cmd+S), then press Play.", "OK");
        }

        /// <summary>
        /// Looks for a TaskScript already present in the active scene with valid (non-empty,
        /// non-null-element) start and end arrays - i.e. a scene someone already hand-wired for
        /// ParticipantRecorder/ImitationAgent, like CentralPillar_Code_927712. Returns null if
        /// none is found, so the caller falls back to bootstrapping a minimal one.
        /// </summary>
        private static TaskScript FindExistingUsableTask(Scene scene)
        {
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                foreach (TaskScript candidate in root.GetComponentsInChildren<TaskScript>(true))
                {
                    if (IsNonEmpty(candidate.start) && IsNonEmpty(candidate.end))
                    {
                        return candidate;
                    }
                }
            }
            return null;
        }

        private static bool IsNonEmpty(GameObject[] markers)
        {
            if (markers == null || markers.Length == 0) return false;
            foreach (var m in markers)
            {
                if (m == null) return false;
            }
            return true;
        }

        private static TaskScript BootstrapMinimalTask(Scene targetScene)
        {
            if (!TryReadReferenceTaskBehaviour(targetScene, out TaskBehaviourSnapshot behaviour))
            {
                return null; // error already logged
            }

            GameObject spawnDoorLower = GameObject.Find("SpawnRoomDoorWall_Lower");
            GameObject spawnDoorUpper = GameObject.Find("SpawnRoomDoorWall_Upper");
            GameObject goalPoi = GameObject.Find("Goal_POI");

            if (spawnDoorLower == null || spawnDoorUpper == null || goalPoi == null)
            {
                Debug.LogError("[SetupEvalScene] This scene has no existing TaskScript, and is missing at least one of the " +
                    "expected LayoutBuilder markers ('SpawnRoomDoorWall_Lower', 'SpawnRoomDoorWall_Upper', 'Goal_POI') needed " +
                    "to bootstrap one. Aborting.");
                return null;
            }

            var triangulation = NavMesh.CalculateTriangulation();
            if (triangulation.vertices == null || triangulation.vertices.Length == 0)
            {
                Debug.LogError("[SetupEvalScene] This scene's NavMesh has no vertices - is it baked? Aborting.");
                return null;
            }

            Vector3 spawnDoorMid = (spawnDoorLower.transform.position + spawnDoorUpper.transform.position) / 2f;
            GameObject startMarker = FindOrCreateMarker("Start_Point", SnapToNavMesh(spawnDoorMid));

            GameObject taskGO = GameObject.Find("EvalTask") ?? new GameObject("EvalTask");
            TaskScript task = taskGO.GetComponent<TaskScript>();
            if (task == null) task = taskGO.AddComponent<TaskScript>();

            task.start = new[] { startMarker };
            task.pointsOfInterest = Array.Empty<GameObject>();
            task.end = new[] { goalPoi };
            task.numberOfNeeds = 0;
            task.revisit = false;
            task.chooseNonDeterministically = false;
            task.taskName = "ZeroShotEval_" + targetScene.name;
            ApplyBehaviour(task, behaviour);
            EditorUtility.SetDirty(task);

            return task;
        }

        private static void ApplyBehaviour(TaskScript task, TaskBehaviourSnapshot behaviour)
        {
            task.numberOfAgents = behaviour.numberOfAgents;
            task.spawnInterval = behaviour.spawnInterval;
            task.taskColor = behaviour.taskColor;
            task.agentType = behaviour.agentType;
            task.agentSpeed = behaviour.agentSpeed;
            task.agentSize = behaviour.agentSize;
            task.agentRadius = behaviour.agentRadius;
            task.privacyRadius = behaviour.privacyRadius;
            task.poiTime = behaviour.poiTime;
        }

        /// <summary>
        /// Reads the reference scene's TaskScript behavioural values. If the target scene IS the
        /// reference scene (and somehow has no usable TaskScript of its own - FindExistingUsableTask
        /// already returned null), reads directly from the already-loaded targetScene instead of
        /// re-opening it additively, to avoid loading a second copy of the same scene.
        /// </summary>
        private static bool TryReadReferenceTaskBehaviour(Scene targetScene, out TaskBehaviourSnapshot behaviour)
        {
            behaviour = default;
            bool isSelfReference = targetScene.path == ReferenceScenePath;

            Scene refScene = isSelfReference ? targetScene : EditorSceneManager.OpenScene(ReferenceScenePath, OpenSceneMode.Additive);
            if (!refScene.IsValid())
            {
                Debug.LogError($"[SetupEvalScene] Could not open reference scene '{ReferenceScenePath}'.");
                return false;
            }

            try
            {
                TaskScript refTask = null;
                foreach (GameObject root in refScene.GetRootGameObjects())
                {
                    refTask = root.GetComponentInChildren<TaskScript>(true);
                    if (refTask != null) break;
                }

                if (refTask == null)
                {
                    Debug.LogError($"[SetupEvalScene] Reference scene '{ReferenceScenePath}' has no TaskScript component - cannot copy behavioural values.");
                    return false;
                }

                behaviour = new TaskBehaviourSnapshot
                {
                    numberOfAgents = refTask.numberOfAgents,
                    spawnInterval = refTask.spawnInterval,
                    taskColor = refTask.taskColor,
                    agentType = refTask.agentType,
                    agentSpeed = refTask.agentSpeed,
                    agentSize = refTask.agentSize,
                    agentRadius = refTask.agentRadius,
                    privacyRadius = refTask.privacyRadius,
                    poiTime = refTask.poiTime
                };
                return true;
            }
            finally
            {
                if (!isSelfReference) EditorSceneManager.CloseScene(refScene, true);
            }
        }

        private static GameObject FindOrCreateMarker(string name, Vector3 position)
        {
            GameObject go = GameObject.Find(name);
            if (go == null) go = new GameObject(name);
            go.transform.position = position;
            return go;
        }

        private static Vector3 SnapToNavMesh(Vector3 approximatePosition)
        {
            if (NavMesh.SamplePosition(approximatePosition, out NavMeshHit hit, NavMeshSnapRadius, NavMesh.AllAreas))
            {
                return hit.position;
            }
            Debug.LogWarning($"[SetupEvalScene] Could not snap {approximatePosition} onto the NavMesh within {NavMeshSnapRadius}m - using the raw position. " +
                "This only matters if you later run with randomizeMarkersPerConfiguration disabled.");
            return approximatePosition;
        }
    }
}
