using System;
using System.IO;
using System.Linq;
using Assets.Scripts.DataRecording;
using Unity.Sentis;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Sets up the open layout scene to run agents through the toolkit's EngineScript, the same way for both agent kinds:
    /// the shortest-path agent (AgentScript on a NavMeshAgent) and the imitation agent (ImitationAgent running a trained
    /// model) differ only in EngineScript.simulationMode. EngineScript sits next to the scene's TaskScript
    /// (Environment_Root/Task, from Prepare Scene), spawns the task's agents on Play and writes their trajectories to
    /// Data_ABM/, where Add Heatmap Visualizer picks them up. Re-running a menu item switches the scene to that agent kind
    /// and resets the values below.
    /// </summary>
    internal static class AgentSimulationSetup
    {
        private const string DataFolder = "Data_ABM"; // relative to the Unity project folder; ABMVisualizer reads it too
        private const string LatestModelAssetPath = "Assets/ImitationModel/" + RetrainPipeline.LatestModelFileName;
        private const string VisualizerName = "ABMVisualizer";
        private const float VisualizerResolution = 0.5f;

        [MenuItem(ToolsMenu.ShortestPath + "Set Up Agent Simulation", priority = ToolsMenu.ShortestPathPriority)]
        private static void SetUpShortestPath()
        {
            SetUp(EngineScript.AgentSimulationMode.RuleBasedNavMesh, null, "Shortest Path Agent");
        }

        [MenuItem(ToolsMenu.RunAgent + "Set Up Agent Simulation", priority = ToolsMenu.RunAgentPriority)]
        private static void SetUpImitation()
        {
            ModelAsset model = PickModel(out string modelSource);
            if (model == null)
            {
                EditorUtility.DisplayDialog("Run Imitation Agent",
                    "No model found. Train one (Imitation Learning Agent > 3 · Train Model) or select a .onnx model in the Project " +
                    "window, then run this again.", "OK");
                return;
            }
            SetUp(EngineScript.AgentSimulationMode.ImitationLearning, model, "Run Imitation Agent",
                $"Model: {model.name} ({modelSource}). To run another one, select it in the Project window and run this again.");
        }

        [MenuItem(ToolsMenu.ShortestPath + "Add Heatmap Visualizer", priority = ToolsMenu.ShortestPathPriority + 1)]
        [MenuItem(ToolsMenu.EvaluateDesign + "Add Heatmap Visualizer", priority = ToolsMenu.EvaluateDesignPriority + 1)]
        private static void AddHeatmapVisualizer()
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            GameObject root = scene.GetRootGameObjects().FirstOrDefault(go => go.name == ScenePreparation.EnvironmentRootName);
            if (root == null)
            {
                EditorUtility.DisplayDialog("Add Heatmap Visualizer",
                    $"'{scene.name}' has no '{ScenePreparation.EnvironmentRootName}'. Open a generated layout scene.", "OK");
                return;
            }
            Bounds bounds = RendererBounds(root);

            GameObject go = scene.GetRootGameObjects().FirstOrDefault(g => g.name == VisualizerName);
            if (go == null)
            {
                go = new GameObject(VisualizerName);
                Undo.RegisterCreatedObjectUndo(go, "Add Heatmap Visualizer");
            }
            ABMVisualizer visualizer = go.GetComponent<ABMVisualizer>();
            bool created = visualizer == null;
            if (created) visualizer = Undo.AddComponent<ABMVisualizer>(go);
            Undo.RecordObject(visualizer, "Add Heatmap Visualizer");

            // One layer of cells just above the floor: the grid needs a y range of at least one cell.
            float y = bounds.min.y + 0.05f;
            visualizer.corner1 = EnsureCorner(go.transform, "Corner1", new Vector3(bounds.min.x, y, bounds.min.z));
            visualizer.corner2 = EnsureCorner(go.transform, "Corner2", new Vector3(bounds.max.x, y + VisualizerResolution, bounds.max.z));
            if (created)
            {
                visualizer.resolution = VisualizerResolution;
                visualizer.gradient = BlueToRed();
                visualizer.gradientCompare = BlueToRed();
            }
            string latestRun = LatestRun(scene.name);
            if (latestRun != null) visualizer.fileName = latestRun;

            EditorUtility.SetDirty(visualizer);
            EditorSceneManager.MarkSceneDirty(scene);
            // ABMVisualizer's Inspector throws while Data_ABM is empty, so it is only selected once there is data.
            if (latestRun != null) Selection.activeGameObject = go;
            string file = latestRun ?? $"none yet - run a simulation first (it writes to {DataFolder}/)";
            Debug.Log($"[AgentSimulationSetup] Heatmap visualizer covers {bounds.size.x:F1} x {bounds.size.z:F1} m. Data: {file}. " +
                      "Pick agent types/tasks in its Inspector, then press Play.");
        }

        private static void SetUp(EngineScript.AgentSimulationMode mode, ModelAsset model, string title, string extraInfo = null)
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            TaskScript task = ScenePreparation.FindTask(scene);
            if (task == null)
            {
                try
                {
                    ScenePreparation.Prepare(scene);
                }
                catch (Exception ex)
                {
                    Debug.LogException(ex);
                    EditorUtility.DisplayDialog(title, $"The scene has no task and Prepare Scene failed:\n{ex.Message}", "OK");
                    return;
                }
                task = ScenePreparation.FindTask(scene);
            }

            // EngineScript reads the TaskScripts on its own GameObject.
            EngineScript engine = task.GetComponent<EngineScript>() ?? Undo.AddComponent<EngineScript>(task.gameObject);
            Undo.RecordObject(engine, title);
            bool shortestPath = mode == EngineScript.AgentSimulationMode.RuleBasedNavMesh;
            engine.simulationMode = mode;
            if (model != null) engine.defaultBaselineModel = model;
            engine.visualizePOIs = true;
            engine.visualizeTrajectories = true;
            engine.visualizePaths = shortestPath; // only AgentScript has a planned path to draw
            engine.saveHeatMap = true;
            engine.fontSize = 5;
            engine.resize = 1f;
            engine.offset = 0.5f;
            engine.traceLength = 100;
            engine.sampleInterval = 0.1f;
            engine.dataFolder = DataFolder;
            engine.fileName = $"{scene.name}_{(shortestPath ? "shortestpath" : "imitation")}";
            EditorUtility.SetDirty(engine);
            EditorSceneManager.MarkSceneDirty(scene);
            Selection.activeGameObject = task.gameObject;

            string message = $"EngineScript on '{task.name}' runs {task.numberOfAgents} {(shortestPath ? "shortest-path" : "imitation")} " +
                             $"agent(s) on the scene's task when you press Play (set the count on the TaskScript).\n" +
                             $"Trajectories: {DataFolder}/{engine.fileName}.csv";
            if (extraInfo != null) message += "\n\n" + extraInfo;
            string conflicts = OtherAgentRunners(scene);
            if (conflicts != null) message += "\n\nAlso in this scene and active on Play: " + conflicts + ". Disable them for a clean run.";
            Debug.Log($"[AgentSimulationSetup] {message}");
            EditorUtility.DisplayDialog(title, message + "\n\nSave the scene to keep the changes.", "OK");
        }

        /// <summary>The ModelAsset selected in the Project window, else the latest trained model, else the one already set.</summary>
        private static ModelAsset PickModel(out string source)
        {
            if (Selection.activeObject is ModelAsset selected)
            {
                source = "selected in the Project window";
                return selected;
            }
            ModelAsset latest = AssetDatabase.LoadAssetAtPath<ModelAsset>(LatestModelAssetPath);
            if (latest != null)
            {
                source = "latest trained model";
                return latest;
            }
            TaskScript task = ScenePreparation.FindTask(EditorSceneManager.GetActiveScene());
            EngineScript engine = task != null ? task.GetComponent<EngineScript>() : null;
            source = "already set on EngineScript";
            return engine != null ? engine.defaultBaselineModel : null;
        }

        /// <summary>Other components that spawn or record agents on Play, as a readable list, or null.</summary>
        private static string OtherAgentRunners(Scene scene)
        {
            var found = scene.GetRootGameObjects()
                .SelectMany(go => go.GetComponentsInChildren<MonoBehaviour>(false))
                .Where(c => c != null && c.enabled &&
                            (c is EvalHarness || c is RecordRandomWalksHarness || c is DaggerHarness ||
                             (c is ParticipantRecorder recorder && recorder.AutoStartForTesting)))
                .Select(c => $"{c.GetType().Name} on '{c.name}'")
                .ToList();
            return found.Count > 0 ? string.Join(", ", found) : null;
        }

        private static string LatestRun(string sceneName)
        {
            if (!Directory.Exists(DataFolder)) return null;
            return new DirectoryInfo(DataFolder).GetFiles(sceneName + "_*.csv")
                .OrderByDescending(f => f.LastWriteTimeUtc)
                .Select(f => Path.Combine(DataFolder, f.Name))
                .FirstOrDefault();
        }

        private static Bounds RendererBounds(GameObject root)
        {
            Renderer[] renderers = root.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0) return new Bounds(root.transform.position, Vector3.zero);
            Bounds bounds = renderers[0].bounds;
            foreach (Renderer r in renderers) bounds.Encapsulate(r.bounds);
            return bounds;
        }

        private static GameObject EnsureCorner(Transform parent, string name, Vector3 position)
        {
            Transform corner = parent.Find(name);
            if (corner == null)
            {
                corner = new GameObject(name).transform;
                Undo.RegisterCreatedObjectUndo(corner.gameObject, "Add Heatmap Visualizer");
                corner.SetParent(parent, false);
            }
            Undo.RecordObject(corner, "Add Heatmap Visualizer");
            corner.position = position;
            return corner.gameObject;
        }

        private static Gradient BlueToRed()
        {
            var gradient = new Gradient();
            gradient.SetKeys(
                new[] { new GradientColorKey(Color.blue, 0f), new GradientColorKey(Color.yellow, 0.5f), new GradientColorKey(Color.red, 1f) },
                new[] { new GradientAlphaKey(0f, 0f), new GradientAlphaKey(1f, 0.1f), new GradientAlphaKey(1f, 1f) });
            return gradient;
        }
    }
}
