using Assets.Scripts.DataRecording;
using Unity.Sentis;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// One-click setup for running EvalHarness against the open layout scene. Runs ScenePreparation.Prepare first (idempotent:
    /// keeps the scene's Start/POI_0/End if still valid, resets the TaskScript's behaviour values), then points a fresh or
    /// existing EvalHarness at that task. Every layout is evaluated on its own start -> POI -> end task, so the harness's
    /// canonical-task override is switched off.
    /// </summary>
    internal static class SetupEvalScene
    {
        private const string ModelAssetPath = "Assets/ImitationModel/ImitationAgentModel.onnx";
        private const int DefaultConfigurationCount = 20;

        [MenuItem("Tools/Eval Harness/Wire Up Current Scene For Eval")]
        public static void WireUpCurrentScene()
        {
            Scene targetScene = EditorSceneManager.GetActiveScene();
            if (!targetScene.IsValid())
            {
                Debug.LogError("[SetupEvalScene] No valid active scene - open a scene first.");
                return;
            }

            try
            {
                ScenePreparation.Prepare(targetScene);
            }
            catch (System.Exception ex)
            {
                Debug.LogException(ex);
                EditorUtility.DisplayDialog("Eval scene setup failed", ex.Message, "OK");
                return;
            }
            TaskScript task = ScenePreparation.FindTask(targetScene);

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

            // Eval settings - identical for every scene so runs are apples-to-apples. Each layout keeps its own fixed
            // Start -> POI_0 -> End task. Re-running this menu item resets a previously wired harness to these values.
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
            Debug.Log($"[SetupEvalScene] '{targetScene.name}' wired up: task='{task.taskName}' (start -> 1 POI -> end), " +
                $"configurationCount={harness.configurationCount}, simulationSpeed=3, timeout=90s (sim), nudge off, model='{modelName}'. Save the scene, then press Play.");
            EditorUtility.DisplayDialog("Eval scene wired up",
                $"'{targetScene.name}' is ready: EvalHarness -> task '{task.taskName}' (start -> 1 POI -> end), " +
                $"{harness.configurationCount} configurations x 1 agent, 3x speed, 90s sim timeout, nudge off, model = '{modelName}'." +
                "\n\nSave the scene (Ctrl/Cmd+S), then press Play.", "OK");
        }
    }
}
