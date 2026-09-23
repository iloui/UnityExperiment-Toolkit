using Assets.Scripts.DataRecording;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// One-click setup for RecordRandomWalksHarness on whichever layout scene is currently open.
    /// The harness builds everything else (task, markers, agent, camera, recorder) itself at Play time,
    /// so the scene only needs the harness component plus a baked NavMesh and Architecture_Obstacle geometry.
    /// </summary>
    internal static class SetupRandomWalksScene
    {
        [MenuItem("Tools/Random Walks/Add RecordRandomWalksHarness To Current Scene")]
        public static void AddHarness()
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            if (!scene.IsValid())
            {
                Debug.LogError("[SetupRandomWalksScene] No valid active scene - open a layout scene first.");
                return;
            }

            var existing = Object.FindObjectOfType<RecordRandomWalksHarness>();
            GameObject go = existing != null ? existing.gameObject : new GameObject("RecordRandomWalksHarness");
            RecordRandomWalksHarness harness = existing != null ? existing : go.AddComponent<RecordRandomWalksHarness>();
            if (harness.fpsControllerPrefab == null) harness.fpsControllerPrefab = SceneExporter.FindDefaultControllerPrefab();

            EditorUtility.SetDirty(harness);
            EditorSceneManager.MarkSceneDirty(scene);
            Selection.activeGameObject = go;

            Debug.Log($"[SetupRandomWalksScene] RecordRandomWalksHarness ready in '{scene.name}' (k={harness.pairCount}, m={harness.agentsPerPair}). " +
                "Save the scene, then press Play. Episodes are written to imitationagent_training/zarr_recordings.");
        }

        /// <summary>
        /// A harness already saved in a scene keeps its serialized Inspector values, so changed code defaults (e.g. 25x1
        /// pairs, the FPSController bot settings) never reach it. This copies a fresh component's
        /// defaults over it (undoable).
        /// </summary>
        [MenuItem("Tools/Random Walks/Reset RecordRandomWalksHarness To Code Defaults")]
        public static void ResetHarnessToDefaults()
        {
            var harness = Object.FindObjectOfType<RecordRandomWalksHarness>();
            if (harness == null)
            {
                Debug.LogError("[SetupRandomWalksScene] No RecordRandomWalksHarness in the open scene.");
                return;
            }

            var temp = new GameObject("RRW_DefaultsTemp") { hideFlags = HideFlags.HideAndDontSave };
            try
            {
                var defaults = temp.AddComponent<RecordRandomWalksHarness>();
                Undo.RecordObject(harness, "Reset RecordRandomWalksHarness");
                EditorUtility.CopySerialized(defaults, harness);
                if (harness.fpsControllerPrefab == null) harness.fpsControllerPrefab = SceneExporter.FindDefaultControllerPrefab();
            }
            finally
            {
                Object.DestroyImmediate(temp);
            }

            EditorUtility.SetDirty(harness);
            EditorSceneManager.MarkSceneDirty(harness.gameObject.scene);
            Debug.Log($"[SetupRandomWalksScene] RecordRandomWalksHarness in '{harness.gameObject.scene.name}' reset to code defaults " +
                $"(k={harness.pairCount}, m={harness.agentsPerPair}). Save the scene.");
        }

        [MenuItem("Tools/DAgger/Add DaggerHarness To Current Scene")]
        public static void AddDaggerHarness()
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            if (!scene.IsValid())
            {
                Debug.LogError("[SetupRandomWalksScene] No valid active scene - open a layout scene first.");
                return;
            }

            var existing = Object.FindObjectOfType<DaggerHarness>();
            GameObject go = existing != null ? existing.gameObject : new GameObject("DaggerHarness");
            DaggerHarness harness = existing != null ? existing : go.AddComponent<DaggerHarness>();

            EditorUtility.SetDirty(harness);
            EditorSceneManager.MarkSceneDirty(scene);
            Selection.activeGameObject = go;

            Debug.Log($"[SetupRandomWalksScene] DaggerHarness ready in '{scene.name}'. Assign 'modelAsset' (the policy to collect with), " +
                "disable EvalHarness/RecordRandomWalksHarness in this scene, then press Play. Episodes are written to imitationagent_training/zarr_recordings.");
        }
    }
}
