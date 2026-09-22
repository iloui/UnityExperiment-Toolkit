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

            EditorUtility.SetDirty(harness);
            EditorSceneManager.MarkSceneDirty(scene);
            Selection.activeGameObject = go;

            Debug.Log($"[SetupRandomWalksScene] RecordRandomWalksHarness ready in '{scene.name}' (k={harness.pairCount}, m={harness.agentsPerPair}). " +
                "Save the scene, then press Play. Episodes are written to imitationagent_training/zarr_recordings.");
        }
    }
}
