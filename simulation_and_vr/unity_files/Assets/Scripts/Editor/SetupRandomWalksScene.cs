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
        [MenuItem(ToolsMenu.SyntheticWalks + "Add Random Walks Harness", priority = ToolsMenu.SyntheticWalksPriority)]
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
        /// Same harness, but with the bot refitted to the thesis participants (RandomWalkBotSettings.RecalibratedV4) and
        /// recording as participant_id "synthetic_rrw_v4", so its episodes never mix with the v3 ones.
        /// </summary>
        [MenuItem(ToolsMenu.SyntheticWalks + "Record Random Walks v4", priority = ToolsMenu.SyntheticWalksPriority + 1)]
        public static void AddHarnessV4()
        {
            ConfigureHarness("v4", RandomWalkBotSettings.RecalibratedV4(), RandomWalkBotSettings.V4ParticipantId, 0f);
        }

        /// <summary>
        /// v4 plus pocket handling (RandomWalkBotSettings.RecalibratedV5) and 40% hidden-goal tasks, recording as
        /// participant_id "synthetic_rrw_v5".
        /// </summary>
        [MenuItem(ToolsMenu.SyntheticWalks + "Record Random Walks v5", priority = ToolsMenu.SyntheticWalksPriority + 2)]
        public static void AddHarnessV5()
        {
            ConfigureHarness("v5", RandomWalkBotSettings.RecalibratedV5(), RandomWalkBotSettings.V5ParticipantId, 0.4f);
        }

        /// <summary>
        /// Adds a RecordRandomWalksHarness (or takes the one in the scene) and overwrites its bot settings, participant id and
        /// hidden-goal share (undoable); the run configuration (pairs, seed, timeouts) is kept. "Reset Harness To Code
        /// Defaults" turns it back into the v3 bot.
        /// </summary>
        private static void ConfigureHarness(string version, RandomWalkBotSettings bot, string participantId, float hiddenGoalShare)
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            if (!scene.IsValid())
            {
                Debug.LogError("[SetupRandomWalksScene] No valid active scene - open a layout scene first.");
                return;
            }

            var harness = Object.FindObjectOfType<RecordRandomWalksHarness>();
            if (harness == null)
            {
                var go = new GameObject("RecordRandomWalksHarness");
                Undo.RegisterCreatedObjectUndo(go, $"Add RecordRandomWalksHarness {version}");
                harness = go.AddComponent<RecordRandomWalksHarness>();
            }
            else
            {
                Undo.RecordObject(harness, $"Configure RecordRandomWalksHarness {version}");
            }
            harness.bot = bot;
            harness.participantId = participantId;
            harness.hiddenGoalShare = hiddenGoalShare;
            if (harness.fpsControllerPrefab == null) harness.fpsControllerPrefab = SceneExporter.FindDefaultControllerPrefab();

            EditorUtility.SetDirty(harness);
            EditorSceneManager.MarkSceneDirty(scene);
            Selection.activeGameObject = harness.gameObject;

            Debug.Log($"[SetupRandomWalksScene] RecordRandomWalksHarness {version} ready in '{scene.name}' " +
                $"(participant_id '{harness.participantId}', hidden-goal share {harness.hiddenGoalShare:F2}, k={harness.pairCount}, m={harness.agentsPerPair}). " +
                "Save the scene, then press Play. Episodes are written to imitationagent_training/zarr_recordings.");
        }

        /// <summary>
        /// A harness already saved in a scene keeps its serialized Inspector values, so changed code defaults (e.g. 25x1
        /// pairs, the FPSController bot settings) never reach it. This copies a fresh component's
        /// defaults over it (undoable).
        /// </summary>
        [MenuItem(ToolsMenu.SyntheticWalks + "Reset Harness To Code Defaults", priority = ToolsMenu.SyntheticWalksPriority + 3)]
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

        [MenuItem(ToolsMenu.ImitationAdvanced + "Add DAgger Harness", priority = ToolsMenu.ImitationAdvancedPriority)]
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
