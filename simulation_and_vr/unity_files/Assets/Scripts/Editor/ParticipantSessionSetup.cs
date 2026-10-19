using Assets.Scripts.ExperimentFlow;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Creates the bootstrap scene for human recording sessions (a camera plus a ParticipantSession) and its
    /// ParticipantSessionConfig asset. Afterwards, assign the practice scene and the 3 layout scenes in the config once;
    /// every participant is then run by opening this scene and pressing Play.
    /// </summary>
    internal static class ParticipantSessionSetup
    {
        private const string ScenePath = "Assets/Scenes/ParticipantSession.unity";
        private const string ConfigPath = "Assets/Scenes/ParticipantSessionConfig.asset";

        [MenuItem(ToolsMenu.HumanWalks + "Create Participant Session Scene", priority = ToolsMenu.HumanWalksPriority + 3)]
        public static void CreateSessionScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;

            var config = AssetDatabase.LoadAssetAtPath<ParticipantSessionConfig>(ConfigPath);
            if (config == null)
            {
                config = ScriptableObject.CreateInstance<ParticipantSessionConfig>();
                AssetDatabase.CreateAsset(config, ConfigPath);
                AssetDatabase.SaveAssets();
            }

            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath) != null &&
                !EditorUtility.DisplayDialog("Participant Session", $"{ScenePath} already exists. Replace it?", "Replace", "Cancel"))
            {
                return;
            }

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // Only for the ID/instruction screens; replaced when the first layout scene loads (Single mode).
            var cameraGo = new GameObject("SessionCamera");
            var camera = cameraGo.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.12f, 0.12f, 0.12f);

            var sessionGo = new GameObject("ParticipantSession");
            sessionGo.AddComponent<ParticipantSession>().config = config;

            EditorSceneManager.SaveScene(scene, ScenePath);
            Selection.activeObject = config;
            EditorGUIUtility.PingObject(config);
            EditorUtility.DisplayDialog("Participant Session",
                $"Created {ScenePath}.\n\nAssign the practice scene and the layout scenes in {ConfigPath} (selected), " +
                "then press Play in this scene to run a participant.", "OK");
        }
    }
}
