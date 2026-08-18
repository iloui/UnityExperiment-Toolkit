using System;
using Assets.Scripts.DataRecording;
using UnityEditor;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    public class LayoutGeneratorWindow : EditorWindow
    {
        private const string WindowTitle = "Generate Layout";

        private LayoutTopology topology = LayoutTopology.CentralPillar;
        private Vector2 dimensions = new Vector2(30f, 30f);
        private float wallHeight = 3.5f;
        private GameObject fpsControllerPrefab;

        [MenuItem("Tools/Generate Layout")]
        public static void ShowWindow()
        {
            var window = GetWindow<LayoutGeneratorWindow>();
            window.titleContent = new GUIContent(WindowTitle);
            window.minSize = new Vector2(360f, 220f);
            window.Show();
        }

        private void OnEnable()
        {
            titleContent = new GUIContent(WindowTitle);

            if (fpsControllerPrefab == null)
            {
                fpsControllerPrefab = SceneExporter.FindDefaultControllerPrefab();
            }
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("Layout Settings", EditorStyles.boldLabel);
            topology = (LayoutTopology)EditorGUILayout.EnumPopup(new GUIContent("Topology", "Select the topology to generate."), topology);
            dimensions = EditorGUILayout.Vector2Field(new GUIContent("Dimensions (m)", "Width X depth of the generated floor plan."), dimensions);
            wallHeight = EditorGUILayout.FloatField(new GUIContent("Wall Height (m)", "Height of the generated walls and ceiling."), wallHeight);
            fpsControllerPrefab = (GameObject)EditorGUILayout.ObjectField(new GUIContent("FPSController Prefab", "Prefab to instantiate as the participant agent."), fpsControllerPrefab, typeof(GameObject), false);

            EditorGUILayout.Space(8f);
            EditorGUILayout.HelpBox("Generates a fresh scene, bakes a NavMesh, wires ParticipantRecorder, and saves the result under Assets/Scenes/TestLayouts/.", MessageType.Info);

            using (new EditorGUI.DisabledScope(dimensions.x <= 0f || dimensions.y <= 0f || wallHeight <= 0f))
            {
                if (GUILayout.Button("Generate Layout", GUILayout.Height(28f)))
                {
                    GenerateLayout();
                }
            }
        }

        private void GenerateLayout()
        {
            try
            {
                var settings = new LayoutGenerationSettings
                {
                    Topology = topology,
                    Dimensions = dimensions,
                    WallHeight = wallHeight,
                    FpsControllerPrefab = fpsControllerPrefab
                };

                var scene = SceneExporter.CreateFreshScene();
                var root = SceneExporter.CreateEnvironmentRoot(scene);
                var layoutMaterial = SceneExporter.CreateSharedLayoutMaterial();

                var layoutData = LayoutBuilder.Build(root, settings, layoutMaterial);
                AgentLinker.SpawnAndWireParticipants(root.transform, settings, layoutData);
                SceneExporter.ConfigureAndBakeNavMesh(root);

                var savedPath = SceneExporter.SaveGeneratedScene(scene, topology);
                Selection.activeGameObject = root;
                EditorUtility.DisplayDialog(WindowTitle, $"Layout generated and saved to:\n{savedPath}", "OK");
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorUtility.DisplayDialog(WindowTitle, $"Generation failed:\n{ex.Message}", "OK");
            }
        }
    }
}
