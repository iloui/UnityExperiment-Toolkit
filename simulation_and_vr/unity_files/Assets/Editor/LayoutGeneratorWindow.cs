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
        private KammAlignment combAlignment = KammAlignment.Random;
        private float combAngle = 90f; // Default 90 degrees (Orthogonal)
        private Vector2 dimensions = new Vector2(30f, 30f);
        private float wallHeight = 3.5f;
        private GameObject fpsControllerPrefab;
        private int randomSeed = 0;

        [MenuItem("Tools/Generate Layout")]
        public static void ShowWindow()
        {
            GetWindow<LayoutGeneratorWindow>("Generate Layout");
        }
        
        private void OnGUI()
        {
            EditorGUILayout.LabelField("Layout Settings", EditorStyles.boldLabel);
            topology = (LayoutTopology)EditorGUILayout.EnumPopup(
                new GUIContent("Topology", "Select the topology to generate."), topology);

            if (topology == LayoutTopology.Comb)
            {
                combAlignment = (KammAlignment)EditorGUILayout.EnumPopup(
                    new GUIContent("Comb Alignment", "Select single, double, or random alignment."),
                    combAlignment
                );

                combAngle = EditorGUILayout.Slider(
                    new GUIContent("Comb Angle (°)", "Anstellwinkel of fingers relative to magistrale (75° - 105°)."),
                    combAngle, 75f, 105f
                );
            }

            dimensions =
                EditorGUILayout.Vector2Field(
                    new GUIContent("Dimensions (m)", "Width X depth of the generated floor plan."), dimensions);
            wallHeight =
                EditorGUILayout.FloatField(
                    new GUIContent("Wall Height (m)", "Height of the generated walls and ceiling."), wallHeight);
            fpsControllerPrefab = (GameObject)EditorGUILayout.ObjectField(
                new GUIContent("FPSController Prefab", "Prefab to instantiate as the participant agent."),
                fpsControllerPrefab, typeof(GameObject), false);

            EditorGUILayout.Space(8f);
            EditorGUILayout.HelpBox(
                "Generates a fresh scene, bakes a NavMesh, wires ParticipantRecorder, and saves the result under Assets/Scenes/TestLayouts/.",
                MessageType.Info);

            using (new EditorGUI.DisabledScope(dimensions.x <= 0f || dimensions.y <= 0f || wallHeight <= 0f))
            {
                if (GUILayout.Button("Generate Layout"))
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
                    CombAlignment = combAlignment,
                    CombAngle = combAngle, // Pass specified angle
                    Dimensions = dimensions,
                    WallHeight = wallHeight,
                    FpsControllerPrefab = fpsControllerPrefab,
                    RandomSeed = randomSeed
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