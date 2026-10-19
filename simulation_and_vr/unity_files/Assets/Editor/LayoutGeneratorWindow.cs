using System;
using System.Globalization;
using Assets.Scripts.DataRecording;
using UnityEditor;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    public class LayoutGeneratorWindow : EditorWindow
    {
        private const string WindowTitle = "Generate Layout";

        private bool useSeedMode = false;
        private string stringSeed = "";

        private LayoutTopology topology = LayoutTopology.Comb;
        private KammAlignment combAlignment = KammAlignment.Random;
        private float combAngle = 90f;
        private Vector2 dimensions = new Vector2(30f, 30f);
        private float wallHeight = 3.5f;
        private GameObject fpsControllerPrefab;

        [MenuItem(ToolsMenu.GenerativeDesign + "Generate Layout...", priority = ToolsMenu.GenerativeDesignPriority)]
        public static void ShowWindow()
        {
            GetWindow<LayoutGeneratorWindow>(WindowTitle);
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("Layout durch Seed", EditorStyles.boldLabel);
            
            EditorGUI.BeginChangeCheck();
            useSeedMode = EditorGUILayout.Toggle(new GUIContent("Seed-Modus aktivieren"), useSeedMode);

            if (useSeedMode)
            {
                EditorGUI.indentLevel++;
                stringSeed = EditorGUILayout.TextField(new GUIContent("Seed Code"), stringSeed);
                EditorGUI.indentLevel--;
            }

            if (EditorGUI.EndChangeCheck() && useSeedMode && !string.IsNullOrEmpty(stringSeed))
            {
                ParseSeedString(stringSeed);
            }

            EditorGUILayout.Space(10f);
            EditorGUILayout.LabelField("Layout Settings", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledScope(useSeedMode))
            {
                topology = (LayoutTopology)EditorGUILayout.EnumPopup("Topology", topology);

                if (topology == LayoutTopology.Comb)
                {
                    combAlignment = (KammAlignment)EditorGUILayout.EnumPopup("Comb Alignment", combAlignment);
                    combAngle = EditorGUILayout.Slider("Comb Angle (°)", combAngle, 75f, 105f);
                }

                dimensions = EditorGUILayout.Vector2Field("Dimensions (m)", dimensions);
                wallHeight = EditorGUILayout.FloatField("Wall Height (m)", wallHeight);
            }

            fpsControllerPrefab = (GameObject)EditorGUILayout.ObjectField(
                "FPSController Prefab", fpsControllerPrefab, typeof(GameObject), false);

            EditorGUILayout.Space(8f);

            if (GUILayout.Button("Generate Layout"))
            {
                GenerateLayout();
            }
        }

        private string BuildSeedString(int internalRngSeed)
        {
            // Format: TOPOLOGY_WIDTHxDEPTH_ANGLE_ALIGNMENT_INTERNALSEED
            return $"{topology}_{dimensions.x}x{dimensions.y}_{combAngle.ToString("F1", CultureInfo.InvariantCulture)}_{combAlignment}_{internalRngSeed}";
        }

        private void ParseSeedString(string seed)
        {
            try
            {
                string[] parts = seed.Split('_');
                if (parts.Length >= 5)
                {
                    topology = (LayoutTopology)Enum.Parse(typeof(LayoutTopology), parts[0]);
                    
                    string[] dim = parts[1].Split('x');
                    dimensions = new Vector2(float.Parse(dim[0], CultureInfo.InvariantCulture), float.Parse(dim[1], CultureInfo.InvariantCulture));
                    
                    combAngle = float.Parse(parts[2], CultureInfo.InvariantCulture);
                    combAlignment = (KammAlignment)Enum.Parse(typeof(KammAlignment), parts[3]);
                }
            }
            catch
            {
                // Ungültiges Format beibehalten/ignorieren
            }
        }

        private int ExtractInternalSeed(string seed)
        {
            string[] parts = seed.Split('_');
            if (parts.Length >= 5 && int.TryParse(parts[4], out int result))
            {
                return result;
            }
            return UnityEngine.Random.Range(1, 1000000);
        }

        private void GenerateLayout()
        {
            try
            {
                int internalSeed;
                string finalSeedCode;

                if (useSeedMode && !string.IsNullOrEmpty(stringSeed))
                {
                    ParseSeedString(stringSeed);
                    internalSeed = ExtractInternalSeed(stringSeed);
                    finalSeedCode = stringSeed;
                }
                else
                {
                    internalSeed = UnityEngine.Random.Range(1, 1000000);
                    finalSeedCode = BuildSeedString(internalSeed);
                    stringSeed = finalSeedCode;
                }

                var settings = new LayoutGenerationSettings
                {
                    Topology = topology,
                    CombAlignment = combAlignment,
                    CombAngle = combAngle,
                    Dimensions = dimensions,
                    WallHeight = wallHeight,
                    FpsControllerPrefab = fpsControllerPrefab,
                    RandomSeed = internalSeed
                };

                var scene = SceneExporter.CreateFreshScene();
                var root = SceneExporter.CreateEnvironmentRoot(scene);
                var layoutMaterial = SceneExporter.CreateSharedLayoutMaterial();

                var layoutData = LayoutBuilder.Build(root, settings, layoutMaterial);
                // Copied into every episode manifest by ParticipantRecorder (layout_typology, layout_seed).
                var layoutInfo = root.AddComponent<LayoutInfo>();
                layoutInfo.typology = topology.ToString();
                layoutInfo.seed = internalSeed;
                layoutInfo.seedCode = finalSeedCode;
                AgentLinker.SpawnAndWireParticipants(root.transform, settings, layoutData);
                // Bakes the NavMesh, places Start/POI_0/End + the TaskScript and puts the FPSController on Start at the
                // contract eye height (the same steps as Tools > Scene Preparation > Prepare Scene).
                ScenePreparation.Prepare(scene);

                string sceneFileName = $"{topology}_Code_{internalSeed}";
                var savedPath = SceneExporter.SaveGeneratedScene(scene, sceneFileName);

                Selection.activeGameObject = root;
                EditorUtility.DisplayDialog(WindowTitle, $"Layout generated!\n\nSeed Code zum Wiederholen:\n{finalSeedCode}", "OK");
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorUtility.DisplayDialog(WindowTitle, $"Generation failed:\n{ex.Message}", "OK");
            }
        }
    }
}