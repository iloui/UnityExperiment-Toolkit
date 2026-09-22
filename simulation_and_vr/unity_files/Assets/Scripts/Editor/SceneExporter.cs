using System;
using System.IO;
using System.Linq;
using Unity.AI.Navigation;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    internal static class SceneExporter
    {
        private const string EnvironmentRootName = "Environment_Root";
        private const string SaveFolder = "Assets/Scenes/TestLayouts";

        public static Scene CreateFreshScene()
        {
            return EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        }

        public static GameObject CreateEnvironmentRoot(Scene scene)
        {
            var existing = GameObject.Find(EnvironmentRootName);
            if (existing != null)
            {
                UnityEngine.Object.DestroyImmediate(existing);
            }

            var root = new GameObject(EnvironmentRootName);
            SceneManager.MoveGameObjectToScene(root, scene);
            root.AddComponent<NavMeshSurface>();
            return root;
        }

        public static void ConfigureAndBakeNavMesh(GameObject root)
        {
            if (root == null) throw new ArgumentNullException(nameof(root));

            var surface = root.GetComponent<NavMeshSurface>();
            if (surface == null)
            {
                surface = root.AddComponent<NavMeshSurface>();
            }

            var walkableLayer = EnsureLayer("Walkable_Floor");
            // Architecture_Obstacle (walls/pillars/doorway partitions) must also be included, not just the floor -
            // the bake needs their geometry to actually carve their footprint out of the walkable surface.
            // Walkable_Floor-only was the original (buggy) setting: it produced a NavMesh that ignored walls
            // entirely and let agents path straight through them, since the floor mesh pieces on their own don't
            // reliably match the current wall layout.
            var obstacleLayer = EnsureLayer("Architecture_Obstacle");
            surface.collectObjects = CollectObjects.Children;
            surface.layerMask = (1 << walkableLayer) | (1 << obstacleLayer);
            surface.overrideTileSize = false;
            surface.overrideVoxelSize = false;
            surface.Bake();
        }

        public static string SaveGeneratedScene(Scene scene, string fileName)
        {
            string directory = "Assets/Scenes/TestLayouts";
            if (!System.IO.Directory.Exists(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }

            string fullPath = $"{directory}/{fileName}.unity";
            UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene, fullPath);
            return fullPath;
        }

        public static Material CreateSharedLayoutMaterial()
        {
            var material = new Material(Shader.Find("Standard"));
            material.color = new Color(0.55f, 0.55f, 0.55f, 1f);
            if (material.HasProperty("_Glossiness"))
            {
                material.SetFloat("_Glossiness", 0f);
            }

            if (material.HasProperty("_Metallic"))
            {
                material.SetFloat("_Metallic", 0f);
            }

            return material;
        }

        public static GameObject FindDefaultControllerPrefab()
        {
            var candidatePaths = new[]
            {
                "Assets/Scenes/FPSController.prefab",
                "Assets/Standard Assets/Characters/FirstPersonCharacter/Prefabs/FPSController.prefab"
            };

            foreach (var path in candidatePaths)
            {
                var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
                if (prefab != null)
                {
                    return prefab;
                }
            }

            return null;
        }

        public static int EnsureLayer(string layerName)
        {
            var current = LayerMask.NameToLayer(layerName);
            if (current >= 0)
            {
                return current;
            }

            var tagManagerAsset = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset").FirstOrDefault();
            if (tagManagerAsset == null)
            {
                throw new InvalidOperationException("Could not access ProjectSettings/TagManager.asset");
            }

            var tagManager = new SerializedObject(tagManagerAsset);
            var layersProp = tagManager.FindProperty("layers");
            if (layersProp == null || !layersProp.isArray)
            {
                throw new InvalidOperationException("Could not access the layers array in TagManager.asset");
            }

            var index = FindEmptyLayerSlot(layersProp, layerName);
            if (index < 0)
            {
                throw new InvalidOperationException($"No empty layer slots available for {layerName}");
            }

            layersProp.GetArrayElementAtIndex(index).stringValue = layerName;
            tagManager.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(tagManagerAsset);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            return index;
        }

        private static int FindEmptyLayerSlot(SerializedProperty layersProp, string layerName)
        {
            for (var i = 8; i < layersProp.arraySize; i++)
            {
                var element = layersProp.GetArrayElementAtIndex(i);
                if (string.IsNullOrEmpty(element.stringValue))
                {
                    return i;
                }

                if (string.Equals(element.stringValue, layerName, StringComparison.Ordinal))
                {
                    return i;
                }
            }

            return -1;
        }

        private static void EnsureFolderExists()
        {
            if (AssetDatabase.IsValidFolder(SaveFolder))
            {
                return;
            }

            Directory.CreateDirectory(SaveFolder);
            AssetDatabase.Refresh();
        }
    }

    internal static class NavMeshSurfaceExtensions
    {
        public static void Bake(this NavMeshSurface surface)
        {
            if (surface == null)
            {
                throw new ArgumentNullException(nameof(surface));
            }

            surface.BuildNavMesh();
        }
    }
}
