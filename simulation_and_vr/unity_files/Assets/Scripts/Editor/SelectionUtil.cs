using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Assets.Scripts.Editor
{
    public static class SelectionUtil
    {
        [NotNull]
        private static readonly List<GameObject> goCache = new List<GameObject>();

        [NotNull]
        private static readonly List<Component> compCache = new List<Component>();

        private static readonly HashSet<GameObject> goLookup = new HashSet<GameObject>();

        #region Selection

        [MenuItem(ToolsMenu.Select + "All", priority = ToolsMenu.SelectPriority + 50)]
        public static void SelectAll()
        {
            SelectByMatchingComponents(comps => true);
        }

        [MenuItem(ToolsMenu.Select + "Empty", priority = ToolsMenu.SelectPriority + 51)]
        public static void SelectEmpty()
        {
            SelectByMatchingComponents(comps => comps.Count == 0 || (comps.Count == 1 && comps[0] is Transform));
        }

        [MenuItem(ToolsMenu.Select + "Non Empty", priority = ToolsMenu.SelectPriority + 52)]
        public static void SelectNonEmpty()
        {
            SelectByMatchingComponents(comps => comps.Count > 1);
        }

        [MenuItem(ToolsMenu.Select + "Invert Selection", priority = ToolsMenu.SelectPriority + 53)]
        public static void InvertSelection()
        {
            goLookup.Clear();
            goCache.Clear();

            var selectedGos = Selection.gameObjects;
            var allGos = Object.FindObjectsOfType<GameObject>();

            foreach (var go in selectedGos)
            {
                goLookup.Add(go);
            }

            foreach (var go in allGos)
            {
                if (!goLookup.Contains(go))
                {
                    goCache.Add(go);
                    goLookup.Add(go);
                }
            }

            SetSelection(goCache);
            
            goLookup.Clear();
            goCache.Clear();
        } 

        #endregion

        #region Render selection

        [MenuItem(ToolsMenu.Select + "Renderer/Any", priority = ToolsMenu.SelectPriority + 100)]
        public static void SelectRenderer()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Renderer));
        }

        [MenuItem(ToolsMenu.Select + "Renderer/Mesh Renderer", priority = ToolsMenu.SelectPriority + 101)]
        public static void SelectMeshRenderer()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is MeshRenderer));
        }

        [MenuItem(ToolsMenu.Select + "Renderer/Skinned Mesh Renderer", priority = ToolsMenu.SelectPriority + 102)]
        public static void SelectSkinnedMeshRenderer()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is SkinnedMeshRenderer));
        }

        #endregion

        #region Collider selection

        [MenuItem(ToolsMenu.Select + "Collider/Any", priority = ToolsMenu.SelectPriority + 110)]
        public static void SelectCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Collider));
        }

        [MenuItem(ToolsMenu.Select + "Collider/Box", priority = ToolsMenu.SelectPriority + 111)]
        public static void SelectBoxCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is BoxCollider));
        }

        [MenuItem(ToolsMenu.Select + "Collider/Sphere", priority = ToolsMenu.SelectPriority + 112)]
        public static void SelectSphereCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is SphereCollider));
        }

        [MenuItem(ToolsMenu.Select + "Collider/Capsule", priority = ToolsMenu.SelectPriority + 113)]
        public static void SelectCapsuleCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is CapsuleCollider));
        }

        [MenuItem(ToolsMenu.Select + "Collider/Terrain", priority = ToolsMenu.SelectPriority + 114)]
        public static void SelectTerrainCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is TerrainCollider));
        }

        [MenuItem(ToolsMenu.Select + "Collider/Mesh", priority = ToolsMenu.SelectPriority + 115)]
        public static void SelectMeshCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is MeshCollider));
        }

        [MenuItem(ToolsMenu.Select + "Collider/Convex Mesh", priority = ToolsMenu.SelectPriority + 116)]
        public static void SelectConvexMeshCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is MeshCollider m && m.convex));
        } 

        [MenuItem(ToolsMenu.Select + "Collider/Concave Mesh", priority = ToolsMenu.SelectPriority + 117)]
        public static void SelectConcaveMeshCollider()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is MeshCollider m && m.convex));
        }

        #endregion

        #region Light selection

        [MenuItem(ToolsMenu.Select + "Lights/Any", priority = ToolsMenu.SelectPriority + 120)]
        public static void SelectLight()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Light));
        }

        [MenuItem(ToolsMenu.Select + "Lights/Point", priority = ToolsMenu.SelectPriority + 121)]
        public static void SelectPointLight()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Light l && l.type == LightType.Point));
        }

        [MenuItem(ToolsMenu.Select + "Lights/Spot", priority = ToolsMenu.SelectPriority + 122)]
        public static void SelectSpotLight()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Light l && l.type == LightType.Spot));
        }

        [MenuItem(ToolsMenu.Select + "Lights/Directional", priority = ToolsMenu.SelectPriority + 123)]
        public static void SelectDirectionalLight()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Light l && l.type == LightType.Directional));
        }

        [MenuItem(ToolsMenu.Select + "Lights/Area", priority = ToolsMenu.SelectPriority + 124)]
        public static void SelectAreaLight()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Light l && l.type == LightType.Area));
        }

        [MenuItem(ToolsMenu.Select + "Lights/Rectangle", priority = ToolsMenu.SelectPriority + 125)]
        public static void SelectRectangleLight()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Light l && l.type == LightType.Rectangle));
        }

        [MenuItem(ToolsMenu.Select + "Lights/Disc", priority = ToolsMenu.SelectPriority + 126)]
        public static void SelectDiscLight()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Light l && l.type == LightType.Disc));
        }

        #endregion

        #region Other Selection

        [MenuItem(ToolsMenu.Select + "Rigid body", priority = ToolsMenu.SelectPriority + 1000)]
        public static void SelectRigidBody()
        {
            SelectByMatchingComponents(comps => comps.Count > 1 && comps.Any(c => c is Rigidbody));
        } 

        #endregion

        private static void SelectByMatchingComponents(Predicate<IList<Component>> selectByComponents)
        {
            goCache.Clear();
            compCache.Clear();

            var gos = Object.FindObjectsOfType<GameObject>();

            foreach (var go in gos)
            {
                go.GetComponents(compCache);

                if (selectByComponents(compCache))
                {
                    goCache.Add(go);
                }

                compCache.Clear();
            }

            SetSelection(goCache);
            goCache.Clear();
        }

        private static void SetSelection(IList<GameObject> gos)
        {
            var selection = new Object[gos.Count];
            for (var i = 0; i < gos.Count; i++)
            {
                selection[i] = gos[i];
            }

            Selection.objects = selection;
        }
    }
}
