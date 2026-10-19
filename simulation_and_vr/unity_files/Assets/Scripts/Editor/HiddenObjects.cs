using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    public static class HiddenObjects
    {
        [MenuItem(ToolsMenu.Objects + "Make hidden visible", priority = ToolsMenu.ObjectsPriority + 0)]
        public static void MakeHiddenObjectsVisible()
        {
            var gos = GameObject.FindObjectsOfType<GameObject>();
            foreach (var go in gos)
            {
                if (go.hideFlags.HasFlag(HideFlags.HideInHierarchy) || go.hideFlags.HasFlag(HideFlags.HideInInspector))
                {
                    go.hideFlags &= ~(HideFlags.HideInHierarchy | HideFlags.HideInInspector);
                }
            }
        }

        [MenuItem(ToolsMenu.Objects + "Make selection invisible", priority = ToolsMenu.ObjectsPriority + 1)]
        public static void MakeSelectionInvisible()
        {
            var gos = Selection.gameObjects;
            foreach (var go in gos)
            {
                go.hideFlags |= HideFlags.HideInHierarchy | HideFlags.HideInInspector;
            }
        }

        [MenuItem(ToolsMenu.Objects + "Make selection editable", priority = ToolsMenu.ObjectsPriority + 2)]
        public static void MakeSelectionEditable()
        {
            var gos = Selection.gameObjects;
            foreach (var go in gos)
            {
                go.hideFlags &= ~(HideFlags.NotEditable);
            }
        }

        [MenuItem(ToolsMenu.Objects + "Make selection non editable", priority = ToolsMenu.ObjectsPriority + 3)]
        public static void MakeSelectionNonEditable()
        {
            var gos = Selection.gameObjects;
            foreach (var go in gos)
            {
                go.hideFlags |= HideFlags.NotEditable;
            }
        }
    }
}
