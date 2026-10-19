using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Assets.Scripts.Editor
{
    public class SetPositionWindow : EditorWindow
    {
        private const string TpXKey = "SetPositionWindow.targetPosition.x";
        private const string TpYKey = "SetPositionWindow.targetPosition.y";
        private const string TpZKey = "SetPositionWindow.targetPosition.z";
        private const string McKey = "SetPositionWindow.moveChildren";

        private readonly List<Object> objectCache = new List<Object>();

        private readonly List<GameObject> goCache = new List<GameObject>();

        private readonly List<Component> compCache = new List<Component>();

        private readonly HashSet<Transform> transformLookup = new HashSet<Transform>();
        
        private Vector3 targetPosition = Vector3.zero;

        private bool moveChildren = false;

        [MenuItem(ToolsMenu.Utilities + "Set Position", priority = ToolsMenu.UtilitiesWindowPriority)]
        static void Init()
        {
            // Get existing open window or if none, make a new one:
            var window = (SetPositionWindow)EditorWindow.GetWindow(typeof(SetPositionWindow));
            window.minSize = new Vector2(200, 146);
            window.Show();
        }

        private void OnEnable()
        {
            this.titleContent = new GUIContent("Set Position", "Utility window to change the position of certain game objects.");

            if (EditorPrefs.HasKey(TpXKey))
            {
                var x = EditorPrefs.GetFloat(TpXKey);
                var y = EditorPrefs.GetFloat(TpYKey);
                var z = EditorPrefs.GetFloat(TpZKey);

                targetPosition = new Vector3(x, y, z);
                moveChildren = EditorPrefs.GetBool(McKey);
            }
            else
            {
                targetPosition = Vector3.zero;
                moveChildren = false;

                var pos = this.position.position;
                this.position = new Rect(pos, new Vector2(350, 132));
            }

            Selection.selectionChanged += OnSelectionChanged;
        }

        private void OnDisable()
        {
            EditorPrefs.SetFloat(TpXKey, targetPosition.x);
            EditorPrefs.SetFloat(TpYKey, targetPosition.y);
            EditorPrefs.SetFloat(TpZKey, targetPosition.z);
            EditorPrefs.SetBool(McKey, moveChildren);

            Selection.selectionChanged -= OnSelectionChanged;
        }

        [ContextMenu("Reset")]
        private void Reset()
        {
            EditorPrefs.DeleteKey(TpXKey);
            EditorPrefs.DeleteKey(TpYKey);
            EditorPrefs.DeleteKey(TpZKey);
            EditorPrefs.DeleteKey(McKey);
        }

        private void OnGUI()
        {
            if (position.width > 350)
            {
                EditorGUIUtility.wideMode = true;
            }

            EditorGUILayout.LabelField("Settings", EditorStyles.boldLabel);

            targetPosition = EditorGUILayout.Vector3Field(
                new GUIContent("Target Position",
                    "The target position (in world coordinates) where the objects should be placed."), targetPosition);

            moveChildren = EditorGUILayout.Toggle(
                new GUIContent("Move Children",
                    "Should the position change propagate to the children?\nEnabled: The children get moved together with the parents.\nDisabled: Children stay where they are in world space."),
                moveChildren);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Set position for:", EditorStyles.boldLabel);

            if (GUILayout.Button(new GUIContent("Empty Game Objects",
                "Sets the position of all empty objects to the target position."),
                GUILayout.MinHeight(24)))
            {
                var gos = FindEmptyGos();
                MoveSelection(gos);
            }

            if (Selection.gameObjects?.Length < 1)
            {
                GUI.enabled = false;
            }

            if (GUILayout.Button(new GUIContent("Selection",
                "Set the position of all selected objects to the target position."),
                GUILayout.MinHeight(24)))
            {
                MoveSelection(Selection.gameObjects);
            }

            GUI.enabled = true;
        }

        private void MoveSelection(IList<GameObject> gos)
        {
            if (gos == null || gos.Count == 0)
            {
                return;
            }

            RegisterUndo(gos);

            foreach (var go in gos)
            {
                if (!moveChildren)
                {
                    var offset = go.transform.position - targetPosition;

                    for (var i = 0; i < go.transform.childCount; i++)
                    {
                        var childTransform = go.transform.GetChild(i);
                        childTransform.position += offset;
                    }
                }

                go.transform.position = targetPosition;
            }

            EditorUtility.SetDirty(gos[0]);
        }

        private void RegisterUndo(IList<GameObject> source)
        {
            objectCache.Clear();
            transformLookup.Clear();

            foreach (var go in source)
            {
                objectCache.Add(go.transform);
                transformLookup.Add(go.transform);
            }

            if (!moveChildren)
            {
                foreach (var go in source)
                {
                    for (var i = 0; i < go.transform.childCount; i++)
                    {
                        var childTransform = go.transform.GetChild(i);

                        if (!transformLookup.Contains(childTransform))
                        {
                            objectCache.Add(childTransform);
                            transformLookup.Add(childTransform);
                        }
                    }
                }
            }

            Undo.RecordObjects(objectCache.ToArray(), "Move to target position");

            objectCache.Clear();
            transformLookup.Clear();
        }

        private GameObject[] FindEmptyGos()
        {
            goCache.Clear();
            compCache.Clear();

            var gos = Object.FindObjectsOfType<GameObject>();

            foreach (var go in gos)
            {
                go.GetComponents(compCache);

                if (compCache.Count == 0 || compCache.Count == 1 && compCache[0] is Transform)
                {
                    goCache.Add(go);
                }

                compCache.Clear();
            }

            var result = goCache.ToArray();

            goCache.Clear();
            compCache.Clear();

            return result;
        }
        
        private void OnSelectionChanged()
        {
            this.Repaint();
        }
    }
}
