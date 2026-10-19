using System;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Object = UnityEngine.Object;

public class ReplacerWindow : EditorWindow
{
    private Object replacerObject;

    private GameObject[] original;

    private GameObject[] current;

    [MenuItem(Assets.Scripts.Editor.ToolsMenu.Utilities + "Replacer", priority = Assets.Scripts.Editor.ToolsMenu.UtilitiesWindowPriority + 1)]
    static void Init()
    {
        // Get existing open window or if none, make a new one:
        var window = (ReplacerWindow)EditorWindow.GetWindow(typeof(ReplacerWindow));
        window.Show();
    }

    private void OnEnable()
    {
        this.original = this.current = Selection.gameObjects;
        Selection.selectionChanged += this.OnSelectionChanged;
    }

    private void OnDisable()
    {
        Selection.selectionChanged -= this.OnSelectionChanged;
        this.original = this.current = Array.Empty<GameObject>();
    }

    private void OnSelectionChanged()
    {
        Debug.Log("Selection changed, applying changes.");

        this.ApplyChanges();
        this.original = Selection.gameObjects;
        this.current = this.original;
        this.replacerObject = null;
    }

    void OnGUI()
    {
        var prev = this.replacerObject;

        this.replacerObject = EditorGUILayout.ObjectField(
            new GUIContent("Replace with", "Select an object to replace all selected instances with."),
            this.replacerObject, typeof(GameObject), false);

        if(original == null ||original.Length == 0)
        {
            this.replacerObject = null;
            return;
        }

        if (this.replacerObject != null)
        {
            var revert = GUILayout.Button(new GUIContent("Revert", "Reverts the changes made to the current selection."));
            if (revert)
            {
                this.replacerObject = null;
            }
        }

        if (prev == this.replacerObject)
        {
            return;
        }

        if (this.replacerObject == null)
        {
            if (this.current == this.original)
            {
                return;
            }

            foreach (var obj in this.original)
            {
                obj.SetActive(true);
            }

            for (var i = 0; i < this.current.Length; i++)
            {
                var go = this.current[i];
                for (var j = go.transform.childCount - 1; j >= 0; j--)
                {
                    go.transform.GetChild(j).parent = this.original[i].transform;
                }

                DestroySafe(go);
            }

            this.current = this.original;
        }
        else
        {
            if (this.current == this.original)
            {
                this.current = new GameObject[this.original.Length];

                for (var i = 0; i < this.original.Length; i++)
                {
                    var go = this.original[i];
                    go.SetActive(false);
                }
            }
            else
            {
                for (var i = 0; i < this.current.Length; i++)
                {
                    var go = this.current[i];
                    for (var j = go.transform.childCount - 1; j >= 0; j--)
                    {
                        go.transform.GetChild(j).parent = this.original[i].transform;
                    }

                    DestroySafe(go);
                }
            }

            for (var i = 0; i < this.original.Length; i++)
            {
                var go = this.original[i]; 

                this.current[i] = InstantiateWithLink((GameObject)this.replacerObject);
                this.current[i].transform.position = go.transform.position;
                this.current[i].transform.rotation = go.transform.rotation;
                this.current[i].transform.parent = go.transform.parent;
                this.current[i].transform.localScale = go.transform.localScale;

                for (var j = go.transform.childCount - 1; j >= 0; j--)
                {
                    go.transform.GetChild(j).parent = this.current[i].transform;
                }
            }
        }
    }

    bool HasSelectionChanged()
    {
        var s = Selection.gameObjects;
        if (s.Length != this.original.Length)
        {
            return true;
        }

        for (var i = 0; i < s.Length; i++)
        {
            if (s[i] != this.original[i])
            {
                return true;
            }
        }

        return false;
    }

    void OnDestroy()
    {
        if (this.current == this.original)
        {
            return;
        }

        this.ApplyChanges();

        Selection.objects = this.current.Select(c => (Object) c).ToArray();
        Selection.selectionChanged -= this.OnSelectionChanged;
    }

    private static void DestroySafe(GameObject go)
    {
        if (Application.isPlaying)
        {
            Destroy(go);
        }
        else
        {
            DestroyImmediate(go);
        }
    }

    private void ApplyChanges()
    {
        if (this.current == this.original)
        {
            return;
        }

        var startIndex = Undo.GetCurrentGroup();

        foreach (var go in this.original)
        {
            go.SetActive(true);
        }

        for (var i = 0; i < this.current.Length; i++)
        {
            var go = this.current[i];

            for (var j = go.transform.childCount - 1; j >= 0; j--)
            {
                go.transform.GetChild(j).parent = this.original[i].transform;
            }
        }

        foreach (var go in this.current)
        {
            Undo.RegisterCreatedObjectUndo(go, "Replace objects.");
        }

        for (var i = 0; i < this.current.Length; i++)
        {
            var go = this.current[i];

            for (var j = this.original[i].transform.childCount - 1; j >= 0; j--)
            {
                Undo.SetTransformParent(this.original[i].transform.GetChild(j), go.transform, "Replace objects.");
            }
        }

        foreach (var go in this.original)
        {
            if (go == null)
            {
                continue;
            }

            Undo.DestroyObjectImmediate(go);
        }

        Undo.CollapseUndoOperations(startIndex + 1);
    }

    private static GameObject InstantiateWithLink(GameObject go)
    {
        if (go == null)
        { 
            return null;
        }

        if (IsPrefab(go))
        {
            return PrefabUtility.InstantiatePrefab(go) as GameObject;
        }

        return Instantiate(go);
    }

    private static bool IsPrefab(GameObject go)
    {
        return PrefabUtility.IsPartOfAnyPrefab(go);
    }
}
