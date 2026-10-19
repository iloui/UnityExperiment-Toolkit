/*
DesignMind2: A Toolkit for Evidence-Based, Cognitively- Informed and Human-Centered Architectural Design
Copyright (C) 2023-2026  michal Gath-Morad, Christoph Hölscher, Raphaël Baur, Leonel Aguilar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class POIMarkerScript : MonoBehaviour
{
    // Properties of this POI.
    public List<TaskScript> tasks;              // List of tasks associated with this location.
    private List<List<string>> taskRoles;       // List of roles that this POI takes in the task.
    private List<string> taskNames;             // List of strings describing the exact role of this POI.
    private List<Color> taskColors;             // List of colors associated with this task.

    // Fields concerned with visualization of this POI.
    private GameObject[] cylinders;             // Cylinders that will be stacked on top of each other to visualize POI.

    // Fields concerned with diplay of text.
    public int fontSize;                        // Size of the text that should be displayed.
    public Vector3 baseOffset;                  // Base offset used to place text above the POI.
    private Vector3 incrementalOffset;          // Offset added for each text corresponding to a role.
    public float resize;                        // Factor used to resize the text.
    private List<GameObject> textMeshParents;   // The parent object which holds the text mesh.
    private List<TextMesh> textMeshes;          // The text mesh used to display the text.
    private List<string> displayTexts;          // All descriptors. Each entry corresponds to a task.

    // Start is called before the first frame update
    void Start()
    {
        taskRoles = new List<List<string>>();
        taskColors = new List<Color>();
        taskNames = new List<string>();

        getRoles();                             // Get all roles of this POI.
        getTaskColors();                        // Get all colors of this POI.
        getTaskNames();                         // Get all names of this task.

        // Generate the texts to be displayed above this POI.
        displayTexts = makeDisplayText(taskRoles, taskNames);

        // Generate as many TextMeshes as needed to display all texts.
        textMeshParents = new List<GameObject>();
        textMeshes = new List<TextMesh>();
        for (int i = 0; i < displayTexts.Count; i++) {
            textMeshParents.Add(new GameObject());
            TextMesh textMesh = textMeshParents[i].AddComponent<TextMesh>();
            textMesh.anchor = TextAnchor.MiddleCenter;
            textMesh.text = displayTexts[i];
            textMesh.fontSize = fontSize;
            textMesh.transform.position = transform.position + 2 * baseOffset;
            textMesh.transform.localScale = (1.0f / resize) * Vector3.one;
            textMesh.color = taskColors[i];
            textMeshes.Add(textMesh);
        }
        
        // Determine the incrementalOffset based on the height of the text meshes.
        float textHeight = textMeshes[0].GetComponent<Renderer>().bounds.size.y;
        incrementalOffset = Vector3.up * textHeight * 1.2f;

        for (int i = 0; i < textMeshes.Count; i++) {
            textMeshes[i].transform.position += incrementalOffset * i;
        }

        // Draw cylinder and Destroy underlying POI.
        cylinders = new GameObject[displayTexts.Count];
        for (int i = 0; i < cylinders.Length; i++) {
            GameObject cylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            cylinder.transform.localScale = new Vector3(3.0f, baseOffset.y / cylinders.Length, 3.0f);
            cylinder.transform.position = transform.position + baseOffset * ((1.0f / (cylinders.Length)));
            cylinder.transform.position += 2 * baseOffset * (1.0f / cylinders.Length) * i;
            Color color = taskColors[i];
            color.a = 0.5f;
            Material material = new Material(Shader.Find("Transparent/Specular"));
            material.color = color;
            cylinder.GetComponent<MeshRenderer>().material = material;
            cylinders[i] = cylinder;
        }

        // Markers placed by Prepare Scene can be invisible (no MeshRenderer).
        MeshRenderer markerRenderer = GetComponent<MeshRenderer>();
        if (markerRenderer != null) markerRenderer.enabled = false;
    }

    // Update is called once per frame
    void Update()
    {
    #if UNITY_EDITOR
        for (int i = 0; i < textMeshParents.Count; i++) {
            textMeshes[i].transform.rotation = Quaternion.LookRotation(textMeshes[i].transform.position - SceneView.lastActiveSceneView.camera.transform.position);
        }
    #endif
    }

    // Identifies the roles this POI takes 
    private void getRoles() {
        for (int i = 0; i < tasks.Count; i++) {
            
            List<string> roles = new List<string>();

            // Check if it is a starting location for this task.
            for (int j = 0; j < tasks[i].start.Length; j++) {
                if (gameObject == tasks[i].start[j]) {
                    roles.Add("Start");
                    break;
                }
            }

            // Check if it is a POI for this task.
            for (int j = 0; j < tasks[i].pointsOfInterest.Length; j++) {
                if (gameObject == tasks[i].pointsOfInterest[j]) {
                    roles.Add("POI");
                    break;
                }
            }

            // Check if it is a ending location for this task.
            for (int j = 0; j < tasks[i].end.Length; j++) {
                if (gameObject == tasks[i].end[j]) {
                    roles.Add("End");
                    break;
                }
            }
            taskRoles.Add(roles);
        }
    }

    private void getTaskColors() {
        for (int i = 0; i < tasks.Count; i++) {
            Color color = tasks[i].taskColor;
            color.a = 1.0f;
            taskColors.Add(color);
        }
    }

    private void getTaskNames() {
        for (int i = 0; i < tasks.Count; i++) {
            taskNames.Add(tasks[i].taskName);
        }
    }

    private List<string> makeDisplayText(List<List<string>> roles, List<string> taskNames) {
        List<string> displayTexts = new List<string>();
        for (int i = 0; i < roles.Count; i++) {
            string text = "";
            for (int j = 0; j < roles[i].Count - 1; j++) {
                text += roles[i][j] + ", ";
            }
            text += roles[i][roles[i].Count - 1] + ": ";
            text += taskNames[i];
            displayTexts.Add(text);
        }
        return displayTexts;
    }
}


