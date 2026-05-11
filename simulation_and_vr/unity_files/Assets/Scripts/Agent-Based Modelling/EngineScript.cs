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
using UnityEngine.AI;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using System.IO;
using System;
using System.Globalization;
using System.Linq;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class EngineScript : MonoBehaviour
{
    private TaskScript[] tasks;                     // The list of tasks that are active on this object.

    // State of the engine.
    private int activeTasks;                        // Number of active tasks.
    private bool allTasksCompleted;                 // Set if all tasks have been completed.
    private float startTime;                        // Time when simulation was started.

    // These fields maintain the recorded quantitities of each agent. They are arrays (fixed-size) of lists
    // (variable-size). The array entries corresponds to lists of agent-related quantities for each task. This design
    // decision was made since at runtime, there are always a fixed number of tasks, but a variable number of active
    // agents.
    private List<GameObject>[] agents;              // Tracks all the agents.
    private List<List<Vector3>>[] agentToPos;       // Records the positions.
    private List<TaskScript>[] agentToTask;         // Records the tasks.
    private List<int>[] agentToColl;                // Records collisions.
    private List<int[]>[] agentToAllColl;           // Records collisions, differentiated by task.
    private List<float>[] agentToTime;              // Records the duration of tasks.

    // POI-related fields.
    private List<GameObject> POIs;                  // List of all points of interest.
    private List<List<TaskScript>> taskPerPOI;      // Mapping between POI and tasks.

    // Visualisation-related fields.
    public bool visualizePOIs;                      // Set if you want to visualize POIs.
    public bool visualizeTrajectories;              // Set if you want to visualize trajectories.
    public bool visualizePaths;                     // Set if you want to visualize the paths that the agent will take.
    public bool saveHeatMap;                        // Toggle for automated heatmap export.
    public int fontSize;                            // Size of the font displayed above each POI.
    public float resize;                            // Factor used to resize the font above each POI.
    public float offset;                            // Factor to multiply offset with.
    public int traceLength;                         // Length of the trace in positions to be visualized.
    private Vector3 baseOffset;                     // Base offset at which the first text mesh gets displayed.

    // IO-related fields.
    public string dataFolder;                       // Directory where data file gets saved to.
    public string fileName;                         // Name of the file where data gets written to.
    public float sampleInterval;                    // Interval which needs to pass until new data is sampled.
    private float lastSample;                       // Last time a sample was taken.
    private StreamWriter file;                      // Stream that writes data to file.
    private string path;                            // Unique path where file gets saved to.
    private string testPath;
    private float latestTime;
    private Text simIdText;

    // Start is called before the first frame update
    void Start()
    {
        ShowSimInfo();
        if (ConfigManager.Instance != null)
        {
            // Initialize Engine Parameters
            Debug.Log("### Engine Script: Config file information found overwriting values");
            sampleInterval=ConfigManager.Instance.Config.EngineScript_sampleInterval;
            dataFolder=CreateDataPath("data_abm_batch",ConfigManager.Instance.Config.Purpose,ConfigManager.Instance.Config.Scene, ConfigManager.Instance.Config.Scenario);
            fileName=SanitizeName(ConfigManager.Instance.Config.Purpose+"_batch_simId_"+ConfigManager.Instance.Config.simId+"_sampleNum_"+ConfigManager.Instance.Config.sampleNum);

            Time.timeScale = 1.0f;
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1; // Unlimited frame rate
            // foreach (Camera cam in Camera.allCameras)
            //     cam.enabled = false;

            // Load external configuration if it exists
            TaskScript[] c_tasks = GetComponents<TaskScript>();

            // Remove each component
            foreach (TaskScript c_task in c_tasks)
            {
                DestroyImmediate(c_task);
            }

            foreach (TaskData t_task in ConfigManager.Instance.Config.tasks)
            {
                TaskScript mytask = gameObject.AddComponent<TaskScript>();
                Debug.Log("Task Name: " + t_task.taskName);
                mytask.numberOfAgents=t_task.numberOfAgents;
                mytask.spawnInterval=t_task.spawnInterval;
                mytask.taskName=t_task.taskName;
                mytask.agentType=t_task.agentType;
                mytask.numberOfNeeds=t_task.numberOfNeeds;
                mytask.agentSpeed=t_task.agentSpeed;
                mytask.agentSize=t_task.agentSize;
                mytask.agentRadius=t_task.agentRadius;
                mytask.privacyRadius=t_task.privacyRadius;
                mytask.poiTime=t_task.poiTime;
                mytask.revisit=t_task.revisit;
                mytask.chooseNonDeterministically=t_task.chooseNonDeterministically;
                PopulateObjectArray(t_task.start,out mytask.start);
                PopulateObjectArray(t_task.end,out mytask.end);
                PopulateObjectArray(t_task.pointsOfInterest,out mytask.pointsOfInterest);
                mytask.taskColor = ParseColor(t_task.taskColor);
            }
        }

        // Get all tasks on this object.
        tasks = GetComponents<TaskScript>();
        tasks = tasks.Where(task => task != null && task.gameObject != null).ToArray();

        foreach (TaskScript task in tasks)
        {
            if (task == null)
            {
                Debug.LogError("Error: Found a null TaskScript reference.");
                continue;
            }

            // Check if task properties are valid
            if (string.IsNullOrEmpty(task.taskName))
            {
                Debug.LogWarning($"Warning: Task {task} has no name assigned.");
                continue;
            }

            if (task.numberOfAgents <= 0)
            {
                Debug.LogWarning($"Warning: Task {task.taskName} has zero agents assigned.");
                continue;
            }
        }

        //

        // Collect all POIs.
        POIs = new List<GameObject>();
        taskPerPOI = new List<List<TaskScript>>();
        for (int i = 0; i < tasks.Length; i++) {
            checkAndInsert(tasks[i], tasks[i].start);
            checkAndInsert(tasks[i], tasks[i].end);
            checkAndInsert(tasks[i], tasks[i].pointsOfInterest);
        }

        // Initialize Visualisation-related fields.
        baseOffset = Vector3.up * offset;

        // Assign a POIMarkerScript-component to all the POIs and supply the relevant attributes.
        if (visualizePOIs) {
            for (int i = 0; i < POIs.Count; i++) {
                POIMarkerScript POIMarker = POIs[i].AddComponent<POIMarkerScript>();
                POIMarker.tasks = taskPerPOI[i];
                POIMarker.baseOffset = baseOffset;
                POIMarker.fontSize = fontSize;
                POIMarker.resize = resize;
            }
        }

        startTime = Time.realtimeSinceStartup;
        agents = new List<GameObject>[tasks.Length];
        for (int i = 0; i < agents.Length; i++) {
            agents[i] = new List<GameObject>();
        }
        path = makeFileNameUnique(dataFolder, fileName, "csv");

        agentToPos = new List<List<Vector3>>[tasks.Length];
        agentToColl = new List<int>[tasks.Length];
        agentToTask = new List<TaskScript>[tasks.Length];
        agentToTime = new List<float>[tasks.Length];
        agentToAllColl = new List<int[]>[tasks.Length];
        for (int i = 0; i < agentToPos.Length; i++) {
            agentToPos[i] = new List<List<Vector3>>();
            agentToColl[i] = new List<int>();
            agentToTask[i] = new List<TaskScript>();
            agentToTime[i] = new List<float>();
            agentToAllColl[i] = new List<int[]>();
        }
    }

    Color ParseColor(string colorString)
    {
        if (string.IsNullOrEmpty(colorString))
        {
            Debug.LogWarning("Color string is empty, defaulting to black.");
            return Color.black; // Default color if empty
        }

        // HEX format (e.g., "#FF5733")
        if (colorString.StartsWith("#"))
        {
            if (ColorUtility.TryParseHtmlString(colorString, out Color hexColor))
            {
                return hexColor;
            }
        }

        // RGB format (e.g., "255,128,0")
        else if (colorString.Contains(","))
        {
            string[] rgb = colorString.Split(',');
            if (rgb.Length == 3 &&
                int.TryParse(rgb[0], out int r) &&
                int.TryParse(rgb[1], out int g) &&
                int.TryParse(rgb[2], out int b))
            {
                return new Color(r / 255f, g / 255f, b / 255f);
            }
        }

        // Named color format (e.g., "red", "blue", "yellow")
        else
        {
            Dictionary<string, Color> namedColors = new Dictionary<string, Color>(StringComparer.OrdinalIgnoreCase)
            {
                { "red", Color.red },
                { "green", Color.green },
                { "blue", Color.blue },
                { "yellow", Color.yellow },
                { "cyan", Color.cyan },
                { "magenta", Color.magenta },
                { "black", Color.black },
                { "white", Color.white },
                { "gray", Color.gray },
                { "grey", Color.grey },
                { "orange", new Color(1f, 0.5f, 0f) },
                { "purple", new Color(0.5f, 0f, 0.5f) },
                { "pink", new Color(1f, 0.4f, 0.7f) },
                { "brown", new Color(0.6f, 0.3f, 0.1f) }
            };

            if (namedColors.TryGetValue(colorString.Trim(), out Color namedColor))
            {
                return namedColor;
            }
        }

        Debug.LogWarning($"Invalid color format: {colorString}, defaulting to white.");
        return Color.black; // Fallback color
    }

public void ShowSimInfo()
    {
        // Create a new Canvas GameObject.
        GameObject canvasObj = new GameObject("HUDCanvas");
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceCamera;

        // Add a CanvasScaler for responsive UI scaling.
        CanvasScaler scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(800, 600);

        // Ensure the Canvas fills the entire screen.
        RectTransform canvasRect = canvasObj.GetComponent<RectTransform>();
        canvasRect.anchorMin = Vector2.zero;
        canvasRect.anchorMax = Vector2.one;
        canvasRect.offsetMin = Vector2.zero;
        canvasRect.offsetMax = Vector2.zero;

        // Create a new GameObject for the Text element.
        GameObject textObj = new GameObject("SimIdText");
        textObj.transform.SetParent(canvasObj.transform);

        // Add the Text component and set its properties.
        Text simIdText = textObj.AddComponent<Text>();
        simIdText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        simIdText.fontSize = 24;
        simIdText.alignment = TextAnchor.UpperLeft;
        simIdText.color = Color.white;

        // Configure the RectTransform for proper positioning.
        RectTransform textRect = simIdText.GetComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0, 1);  // Top-left corner
        textRect.anchorMax = new Vector2(0, 1);
        textRect.pivot = new Vector2(0, 1);
        textRect.anchoredPosition = new Vector2(10, -10); // 10 pixels offset from top-left
        textRect.sizeDelta = new Vector2(400, 100);

        // Display the simId value.
        simIdText.text = "simId: " + ConfigManager.Instance.Config.simId;
    }



    public void PopulateObjectArray(string allObjectNames, out GameObject[] objArray)
    {
        List<GameObject> tempList = new List<GameObject>();

        string[] names = allObjectNames.Split(','); // Split string by comma

        foreach (string objectName in names) // loop over names
        {
            string trimmedName = objectName.Trim(); // Remove extra spaces
            GameObject targetObject = GameObject.Find(trimmedName);

            if (targetObject == null)
            {
                Debug.LogWarning("Object not found: " + trimmedName);
                continue; // Skip to the next name instead of returning immediately
            }

            if (targetObject.transform.childCount > 0)
            {
                foreach (Transform child in targetObject.transform)
                {
                    tempList.Add(child.gameObject);
                }
            }
            else
            {
                tempList.Add(targetObject); // No children, add only the object itself
            }
        }

        objArray = tempList.ToArray();

        Debug.Log("Populated array with " + objArray.Length + " elements.");
    }

    string CreateDataPath(params string[] pathSegments)
    {
        return Path.Combine(pathSegments.Select(SanitizeName).ToArray());
    }

    string SanitizeName(string input)
    {
        return input.ToLower().Replace(" ", "_");
    }

    // Update is called once per frame
    void Update()
    {
        /* Go through all agents per task and check,
         * if the agents want to be destroyed. In that case, save their data and destroy them.
         */
        bool anyAgentsLeft = false;

        Boolean record = lastSample + sampleInterval < Time.realtimeSinceStartup;
        if (record) {
            lastSample = Time.realtimeSinceStartup;
        }
        for (int i = 0; i < agents.Length; i++) {
            for (int j = 0; j < agents[i].Count; j++) {
                AgentScript currAgent = agents[i][j].GetComponent<AgentScript>();
                GameObject currAgentHolder = agents[i][j];
                int containerIdx = currAgent.startIndex;
                //Debug.Log("# i: "+i+" j:"+j+" - "+currAgentHolder.transform.position);
                if (record) {
                    agentToPos[i][containerIdx].Add(currAgentHolder.transform.position);
                    agentToColl[i][containerIdx] += collisions(currAgentHolder, i, j);
                    int[] cs = CheckCollisions(i, j);
                    for (int t = 0; t < cs.Length; t++)
                    {
                        agentToAllColl[i][j][t] += cs[t];
                    }
                }
                if (currAgent.destroyRequest) {
                    Destroy(agents[i][j]);
                    agents[i].RemoveAt(j);
                    j--;
                } else {
                    anyAgentsLeft = true; // At least one agent is still active
                }
            }
        }

        // Go through all tasks.
        for (int i = 0; i < tasks.Length; i++) {
            activeTasks = 0;
            // Check if we need to spawn a new agent for this task.
            if (tasks[i].agentsSpawned < tasks[i].numberOfAgents) {
                activeTasks++;
                if ((Time.realtimeSinceStartup - startTime)  > tasks[i].agentsSpawned * tasks[i].spawnInterval) {
                    GameObject newAgent = SpawnAgent(tasks[i], tasks[i].agentsSpawned);
                    agents[i].Add(newAgent);
                    agentToPos[i].Add(new List<Vector3>());
                    agentToColl[i].Add(0);
                    agentToTask[i].Add(tasks[i]);
                    agentToTime[i].Add(Time.realtimeSinceStartup);
                    agentToAllColl[i].Add(new int[tasks.Length]);
                    tasks[i].agentsSpawned++;
                    anyAgentsLeft = true; // A new agent has been added

                }
            }else {
                //TODO
                allTasksCompleted = true;
            }
        }


        // If there are no more active tasks, we can stop the simulation.
        //if (activeTasks == 0 && !allTasksCompleted) {
        //    allTasksCompleted = true;
        //}

        if(!anyAgentsLeft && allTasksCompleted){
        Debug.Log("# No Agents Left!");
        
        if (saveHeatMap)
        {
            // Use the same base filename as the CSV, but with .png extension
            string pngPath = path.Substring(0, path.LastIndexOf('.')) + ".png";
            AnalyticsManager.ExportHeatMap(agentToPos, POIs, pngPath);
        }

        #if UNITY_EDITOR
        EditorApplication.ExitPlaymode();  // Exit Play mode in Unity Editor
        #else
        Application.Quit();  // Quit application in standalone build
        #endif
        }
    }

    public string indexToTaskname(int index) {
        return tasks[index].name;
    }

    // Spawns an agent from a task.
    private GameObject SpawnAgent(TaskScript task, int startIdx) {

        if (task == null)
        {
            Debug.LogError("Error: Trying to spawn an agent without a task assigned.");
            return null;
        }

        // Create the GameObject that will be the agent.
        GameObject agent = GameObject.CreatePrimitive(PrimitiveType.Capsule);

        // Apply a material in the with the color of the specific task.
        //Shader shader = Shader.Find("Standard");
        Shader shader = Shader.Find("Diffuse");
        //Debug.Log(shader != null ? "Shader found!" : "Shader is null!");
        //Material material = new Material(Shader.Find("Diffuse"));
        Material material = new Material(shader);
        material.color = task.taskColor;
        agent.GetComponent<MeshRenderer>().material = material;

        // Set the position to somewhere on the NavMesh, else Unity complains about NavMeshAgent component being added
        // despite the agent being far away from the NavMesh.
        NavMeshHit hit;
        NavMesh.SamplePosition(agent.transform.position, out hit, 1000.0f, NavMesh.AllAreas);
        agent.transform.position = hit.position;

        // Add the necessary components to execute the task.
        agent.AddComponent<NavMeshAgent>();
        agent.AddComponent<AgentScript>();

        // Set the attributes of the agent.
        agent.GetComponent<AgentScript>().task = task;
        agent.GetComponent<AgentScript>().traceLength = traceLength;
        agent.GetComponent<AgentScript>().visualizePaths = visualizePaths;
        agent.GetComponent<AgentScript>().visualizeTrajectories = visualizeTrajectories;
        agent.GetComponent<AgentScript>().startIndex = startIdx;
        return agent;
    }

    // Checks for new POIs in list and adds if new.
    private void checkAndInsert(TaskScript taskScript, GameObject[] toBeChecked) {
       for (int i = 0; i < toBeChecked.Length; i++) {

            // If the POI is not in the list, we add it and remember which task it corresponds to.
            int index = POIs.IndexOf(toBeChecked[i]);
            if (index == -1) {
                POIs.Add(toBeChecked[i]);
                taskPerPOI.Add(new List<TaskScript>());
                taskPerPOI[taskPerPOI.Count - 1].Add(taskScript);
            }

            // Else it could still be that it gets referenced in a different task.
            else {
                if (!taskPerPOI[index].Contains(taskScript)) {
                    taskPerPOI[index].Add(taskScript);
                }
            }
        }
    }

    string makeFileNameUnique(string dirName, string fileName, string format)
    {
        // Create directory if does not exist.
        Directory.CreateDirectory(dirName);

        // This is the path the file will be written to.
        string path = dirName + Path.DirectorySeparatorChar + fileName + "." + format;

        // Check if specified file exists yet and if user wants to overwrite.
        if (File.Exists(path))
        {
            /* In this case we need to make the filename unique.
             * We will achieve that by:
             * foldername + sep + filename + . + format -> foldername + sep + filename + _x + . format
             * x will be increased in case of multiple overwrites.
             */

            // Check if there was a previous overwrite and get highest identifier.
            int id = 0;
            while (File.Exists(dirName + Path.DirectorySeparatorChar + fileName + "_" + id.ToString() + "." + format))
            {
                id++;
            }

            // Now we have found a unique identifier and create the new name.
            path = dirName + Path.DirectorySeparatorChar + fileName + "_" + id.ToString() + "." + format;
        }
        return path;
    }

    void OnDestroy() {

        // Find longest trajectory.
        int maxLen = 0;

        for (int i = 0; i < agentToPos.Length; i++) {
            for (int j = 0; j < agentToPos[i].Count; j++) {
                maxLen = agentToPos[i][j].Count > maxLen ? agentToPos[i][j].Count : maxLen;
            }
        }

        // For each agent, create a new line.
        List<string> lines = new List<string>();

        // Create header-line.
        string header = "";
        header += "AgentType;";
        header += "TaskName;";
        header += "Distance;";
        header += "Collisions;";

        // Add new columns for each combination of possible collisions. Each column corresponds to the other agent group
        // the current agent was colliding with.
        for (int t = 0; t < tasks.Length; t++)
        {
            header += "Collisions_" + tasks[t].taskName + ";";
        }

        header += "Duration;";
        for (int i = 0; i < maxLen; i++) {
            header += "pos" + i.ToString(CultureInfo.InvariantCulture) + "x;";
            header += "pos" + i.ToString(CultureInfo.InvariantCulture) + "y;";
            header += "pos" + i.ToString(CultureInfo.InvariantCulture) + "z;";
        }
        header = header.Remove(header.Length - 1);
        lines.Add(header);

        for (int i = 0; i < agentToPos.Length; i++) {
            for (int j = 0; j < agentToPos[i].Count; j++) {
                string line = "";
                line += agentToTask[i][j].agentType + ";";
                line += agentToTask[i][j].taskName + ";";
                line += distance(agentToPos[i][j]).ToString(CultureInfo.InvariantCulture) + ";";
                line += agentToColl[i][j].ToString(CultureInfo.InvariantCulture) + ";";
                for (int t = 0; t < tasks.Length; t++)
                {
                    line += agentToAllColl[i][j][t] + ";";
                }
                line += (lastSample - agentToTime[i][j]).ToString(CultureInfo.InvariantCulture) + ";";
                int k = 0;
                for (; k < agentToPos[i][j].Count; k++) {
                    line += agentToPos[i][j][k].x.ToString(CultureInfo.InvariantCulture) + ";" +
                            agentToPos[i][j][k].y.ToString(CultureInfo.InvariantCulture) + ";" +
                            agentToPos[i][j][k].z.ToString(CultureInfo.InvariantCulture) + ";";
                }
                for (; k < maxLen; k++) {
                    line += ";;;";
                }
                if (line.EndsWith(";")) {
                    line = line.Remove(line.Length-1);
                }
                lines.Add(line);
            }
        }

        File.WriteAllLines(path, lines);
        Debug.Log("# DONE!");
    }

    private int collisions(GameObject refAgent, int I, int J) {
        int collCount = 0;
        for (int i = 0; i < agents.Length; i++) {
            for (int j = 0; j < agents[i].Count; j++) {
                if (i == I && j == J) {
                    continue;
                }

                // Only count collisions if:
                // - the other agent is closer than privacy-radius.
                // - the tasks of the agents differ.
                if (Vector3.Distance(refAgent.transform.position, agents[i][j].transform.position) < refAgent.GetComponent<AgentScript>().task.privacyRadius
                                    && refAgent.GetComponent<AgentScript>().task.agentType != agents[i][j].GetComponent<AgentScript>().task.agentType) {
                    collCount++;
                }
            }
        }
        return collCount;
    }

    /**
     * @brief Returns an array of collision numbers for the agent identified by taskIdx and agentIdx.
     *
     * @param taskIdx: Index for the task this agent belongs to.
     * @param agentIdx: Index for the agent within that task.
     *
     * @returns An array where each entry corresponds to the number collisions at this moment of the simulation for each
     * task.
     */
    private int[] CheckCollisions(int taskIdx, int agentIdx)
    {
        GameObject currAgent = agents[taskIdx][agentIdx];
        int[] collisionsPerTask = new int[tasks.Length];
        for (int t = 0; t < tasks.Length; t++)
        {
            for (int j = 0; j < agents[t].Count; j++)
            {
                if (t == taskIdx && j == agentIdx)
                {
                    // Ignore collisions with itself.
                    continue;
                }

                // Distance to the other agent.
                float dist = Vector3.Distance(currAgent.transform.position, agents[t][j].transform.position);

                // Only count collisions if the agent is closer than the current agent's privacy radius.
                if (dist < tasks[taskIdx].privacyRadius)
                {
                    collisionsPerTask[t] += 1;
                }
            }
        }
        return collisionsPerTask;
    }

    private float distance(List<Vector3> trajectory) {
        float distance = 0.0f;
        for (int i = 1; i < trajectory.Count; i++) {
            distance += Vector3.Distance(trajectory[i-1], trajectory[i]);
        }
        return distance;
    }
}
