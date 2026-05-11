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

public class AgentScript : MonoBehaviour
{
    public TaskScript task;                         // The task this agent is pursuing.
    private NavMeshAgent navMeshAgent;              // The NavMeshAgent component of this agent.
    private MeshRenderer meshRenderer;              // The MeshRenderer component of this agent.

    // Task-locations.
    private GameObject[] start;                     // The start of this agent.
    private GameObject[] end;                       // The end of this agent.
    private GameObject[] POIs;                      // Points of interest of the agent.
    
    // Attributes defining the interest of the agent.
    private float poiTime;                          // How long the agent is at each point of interest.
    private bool revisit;                           // Can the agent revisit targets?
    private bool cnd;                               // Does the agent choose its targets non-deterministically?
    private int numberOfNeeds;                      // Number of needs that this agent needs to fulfill.

    // Attributes defining shape and locomotion of agent.
    private float agentSize;                        // Size of the agent.
    private float agentRadius;                      // Radius from center in which no other agent can intrude.
    private float agentSpeed;                       // Speed of this agent.

    // Attributes for visualization.
    public bool visualizeTrajectories;              // Set if you want to visualize trajectories.
    public bool visualizePaths;                     // Set if you want to visualize paths.
    public int traceLength;                         // How many past positions should be considered.
    private LineRenderer lineRenderer;              // Renderer used to visualize trajectory.
    private Gradient gradient;                      // Gradient used to color trace.
    private Color agentColor;                       // Color of this specific agent.

    // Technical stuff.
    //private int failSave = 100;                     // Upper bound on the retries in a while loop to avoid infinity-loop.
    private float displacementInterval = 10.0f;      // After how many seconds should we change the agents position a bit to avoid deadlock.
    private float displacement = 0.1f;              // In this range the x and z component of the displacement-vector will be chosen.
    public float lastDisplacementTime;                 // Last time the agent was displaced.
    private float displacementDelta = 0.1f;         // Length of the vector that the agents needs to have traveled to not be displaced.
    public List<Vector3> trajectory;                // A list of past positions of the agent, constituting the trajectories.
    private float sampleInterval;                   // Interval that needs to pass until new location gets sampled.
    private float lastSample;                       // Last time a sample was taken.
    private Vector3 firstPos;                       // First position in simulation.

    // State of the agent.
    private bool choosingPOI = false;               // True: Needs to choose new POI.
    private bool findingPOI = false;                // Is currently walking towards the POI.
    private bool fulfillingNeed = false;            // Is currently fulfilling its need.
    private bool taskCompleted = false;             // Has fulfilled all needs.
    public bool destroyRequest = false;             // Indicates to the engine if this agent wants to be destroyed.
    private bool[] poiMask;                         // Masks the POIs that are invalid.
    private int currPOI;                            // The index of the current point of interest.
    private int needsFulfilled;                     // Number needs that this agent has already fulfilled.
    private float arrivalTime;                      // The last time the agent arrived at a POI.
    public int startIndex;
    private NavMeshPath lastDrawnPath;              // Store the last path for drawing in pause mode

    // Start is called before the first frame update
    void Start()
    {
        // Initializing the task locations.
        start = task.start;
        end = task.end;
        POIs = task.pointsOfInterest;

        // Initializing the fields defining the interest of the agent.
        poiTime = task.poiTime;
        revisit = task.revisit;
        cnd = task.chooseNonDeterministically;
        numberOfNeeds = task.numberOfNeeds;

        // Initializing the fields defining shape and locomotion of the agent.
        agentSize = task.agentSize;
        agentRadius = task.agentRadius;
        agentSpeed = task.agentSpeed;

        // Initializing technical stuff.
        lastDisplacementTime=Time.realtimeSinceStartup;
        trajectory = new List<Vector3>();
        lastDrawnPath = new NavMeshPath();

        // Initializing the state of the agent.
        poiMask = new bool[POIs.Length];
        for (int i = 0; i < poiMask.Length; i++) {
            poiMask[i] = true;
        }
        currPOI = 0;
        needsFulfilled = 0;
        choosingPOI = true;

        // Choosing the starting location of the agent.
        GameObject chosenStart = start[Random.Range(0, start.Length)];
        
        // Transferring the agent to the starting-location.
        NavMeshHit hit;
        NavMesh.SamplePosition(chosenStart.transform.position, out hit, 100.0f, NavMesh.AllAreas);
        navMeshAgent = GetComponent<NavMeshAgent>();
        navMeshAgent.Warp(hit.position);

        // Initializing the visualization-related field.
        initializeAgentColor();
        meshRenderer = GetComponent<MeshRenderer>();
        meshRenderer.material.color = agentColor;
        firstPos = hit.position;
        gradient = new Gradient();
        GradientColorKey[] colorKey = new GradientColorKey[2];
        colorKey[0].color = agentColor;
        colorKey[0].time = 0.0f;
        colorKey[1].color = agentColor;
        colorKey[1].time = 1.0f;
        GradientAlphaKey[] alphaKey = new GradientAlphaKey[2];
        alphaKey[0].alpha = 0.0f;
        alphaKey[0].time = 0.0f;
        alphaKey[1].alpha = 1.0f;
        alphaKey[1].time = 1.0f;
        gradient.colorKeys = colorKey;
        gradient.alphaKeys = alphaKey;
        lineRenderer = gameObject.AddComponent<LineRenderer>();
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.colorGradient = gradient;
        lineRenderer.positionCount = traceLength;
        Vector3[] startArray = new Vector3[traceLength];
        for (int i = 0; i < traceLength; i++) {
            startArray[i] = transform.position;
        }

        // Assigning shape and locomotion properties to agent.
        navMeshAgent.avoidancePriority = Random.Range(0, 99);
        navMeshAgent.radius = agentRadius;
        navMeshAgent.speed = agentSpeed;
        transform.localScale = new Vector3(agentSize, agentSize, agentSize);
    }

    void Update()
    {
         // Remembering current position and attempt to displace to avoid deadlock (only if the agents has traveled at least one update).
        trajectory.Add(transform.position);
        if (trajectory.Count > 1) {
            displace();
        }

        //Debug.Log("### "+showState());

        //TODO
        // Visualize trajectories.
        if (visualizeTrajectories) {
            visualizeTrajectoryHeatMap();
        }

        // State: Agent is currently choosing a new point of interest.
        if (choosingPOI) {

            // If we need to fulfill more needs, we choose a new POI.
            if (needsFulfilled < numberOfNeeds) {

                // Get next POI and try to get path. Repeat this process if fails.
                currPOI = choosePOI();
                NavMeshPath path = new NavMeshPath();

                // Find the closest point on NavMesh for current and goal location. This is necessary, as goal locations
                // are not necessarily placed on NavMesh.
                Vector3 startPos = ClosestPointOnNavMesh(transform.position);
                Vector3 endPos = ClosestPointOnNavMesh(POIs[currPOI].transform.position);
                //Debug.Log("$$ CALCULATING PATH");
                if (!NavMesh.CalculatePath(startPos, endPos, NavMesh.AllAreas, path))
                {
                    throw new System.Exception($"No valid path was found to POI {currPOI}");
                }

                // Setting the new path of the agent.
                navMeshAgent.path = path;
                choosingPOI = false;
                findingPOI = true;
            }

            // Else we have fulfilled all needs, and we can set the end as target.
            else {
                NavMeshPath path = new NavMeshPath();
                //Debug.Log("@@@#")
                GameObject chosenEnd = end[Random.Range(0, end.Length)];
                //Debug.Log("## CALCULATING PATH");
                while (!navMeshAgent.CalculatePath(chosenEnd.transform.position, path)) {
                    throw new System.Exception(task.name + ": End is not located properly. Please readjust its position.");
                }
                navMeshAgent.path = path;
                choosingPOI = false;
                taskCompleted = true;
            }
        }

        // State: Agent is currently walking towards the POI.
        else if (findingPOI) {
            
            // Visualize path during movement
            if (visualizePaths) {
                visualizePath(navMeshAgent.path);
            }
            
            // Has completed the search.
            if (hasArrivedAtPOI()) {
                //Debug.Log("@@@ ARRIVED ");
                arrivalTime = Time.realtimeSinceStartup;
                findingPOI = false;
                fulfillingNeed = true;
            }
        }

        // State: Agent is fulfilling need.
        else if (fulfillingNeed) {
            
            // Visualize path while waiting at POI
            if (visualizePaths) {
                visualizePath(navMeshAgent.path);
            }
            
            if (hasFulfilledNeed()) {
                needsFulfilled++;
                if (!revisit) {
                    poiMask[currPOI] = false;
                }
                fulfillingNeed = false;
                choosingPOI = true;
            }
        }

        // State: Task is completed.
        else if (taskCompleted) {
            
            // Visualize path during final movement to end
            if (visualizePaths) {
                visualizePath(navMeshAgent.path);
            }
            
            if (hasArrivedAtPOI()) {
                destroyRequest = true;
            }
        }

    }

    // Returns the index of the next point of interest.
    private int choosePOI() {

        // If we choose non-deterministically, we identify all valid POIs and choose one randomly.
        if (cnd) {

            // Generate list of all valid POIs.
            List<int> validPOIs = new List<int>();
            for (int i = 0; i < POIs.Length; i++) {
                if (poiMask[i]) {
                    validPOIs.Add(i);
                }
            }

            // Choose POI randomly.
            int randomPOIIndex = Random.Range(0, validPOIs.Count);
            return validPOIs[randomPOIIndex];
        }

        // Else we are just picking the next POI that has not been visited yet.
        else {

            // Go through all POIs and choose the next unvisited one.
            for (int i = 0; i < POIs.Length; i++) {
                if (poiMask[i]) {
                    return i;
                }
            }
        }

        // Should never happen.
        return 0;
    }

    // Checks if agent has arrived at POI.
    private bool hasArrivedAtPOI() {

        // Check that the path is not pending.
        bool c1 = !navMeshAgent.pathPending;

        // The remaining path is shorter than the epsilon-distance to the target.
        bool c2 = navMeshAgent.remainingDistance <= navMeshAgent.stoppingDistance + 0.5; //TODO hacky way to make it stop at most at 0.5m 

        // The agent has no path.
        bool c3 = !navMeshAgent.hasPath;

        // The agent velocity is zero.
        bool c4 = navMeshAgent.velocity.sqrMagnitude == 0;
        //Debug.Log("$$ "+navMeshAgent.remainingDistance+" | "+navMeshAgent.stoppingDistance);
        //Debug.Log("$$ c1 "+c1+" c2 "+c2+" c3 "+c3+" c4 "+c4);

        return c1 && c2; // && (c3 || c4);
    }

    // Checks if agent has fulfilled need.
    private bool hasFulfilledNeed() {
        return Time.realtimeSinceStartup - arrivalTime >= poiTime;
    }

    public string showState() {
        if (choosingPOI) {
            return "choosingPOI";
        } else if (findingPOI) {
            return "findingPOI";
        } else if (fulfillingNeed) {
            return "fulfillingNeed";
        } else if (taskCompleted) {
            return "taskCompleted";
        } else {
            return "ERROR";
        }
    }

    public void visualizePath(NavMeshPath path) {
        // Store the path for drawing in pause mode
        lastDrawnPath = path;
        
        // Determine the color based on path validity and agent selection
        Color pathColor = GetPathColor();
        
        // Draw line segments for each corner of the path
        for (int i = 1; i < path.corners.Length; i++) {
            Vector3 startPoint = path.corners[i-1] + Vector3.up * 1.0f;
            Vector3 endPoint = path.corners[i] + Vector3.up * 1.0f;
            
            // Draw with duration 0 to redraw every frame, using the determined color
            Debug.DrawLine(startPoint, endPoint, pathColor, 0.0f);
        }
    }

    // Helper function to determine the path color based on path validity and selection state
    private Color GetPathColor() {
        Color color = agentColor;
        
        // Check if the path is invalid and apply blinking effect
        if (navMeshAgent.pathStatus == NavMeshPathStatus.PathInvalid || navMeshAgent.pathStatus == NavMeshPathStatus.PathPartial) {
            // Blink between agent color and red
            float blinkSpeed = 1.5f; // Controls blink frequency
            float blinkValue = Mathf.Sin(Time.time * blinkSpeed * Mathf.PI) * 0.5f + 0.5f;
            color = Color.Lerp(agentColor, Color.red, blinkValue);
        }
        
        // Check if agent is selected in editor to highlight the path
        #if UNITY_EDITOR
        if (UnityEditor.Selection.Contains(gameObject)) {
            // Brighten the color when selected
            color = Color.Lerp(color, Color.white, 0.5f);
            // Also highlight the agent material
            meshRenderer.material.color = Color.Lerp(agentColor, Color.white, 0.5f);
        } else {
            // Restore original material color when not selected
            meshRenderer.material.color = agentColor;
        }
        #endif
        
        return color;
    }
    
    /*
    // OnDrawGizmos is called in pause mode to visualize paths
    private void OnDrawGizmos() {
        #if UNITY_EDITOR
        if (lastDrawnPath != null && lastDrawnPath.corners.Length > 0) {
            // Determine the color based on path validity and agent selection
            Color pathColor = GetPathColor();
            
            // Draw line segments for each corner of the path
            for (int i = 1; i < lastDrawnPath.corners.Length; i++) {
                Vector3 startPoint = lastDrawnPath.corners[i-1] + Vector3.up * 1.0f;
                Vector3 endPoint = lastDrawnPath.corners[i] + Vector3.up * 1.0f;
                
                // Draw gizmo line (visible in pause mode)
                Gizmos.color = pathColor;
                Gizmos.DrawLine(startPoint, endPoint);
            }
        }
        #endif
    }
    */
    
    private void displace() {
        // Enough time has passed such that we can attempt displacement.
        if (Time.realtimeSinceStartup >= lastDisplacementTime + displacementInterval) {
            navMeshAgent.avoidancePriority = Random.Range(0, 99);
            // But only if the distance to the last position is small enough and the agent is not currently fulfilling its needs.
            if (Vector3.Distance(trajectory[trajectory.Count - 2], trajectory[trajectory.Count - 1]) < displacementDelta && !fulfillingNeed) {

                // Set the last time the agent was displaced to now.
                lastDisplacementTime = Time.realtimeSinceStartup;

                // Add a randomized vector on the x-z plane.
                transform.position += new Vector3(Random.value * displacement, 0.05f, Random.value * displacement);
            }
        }
    }

    private void visualizeTrajectory() {
        
    }
    // Draws trajectory of agent with heatmap visualization showing dwell time at locations.
    // Warm colors (red/yellow) indicate areas where the agent spent more time.
    // Cool colors (blue/cyan) indicate transit corridors.
    private void visualizeTrajectoryHeatMap() {
        if (trajectory.Count < 2) return;
        
        // Determine if agent is selected for highlighting
        #if UNITY_EDITOR
        bool isSelected = UnityEditor.Selection.Contains(gameObject);
        #else
        bool isSelected = false;
        #endif
        
        // Calculate dwell time at each position (how many times the agent was near it)
        float dwellThreshold = 0.5f; // Distance threshold for considering same location
        float[] dwellTimes = new float[trajectory.Count];
        
        // Korrigierte Version:
        for (int i = 0; i < trajectory.Count; i++) {
            int dwellCount = 0;
            // Wir vergleichen Punkt 'i' mit allen anderen Punkten 'j'
            for (int j = 0; j < trajectory.Count; j++) {
                if (Vector3.Distance(trajectory[i], trajectory[j]) < dwellThreshold) {
                    dwellCount++;
                }
            }
            dwellTimes[i] = (float)dwellCount / trajectory.Count;
        }
        
        // Find max dwell time for normalization
        float maxDwellTime = Mathf.Max(dwellTimes);
        if (maxDwellTime == 0) maxDwellTime = 1f;
        
        // Draw the complete trajectory with heatmap coloring
        for (int i = 1; i < trajectory.Count; i++) {
            Vector3 startPoint = trajectory[i-1] + Vector3.up * 1.0f;
            Vector3 endPoint = trajectory[i] + Vector3.up * 1.0f;
            
            // Calculate heatmap color based on average dwell time of the two points
            float avgDwell = (dwellTimes[i-1] + dwellTimes[i]) / 2.0f;
            float normalizedDwell = avgDwell / maxDwellTime;
            
            // Create heatmap color: Blue (low dwell) -> Green -> Yellow -> Red (high dwell)
            Color heatmapColor = GetHeatmapColor(normalizedDwell, agentColor);
            
            // Highlight when agent is selected
            if (isSelected) {
                heatmapColor = Color.Lerp(heatmapColor, Color.white, 0.4f);
            }
            
            // Draw line with slight transparency (duration 0 for continuous redraw)
            Debug.DrawLine(startPoint, endPoint, heatmapColor, 0.0f);
        }
    }
    
    // Helper function to convert normalized dwell time (0-1) to a heatmap color
    private Color GetHeatmapColor(float normalizedDwell, Color baseColor) {
        if (normalizedDwell < 0.25f) {
            // Cool: Blue/Cyan - transit corridors
            return Color.Lerp(new Color(0.0f, 0.5f, 1.0f), new Color(0.0f, 1.0f, 1.0f), normalizedDwell * 4f);
        } else if (normalizedDwell < 0.5f) {
            // Green - moderate dwell time
            return Color.Lerp(new Color(0.0f, 1.0f, 1.0f), new Color(0.0f, 1.0f, 0.0f), (normalizedDwell - 0.25f) * 4f);
        } else if (normalizedDwell < 0.75f) {
            // Yellow - higher dwell time
            return Color.Lerp(new Color(0.0f, 1.0f, 0.0f), new Color(1.0f, 1.0f, 0.0f), (normalizedDwell - 0.5f) * 4f);
        } else {
            // Red/Orange - hotspots (long dwell times)
            return Color.Lerp(new Color(1.0f, 1.0f, 0.0f), new Color(1.0f, 0.0f, 0.0f), (normalizedDwell - 0.75f) * 4f);
        }
    }
    
    // Modifies agent color.
    private void initializeAgentColor() {
        Color taskColor = task.taskColor;
        /*
        TODO: Remove or make optional.
        float tR = taskColor.r;
        float tG = taskColor.g;
        float tB = taskColor.b;
        float randomValue = Random.value;
        float dark = 0.3f;
        float aR = Mathf.Clamp(tR - dark + randomValue * dark, 0.0f, 1.0f);
        float aG = Mathf.Clamp(tG - dark + randomValue * dark, 0.0f, 1.0f);
        float aB = Mathf.Clamp(tB - dark + randomValue * dark, 0.0f, 1.0f);
        agentColor.r = aR;
        agentColor.g = aG;
        agentColor.b = aB;
        agentColor.a = 1.0f;
        */
        agentColor = taskColor;
    }

    // Finds closest point on NavMesh, assuming that proposed position is not further than 100 units from NavMesh.
    private Vector3 ClosestPointOnNavMesh(Vector3 proposal)
    {
        NavMeshHit hit;
        bool success = NavMesh.SamplePosition(proposal, out hit, 100.0f, NavMesh.AllAreas);  // Hardcoded to 100 units of maximal distance.
        return hit.position;
    }
}
