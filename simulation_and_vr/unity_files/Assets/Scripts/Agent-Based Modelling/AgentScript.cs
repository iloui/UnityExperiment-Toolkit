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
    public TaskScript task; // The task this agent is pursuing.
    private NavMeshAgent navMeshAgent; // The NavMeshAgent component of this agent.
    private MeshRenderer meshRenderer; // The MeshRenderer component of this agent.

    // Task-locations.
    private GameObject[] start; // The start of this agent.
    private GameObject[] end; // The end of this agent.
    private GameObject[] POIs; // Points of interest of the agent.

    // Attributes defining the interest of the agent.
    private float poiTime; // How long the agent is at each point of interest.
    private bool revisit; // Can the agent revisit targets?
    private bool cnd; // Does the agent choose its targets non-deterministically?
    private int numberOfNeeds; // Number of needs that this agent needs to fulfill.

    // Attributes defining shape and locomotion of agent.
    private float agentSize; // Size of the agent.
    private float agentRadius; // Radius from center in which no other agent can intrude.
    private float agentSpeed; // Speed of this agent.

    // Attributes for visualization.
    public bool visualizeTrajectories; // Set if you want to visualize trajectories.
    public bool visualizePaths; // Set if you want to visualize paths.
    public int traceLength; // How many past positions should be considered.
    private LineRenderer lineRenderer; // Renderer used to visualize trajectory.
    private Gradient gradient; // Gradient used to color trace.
    private Color agentColor; // Color of this specific agent.

    [Header("Architecture-Agnostic Goal Indicator (Option B)")]
    [Tooltip("Calculated normalized local vector pointing to the current target destination.")]
    public Vector3 localGoalVector3D;
    private GameObject arrowIndicatorRoot;
    private Transform currentActiveTargetTransform;

    // Technical stuff.
    private float displacementInterval = 10.0f; // After how many seconds should we change the agents position a bit to avoid deadlock.
    private float displacement = 0.1f; // In this range the x and z component of the displacement-vector will be chosen.
    public float lastDisplacementTime; // Last time the agent was displaced.
    private float displacementDelta = 0.1f; // Length of the vector that the agents needs to have traveled to not be displaced.
    public List<Vector3> trajectory; // A list of past positions of the agent, constituting the trajectories.
    private float sampleInterval; // Interval that needs to pass until new location gets sampled.
    private float lastSample; // Last time a sample was taken.
    private Vector3 firstPos; // First position in simulation.

    // State of the agent.
    private bool choosingPOI = false; // True: Needs to choose new POI.
    private bool findingPOI = false; // Is currently walking towards the POI.
    private bool fulfillingNeed = false; // Is currently fulfilling its need.
    private bool taskCompleted = false; // Has fulfilled all needs.
    public bool destroyRequest = false; // Indicates to the engine if this agent wants to be destroyed.
    private bool[] poiMask; // Masks the POIs that are invalid.
    private int currPOI; // The index of the current point of interest.
    private int needsFulfilled; // Number needs that this agent has already fulfilled.
    private float arrivalTime; // The last time the agent arrived at a POI.
    public int startIndex;
    private NavMeshPath lastDrawnPath; // Store the last path for drawing in pause mode

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
        lastDisplacementTime = Time.realtimeSinceStartup;
        trajectory = new List<Vector3>();
        lastDrawnPath = new NavMeshPath();

        // Initializing the state of the agent.
        poiMask = new bool[POIs.Length];
        for (int i = 0; i < poiMask.Length; i++)
        {
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
        for (int i = 0; i < traceLength; i++)
        {
            startArray[i] = transform.position;
        }

        // Assigning shape and locomotion properties to agent.
        navMeshAgent.avoidancePriority = Random.Range(0, 99);
        navMeshAgent.radius = agentRadius;
        navMeshAgent.speed = agentSpeed;
        transform.localScale = new Vector3(agentSize, agentSize, agentSize);

        // Option B: Programmatically spawn the 3D Goal Arrow on top of the agent
        Build3DArrowIndicator();
    }

    private void Build3DArrowIndicator()
    {
        // Create an anchor container centered above the agent's head geometry
        arrowIndicatorRoot = new GameObject("Goal_Direction_Arrow");
        arrowIndicatorRoot.transform.SetParent(transform, false);
        arrowIndicatorRoot.transform.localPosition = new Vector3(0f, agentSize * 1.5f, 0f);

        // Generate the shaft of the arrow indicator
        GameObject shaft = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        Destroy(shaft.GetComponent<CapsuleCollider>()); // Strip collision matrices
        shaft.transform.SetParent(arrowIndicatorRoot.transform, false);
        shaft.transform.localScale = new Vector3(0.08f, 0.25f, 0.08f);
        shaft.transform.localRotation = Quaternion.Euler(90f, 0f, 0f); // Rotate cylinder horizontal forward
        shaft.transform.localPosition = new Vector3(0f, 0f, 0.25f);

        // Generate the point/cone head of the arrow indicator
        GameObject head = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        // Note: Unity lacks built-in PrimitiveType.Cone, so we use a small scaled cylinder or capsule tipped forward
        Destroy(head.GetComponent<Collider>());
        head.transform.SetParent(arrowIndicatorRoot.transform, false);
        head.transform.localScale = new Vector3(0.18f, 0.12f, 0.18f);
        head.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        head.transform.localPosition = new Vector3(0f, 0f, 0.55f);

        // Uniformly color the indicator components to stand out
        Color indicatorColor = Color.cyan;
        shaft.GetComponent<MeshRenderer>().material.color = indicatorColor;
        head.GetComponent<MeshRenderer>().material.color = indicatorColor;

        // Default to hidden state until navigating
        arrowIndicatorRoot.SetActive(false);
    }

    void Update()
    {
        // Remembering current position and attempt to displace to avoid deadlock (only if the agents has traveled at least one update).
        trajectory.Add(transform.position);
        if (trajectory.Count > 1)
        {
            displace();
        }

        // Visualize trajectories.
        if (visualizeTrajectories)
        {
            visualizeTrajectoryHeatMap();
        }

        // State: Agent is currently choosing a new point of interest.
        if (choosingPOI)
        {
            // If we need to fulfill more needs, we choose a new POI.
            if (needsFulfilled < numberOfNeeds)
            {
                currPOI = choosePOI();
                NavMeshPath path = new NavMeshPath();

                Vector3 startPos = ClosestPointOnNavMesh(transform.position);
                Vector3 endPos = ClosestPointOnNavMesh(POIs[currPOI].transform.position);
                
                if (!NavMesh.CalculatePath(startPos, endPos, NavMesh.AllAreas, path))
                {
                    throw new System.Exception($"No valid path was found to POI {currPOI}");
                }

                navMeshAgent.path = path;
                choosingPOI = false;
                findingPOI = true;

                // Target Lock: Track the active POI transform node
                currentActiveTargetTransform = POIs[currPOI].transform;
            }
            // Else we have fulfilled all needs, and we can set the end as target.
            else
            {
                NavMeshPath path = new NavMeshPath();
                GameObject chosenEnd = end[Random.Range(0, end.Length)];
                
                while (!navMeshAgent.CalculatePath(chosenEnd.transform.position, path))
                {
                    throw new System.Exception(task.name + ": End is not located properly. Please readjust its position.");
                }

                navMeshAgent.path = path;
                choosingPOI = false;
                taskCompleted = true;

                // Target Lock: Track the active destination exit node
                currentActiveTargetTransform = chosenEnd.transform;
            }
        }

        // State: Agent is currently walking towards the POI.
        else if (findingPOI)
        {
            if (visualizePaths)
            {
                visualizePath(navMeshAgent.path);
            }

            if (hasArrivedAtPOI())
            {
                arrivalTime = Time.realtimeSinceStartup;
                findingPOI = false;
                fulfillingNeed = true;
            }
        }

        // State: Agent is fulfilling need.
        else if (fulfillingNeed)
        {
            if (visualizePaths)
            {
                visualizePath(navMeshAgent.path);
            }

            if (hasFulfilledNeed())
            {
                needsFulfilled++;
                if (!revisit)
                {
                    poiMask[currPOI] = false;
                }

                fulfillingNeed = false;
                choosingPOI = true;
            }
        }

        // State: Task is completed.
        else if (taskCompleted)
        {
            if (visualizePaths)
            {
                visualizePath(navMeshAgent.path);
            }

            if (hasArrivedAtPOI())
            {
                destroyRequest = true;
            }
        }

        // --- DYNAMIC GEOMETRICAL VECTOR CALCULATION & INDICATOR DRIVER ---
        // Hide arrow if stationary (choosing a POI or actively fulfilling a need)
        if (choosingPOI || fulfillingNeed || currentActiveTargetTransform == null)
        {
            localGoalVector3D = Vector3.zero;
            if (arrowIndicatorRoot.activeSelf) arrowIndicatorRoot.SetActive(false);
        }
        else // Show arrow when moving toward a POI or final destination
        {
            if (!arrowIndicatorRoot.activeSelf) arrowIndicatorRoot.SetActive(true);

            // Compute the absolute 3D world direction vector toward the current locked goal
            Vector3 worldGoalDir = (currentActiveTargetTransform.position - transform.position).normalized;

            // Map it architecture-agnostically into full local 3D space based on agent's tracking forward
            localGoalVector3D = transform.InverseTransformDirection(worldGoalDir);

            // Rotate the physical 3D arrow pointer container to project directly along the world heading vector
            arrowIndicatorRoot.transform.rotation = Quaternion.LookRotation(worldGoalDir, Vector3.up);
        }
    }

    // Returns the index of the next point of interest.
    private int choosePOI()
    {
        if (cnd)
        {
            List<int> validPOIs = new List<int>();
            for (int i = 0; i < POIs.Length; i++)
            {
                if (poiMask[i])
                {
                    validPOIs.Add(i);
                }
            }

            int randomPOIIndex = Random.Range(0, validPOIs.Count);
            return validPOIs[randomPOIIndex];
        }
        else
        {
            for (int i = 0; i < POIs.Length; i++)
            {
                if (poiMask[i])
                {
                    return i;
                }
            }
        }
        return 0;
    }

    // Checks if agent has arrived at POI.
    private bool hasArrivedAtPOI()
    {
        bool c1 = !navMeshAgent.pathPending;
        bool c2 = navMeshAgent.remainingDistance <= navMeshAgent.stoppingDistance + 0.5; 
        return c1 && c2;
    }

    // Checks if agent has fulfilled need.
    private bool hasFulfilledNeed()
    {
        return Time.realtimeSinceStartup - arrivalTime >= poiTime;
    }

    public string showState()
    {
        if (choosingPOI) return "choosingPOI";
        else if (findingPOI) return "findingPOI";
        else if (fulfillingNeed) return "fulfillingNeed";
        else if (taskCompleted) return "taskCompleted";
        else return "ERROR";
    }

    public void visualizePath(NavMeshPath path)
    {
        lastDrawnPath = path;
        Color pathColor = GetPathColor();

        for (int i = 1; i < path.corners.Length; i++)
        {
            Vector3 startPoint = path.corners[i - 1] + Vector3.up * 1.0f;
            Vector3 endPoint = path.corners[i] + Vector3.up * 1.0f;
            Debug.DrawLine(startPoint, endPoint, pathColor, 0.0f);
        }
    }

    private Color GetPathColor()
    {
        Color color = agentColor;

        if (navMeshAgent.pathStatus == NavMeshPathStatus.PathInvalid || navMeshAgent.pathStatus == NavMeshPathStatus.PathPartial)
        {
            float blinkSpeed = 1.5f; 
            float blinkValue = Mathf.Sin(Time.time * blinkSpeed * Mathf.PI) * 0.5f + 0.5f;
            color = Color.Lerp(agentColor, Color.red, blinkValue);
        }

#if UNITY_EDITOR
        if (UnityEditor.Selection.Contains(gameObject))
        {
            color = Color.Lerp(color, Color.white, 0.5f);
            meshRenderer.material.color = Color.Lerp(agentColor, Color.white, 0.5f);
        }
        else
        {
            meshRenderer.material.color = agentColor;
        }
#endif
        return color;
    }

    private void displace()
    {
        if (Time.realtimeSinceStartup >= lastDisplacementTime + displacementInterval)
        {
            navMeshAgent.avoidancePriority = Random.Range(0, 99);
            if (Vector3.Distance(trajectory[trajectory.Count - 2], trajectory[trajectory.Count - 1]) < displacementDelta && !fulfillingNeed)
            {
                lastDisplacementTime = Time.realtimeSinceStartup;
                transform.position += new Vector3(Random.value * displacement, 0.05f, Random.value * displacement);
            }
        }
    }

    private void visualizeTrajectoryHeatMap()
    {
        if (trajectory.Count < 2) return;

#if UNITY_EDITOR
        bool isSelected = UnityEditor.Selection.Contains(gameObject);
#else
        bool isSelected = false;
#endif

        float dwellThreshold = 0.5f; 
        float[] dwellTimes = new float[trajectory.Count];

        for (int i = 0; i < trajectory.Count; i++)
        {
            int dwellCount = 0;
            for (int j = 0; j < trajectory.Count; j++)
            {
                if (Vector3.Distance(trajectory[i], trajectory[j]) < dwellThreshold)
                {
                    dwellCount++;
                }
            }
            dwellTimes[i] = (float)dwellCount / trajectory.Count;
        }

        float maxDwellTime = Mathf.Max(dwellTimes);
        if (maxDwellTime == 0) maxDwellTime = 1f;

        for (int i = 1; i < trajectory.Count; i++)
        {
            Vector3 startPoint = trajectory[i - 1] + Vector3.up * 1.0f;
            Vector3 endPoint = trajectory[i] + Vector3.up * 1.0f;

            float avgDwell = (dwellTimes[i - 1] + dwellTimes[i]) / 2.0f;
            float normalizedDwell = avgDwell / maxDwellTime;

            Color heatmapColor = GetHeatmapColor(normalizedDwell, agentColor);

            if (isSelected)
            {
                heatmapColor = Color.Lerp(heatmapColor, Color.white, 0.4f);
            }

            Debug.DrawLine(startPoint, endPoint, heatmapColor, 0.0f);
        }
    }

    private Color GetHeatmapColor(float normalizedDwell, Color baseColor)
    {
        if (normalizedDwell < 0.25f)
        {
            return Color.Lerp(new Color(0.0f, 0.5f, 1.0f), new Color(0.0f, 1.0f, 1.0f), normalizedDwell * 4f);
        }
        else if (normalizedDwell < 0.5f)
        {
            return Color.Lerp(new Color(0.0f, 1.0f, 1.0f), new Color(0.0f, 1.0f, 0.0f), (normalizedDwell - 0.25f) * 4f);
        }
        else if (normalizedDwell < 0.75f)
        {
            return Color.Lerp(new Color(0.0f, 1.0f, 0.0f), new Color(1.0f, 1.0f, 0.0f), (normalizedDwell - 0.5f) * 4f);
        }
        else
        {
            return Color.Lerp(new Color(1.0f, 1.0f, 0.0f), new Color(1.0f, 0.0f, 0.0f), (normalizedDwell - 0.75f) * 4f);
        }
    }

    private void initializeAgentColor()
    {
        agentColor = task.taskColor;
    }

    private Vector3 ClosestPointOnNavMesh(Vector3 proposal)
    {
        NavMeshHit hit;
        bool success = NavMesh.SamplePosition(proposal, out hit, 100.0f, NavMesh.AllAreas); 
        return hit.position;
    }
}