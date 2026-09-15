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


/*

namespace Agent_Based_Modelling
{ 
    using System;
    using System.IO;
    using System.Collections;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using System.Diagnostics;
    using UnityEngine;
    using UnityEngine.AI;
    using Unity.Collections;
    using Unity.Jobs;
    using Unity.Sentis;
    using Debug = UnityEngine.Debug;

    public class ImitationAgent : MonoBehaviour
    {
        [Header("Linked Framework Components")]
        public TaskScript task; 
        public LayerMask ArchitectureLayerMask;

        // Task-locations & attributes
        private GameObject[] start;
        private GameObject[] end;
        private GameObject[] POIs;
        private float poiTime;
        private bool revisit;
        private bool cnd;
        private int numberOfNeeds;
        private float agentSize;
        private float agentRadius;
        private float agentSpeed;

        // State Sequencer
        private bool choosingPOI = false;
        private bool findingPOI = false;
        private bool fulfillingNeed = false;
        private bool taskCompleted = false;
        public bool destroyRequest = false;
        private bool[] poiMask;
        private int currPOI;
        private int needsFulfilled;
        private float arrivalTime;
        private Transform currentActiveTargetTransform;

        [HideInInspector]
        public int startIndex;

        [Header("Hardware Simulation (64x64 Complete Eyes)")]
        public Transform AgentHeadAnchor; 
        public float HorizonalFovOverride = 90f;
        public float VerticalFovOverride = 90f;
        public float MaxRayDistance = 50f;

        private const int ImageWidth = 64;
        private const int ImageHeight = 64;
        private const int TotalPixels = ImageWidth * ImageHeight;

        private Camera hiddenCaptureCamera;
        private RenderTexture colorRenderTexture;
        private Texture2D colorReadbackTex;
        private NativeArray<RaycastCommand> rayCommands;
        private NativeArray<RaycastHit> rayResults;

        [Header("Temporal Windowing (Memory Buffer)")]
        public int TemporalWindowSize = 3;
        private Queue<SerializableFrameData> stateHistoryQueue = new Queue<SerializableFrameData>();

        public struct SerializableFrameData
        {
            public Color32[] colorMatrix;
            public float[] depthMatrix;
        }

        [Header("Neural Network Runtime (Unity Sentis)")]
        [Tooltip("Drag your baseline or trained .onnx file here.")]
        public ModelAsset BaselineModelAsset;
        
        private Model runtimeModel;
        private Worker inferenceEngine;
        private bool isModelLoaded = false;
        private object modelLock = new object();

        [Header("Automated Python Lifecycle Pipeline")]
        [Tooltip("The path to the newly exported model that Python writes out.")]
        public string OutputOnnxModelPath = "";

        private static string lastKnownExportPath = "";
        private static DateTime lastLoadedModelWriteTime = DateTime.MinValue;
        private static bool pendingModelReload = false;
        private DateTime lastModelWriteTime;

        // Trajectory Mechanics
        public List<Vector3> trajectory;
        private float displacementInterval = 10.0f;
        private float displacement = 0.1f;
        public float lastDisplacementTime;
        private float displacementDelta = 0.1f;

        // Inference Inputs/Outputs
        public Vector3 localGoalVector3D;
        private float networkForwardAction = 0f;
        private float networkYawAction = 0f;
        private float networkPitchAction = 0f;

        public static void NotifyRecordingCompleted(string exportedModelPath)
        {
            if (string.IsNullOrEmpty(exportedModelPath)) return;
            lastKnownExportPath = exportedModelPath;
            pendingModelReload = true;
        }

        public static bool TryLoadNewestModelOnce()
        {
            if (!pendingModelReload || string.IsNullOrEmpty(lastKnownExportPath) || !File.Exists(lastKnownExportPath))
            {
                return false;
            }

            try
            {
                DateTime currentWriteTime = File.GetLastWriteTime(lastKnownExportPath);
                if (currentWriteTime <= lastLoadedModelWriteTime)
                {
                    pendingModelReload = false;
                    return false;
                }

                lastLoadedModelWriteTime = currentWriteTime;
                pendingModelReload = false;
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Model Swap] Failed to inspect exported model: {ex.Message}");
                pendingModelReload = false;
                return false;
            }
        }

        public bool ApplyLatestExportedModel()
        {
            if (string.IsNullOrEmpty(lastKnownExportPath) || !File.Exists(lastKnownExportPath))
            {
                return false;
            }

            try
            {
                lock (modelLock)
                {
                    if (inferenceEngine != null)
                    {
                        inferenceEngine.Dispose();
                    }

                    runtimeModel = ModelLoader.Load(lastKnownExportPath);
                    inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute);
                    isModelLoaded = true;
                    lastModelWriteTime = File.GetLastWriteTime(lastKnownExportPath);
                    Debug.Log($"[HOT-SWAP SUCCESS] Silent hot-swap completed. Path: {lastKnownExportPath}");
                    return true;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"ImitationAgent: Hot-Swap loading failed: {ex.Message}");
                return false;
            }
        }

        void Start()
        {
            start = task.start;
            end = task.end;
            POIs = task.pointsOfInterest;
            poiTime = task.poiTime;
            revisit = task.revisit;
            cnd = task.chooseNonDeterministically;
            numberOfNeeds = task.numberOfNeeds;
            agentSize = task.agentSize;
            agentRadius = task.agentRadius;
            agentSpeed = task.agentSpeed;

            lastDisplacementTime = Time.realtimeSinceStartup;
            trajectory = new List<Vector3>();
            poiMask = new bool[POIs.Length];
            for (int i = 0; i < poiMask.Length; i++) poiMask[i] = true;

            currPOI = 0;
            needsFulfilled = 0;
            choosingPOI = true;

            // Fallback safety check if ArchitectureLayerMask was left unassigned in Inspector
            if (ArchitectureLayerMask.value == 0)
            {
                ArchitectureLayerMask = LayerMask.GetMask("Architecture_Obstacle"); 
                Debug.LogWarning("[ImitationAgent] ArchitectureLayerMask unassigned. Defaulting to 'Default' layer.");
            }

            GameObject chosenStart = start[UnityEngine.Random.Range(0, start.Length)];
            NavMeshHit hit;
            if (NavMesh.SamplePosition(chosenStart.transform.position, out hit, 100.0f, NavMesh.AllAreas))
            {
                transform.position = hit.position;
            }
            else
            {
                transform.position = chosenStart.transform.position;
            }
            
            transform.localScale = new Vector3(agentSize, agentSize, agentSize);

            if (AgentHeadAnchor == null)
            {
                GameObject headGO = new GameObject("ImitationHeadAnchor");
                headGO.transform.SetParent(transform, false);
                headGO.transform.localPosition = new Vector3(0f, agentSize * 1.5f, 0f);
                AgentHeadAnchor = headGO.transform;
            }

            if (string.IsNullOrEmpty(OutputOnnxModelPath))
            {
                OutputOnnxModelPath = Path.Combine(Application.dataPath, "ImitationModel", "ImitationAgentModel.onnx");
            }

            InitializeHardwareReplication();
            InitializeJobBuffers();
            InitializeSentisEngine();

            if (File.Exists(OutputOnnxModelPath))
            {
                lastModelWriteTime = File.GetLastWriteTime(OutputOnnxModelPath);
            }
            else
            {
                lastModelWriteTime = DateTime.MinValue;
            }
        }

        private void InitializeHardwareReplication()
        {
            GameObject camGO = new GameObject("Hidden_Imitation_CaptureCamera");
            camGO.transform.SetParent(AgentHeadAnchor, false);
            hiddenCaptureCamera = camGO.AddComponent<Camera>();
            hiddenCaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            hiddenCaptureCamera.backgroundColor = Color.black;
            hiddenCaptureCamera.nearClipPlane = 0.1f;
            hiddenCaptureCamera.farClipPlane = MaxRayDistance;
            hiddenCaptureCamera.fieldOfView = VerticalFovOverride; 

            colorRenderTexture = new RenderTexture(ImageWidth, ImageHeight, 24, RenderTextureFormat.ARGB32);
            colorRenderTexture.Create();
            hiddenCaptureCamera.targetTexture = colorRenderTexture;
            colorReadbackTex = new Texture2D(ImageWidth, ImageHeight, TextureFormat.RGB24, false);
        }

        private void InitializeJobBuffers()
        {
            rayCommands = new NativeArray<RaycastCommand>(TotalPixels, Allocator.Persistent);
            rayResults = new NativeArray<RaycastHit>(TotalPixels, Allocator.Persistent);
        }
        
        private void InitializeSentisEngine()
        {
            try
            {
                if (BaselineModelAsset == null)
                {
                    Debug.LogError("[Neural Inference Core] CRITICAL: BaselineModelAsset is not assigned in the Inspector!");
                    return;
                }

                runtimeModel = ModelLoader.Load(BaselineModelAsset);
                inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute); 
                isModelLoaded = true;
                Debug.Log("[Neural Inference Core] Sentis Engine successfully bound and ready for execution.");
            }
            catch (Exception e)
            {
                Debug.LogError($"[Neural Inference Core] Sentis loading exception: {e.Message}");
            }
        }

        void Update()
        {
            trajectory.Add(transform.position);
            if (trajectory.Count > 1) DisplaceDeadlocks();

            EvaluateStateSequencer();
            CaptureCurrentObservations();

            if (stateHistoryQueue.Count == TemporalWindowSize)
            {
                ExecuteNeuralNetworkInference();
            }

            MoveAgentLocomotion(networkForwardAction, networkYawAction, networkPitchAction);
        }

        private void EvaluateStateSequencer()
        {
            if (choosingPOI)
            {
                if (needsFulfilled < numberOfNeeds)
                {
                    currPOI = ChoosePOI();
                    choosingPOI = false;
                    findingPOI = true;
                    currentActiveTargetTransform = POIs[currPOI].transform;
                }
                else
                {
                    choosingPOI = false;
                    taskCompleted = true;
                    currentActiveTargetTransform = end[UnityEngine.Random.Range(0, end.Length)].transform;
                }
            }
            else if (findingPOI)
            {
                if (HasArrivedAtTarget())
                {
                    arrivalTime = Time.realtimeSinceStartup;
                    findingPOI = false;
                    fulfillingNeed = true;
                }
            }
            else if (fulfillingNeed)
            {
                if (Time.realtimeSinceStartup - arrivalTime >= poiTime)
                {
                    needsFulfilled++;
                    if (!revisit) poiMask[currPOI] = false;
                    
                    fulfillingNeed = false;
                    choosingPOI = true;
                }
            }
            else if (taskCompleted)
            {
                if (HasArrivedAtTarget())
                {
                    destroyRequest = true;
                }
            }
        }

        private void CaptureCurrentObservations()
        {
            if (currentActiveTargetTransform == null)
            {
                localGoalVector3D = Vector3.zero;
                return;
            }

            Vector3 worldGoalDir = (currentActiveTargetTransform.position - AgentHeadAnchor.position).normalized;
            localGoalVector3D = AgentHeadAnchor.InverseTransformDirection(worldGoalDir);

            hiddenCaptureCamera.Render();
            RenderTexture.active = colorRenderTexture;
            colorReadbackTex.ReadPixels(new Rect(0, 0, ImageWidth, ImageHeight), 0, 0);
            colorReadbackTex.Apply();
            Color32[] colorPixels = colorReadbackTex.GetPixels32();

            Vector3 origin = AgentHeadAnchor.position + AgentHeadAnchor.forward * 0.1f;
            float fovRadH = HorizonalFovOverride * Mathf.Deg2Rad;
            float fovRadV = VerticalFovOverride * Mathf.Deg2Rad;
            int index = 0;

            for (int y = 0; y < ImageHeight; y++)
            {
                float vAngle = (((ImageHeight > 1) ? ((float)y / (ImageHeight - 1)) - 0.5f : 0f)) * fovRadV;
                for (int x = 0; x < ImageWidth; x++)
                {
                    float hAngle = (((ImageWidth > 1) ? ((float)x / (ImageWidth - 1)) - 0.5f : 0f)) * fovRadH;
                    Vector3 localDir = new Vector3(Mathf.Tan(hAngle), Mathf.Tan(vAngle), 1.0f).normalized;
        
                    rayCommands[index] = new RaycastCommand(
                        origin, 
                        AgentHeadAnchor.TransformDirection(localDir), 
                        new QueryParameters(ArchitectureLayerMask, true, QueryTriggerInteraction.Collide, false), 
                        MaxRayDistance
                    );
                    index++;
                }
            }

            JobHandle handle = RaycastCommand.ScheduleBatch(rayCommands, rayResults, 32);
            handle.Complete();

            float[] frameDepthBuffer = new float[TotalPixels];
            for (int i = 0; i < rayResults.Length; i++)
            {
                float dist = rayResults[i].distance;
                frameDepthBuffer[i] = (dist > 0) ? dist / MaxRayDistance : 1.0f;
            }

            stateHistoryQueue.Enqueue(new SerializableFrameData { colorMatrix = colorPixels, depthMatrix = frameDepthBuffer });
            if (stateHistoryQueue.Count > TemporalWindowSize) stateHistoryQueue.Dequeue();
        }

        private void ExecuteNeuralNetworkInference()
        {
            lock (modelLock)
            {
                if (!isModelLoaded || inferenceEngine == null)
                {
                    if (findingPOI || taskCompleted)
                    {
                        networkForwardAction = 1.0f;
                        networkYawAction = Mathf.Clamp(localGoalVector3D.x * 2.0f, -1f, 1f);
                        networkPitchAction = Mathf.Clamp(localGoalVector3D.y * 2.0f, -1f, 1f);
                    }
                    return;
                }

                int totalSensoryInputs = (TemporalWindowSize * TotalPixels * 4) + 3;
                float[] flattenedFeatureArray = new float[totalSensoryInputs];
                int ptr = 0;

                flattenedFeatureArray[ptr++] = localGoalVector3D.x;
                flattenedFeatureArray[ptr++] = localGoalVector3D.y;
                flattenedFeatureArray[ptr++] = localGoalVector3D.z;

                foreach (var frame in stateHistoryQueue)
                {
                    for (int i = 0; i < TotalPixels; i++)
                    {
                        flattenedFeatureArray[ptr++] = frame.colorMatrix[i].r / 255f;
                        flattenedFeatureArray[ptr++] = frame.colorMatrix[i].g / 255f;
                        flattenedFeatureArray[ptr++] = frame.colorMatrix[i].b / 255f;
                        flattenedFeatureArray[ptr++] = frame.depthMatrix[i];
                    }
                }

                TensorShape tensorShape = new TensorShape(1, totalSensoryInputs);
                using (Tensor<float> sensoryTensor = new Tensor<float>(tensorShape, flattenedFeatureArray))
                {
                    inferenceEngine.SetInput("sensory_inputs", sensoryTensor);
                    inferenceEngine.Schedule();
                    
                    var outputTensor = inferenceEngine.PeekOutput("motor_actions") as Tensor<float>;
                    if (outputTensor != null)
                    {
                        float[] outputData = outputTensor.DownloadToArray();
                        networkForwardAction = outputData[0];
                        networkYawAction     = outputData[1];
                        networkPitchAction   = outputData[2];
                    }
                }
            }
        }

        private void MoveAgentLocomotion(float forward, float yaw, float pitch)
        {
            // 1. Rotation (Yaw & Pitch)
            float rotationSpeed = 120.0f;
            transform.Rotate(0f, yaw * rotationSpeed * Time.deltaTime, 0f, Space.Self);
    
            float targetPitch = AgentHeadAnchor.localEulerAngles.x + (-pitch * rotationSpeed * Time.deltaTime);
            if (targetPitch > 180f) targetPitch -= 360f;
            targetPitch = Mathf.Clamp(targetPitch, -45f, 45f);
            AgentHeadAnchor.localEulerAngles = new Vector3(targetPitch, 0f, 0f);

            // 2. Forward Movement Calculation
            float moveAmount = forward * agentSpeed * Time.deltaTime;
            if (Mathf.Abs(moveAmount) < 0.001f) return;

            Vector3 targetDisplacement = transform.forward * moveAmount;
            Vector3 proposedPosition = transform.position + targetDisplacement;

            // 3. Physical Wall Collision & NavMesh Surface Constraint
            // Cast a sphere ahead to stop before hitting physical walls/colliders
            Vector3 castOrigin = transform.position + Vector3.up * 0.5f;
            bool hitWall = Physics.SphereCast(castOrigin, agentRadius * 0.8f, transform.forward, out RaycastHit wallHit, Mathf.Abs(moveAmount) + 0.1f, ArchitectureLayerMask);

            if (!hitWall)
            {
                // Safe to move: sample the closest valid NavMesh point to retain ground positioning
                NavMeshHit navHit;
                if (NavMesh.SamplePosition(proposedPosition, out navHit, agentRadius * 2.0f, NavMesh.AllAreas))
                {
                    transform.position = navHit.position;
                }
            }
        }

        private void CheckForPendingModelHotSwaps()
        {
            if (File.Exists(OutputOnnxModelPath))
            {
                try
                {
                    DateTime currentWriteTime = File.GetLastWriteTime(OutputOnnxModelPath);

                    if (currentWriteTime > lastModelWriteTime)
                    {
                        lastModelWriteTime = currentWriteTime;

                        lock (modelLock)
                        {
                            if (inferenceEngine != null) inferenceEngine.Dispose();

                            runtimeModel = ModelLoader.Load(OutputOnnxModelPath);
                            inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute);
                            isModelLoaded = true;
                            
                            Debug.Log($"[HOT-SWAP SUCCESS] Silent hot-swap completed. Path: {OutputOnnxModelPath}");
                        }
                    }
                }
                catch (IOException) { }
                catch (Exception ex)
                {
                    Debug.LogError($"ImitationAgent: Hot-Swap loading failed: {ex.Message}");
                }
            }
        }

        private int ChoosePOI()
        {
            if (cnd)
            {
                List<int> validPOIs = new List<int>();
                for (int i = 0; i < POIs.Length; i++) if (poiMask[i]) validPOIs.Add(i);
                return validPOIs[UnityEngine.Random.Range(0, validPOIs.Count)];
            }
            else
            {
                for (int i = 0; i < POIs.Length; i++) if (poiMask[i]) return i;
            }
            return 0;
        }

        private bool HasArrivedAtTarget()
        {
            if (currentActiveTargetTransform == null) return false;
            return Vector3.Distance(transform.position, currentActiveTargetTransform.position) <= 1.5f;
        }

        private void DisplaceDeadlocks()
        {
            if (Time.realtimeSinceStartup >= lastDisplacementTime + displacementInterval)
            {
                if (Vector3.Distance(trajectory[trajectory.Count - 2], trajectory[trajectory.Count - 1]) < displacementDelta && !fulfillingNeed)
                {
                    lastDisplacementTime = Time.realtimeSinceStartup;
                    Vector3 targetPos = transform.position + new Vector3(UnityEngine.Random.value * displacement, 0f, UnityEngine.Random.value * displacement);
                    
                    NavMeshHit hit;
                    if (NavMesh.SamplePosition(targetPos, out hit, 1.0f, NavMesh.AllAreas))
                    {
                        transform.position = hit.position;
                    }
                }
            }
        }

        private void OnDestroy()
        {
            lock (modelLock)
            {
                if (inferenceEngine != null) inferenceEngine.Dispose();
            }
            if (rayCommands.IsCreated) rayCommands.Dispose();
            if (rayResults.IsCreated) rayResults.Dispose();
            if (colorRenderTexture != null) colorRenderTexture.Release();
            if (colorReadbackTex != null) Destroy(colorReadbackTex);
        }
    }
}

*/


























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

namespace Agent_Based_Modelling
{ 
    using System;
    using System.IO;
    using System.Collections;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using System.Diagnostics;
    using UnityEngine;
    using UnityEngine.AI;
    using Unity.Collections;
    using Unity.Jobs;
    using Unity.Sentis;
    using Assets.Scripts.DataRecording;
    using Debug = UnityEngine.Debug;

    public class ImitationAgent : MonoBehaviour
    {
        [Header("Linked Framework Components")]
        public TaskScript task; 
        public LayerMask ArchitectureLayerMask;

        // Task-locations & attributes
        private GameObject[] start;
        private GameObject[] end;
        private GameObject[] POIs;
        private float poiTime;
        private bool revisit;
        private bool cnd;
        private int numberOfNeeds;
        private float agentSize;
        private float agentRadius;
        private float agentSpeed;

        // State Sequencer
        private bool choosingPOI = false;
        private bool findingPOI = false;
        private bool fulfillingNeed = false;
        private bool taskCompleted = false;
        public bool destroyRequest = false;
        private bool[] poiMask;
        private int currPOI;
        private int needsFulfilled;
        private float arrivalTime;
        private Transform currentActiveTargetTransform;

        [HideInInspector]
        public int startIndex;

        // All fields below are initialized from ZarrDataContract (the same contract
        // ParticipantRecorder writes its training-data manifest against) rather than
        // separately-hardcoded literals, so live inference cannot silently drift out of
        // sync with the format the model was trained on (e.g. the depth channel used to
        // normalize by a 2000f MaxRayDistance here vs. the recorder's/training data's 50f,
        // a ~40x scale mismatch on every depth value fed to the network).
        [Header("Hardware Simulation (64x64 Complete Eyes)")]
        public Transform AgentHeadAnchor;
        [Tooltip("Camera whose exposure/HDR/FOV/post-processing settings the hidden capture camera should " +
                 "copy - must be the same camera ParticipantRecorder used as MainVRCamera during data " +
                 "collection (the 'FirstPersonCharacter' camera under FPSController.prefab, FOV 60 with the " +
                 "Post-Processing Stack v2 layer attached), NOT the scene's Camera.main (e.g. a simulation " +
                 "overview/spectator camera), or live RGB frames will not match the distribution the model " +
                 "was trained on.")]
        public Camera ReferenceCameraSettings;
        public float HorizonalFovOverride = ZarrDataContract.HorizontalFovDegrees;
        public float VerticalFovOverride = ZarrDataContract.VerticalFovDegrees;
        public float MaxRayDistance = ZarrDataContract.MaxRayDistance;

        private const int ImageWidth = ZarrDataContract.ImageWidth;
        private const int ImageHeight = ZarrDataContract.ImageHeight;
        private const int TotalPixels = ImageWidth * ImageHeight;

        private Camera hiddenCaptureCamera;
        private RenderTexture colorRenderTexture;
        private Texture2D colorReadbackTex;
        private NativeArray<RaycastCommand> rayCommands;
        private NativeArray<RaycastHit> rayResults;

        [Header("Temporal Windowing (Memory Buffer)")]
        public int TemporalWindowSize = ZarrDataContract.TemporalWindowSize;
        private Queue<SerializableFrameData> stateHistoryQueue = new Queue<SerializableFrameData>();

        public struct SerializableFrameData
        {
            public Color32[] colorMatrix;
            public float[] depthMatrix;
        }

        [Header("Neural Network Runtime (Unity Sentis)")]
        [Tooltip("Drag your baseline or trained .onnx file here.")]
        public ModelAsset BaselineModelAsset;
        [Tooltip("Must match ParticipantRecorder.SamplingInterval: action_rot labels are a delta angle accumulated over one recording sample, not a rate.")]
        public float RecordingSamplingInterval = ZarrDataContract.SamplingIntervalSeconds;
        private const float HeuristicMaxTurnRateDegPerSec = 120.0f;

        // Sanity bounds on the raw model output, taken from the actual recorded training data
        // (20 episodes: forward 0.0-1.56 m/s, yaw/pitch deltas within +-37.1/+-28.2 deg per
        // RecordingSamplingInterval), with a little headroom. Without this an undercalibrated
        // model can output forward speeds many times faster than any recorded human walking
        // speed, which is what was causing the agent to charge straight into walls instead of
        // slowing/turning through openings, and occasionally tunnel through thin walls in a
        // single frame.
        private const float MaxModelForwardMetersPerSec = 2.0f;
        private const float MaxModelRawYawDegPerSample = 45.0f;
        private const float MaxModelRawPitchDegPerSample = 35.0f;
        
        private Model runtimeModel;
        private Worker inferenceEngine;
        private bool isModelLoaded = false;
        private object modelLock = new object();

        [Header("Automated Python Lifecycle Pipeline")]
        [Tooltip("The path to the newly exported model that Python writes out.")]
        public string OutputOnnxModelPath = "";

        private static string lastKnownExportPath = "";
        private static DateTime lastLoadedModelWriteTime = DateTime.MinValue;
        private static bool pendingModelReload = false;
        private DateTime lastModelWriteTime;

        // Trajectory Mechanics
        public List<Vector3> trajectory;
        private float displacementInterval = 10.0f;
        private float displacement = 0.1f;
        public float lastDisplacementTime;
        private float displacementDelta = 0.1f;

        // Inference Inputs/Outputs
        public Vector3 localGoalVector3D;
        private float networkForwardAction = 0f;
        private float networkYawAction = 0f;
        private float networkPitchAction = 0f;

        // Diagnostic Throttling
        [Header("Debug Settings")]
        [Tooltip("Log neural network outputs every N frames to avoid flooding the console.")]
        public int DiagnosticLogInterval = 60;
        private bool hasLoggedModelFallbackWarning = false;

        public static void NotifyRecordingCompleted(string exportedModelPath)
        {
            if (string.IsNullOrEmpty(exportedModelPath)) return;
            lastKnownExportPath = exportedModelPath;
            pendingModelReload = true;
            Debug.Log($"[Pipeline Debug] New model export notification received at path: {exportedModelPath}");
        }

        public static bool TryLoadNewestModelOnce()
        {
            if (!pendingModelReload || string.IsNullOrEmpty(lastKnownExportPath) || !File.Exists(lastKnownExportPath))
            {
                return false;
            }

            try
            {
                DateTime currentWriteTime = File.GetLastWriteTime(lastKnownExportPath);
                if (currentWriteTime <= lastLoadedModelWriteTime)
                {
                    pendingModelReload = false;
                    return false;
                }

                lastLoadedModelWriteTime = currentWriteTime;
                pendingModelReload = false;
                return true;
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Model Swap] Failed to inspect exported model: {ex.Message}");
                pendingModelReload = false;
                return false;
            }
        }

        public bool ApplyLatestExportedModel()
        {
            if (string.IsNullOrEmpty(lastKnownExportPath) || !File.Exists(lastKnownExportPath))
            {
                Debug.LogWarning($"[Hot-Swap] Attempted hot-swap, but invalid file path: '{lastKnownExportPath}'");
                return false;
            }

            try
            {
                lock (modelLock)
                {
                    if (inferenceEngine != null)
                    {
                        inferenceEngine.Dispose();
                    }

                    runtimeModel = ModelLoader.Load(lastKnownExportPath);
                    inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute);
                    isModelLoaded = true;
                    lastModelWriteTime = File.GetLastWriteTime(lastKnownExportPath);
                    hasLoggedModelFallbackWarning = false;
                    Debug.Log($"[HOT-SWAP SUCCESS] Model hot-swap completed successfully from path: {lastKnownExportPath}");
                    return true;
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[Hot-Swap Failure] Loading failed: {ex.Message}\n{ex.StackTrace}");
                return false;
            }
        }

        void Start()
        {
            if (task == null)
            {
                Debug.LogError($"[ImitationAgent Debug] CRITICAL: 'TaskScript task' field is NOT assigned on {gameObject.name}!");
                return;
            }

            start = task.start;
            end = task.end;
            POIs = task.pointsOfInterest;
            poiTime = task.poiTime;
            revisit = task.revisit;
            cnd = task.chooseNonDeterministically;
            numberOfNeeds = task.numberOfNeeds;
            agentSize = task.agentSize;
            agentRadius = task.agentRadius;
            agentSpeed = task.agentSpeed;

            Debug.Log($"[ImitationAgent Start] Agent Initialized. Setup: POIs={POIs.Length}, NeedsRequired={numberOfNeeds}, Speed={agentSpeed}");

            lastDisplacementTime = Time.realtimeSinceStartup;
            trajectory = new List<Vector3>();
            poiMask = new bool[POIs.Length];
            for (int i = 0; i < poiMask.Length; i++) poiMask[i] = true;

            currPOI = 0;
            needsFulfilled = 0;
            choosingPOI = true;

            if (ArchitectureLayerMask.value == 0)
            {
                ArchitectureLayerMask = LayerMask.GetMask("Architecture_Obstacle"); 
                Debug.LogWarning("[ImitationAgent Debug] ArchitectureLayerMask unassigned. Defaulted to 'Architecture_Obstacle'.");
            }

            GameObject chosenStart = start[UnityEngine.Random.Range(0, start.Length)];
            NavMeshHit hit;
            if (NavMesh.SamplePosition(chosenStart.transform.position, out hit, 100.0f, NavMesh.AllAreas))
            {
                transform.position = hit.position;
                Debug.Log($"[ImitationAgent Start] Spawned on NavMesh at {transform.position}");
            }
            else
            {
                transform.position = chosenStart.transform.position;
                Debug.LogWarning($"[ImitationAgent Start] Could not find NavMesh near start position! Teleported to {transform.position}");
            }

            // Face the first goal at spawn instead of inheriting an arbitrary world rotation.
            // In the recorded human data, the "large reorientation from a bad starting heading"
            // pattern makes up only ~7% of frames (vs. steady walking with small corrections for
            // the rest), so the trained model's turning response is far weaker than what a human
            // shows during that phase. Spawning already facing the goal avoids needing that
            // under-trained large-turn behavior at all.
            if (needsFulfilled < numberOfNeeds && POIs.Length > 0)
            {
                currPOI = ChoosePOI();
                currentActiveTargetTransform = POIs[currPOI].transform;
                choosingPOI = false;
                findingPOI = true;
                Vector3 towardGoal = currentActiveTargetTransform.position - transform.position;
                towardGoal.y = 0f;
                if (towardGoal.sqrMagnitude > 0.0001f)
                {
                    transform.rotation = Quaternion.LookRotation(towardGoal.normalized, Vector3.up);
                }
            }

            transform.localScale = new Vector3(agentSize, agentSize, agentSize);

            if (AgentHeadAnchor == null)
            {
                GameObject headGO = new GameObject("ImitationHeadAnchor");
                headGO.transform.SetParent(transform, false);
                headGO.transform.localPosition = new Vector3(0f, agentSize * 1.5f, 0f);
                AgentHeadAnchor = headGO.transform;
                Debug.Log($"[ImitationAgent Start] Created default AgentHeadAnchor at local position {AgentHeadAnchor.localPosition}");
            }

            if (string.IsNullOrEmpty(OutputOnnxModelPath))
            {
                OutputOnnxModelPath = Path.Combine(Application.dataPath, "ImitationModel", "ImitationAgentModel.onnx");
            }

            InitializeHardwareReplication();
            InitializeJobBuffers();
            InitializeSentisEngine();

            if (File.Exists(OutputOnnxModelPath))
            {
                lastModelWriteTime = File.GetLastWriteTime(OutputOnnxModelPath);
            }
            else
            {
                lastModelWriteTime = DateTime.MinValue;
            }
        }

        private void InitializeHardwareReplication()
        {
            GameObject camGO = new GameObject("Hidden_Imitation_CaptureCamera");
            camGO.transform.SetParent(AgentHeadAnchor, false);
            hiddenCaptureCamera = camGO.AddComponent<Camera>();

            // Mirror ParticipantRecorder.InitializeHardwareReplication: copy from the SAME camera
            // configuration used to record the training data (FOV, clip planes, HDR, culling mask,
            // post-processing), then only override clear flags/background exactly as the recorder
            // does. Camera.main is NOT a substitute for this - in a simulation scene it is typically
            // an overview/spectator camera with its own exposure and post-processing, and copying
            // from it silently shifts every RGB pixel fed to the model out of the training
            // distribution (see the [RGB DEBUG] mean-brightness diagnostic below).
            Camera referenceCamera = ReferenceCameraSettings;
            if (referenceCamera == null)
            {
                referenceCamera = Camera.main;
                Debug.LogWarning("[Sensory Hardware] ReferenceCameraSettings not assigned! Falling back to " +
                    "Camera.main, which will NOT match the MainVRCamera used during recording and will " +
                    "produce out-of-distribution RGB input for the model. Assign the recording camera's " +
                    "settings explicitly.");
            }
            if (referenceCamera != null)
            {
                hiddenCaptureCamera.CopyFrom(referenceCamera);
                // CopyFrom also copies the source camera's transform. ParticipantRecorder gets away
                // with this because it copies from its own direct parent (a no-op offset-wise); here
                // the reference camera is a different object entirely, so pin this camera back to the
                // agent's head instead of wherever the reference camera happens to be.
                camGO.transform.localPosition = Vector3.zero;
                camGO.transform.localRotation = Quaternion.identity;
            }
            hiddenCaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            hiddenCaptureCamera.backgroundColor = Color.black;
            // Deliberately do NOT reset fieldOfView here. The RGB camera's FOV is whatever
            // CopyFrom pulled from the reference camera (e.g. FirstPersonCharacter's 60 deg in
            // FPSController.prefab) - that is what actually produced the training images.
            // HorizonalFovOverride/VerticalFovOverride (90 deg) only shape the separate depth
            // raycast fan below; the two are intentionally different and both must match what
            // ParticipantRecorder used, not each other.

            colorRenderTexture = new RenderTexture(ImageWidth, ImageHeight, 24, RenderTextureFormat.ARGB32);
            colorRenderTexture.Create();
            hiddenCaptureCamera.targetTexture = colorRenderTexture;
            colorReadbackTex = new Texture2D(ImageWidth, ImageHeight, TextureFormat.RGB24, false);

            Debug.Log($"[Sensory Hardware] Visual Camera initialized ({ImageWidth}x{ImageHeight} render target).");
        }

        private void InitializeJobBuffers()
        {
            rayCommands = new NativeArray<RaycastCommand>(TotalPixels, Allocator.Persistent);
            rayResults = new NativeArray<RaycastHit>(TotalPixels, Allocator.Persistent);
        }
        
        private void InitializeSentisEngine()
        {
            try
            {
                if (BaselineModelAsset == null)
                {
                    Debug.LogError("[Neural Inference Core] CRITICAL: BaselineModelAsset is missing! Agent will fallback to heuristic control.");
                    isModelLoaded = false;
                    return;
                }

                runtimeModel = ModelLoader.Load(BaselineModelAsset);
                inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute); 
                isModelLoaded = true;
                Debug.Log($"[Neural Inference Core] SUCCESS: Sentis engine loaded '{BaselineModelAsset.name}'. Input/Output nodes mapped.");
            }
            catch (Exception e)
            {
                isModelLoaded = false;
                Debug.LogError($"[Neural Inference Core] Sentis loading exception: {e.Message}\n{e.StackTrace}");
            }
        }

        void Update()
        {
            trajectory.Add(transform.position);
            if (trajectory.Count > 1) DisplaceDeadlocks();

            EvaluateStateSequencer();
            CaptureCurrentObservations();

            if (stateHistoryQueue.Count == TemporalWindowSize)
            {
                ExecuteNeuralNetworkInference();
            }
            else
            {
                if (Time.frameCount % DiagnosticLogInterval == 0)
                {
                    Debug.Log($"[Temporal Buffer] Filling observation window: {stateHistoryQueue.Count}/{TemporalWindowSize} frames ready.");
                }
            }

            MoveAgentLocomotion(networkForwardAction, networkYawAction, networkPitchAction);
        }

        private void EvaluateStateSequencer()
        {
            if (choosingPOI)
            {
                if (needsFulfilled < numberOfNeeds)
                {
                    currPOI = ChoosePOI();
                    choosingPOI = false;
                    findingPOI = true;
                    currentActiveTargetTransform = POIs[currPOI].transform;
                    Debug.Log($"[State Sequencer] POI Selected: Index={currPOI} ({POIs[currPOI].name}). Progress: {needsFulfilled}/{numberOfNeeds} needs.");
                }
                else
                {
                    choosingPOI = false;
                    taskCompleted = true;
                    int chosenEndIdx = UnityEngine.Random.Range(0, end.Length);
                    currentActiveTargetTransform = end[chosenEndIdx].transform;
                    Debug.Log($"[State Sequencer] All needs fulfilled ({needsFulfilled}/{numberOfNeeds}). Heading to Exit Target: Index={chosenEndIdx} ({end[chosenEndIdx].name}).");
                }
            }
            else if (findingPOI)
            {
                if (HasArrivedAtTarget())
                {
                    arrivalTime = Time.realtimeSinceStartup;
                    findingPOI = false;
                    fulfillingNeed = true;
                    Debug.Log($"[State Sequencer] Target reached: {currentActiveTargetTransform.name}. Fulfilling need for {poiTime}s...");
                }
            }
            else if (fulfillingNeed)
            {
                if (Time.realtimeSinceStartup - arrivalTime >= poiTime)
                {
                    needsFulfilled++;
                    if (!revisit) poiMask[currPOI] = false;
                    
                    fulfillingNeed = false;
                    choosingPOI = true;
                    Debug.Log($"[State Sequencer] Need fulfilled. Total needs satisfied: {needsFulfilled}/{numberOfNeeds}. Masking status for POI {currPOI}: Available={poiMask[currPOI]}");
                }
            }
            else if (taskCompleted)
            {
                if (HasArrivedAtTarget())
                {
                    destroyRequest = true;
                    Debug.Log("[State Sequencer] Agent reached end location! Task sequence completed. Requesting destruction.");
                }
            }
        }

        private void CaptureCurrentObservations()
        {
            if (currentActiveTargetTransform == null)
            {
                localGoalVector3D = Vector3.zero;
                if (Time.frameCount % DiagnosticLogInterval == 0)
                {
                    Debug.LogWarning("[Observation Pipeline] currentActiveTargetTransform is NULL! localGoalVector3D set to Vector3.zero.");
                }
                return;
            }

            Vector3 worldGoalDir = (currentActiveTargetTransform.position - AgentHeadAnchor.position).normalized;
            localGoalVector3D = AgentHeadAnchor.InverseTransformDirection(worldGoalDir);

            hiddenCaptureCamera.Render();
            RenderTexture.active = colorRenderTexture;
            colorReadbackTex.ReadPixels(new Rect(0, 0, ImageWidth, ImageHeight), 0, 0);
            colorReadbackTex.Apply();
            Color32[] colorPixels = colorReadbackTex.GetPixels32();

            Vector3 origin = AgentHeadAnchor.position + AgentHeadAnchor.forward * 0.1f;
            float fovRadH = HorizonalFovOverride * Mathf.Deg2Rad;
            float fovRadV = VerticalFovOverride * Mathf.Deg2Rad;
            int index = 0;

            for (int y = 0; y < ImageHeight; y++)
            {
                float vAngle = (((ImageHeight > 1) ? ((float)y / (ImageHeight - 1)) - 0.5f : 0f)) * fovRadV;
                for (int x = 0; x < ImageWidth; x++)
                {
                    float hAngle = (((ImageWidth > 1) ? ((float)x / (ImageWidth - 1)) - 0.5f : 0f)) * fovRadH;
                    Vector3 localDir = new Vector3(Mathf.Tan(hAngle), Mathf.Tan(vAngle), 1.0f).normalized;
        
                    rayCommands[index] = new RaycastCommand(
                        origin, 
                        AgentHeadAnchor.TransformDirection(localDir), 
                        new QueryParameters(ArchitectureLayerMask, true, QueryTriggerInteraction.Collide, false), 
                        MaxRayDistance
                    );
                    index++;
                }
            }

            JobHandle handle = RaycastCommand.ScheduleBatch(rayCommands, rayResults, 32);
            handle.Complete();

            float[] frameDepthBuffer = new float[TotalPixels];
            for (int i = 0; i < rayResults.Length; i++)
            {
                float dist = rayResults[i].distance;
                frameDepthBuffer[i] = (dist > 0) ? dist / MaxRayDistance : 1.0f;
            }

            stateHistoryQueue.Enqueue(new SerializableFrameData { colorMatrix = colorPixels, depthMatrix = frameDepthBuffer });
            if (stateHistoryQueue.Count > TemporalWindowSize) stateHistoryQueue.Dequeue();
        }

        private void ExecuteNeuralNetworkInference()
        {
            lock (modelLock)
            {
                if (!isModelLoaded || inferenceEngine == null)
                {
                    if (!hasLoggedModelFallbackWarning)
                    {
                        Debug.LogWarning("[Inference Diagnostic] MODEL NOT LOADED! Falling back to hardcoded heuristic steering rules.");
                        hasLoggedModelFallbackWarning = true;
                    }

                    if (findingPOI || taskCompleted)
                    {
                        // forward/yaw/pitch are rates (m/s, deg/s, deg/s) - see MoveAgentLocomotion
                        networkForwardAction = agentSpeed;
                        networkYawAction = Mathf.Clamp(localGoalVector3D.x * 2.0f, -1f, 1f) * HeuristicMaxTurnRateDegPerSec;
                        networkPitchAction = Mathf.Clamp(localGoalVector3D.y * 2.0f, -1f, 1f) * HeuristicMaxTurnRateDegPerSec;
                    }
                    
                    if (Time.frameCount % DiagnosticLogInterval == 0)
                    {
                        Debug.Log($"[Heuristic Control Output] Forward: {networkForwardAction:F2}, Yaw: {networkYawAction:F2}, Pitch: {networkPitchAction:F2} | Goal Vector: {localGoalVector3D.ToString("F2")}");
                    }
                    return;
                }

                int totalSensoryInputs = (TemporalWindowSize * TotalPixels * 4) + 3;
                float[] flattenedFeatureArray = new float[totalSensoryInputs];
                int ptr = 0;

                flattenedFeatureArray[ptr++] = localGoalVector3D.x;
                flattenedFeatureArray[ptr++] = localGoalVector3D.y;
                flattenedFeatureArray[ptr++] = localGoalVector3D.z;

                foreach (var frame in stateHistoryQueue)
                {
                    for (int i = 0; i < TotalPixels; i++)
                    {
                        flattenedFeatureArray[ptr++] = frame.colorMatrix[i].r / 255f;
                        flattenedFeatureArray[ptr++] = frame.colorMatrix[i].g / 255f;
                        flattenedFeatureArray[ptr++] = frame.colorMatrix[i].b / 255f;
                        flattenedFeatureArray[ptr++] = frame.depthMatrix[i];
                    }
                }

                // TEMPORARY DIAGNOSTIC: compare against training data's R-channel mean (~0.15) to
                // check for a live-vs-recorded brightness/exposure mismatch. Safe to remove once resolved.
                if (Time.frameCount % DiagnosticLogInterval == 0)
                {
                    float rSum = 0f, rMin = 1f, rMax = 0f;
                    int rCount = 0;
                    for (int k = 3; k < flattenedFeatureArray.Length; k += 4)
                    {
                        float v = flattenedFeatureArray[k];
                        rSum += v;
                        if (v < rMin) rMin = v;
                        if (v > rMax) rMax = v;
                        rCount++;
                    }
                    Debug.Log($"[RGB DEBUG] mean={rSum / rCount:F4} min={rMin:F4} max={rMax:F4}");
                }

                TensorShape tensorShape = new TensorShape(1, totalSensoryInputs);
                using (Tensor<float> sensoryTensor = new Tensor<float>(tensorShape, flattenedFeatureArray))
                {
                    inferenceEngine.SetInput("sensory_inputs", sensoryTensor);
                    inferenceEngine.Schedule();
                    
                    var outputTensor = inferenceEngine.PeekOutput("motor_actions") as Tensor<float>;
                    if (outputTensor != null)
                    {
                        float[] outputData = outputTensor.DownloadToArray();
                        // forward is already m/s (matches ParticipantRecorder action_vel.z); yaw/pitch are
                        // degrees accumulated over one RecordingSamplingInterval, so convert to deg/s here.
                        // Clamp to the range the model was actually trained on before doing anything else
                        // with it - an out-of-range raw prediction is what let the agent dash into (and
                        // occasionally tunnel through) walls instead of walking/turning at a human pace.
                        float rawForward = Mathf.Clamp(outputData[0], 0f, MaxModelForwardMetersPerSec);
                        float rawYaw = Mathf.Clamp(outputData[1], -MaxModelRawYawDegPerSample, MaxModelRawYawDegPerSample);
                        float rawPitch = Mathf.Clamp(outputData[2], -MaxModelRawPitchDegPerSample, MaxModelRawPitchDegPerSample);

                        networkForwardAction = rawForward;
                        networkYawAction     = rawYaw / RecordingSamplingInterval;
                        networkPitchAction   = rawPitch / RecordingSamplingInterval;

                        // Diagnostic check for Model anomalies (NaN, Inf, or Constant Zeroes)
                        if (float.IsNaN(networkForwardAction) || float.IsNaN(networkYawAction) || float.IsNaN(networkPitchAction))
                        {
                            Debug.LogError("[Model Diagnostic] CRITICAL: Model output contains NaN values! Model output is corrupted/exploding.");
                        }

                        if (Time.frameCount % DiagnosticLogInterval == 0)
                        {
                            Debug.Log($"[Neural Model Output] Forward: {networkForwardAction:F3}, Yaw: {networkYawAction:F3}, Pitch: {networkPitchAction:F3} | Local Goal: {localGoalVector3D.ToString("F2")}");
                        }
                    }
                    else
                    {
                        Debug.LogError("[Model Diagnostic] Output tensor 'motor_actions' was NULL or wrong type! Verify ONNX output node name.");
                    }
                }
            }
        }

        private void MoveAgentLocomotion(float forward, float yaw, float pitch)
        {
            // forward/yaw/pitch are rates (m/s, deg/s, deg/s) set in ExecuteNeuralNetworkInference above.
            // 1. Rotation (Yaw & Pitch)
            transform.Rotate(0f, yaw * Time.deltaTime, 0f, Space.Self);
    
            float targetPitch = AgentHeadAnchor.localEulerAngles.x + (-pitch * Time.deltaTime);
            if (targetPitch > 180f) targetPitch -= 360f;
            targetPitch = Mathf.Clamp(targetPitch, -45f, 45f);
            AgentHeadAnchor.localEulerAngles = new Vector3(targetPitch, 0f, 0f);

            // 2. Forward Movement Calculation
            float moveAmount = forward * Time.deltaTime;

            if (Mathf.Abs(moveAmount) < 0.001f)
            {
                if (Time.frameCount % DiagnosticLogInterval == 0)
                {
                    Debug.LogWarning($"[Locomotion Warning] Move amount near zero ({moveAmount:F4}). Agent is standing still. Forward action was: {forward:F3}");
                }
                return;
            }

            Vector3 targetDisplacement = transform.forward * moveAmount;
            Vector3 proposedPosition = transform.position + targetDisplacement;

            // 3. Physical Wall Collision & NavMesh Surface Constraint
            Vector3 castOrigin = transform.position + Vector3.up * 0.5f;
            bool hitWall = Physics.SphereCast(castOrigin, agentRadius * 0.8f, transform.forward, out RaycastHit wallHit, Mathf.Abs(moveAmount) + 0.1f, ArchitectureLayerMask);

            if (!hitWall)
            {
                NavMeshHit navHit;
                if (NavMesh.SamplePosition(proposedPosition, out navHit, agentRadius * 2.0f, NavMesh.AllAreas))
                {
                    transform.position = navHit.position;
                }
                else
                {
                    if (Time.frameCount % DiagnosticLogInterval == 0)
                    {
                        Debug.LogWarning($"[Locomotion Warning] Proposed position {proposedPosition} is OFF the NavMesh. Movement blocked.");
                    }
                }
            }
            else
            {
                if (Time.frameCount % DiagnosticLogInterval == 0)
                {
                    Debug.LogWarning($"[Locomotion Collision] SphereCast blocked forward movement into '{wallHit.collider.gameObject.name}' at distance {wallHit.distance:F2}m.");
                }
            }
        }

        private void CheckForPendingModelHotSwaps()
        {
            if (File.Exists(OutputOnnxModelPath))
            {
                try
                {
                    DateTime currentWriteTime = File.GetLastWriteTime(OutputOnnxModelPath);

                    if (currentWriteTime > lastModelWriteTime)
                    {
                        lastModelWriteTime = currentWriteTime;

                        lock (modelLock)
                        {
                            if (inferenceEngine != null) inferenceEngine.Dispose();

                            runtimeModel = ModelLoader.Load(OutputOnnxModelPath);
                            inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute);
                            isModelLoaded = true;
                            hasLoggedModelFallbackWarning = false;
                            
                            Debug.Log($"[HOT-SWAP SUCCESS] Silent hot-swap completed from path: {OutputOnnxModelPath}");
                        }
                    }
                }
                catch (IOException) { }
                catch (Exception ex)
                {
                    Debug.LogError($"[Hot-Swap Failure] Loading failed: {ex.Message}");
                }
            }
        }

        private int ChoosePOI()
        {
            if (cnd)
            {
                List<int> validPOIs = new List<int>();
                for (int i = 0; i < POIs.Length; i++) if (poiMask[i]) validPOIs.Add(i);

                if (validPOIs.Count == 0)
                {
                    Debug.LogError("[State Sequencer] Non-deterministic POI choice failed! No remaining valid POIs available in poiMask.");
                    return 0;
                }

                return validPOIs[UnityEngine.Random.Range(0, validPOIs.Count)];
            }
            else
            {
                for (int i = 0; i < POIs.Length; i++) if (poiMask[i]) return i;
            }

            Debug.LogWarning("[State Sequencer] ChoosePOI fallback to 0. All POIs may have been marked false.");
            return 0;
        }

        private bool HasArrivedAtTarget()
        {
            if (currentActiveTargetTransform == null) return false;
            float dist = Vector3.Distance(transform.position, currentActiveTargetTransform.position);
            return dist <= 1.5f;
        }

        private void DisplaceDeadlocks()
        {
            if (Time.realtimeSinceStartup >= lastDisplacementTime + displacementInterval)
            {
                float deltaMoved = Vector3.Distance(trajectory[trajectory.Count - 2], trajectory[trajectory.Count - 1]);
                if (deltaMoved < displacementDelta && !fulfillingNeed)
                {
                    lastDisplacementTime = Time.realtimeSinceStartup;
                    Vector3 targetPos = transform.position + new Vector3(UnityEngine.Random.value * displacement, 0f, UnityEngine.Random.value * displacement);
                    
                    Debug.LogWarning($"[Deadlock Triggered] Agent moved only {deltaMoved:F4}m in target direction over {displacementInterval}s. Applying deadlock nudge towards {targetPos}.");

                    NavMeshHit hit;
                    if (NavMesh.SamplePosition(targetPos, out hit, 1.0f, NavMesh.AllAreas))
                    {
                        transform.position = hit.position;
                    }
                }
            }
        }

        private void OnDestroy()
        {
            Debug.Log("[ImitationAgent CleanUp] Disposing model workers, camera textures, and NativeArrays.");
            lock (modelLock)
            {
                if (inferenceEngine != null) inferenceEngine.Dispose();
            }
            if (rayCommands.IsCreated) rayCommands.Dispose();
            if (rayResults.IsCreated) rayResults.Dispose();
            if (colorRenderTexture != null) colorRenderTexture.Release();
            if (colorReadbackTex != null) Destroy(colorReadbackTex);
        }
    }
}