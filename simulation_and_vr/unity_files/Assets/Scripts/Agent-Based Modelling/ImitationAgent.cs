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

    public class ImitationAgent : MonoBehaviour, IEvalAgent
    {
        [Header("Linked Framework Components")]
        [Tooltip("Defines this scene's start/POI(s)/end locations and selection rules. The same " +
                 "TaskScript ParticipantRecorder uses for this scene, so both pipelines walk the " +
                 "identical POI progression via PoiProgressTracker.")]
        public TaskScript task;
        [Tooltip("Must equal exactly LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName) - " +
                 "the agent refuses to start otherwise, since any broader/narrower mask silently corrupts " +
                 "every depth value fed to the model.")]
        public LayerMask ArchitectureLayerMask;

        // Task-locations & attributes. The body itself (radius, height, walk speed) is the recording body from
        // ZarrDataContract.RecordingBody, not task.agentRadius/agentSpeed - see InitializeRecordingBody.
        private float agentSize;
        private float agentSpeed;

        // Moves exactly like FPSController.prefab (human and synthetic recordings): same CharacterController capsule,
        // FixedUpdate movement with wall sliding, stick-to-ground force and gravity. The old SphereCast mover used a
        // 0.4 m cast that stopped dead at walls, so the policy saw states (and wall distances) no recording contains.
        private CharacterController body;
        private float bodyVerticalVelocity;
        private float executedForwardMetersPerSec;

        // Shared start -> POI(s) -> end progression, identical to ParticipantRecorder's.
        private PoiProgressTracker poiTracker;
        public bool destroyRequest = false;

        // Eval-harness instrumentation: read-only counters exposing the two existing failure
        // signals (CharacterController side contact in MoveRecordingBody, DisplaceDeadlocks stuck-nudge)
        // as data instead of just log lines, without changing either's actual behavior.
        // CollisionCount only increments on the rising edge (not-blocked -> blocked), so resting
        // against a wall for many frames counts as one collision event, not hundreds.
        [HideInInspector] public int CollisionCount = 0;
        [HideInInspector] public int StuckCount = 0;
        [HideInInspector] public bool HasCollided = false;
        [HideInInspector] public float FirstCollisionRealtime = 0f;
        private bool wasCollidingLastFrame = false;
        // Latest model output, sampled by EvalHarness's per-tick CSV. NaN until the first model call.
        // Discrete head: LastRawForward = P(walk), LastRawYawPerSample = decoded deg/sample.
        public float LastRawForward { get; private set; } = float.NaN;
        public float LastRawYawPerSample { get; private set; } = float.NaN;
        public float LastRawPitch { get; private set; } = float.NaN;
        /// <summary>Commanded forward speed (m/s) from the latest model call; for the discrete head, the walk key's target speed.</summary>
        public float LastCommandedForwardMps { get; private set; } = float.NaN;
        public float LastCommandedYawDegPerSecond { get; private set; } = float.NaN;
        /// <summary>The body pushed sideways into a wall in the last FixedUpdate.</summary>
        public bool IsWallBlocked => wasCollidingLastFrame;
        /// <summary>
        /// Optional hook set by DaggerHarness: receives the policy's (forward m/s, yaw deg/s, pitch deg/s) and returns
        /// the rates actually executed this frame (e.g. the NavMesh expert's while it is rescuing the agent).
        /// null = the policy's output is executed unchanged.
        /// </summary>
        [HideInInspector] public System.Func<Vector3, Vector3> ExecutedActionOverride;

        /// <summary>Position of the current POI/end target, for eval-harness path-efficiency logging.</summary>
        public Vector3 CurrentGoalPosition => (poiTracker != null && poiTracker.CurrentTarget != null) ? poiTracker.CurrentTarget.position : transform.position;
        /// <summary>How many POIs have been fulfilled so far, for eval-harness partial-credit logging.</summary>
        public int NeedsFulfilledCount => poiTracker != null ? poiTracker.NeedsFulfilled : 0;
        /// <summary>The POIs this agent must visit, in visiting order (empty until Start() has run), for the eval harness's optimal-path computation.</summary>
        public IReadOnlyList<Transform> PlannedPois => poiTracker != null ? poiTracker.PlannedPois : null;
        /// <summary>The end marker this agent finishes at (null until Start() has run).</summary>
        public Transform PlannedEnd => poiTracker != null ? poiTracker.PlannedEnd : null;
        List<Vector3> IEvalAgent.Trajectory => trajectory;
        bool IEvalAgent.DestroyRequest => destroyRequest;
        Transform IEvalAgent.EyeAnchor => AgentHeadAnchor;

        [HideInInspector]
        public int startIndex;

        // All hardware/sampling constants below are read directly from ZarrDataContract - the
        // same contract ParticipantRecorder's manifest is written against - rather than being
        // separately-tunable Inspector fields, so live inference cannot silently drift out of
        // sync with the format the model was trained on (e.g. the depth channel used to
        // normalize by a 2000f MaxRayDistance here vs. the recorder's/training data's 50f,
        // a ~40x scale mismatch on every depth value fed to the network).
        [Header("Hardware Simulation (64x64 Complete Eyes)")]
        public Transform AgentHeadAnchor;
        private const float MaxRayDistance = ZarrDataContract.MaxRayDistance;

        private const int ImageWidth = ZarrDataContract.ImageWidth;
        private const int ImageHeight = ZarrDataContract.ImageHeight;
        private const int TotalPixels = ImageWidth * ImageHeight;

        private Camera hiddenCaptureCamera;
        private RenderTexture colorRenderTexture;
        private Texture2D colorReadbackTex;
        private NativeArray<RaycastCommand> rayCommands;
        private NativeArray<RaycastHit> rayResults;

        [Header("Temporal Windowing (Memory Buffer)")]
        private const int TemporalWindowSize = ZarrDataContract.TemporalWindowSize;
        private Queue<SerializableFrameData> stateHistoryQueue = new Queue<SerializableFrameData>();

        // Decouples observation capture from Update()'s framerate-dependent cadence, so the
        // temporal window spans the same fixed real-world duration (SamplingIntervalSeconds
        // per frame) that the recorded training data's window does, instead of however many
        // frames the engine happens to render.
        private float captureAccumulator = 0f;

        public struct SerializableFrameData
        {
            public Color32[] colorMatrix;
            public float[] depthMatrix;
        }

        [Header("Neural Network Runtime (Unity Sentis)")]
        [Tooltip("Drag your baseline or trained .onnx file here.")]
        public ModelAsset BaselineModelAsset;
        // action_rot labels are a delta angle accumulated over one recording sample, not a
        // rate - RecordingSamplingInterval must match ParticipantRecorder's sampling interval,
        // so both are locked to the same ZarrDataContract constant.
        private const float RecordingSamplingInterval = ZarrDataContract.SamplingIntervalSeconds;
        private const float HeuristicMaxTurnRateDegPerSec = 120.0f;

        // Sanity bounds on the raw model output. Originally taken from the human-only recorded
        // training data (20 episodes: forward 0.0-1.56 m/s, yaw/pitch deltas within
        // +-37.1/+-28.2 deg per RecordingSamplingInterval), with a little headroom. Without this
        // an undercalibrated model can output forward speeds many times faster than any recorded
        // human walking speed, which is what was causing the agent to charge straight into walls
        // instead of slowing/turning through openings, and occasionally tunnel through thin
        // walls in a single frame.
        // Forward is clamped symmetrically (+-MaxModelForwardMetersPerSec), not floored at 0:
        // later training data (synthetic NavMeshAgent + expanded recordings) is not restricted
        // to walking-forward-only, so a negative prediction is a legitimate "back away" action,
        // not noise to be discarded.
        private const float MaxModelForwardMetersPerSec = 2.0f;
        private const float MaxModelRawYawDegPerSample = 45.0f;
        private const float MaxModelRawPitchDegPerSample = 35.0f;

        // Below this forward SPEED (m/s, not per-frame distance) the agent is considered
        // standing still. Framerate-independent, unlike the old per-frame moveAmount check.
        private const float MinForwardSpeedMetersPerSec = 0.01f;

        // Discrete-head models (train_2DCNN_discrete.py, ZarrDataContract.DiscreteHead) are detected from the output size.
        // Walk/stop is a key press like the humans' W key (ramped through the recording body's axis sensitivity), and
        // yaw is decoded by committing to one side first, so "turn left" and "turn right" can no longer average to 0.
        [Header("Discrete Head Decoding")]
        [Tooltip("Walk while P(walk) is at least this.")]
        [Range(0.05f, 0.95f)] public float walkProbabilityThreshold = 0.5f;
        [Tooltip("The side (left/straight/right) chosen last tick is kept unless another side's probability beats it by more " +
                 "than this, so a near-tie at a junction can't flip the turn direction every 0.1 s.")]
        [Range(0f, 0.5f)] public float yawSideHysteresis = 0.1f;
        private bool discreteHead = false;
        private int lastYawSide = 0;
        private bool walkKeyDown = false;
        private float walkAxis = 0f;

        private Model runtimeModel;
        private Worker inferenceEngine;
        private bool isModelLoaded = false;
        private object modelLock = new object();

        [Header("Model hot-reload")]
        [Tooltip("ONNX file watched for a newer write time; left empty, RetrainPipeline's ImitationAgent_discrete_latest.onnx. " +
                 "When it changes (a finished retrain), the agent swaps to it. Editor only (it is loaded as the re-imported ModelAsset).")]
        public string OutputOnnxModelPath = "";
        [Tooltip("Off: the agent keeps BaselineModelAsset for its whole life. EvalHarness turns this off so an eval run never " +
                 "switches models mid-run.")]
        public bool hotReloadModel = true;
        private const float HotReloadCheckIntervalSeconds = 1f;
        private float nextHotReloadCheck;
        private DateTime lastModelWriteTime;

        // Trajectory Mechanics
        public List<Vector3> trajectory;
        private float displacementInterval = 10.0f;
        private float displacement = 0.1f;
        public float lastDisplacementTime;
        private float displacementDelta = 0.1f;
        // The deadlock nudge teleports the agent to the nearest NavMesh point with no wall check, so it can
        // hop across thin walls. EvalHarness turns it off so a stuck agent times out instead of teleporting.
        [HideInInspector] public bool enableDeadlockNudge = true;

        // Inference Inputs/Outputs
        public Vector3 localGoalVector3D;
        private float localGoalDistanceMeters = 0f;
        private float networkForwardAction = 0f;
        private float networkYawAction = 0f;
        private float networkPitchAction = 0f;

        // Diagnostic Throttling
        [Header("Debug Settings")]
        [Tooltip("Log neural network outputs every N frames to avoid flooding the console.")]
        public int DiagnosticLogInterval = 60;
        private bool hasLoggedModelFallbackWarning = false;

        void Start()
        {
            if (task == null)
            {
                Debug.LogError($"[ImitationAgent Debug] CRITICAL: 'TaskScript task' field is NOT assigned on {gameObject.name}!");
                return;
            }

            agentSize = task.agentSize;
            agentSpeed = task.agentSpeed;
            if (Mathf.Abs(agentSpeed - ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond) > 0.01f)
            {
                Debug.LogWarning($"[ImitationAgent Start] task.agentSpeed={agentSpeed} differs from the recording walk speed " +
                    $"({ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond} m/s); the discrete head and heuristic walk at the recording speed.");
            }

            // Dummy-proof precondition: refuse to run rather than silently feed the model
            // depth values raycast against the wrong set of colliders. Mirrors
            // ParticipantRecorder's identical hard-fail check.
            LayerMask requiredMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);
            if (ArchitectureLayerMask.value != requiredMask.value)
            {
                Debug.LogError($"[ImitationAgent Debug] CRITICAL: ArchitectureLayerMask must equal exactly the " +
                    $"'{ZarrDataContract.ArchitectureObstacleLayerName}' layer (mask value {requiredMask.value}), " +
                    $"but is currently set to mask value {ArchitectureLayerMask.value}. Refusing to start - a wrong " +
                    $"mask (including 'Everything') silently corrupts every depth value fed to the model.");
                throw new InvalidOperationException(
                    $"[ImitationAgent] ArchitectureLayerMask ({ArchitectureLayerMask.value}) does not exactly match " +
                    $"required layer '{ZarrDataContract.ArchitectureObstacleLayerName}' ({requiredMask.value}).");
            }

            poiTracker = new PoiProgressTracker(task);

            Debug.Log($"[ImitationAgent Start] Agent Initialized. Setup: POIs={task.pointsOfInterest.Length}, NeedsRequired={task.numberOfNeeds}, Speed={agentSpeed}");

            lastDisplacementTime = Time.realtimeSinceStartup;
            trajectory = new List<Vector3>();

            Vector3 chosenStart = poiTracker.ChooseSpawnPosition();
            NavMeshHit hit;
            if (NavMesh.SamplePosition(chosenStart, out hit, 100.0f, NavMesh.AllAreas))
            {
                transform.position = hit.position;
                Debug.Log($"[ImitationAgent Start] Spawned on NavMesh at {transform.position}");
            }
            else
            {
                transform.position = chosenStart;
                Debug.LogWarning($"[ImitationAgent Start] Could not find NavMesh near start position! Teleported to {transform.position}");
            }

            // Face the first goal at spawn instead of inheriting an arbitrary world rotation.
            // In the recorded human data, the "large reorientation from a bad starting heading"
            // pattern makes up only ~7% of frames (vs. steady walking with small corrections for
            // the rest), so the trained model's turning response is far weaker than what a human
            // shows during that phase. Spawning already facing the goal avoids needing that
            // under-trained large-turn behavior at all.
            if (poiTracker.CurrentTarget != null)
            {
                Vector3 towardGoal = poiTracker.CurrentTarget.position - transform.position;
                towardGoal.y = 0f;
                if (towardGoal.sqrMagnitude > 0.0001f)
                {
                    transform.rotation = Quaternion.LookRotation(towardGoal.normalized, Vector3.up);
                }
            }

            transform.localScale = new Vector3(agentSize, agentSize, agentSize);
            InitializeRecordingBody();

            if (AgentHeadAnchor == null)
            {
                GameObject headGO = new GameObject("ImitationHeadAnchor");
                headGO.transform.SetParent(transform, false);
                // World eye height = ZarrDataContract.CameraHeightAboveNavMeshMeters regardless of agentSize (this
                // transform is scaled by agentSize); the recordings use the same contract height.
                headGO.transform.localPosition = new Vector3(0f, ZarrDataContract.CameraHeightAboveNavMeshMeters / Mathf.Max(0.01f, agentSize), 0f);
                AgentHeadAnchor = headGO.transform;
                Debug.Log($"[ImitationAgent Start] Created default AgentHeadAnchor at local position {AgentHeadAnchor.localPosition}");
            }

            if (string.IsNullOrEmpty(OutputOnnxModelPath))
            {
                OutputOnnxModelPath = RetrainPipeline.LatestModelPath;
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
            // Throws if the scene doesn't have exactly one CanonicalRecordingCamera - no
            // fallback to Camera.main or any other guess is allowed. The same resolution call
            // ParticipantRecorder uses, so both pipelines are guaranteed to copy optical
            // settings from the identical camera.
            Camera canonicalCamera = CanonicalRecordingCamera.ResolveInScene();

            GameObject camGO = new GameObject("Hidden_Imitation_CaptureCamera");
            camGO.transform.SetParent(AgentHeadAnchor, false);
            hiddenCaptureCamera = camGO.AddComponent<Camera>();

            // Mirror ParticipantRecorder.InitializeHardwareReplication: copy from the SAME
            // CanonicalRecordingCamera (FOV, clip planes, HDR, culling mask, post-processing),
            // then only override clear flags/background exactly as the recorder does.
            hiddenCaptureCamera.CopyFrom(canonicalCamera);
            // CopyFrom also copies the source camera's transform. ParticipantRecorder gets away
            // with this because it copies from its own direct parent (a no-op offset-wise); here
            // the canonical camera is a different object entirely, so pin this camera back to the
            // agent's head instead of wherever the canonical camera happens to be.
            camGO.transform.localPosition = Vector3.zero;
            camGO.transform.localRotation = Quaternion.identity;

            hiddenCaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            hiddenCaptureCamera.backgroundColor = Color.black;

            colorRenderTexture = new RenderTexture(ImageWidth, ImageHeight, 24, RenderTextureFormat.ARGB32);
            colorRenderTexture.Create();
            hiddenCaptureCamera.targetTexture = colorRenderTexture;
            // FOV and aspect come from the contract, not from the canonical camera, and the depth rays are cast
            // through this camera's pixel centres (ContractCamera), exactly as ParticipantRecorder does.
            ContractCamera.Configure(hiddenCaptureCamera);
            colorReadbackTex = new Texture2D(ImageWidth, ImageHeight, TextureFormat.RGB24, false);

            Debug.Log($"[Sensory Hardware] Visual Camera initialized ({ImageWidth}x{ImageHeight} render target).");
        }

        /// <summary>
        /// Gives the agent FPSController.prefab's CharacterController. The capsule stands on the floor collider, but the pivot
        /// must stay at NavMesh height (the NavMesh is baked a few cm above the floor), because the head anchor sits
        /// CameraHeightAboveNavMeshMeters above the pivot. So the capsule is raised by half its height plus the skin width,
        /// minus the NavMesh-to-floor gap measured under the spawn point. Radius, height and center are in local units, hence
        /// divided by agentSize (the transform is scaled by it). Call with the transform already on the NavMesh.
        /// </summary>
        private void InitializeRecordingBody()
        {
            float navMeshAboveFloor = 0f;
            if (Physics.Raycast(transform.position + Vector3.up * 0.5f, Vector3.down, out RaycastHit floor, 1.5f, Physics.AllLayers, QueryTriggerInteraction.Ignore)
                && !floor.transform.IsChildOf(transform))
            {
                navMeshAboveFloor = Mathf.Clamp(transform.position.y - floor.point.y, -0.3f, 0.3f);
            }

            body = GetComponent<CharacterController>();
            if (body == null) body = gameObject.AddComponent<CharacterController>();
            float s = Mathf.Max(0.01f, agentSize);
            body.radius = ZarrDataContract.RecordingBody.Radius / s;
            body.height = ZarrDataContract.RecordingBody.Height / s;
            body.center = new Vector3(0f, (ZarrDataContract.RecordingBody.Height * 0.5f + ZarrDataContract.RecordingBody.SkinWidth - navMeshAboveFloor) / s, 0f);
            body.stepOffset = ZarrDataContract.RecordingBody.StepOffset;
            body.slopeLimit = ZarrDataContract.RecordingBody.SlopeLimitDegrees;
            body.skinWidth = ZarrDataContract.RecordingBody.SkinWidth;
            bodyVerticalVelocity = 0f;
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
            if (hotReloadModel && Time.unscaledTime >= nextHotReloadCheck)
            {
                nextHotReloadCheck = Time.unscaledTime + HotReloadCheckIntervalSeconds;
                CheckForPendingModelHotSwaps();
            }

            trajectory.Add(transform.position);
            if (trajectory.Count > 1) DisplaceDeadlocks();

            poiTracker.Tick(transform.position);
            if (poiTracker.IsFinished)
            {
                destroyRequest = true;
            }

            // Decoupled from Update()'s framerate: observations are only captured once every
            // ZarrDataContract.SamplingIntervalSeconds, so the 3-frame temporal window spans
            // the same fixed ~0.3s of real time the recorded training data's window does,
            // instead of however many frames the engine happens to render in that time.
            // ApplyRotation (every Update) and MoveRecordingBody (every FixedUpdate) apply the most recently
            // computed action between capture ticks.
            captureAccumulator += Time.deltaTime;
            if (captureAccumulator >= ZarrDataContract.SamplingIntervalSeconds)
            {
                captureAccumulator -= ZarrDataContract.SamplingIntervalSeconds;
                CaptureCurrentObservations();

                if (stateHistoryQueue.Count == TemporalWindowSize)
                {
                    ExecuteNeuralNetworkInference();
                }
            }

            if (stateHistoryQueue.Count < TemporalWindowSize && Time.frameCount % DiagnosticLogInterval == 0)
            {
                Debug.Log($"[Temporal Buffer] Filling observation window: {stateHistoryQueue.Count}/{TemporalWindowSize} frames ready.");
            }

            if (discreteHead)
            {
                // The walk key goes through the same axis ramp as the humans' W key (1/3 s to full speed, instant release).
                walkAxis = walkKeyDown
                    ? Mathf.Min(1f, walkAxis + ZarrDataContract.RecordingBody.WalkAxisSensitivity * Time.deltaTime)
                    : Mathf.Max(0f, walkAxis - ZarrDataContract.RecordingBody.WalkAxisGravity * Time.deltaTime);
                networkForwardAction = walkAxis * ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond;
            }

            Vector3 executed = new Vector3(networkForwardAction, networkYawAction, networkPitchAction);
            if (ExecutedActionOverride != null) executed = ExecutedActionOverride(executed);
            executedForwardMetersPerSec = Mathf.Clamp(executed.x, -MaxModelForwardMetersPerSec, MaxModelForwardMetersPerSec);
            ApplyRotation(executed.y, executed.z);
        }

        /// <summary>Translation, as FirstPersonController.FixedUpdate does it (50 Hz, CharacterController.Move).</summary>
        void FixedUpdate()
        {
            if (body == null || !body.enabled) return;
            MoveRecordingBody(executedForwardMetersPerSec);
        }

        private void CaptureCurrentObservations()
        {
            if (poiTracker.CurrentTarget == null)
            {
                localGoalVector3D = Vector3.zero;
                localGoalDistanceMeters = 0f;
                if (Time.frameCount % DiagnosticLogInterval == 0)
                {
                    Debug.LogWarning("[Observation Pipeline] poiTracker.CurrentTarget is NULL! localGoalVector3D set to Vector3.zero.");
                }
                return;
            }

            Vector3 worldGoalOffset = poiTracker.CurrentTarget.position - AgentHeadAnchor.position;
            localGoalVector3D = AgentHeadAnchor.InverseTransformDirection(worldGoalOffset.normalized);
            localGoalDistanceMeters = worldGoalOffset.magnitude;

            hiddenCaptureCamera.Render();
            RenderTexture.active = colorRenderTexture;
            colorReadbackTex.ReadPixels(new Rect(0, 0, ImageWidth, ImageHeight), 0, 0);
            colorReadbackTex.Apply();
            Color32[] colorPixels = colorReadbackTex.GetPixels32();

            ContractCamera.FillDepthRays(hiddenCaptureCamera, rayCommands, ArchitectureLayerMask);

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

        /// <summary>
        /// Converts a raw goal distance (meters) into the same bounded, resolution-favoring-close-range
        /// proximity signal fed to the network on both sides - see SingleEpisodeDataset.__getitem__'s
        /// identical transform in dataset.py. Raw distance is recorded/passed
        /// around unconverted so this transform can change later without re-recording.
        /// </summary>
        public static float GoalDistanceToProximity(float goalDistanceMeters)
        {
            return 1f / (1f + Mathf.Max(0f, goalDistanceMeters));
        }

        /// <summary>
        /// Flattens a goal vector + distance + temporal window of RGBD frames into the exact
        /// "sensory_inputs" layout the ONNX model was trained on: [goal.x, goal.y, goal.z,
        /// proximity(goal_distance), then per frame (oldest to newest) per pixel (row-major)
        /// R/255, G/255, B/255, depth]. This must byte-for-byte match
        /// SingleEpisodeDataset.__getitem__ in dataset.py - extracted as
        /// its own static, side-effect-free method (rather than being inlined in
        /// ExecuteNeuralNetworkInference) specifically so SensoryTensorParityTest can call it
        /// directly with synthetic frame data and compare it against the Python training
        /// pipeline's output on the same input, without needing a live camera/raycast/Sentis engine.
        /// </summary>
        public static float[] BuildSensoryTensor(Vector3 goal, float goalDistanceMeters, IEnumerable<SerializableFrameData> frameHistory)
        {
            int totalSensoryInputs = (TemporalWindowSize * TotalPixels * 4) + 4;
            float[] flattenedFeatureArray = new float[totalSensoryInputs];
            int ptr = 0;

            flattenedFeatureArray[ptr++] = goal.x;
            flattenedFeatureArray[ptr++] = goal.y;
            flattenedFeatureArray[ptr++] = goal.z;
            flattenedFeatureArray[ptr++] = GoalDistanceToProximity(goalDistanceMeters);

            foreach (var frame in frameHistory)
            {
                for (int i = 0; i < TotalPixels; i++)
                {
                    flattenedFeatureArray[ptr++] = frame.colorMatrix[i].r / 255f;
                    flattenedFeatureArray[ptr++] = frame.colorMatrix[i].g / 255f;
                    flattenedFeatureArray[ptr++] = frame.colorMatrix[i].b / 255f;
                    flattenedFeatureArray[ptr++] = frame.depthMatrix[i];
                }
            }

            return flattenedFeatureArray;
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

                    if (poiTracker.CurrentTarget != null && !poiTracker.IsFinished)
                    {
                        // forward/yaw/pitch are rates (m/s, deg/s, deg/s) - see ApplyRotation/MoveRecordingBody
                        networkForwardAction = ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond;
                        networkYawAction = Mathf.Clamp(localGoalVector3D.x * 2.0f, -1f, 1f) * HeuristicMaxTurnRateDegPerSec;
                        networkPitchAction = Mathf.Clamp(localGoalVector3D.y * 2.0f, -1f, 1f) * HeuristicMaxTurnRateDegPerSec;
                    }
                    
                    if (Time.frameCount % DiagnosticLogInterval == 0)
                    {
                        Debug.Log($"[Heuristic Control Output] Forward: {networkForwardAction:F2}, Yaw: {networkYawAction:F2}, Pitch: {networkPitchAction:F2} | Goal Vector: {localGoalVector3D.ToString("F2")}");
                    }
                    return;
                }

                float[] flattenedFeatureArray = BuildSensoryTensor(localGoalVector3D, localGoalDistanceMeters, stateHistoryQueue);

                // TEMPORARY DIAGNOSTIC: compare against training data's R-channel mean (~0.15) to
                // check for a live-vs-recorded brightness/exposure mismatch. Safe to remove once resolved.
                if (Time.frameCount % DiagnosticLogInterval == 0)
                {
                    float rSum = 0f, rMin = 1f, rMax = 0f;
                    int rCount = 0;
                    for (int k = 4; k < flattenedFeatureArray.Length; k += 4)
                    {
                        float v = flattenedFeatureArray[k];
                        rSum += v;
                        if (v < rMin) rMin = v;
                        if (v > rMax) rMax = v;
                        rCount++;
                    }
                    Debug.Log($"[RGB DEBUG] mean={rSum / rCount:F4} min={rMin:F4} max={rMax:F4}");
                }

                TensorShape tensorShape = new TensorShape(1, flattenedFeatureArray.Length);
                using (Tensor<float> sensoryTensor = new Tensor<float>(tensorShape, flattenedFeatureArray))
                {
                    inferenceEngine.SetInput("sensory_inputs", sensoryTensor);
                    inferenceEngine.Schedule();
                    
                    var outputTensor = inferenceEngine.PeekOutput("motor_actions") as Tensor<float>;
                    if (outputTensor != null)
                    {
                        float[] outputData = outputTensor.DownloadToArray();
                        if (outputData.Length == ZarrDataContract.DiscreteHead.OutputSize)
                        {
                            ApplyDiscreteHeadOutput(outputData);
                            return;
                        }
                        discreteHead = false;
                        // forward is already m/s (matches ParticipantRecorder action_vel.z); yaw/pitch are
                        // degrees accumulated over one RecordingSamplingInterval, so convert to deg/s here.
                        // Clamp to the range the model was actually trained on before doing anything else
                        // with it - an out-of-range raw prediction is what let the agent dash into (and
                        // occasionally tunnel through) walls instead of walking/turning at a human pace.
                        // Symmetric like yaw/pitch below, NOT floored at 0f: flooring negative predictions
                        // to exactly 0 (instead of letting the agent back away) was what turned a model
                        // that wanted to reverse away from a wall/corner into a permanently frozen agent -
                        // see the CentralPillar_Code_134868 eval run where raw_fwd sat at -0.57 for the
                        // entire episode and MoveAgentLocomotion's "standing still" check fired every tick.
                        float rawForward = Mathf.Clamp(outputData[0], -MaxModelForwardMetersPerSec, MaxModelForwardMetersPerSec);
                        float rawYaw = Mathf.Clamp(outputData[1], -MaxModelRawYawDegPerSample, MaxModelRawYawDegPerSample);
                        float rawPitch = Mathf.Clamp(outputData[2], -MaxModelRawPitchDegPerSample, MaxModelRawPitchDegPerSample);

                        networkForwardAction = rawForward;
                        networkYawAction     = rawYaw / RecordingSamplingInterval;
                        networkPitchAction   = rawPitch / RecordingSamplingInterval;

                        LastRawForward = outputData[0];
                        LastRawYawPerSample = outputData[1];
                        LastRawPitch = outputData[2];
                        LastCommandedForwardMps = networkForwardAction;
                        LastCommandedYawDegPerSecond = networkYawAction;

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

        /// <summary>
        /// Discrete head: [walk_logit, yaw_logit_0..6, pitch_deg_per_sample] (ZarrDataContract.DiscreteHead). Walk/stop sets
        /// the walk key (ramped in Update); yaw commits to one side before averaging (DecodeDiscreteYaw).
        /// </summary>
        private void ApplyDiscreteHeadOutput(float[] outputData)
        {
            if (!discreteHead)
            {
                discreteHead = true;
                Debug.Log("[Neural Inference Core] Discrete walk/yaw head detected (" + outputData.Length + " outputs).");
            }
            float pWalk = 1f / (1f + Mathf.Exp(-outputData[0]));
            walkKeyDown = pWalk >= walkProbabilityThreshold;
            float yawPerSample = DecodeDiscreteYaw(outputData, 1, ref lastYawSide, yawSideHysteresis, out Vector3 sideProbabilities);
            float rawPitch = Mathf.Clamp(outputData[ZarrDataContract.DiscreteHead.OutputSize - 1], -MaxModelRawPitchDegPerSample, MaxModelRawPitchDegPerSample);

            networkYawAction = yawPerSample / RecordingSamplingInterval;
            networkPitchAction = rawPitch / RecordingSamplingInterval;

            LastRawForward = pWalk;
            LastRawYawPerSample = yawPerSample;
            LastRawPitch = rawPitch;
            LastCommandedForwardMps = walkKeyDown ? ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond : 0f;
            LastCommandedYawDegPerSecond = networkYawAction;

            if (Time.frameCount % DiagnosticLogInterval == 0)
            {
                Debug.Log($"[Neural Model Output] P(walk) {pWalk:F2} | P(left/straight/right) {sideProbabilities.x:F2}/{sideProbabilities.y:F2}/{sideProbabilities.z:F2} " +
                          $"-> yaw {yawPerSample:F2} deg/sample | Local Goal: {localGoalVector3D.ToString("F2")}");
            }
        }

        /// <summary>
        /// Softmax over the 7 yaw logits at output[offset..], then: pick the side (left / straight / right) with the largest
        /// summed probability - keeping lastSide unless another side beats it by more than hysteresis - and return the
        /// probability-weighted mean of that side's bin centers (deg per sampling interval; 0 for straight). Averaging only
        /// within one side is what keeps a left/right split from cancelling to ~0 in front of a wall. Mirrors
        /// decode_yaw in train_2DCNN_discrete.py (which has no hysteresis). sideProbabilities = (left, straight, right).
        /// </summary>
        public static float DecodeDiscreteYaw(float[] output, int offset, ref int lastSide, float hysteresis, out Vector3 sideProbabilities)
        {
            int n = ZarrDataContract.DiscreteHead.YawClassCount;
            int straight = ZarrDataContract.DiscreteHead.StraightClass;
            float max = float.NegativeInfinity;
            for (int c = 0; c < n; c++) max = Mathf.Max(max, output[offset + c]);
            float[] p = new float[n];
            float sum = 0f;
            for (int c = 0; c < n; c++) { p[c] = Mathf.Exp(output[offset + c] - max); sum += p[c]; }
            for (int c = 0; c < n; c++) p[c] /= sum;

            float left = 0f, right = 0f, leftValue = 0f, rightValue = 0f;
            for (int c = 0; c < n; c++)
            {
                int k = c - straight;
                if (k == 0) continue;
                float center = Mathf.Sign(k) * ZarrDataContract.DiscreteHead.YawBinCentersDeg[Mathf.Abs(k)];
                if (k < 0) { left += p[c]; leftValue += p[c] * center; }
                else { right += p[c]; rightValue += p[c] * center; }
            }
            sideProbabilities = new Vector3(left, p[straight], right);

            float SideP(int side) => side < 0 ? left : side > 0 ? right : p[straight];
            int best = left >= right && left >= p[straight] ? -1 : right >= p[straight] ? 1 : 0;
            int chosen = SideP(best) > SideP(lastSide) + hysteresis ? best : lastSide;
            lastSide = chosen;
            if (chosen < 0) return leftValue / Mathf.Max(1e-8f, left);
            if (chosen > 0) return rightValue / Mathf.Max(1e-8f, right);
            return 0f;
        }

        private void ApplyRotation(float yaw, float pitch)
        {
            // yaw/pitch are rates (deg/s). Yaw turns the body like MouseLook; pitch tilts only the head.
            transform.Rotate(0f, yaw * Time.deltaTime, 0f, Space.Self);

            float targetPitch = AgentHeadAnchor.localEulerAngles.x + (-pitch * Time.deltaTime);
            if (targetPitch > 180f) targetPitch -= 360f;
            targetPitch = Mathf.Clamp(targetPitch, -45f, 45f);
            AgentHeadAnchor.localEulerAngles = new Vector3(targetPitch, 0f, 0f);
        }

        /// <summary>
        /// FirstPersonController.FixedUpdate for a forward-only input: move along the ground normal at the given speed, stick
        /// to the ground when grounded, fall otherwise, and let the CharacterController slide along walls. A side contact
        /// while trying to move counts as a collision (rising edge only, as before).
        /// </summary>
        private void MoveRecordingBody(float forward)
        {
            float dt = Time.fixedDeltaTime;
            Vector3 desired = transform.forward;
            Vector3 capsuleCenter = transform.TransformPoint(body.center);
            if (Physics.SphereCast(capsuleCenter, ZarrDataContract.RecordingBody.Radius, Vector3.down, out RaycastHit ground,
                    ZarrDataContract.RecordingBody.Height * 0.5f, Physics.AllLayers, QueryTriggerInteraction.Ignore))
            {
                desired = Vector3.ProjectOnPlane(desired, ground.normal).normalized;
            }

            if (body.isGrounded) bodyVerticalVelocity = -ZarrDataContract.RecordingBody.StickToGroundForce;
            else bodyVerticalVelocity += Physics.gravity.y * ZarrDataContract.RecordingBody.GravityMultiplier * dt;

            Vector3 velocity = desired * forward;
            velocity.y = bodyVerticalVelocity;
            CollisionFlags flags = body.Move(velocity * dt);

            bool blocked = (flags & CollisionFlags.Sides) != 0 && Mathf.Abs(forward) >= MinForwardSpeedMetersPerSec;
            if (blocked && !wasCollidingLastFrame)
            {
                CollisionCount++;
                if (!HasCollided)
                {
                    HasCollided = true;
                    FirstCollisionRealtime = Time.realtimeSinceStartup;
                }
            }
            wasCollidingLastFrame = blocked;
        }

        /// <summary>
        /// Swaps to OutputOnnxModelPath once its write time is newer than at Start or the last swap. RetrainPipeline copies a
        /// finished retrain there and re-imports it on the main thread, so the ModelAsset is current by the time this runs.
        /// ModelLoader.Load(path) only reads .sentis files, hence the detour through the asset in the Editor.
        /// </summary>
        private void CheckForPendingModelHotSwaps()
        {
            try
            {
                if (!File.Exists(OutputOnnxModelPath)) return;
                DateTime currentWriteTime = File.GetLastWriteTime(OutputOnnxModelPath);
                if (currentWriteTime <= lastModelWriteTime) return;
                lastModelWriteTime = currentWriteTime;
#if UNITY_EDITOR
                string assetPath = "Assets" + Path.GetFullPath(OutputOnnxModelPath).Substring(Path.GetFullPath(Application.dataPath).Length).Replace('\\', '/');
                ModelAsset asset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>(assetPath);
                if (asset == null)
                {
                    Debug.LogError($"[Hot-Swap Failure] '{OutputOnnxModelPath}' changed, but no ModelAsset at '{assetPath}' (must be under Assets/).");
                    return;
                }
                lock (modelLock)
                {
                    inferenceEngine?.Dispose();
                    BaselineModelAsset = asset;
                    runtimeModel = ModelLoader.Load(asset);
                    inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute);
                    isModelLoaded = true;
                    hasLoggedModelFallbackWarning = false;
                }
                Debug.Log($"[HOT-SWAP SUCCESS] {name} now runs {assetPath} (written {currentWriteTime:HH:mm:ss}).");
#else
                Debug.LogWarning($"[Hot-Swap] '{OutputOnnxModelPath}' changed, but a player build cannot import ONNX at run time - restart with the new model.");
#endif
            }
            catch (IOException) { }
            catch (Exception ex)
            {
                Debug.LogError($"[Hot-Swap Failure] Loading failed: {ex.Message}");
            }
        }

        private void DisplaceDeadlocks()
        {
            if (!enableDeadlockNudge) return;
            if (Time.realtimeSinceStartup >= lastDisplacementTime + displacementInterval)
            {
                float deltaMoved = Vector3.Distance(trajectory[trajectory.Count - 2], trajectory[trajectory.Count - 1]);
                if (deltaMoved < displacementDelta && !poiTracker.IsFinished)
                {
                    lastDisplacementTime = Time.realtimeSinceStartup;
                    Vector3 targetPos = transform.position + new Vector3(UnityEngine.Random.value * displacement, 0f, UnityEngine.Random.value * displacement);
                    StuckCount++;

                    Debug.LogWarning($"[Deadlock Triggered] Agent moved only {deltaMoved:F4}m in target direction over {displacementInterval}s. Applying deadlock nudge towards {targetPos}.");

                    NavMeshHit hit;
                    if (NavMesh.SamplePosition(targetPos, out hit, 1.0f, NavMesh.AllAreas))
                    {
                        if (body != null) body.enabled = false;
                        transform.position = hit.position;
                        if (body != null) body.enabled = true;
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