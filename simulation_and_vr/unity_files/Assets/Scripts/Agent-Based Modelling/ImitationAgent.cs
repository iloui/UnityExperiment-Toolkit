namespace Agent_Based_Modelling
{ 
    /*
    DesignMind2: A Toolkit for Evidence-Based, Cognitively-Informed and Human-Centered Architectural Design
    Adaptive Simulation Agents Extension - Automated Machine Learning Lifecycle Engine
    */

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
        private object modelLock = new object(); // Structural lock safeguarding background hot-swapping threads

        [Header("Automated Python Lifecycle Pipeline (Option A)")]
        [Tooltip("Absolute path to your python executable (e.g. C:/miniconda3/envs/ml/python.exe)")]
        public string PythonExecutablePath = "python";
        [Tooltip("Absolute path to your Python fine-tuning script.")]
        public string PythonScriptPath = "C:/Project/ML/incremental_train.py";
        [Tooltip("The path to the newly exported model that Python writes out.")]
        public string OutputOnnxModelPath = "C:/Project/Assets/Models/ImitationAgentModel.onnx";

        private bool isRetrainingBackgroundActive = false;
        private string pendingHotSwapModelPath = "";

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

        void Start()
        {
            // Inheriting baseline architecture attributes
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

            // Warp agent to starting grid location
            GameObject chosenStart = start[UnityEngine.Random.Range(0, start.Length)];
            NavMeshHit hit;
            NavMesh.SamplePosition(chosenStart.transform.position, out hit, 100.0f, NavMesh.AllAreas);
            transform.position = hit.position;
            transform.localScale = new Vector3(agentSize, agentSize, agentSize);

            if (AgentHeadAnchor == null)
            {
                GameObject headGO = new GameObject("ImitationHeadAnchor");
                headGO.transform.SetParent(transform, false);
                headGO.transform.localPosition = new Vector3(0f, agentSize * 1.5f, 0f);
                AgentHeadAnchor = headGO.transform;
            }

            InitializeHardwareReplication();
            InitializeJobBuffers();
            InitializeSentisEngine();
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
            if (BaselineModelAsset != null)
            {
                lock (modelLock)
                {
                    runtimeModel = ModelLoader.Load(BaselineModelAsset);
                    inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute); // Sentis 2.0+ constructor
                    isModelLoaded = true;
                }
            }
            else
            {
                Debug.LogWarning("ImitationAgent: No initial Baseline Model Asset assigned! Running on mock fallback until hot-swapped.");
            }
        }

        void Update()
        {
            trajectory.Add(transform.position);
            if (trajectory.Count > 1) DisplaceDeadlocks();

            EvaluateStateSequencer();
            CaptureCurrentObservations();
            CheckForPendingModelHotSwaps();

            if (stateHistoryQueue.Count == TemporalWindowSize)
            {
                ExecuteNeuralNetworkInference();
            }

            // Apply Motor controls via physical adjustments frame-by-frame
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
                    OnSimulationRunCompleted(); // Trigger automated lifecycle hook
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

                // 1. Construct feature array
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

                // 2. Sentis 2.0+ Tensor Allocation: Allocation uses the new explicit generic class structure.
                TensorShape tensorShape = new TensorShape(1, totalSensoryInputs);
                using (Tensor<float> sensoryTensor = new Tensor<float>(tensorShape, flattenedFeatureArray)) 
                {
                    // 3. Inference Execution: worker.Execute() is now worker.Schedule()
                    inferenceEngine.Schedule(sensoryTensor);
                    
                    // 4. Retrieve Actions Safely via Output download
                    var outputTensor = inferenceEngine.PeekOutput() as Tensor<float>;
                    if (outputTensor != null)
                    {
                        // In Sentis 2.0+, DownloadToArray() downloads from GPU safely and creates/returns an array
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
            float rotationSpeed = 120.0f;
            transform.Rotate(0f, yaw * rotationSpeed * Time.deltaTime, 0f, Space.Self);
            
            float targetPitch = AgentHeadAnchor.localEulerAngles.x + (-pitch * rotationSpeed * Time.deltaTime);
            if (targetPitch > 180f) targetPitch -= 360f;
            targetPitch = Mathf.Clamp(targetPitch, -45f, 45f);
            AgentHeadAnchor.localEulerAngles = new Vector3(targetPitch, 0f, 0f);

            transform.position += transform.forward * forward * agentSpeed * Time.deltaTime;
        }

        private void OnSimulationRunCompleted()
        {
            if (isRetrainingBackgroundActive)
            {
                Debug.LogWarning("ImitationAgent: Python pipeline is already processing a training run. Skipping overlapping execution.");
                return;
            }

            string csvDirectory = Path.Combine(Application.dataPath, "../VR_Recordings");
            if (!Directory.Exists(csvDirectory)) return;

            string[] recordedFiles = Directory.GetFiles(csvDirectory, "*.csv");
            if (recordedFiles.Length == 0) return;

            Array.Sort(recordedFiles, (a, b) => File.GetLastWriteTime(b).CompareTo(File.GetLastWriteTime(a)));
            string newestCsvPath = recordedFiles[0];

            Debug.Log($"ImitationAgent: Starting background pipeline execution... Training on: {newestCsvPath}");
            isRetrainingBackgroundActive = true;

            Task.Run(() =>
            {
                try
                {
                    ProcessStartInfo startInfo = new ProcessStartInfo();
                    startInfo.FileName = PythonExecutablePath;
                    startInfo.Arguments = $"\"{PythonScriptPath}\" --csv_input \"{newestCsvPath}\" --model_output \"{OutputOnnxModelPath}\"";
                    startInfo.UseShellExecute = false;
                    startInfo.RedirectStandardOutput = true;
                    startInfo.RedirectStandardError = true;
                    startInfo.CreateNoWindow = true;

                    using (Process process = Process.Start(startInfo))
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        string error = process.StandardError.ReadToEnd();
                        process.WaitForExit();

                        if (process.ExitCode == 0)
                        {
                            pendingHotSwapModelPath = OutputOnnxModelPath;
                        }
                        else
                        {
                            Debug.LogError($"ImitationAgent Python Pipeline Error: {error}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.LogError($"ImitationAgent Pipeline Exception: {ex.Message}");
                }
                finally
                {
                    isRetrainingBackgroundActive = false;
                }
            });
        }

        private void CheckForPendingModelHotSwaps()
        {
            if (!string.IsNullOrEmpty(pendingHotSwapModelPath))
            {
                if (File.Exists(pendingHotSwapModelPath))
                {
                    lock (modelLock)
                    {
                        if (inferenceEngine != null) inferenceEngine.Dispose();

                        try
                        {
                            runtimeModel = ModelLoader.Load(pendingHotSwapModelPath);
                            inferenceEngine = new Worker(runtimeModel, BackendType.GPUCompute);
                            isModelLoaded = true;
                            Debug.Log($"SUCCESS: Silent hot-swap completed! Path: {pendingHotSwapModelPath}");
                        }
                        catch (Exception ex)
                        {
                            Debug.LogError($"ImitationAgent: Hot-Swap loading failed on main frame boundary: {ex.Message}");
                        }
                    }
                }
                pendingHotSwapModelPath = ""; 
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
                    transform.position += new Vector3(UnityEngine.Random.value * displacement, 0.02f, UnityEngine.Random.value * displacement);
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