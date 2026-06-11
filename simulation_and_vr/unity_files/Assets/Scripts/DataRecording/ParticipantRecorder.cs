using UnityEngine;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using UnityTools.Core;

namespace Assets.Scripts.DataRecording
{
    public class ParticipantRecorder : MonoBehaviour
    {
        [Header("Linked Framework Components")]
        public Camera MainVRCamera;
        public LayerMask ArchitectureLayerMask;

        [Header("Target Tracking (POIs)")]
        public Transform CurrentPOITarget;

        [Header("Sampling & Storage Settings")]
        public string SaveDirectoryName = "VR_Recordings";
        public float SamplingInterval = 0.1f; // 10Hz
        public float MaxRayDistance = 50f;
        public bool EnableDebugRays = true; // Enabled by default for edge testing
        public bool AutoStartForTesting = false;

        [Header("Custom FOV Controls (Fixes Matrix Collapse)")]
        [Tooltip("The manual horizontal FOV cone angle for the ray beam.")]
        public float HorizonalFovOverride = 90f;
        [Tooltip("The manual vertical FOV cone angle for the ray beam.")]
        public float VerticalFovOverride = 90f;

        [Header("Temporal Windowing")]
        public int TemporalWindowSize = 3;

        private const int ImageWidth = 64;
        private const int ImageHeight = 64;
        private const int TotalPixels = ImageWidth * ImageHeight;

        private bool isRecording;
        private float startTime;
        private Vector3 lastPosition;
        private string currentCsvPath;
        private StreamWriter csvWriter;

        private Camera hiddenCaptureCamera;
        private RenderTexture colorRenderTexture;
        private Texture2D colorReadbackTex;

        private NativeArray<RaycastCommand> rayCommands;
        private NativeArray<RaycastHit> rayResults;
        private float[] currentRayBuffer;

        private Queue<string> stateHistoryQueue = new Queue<string>();
        private float Azimuth;
        private float Elevation;

        private void Start()
        {
            if (AutoStartForTesting)
            {
                StartRecording();
            }
        }

        public void StartRecording()
        {
            if (isRecording) return;

            startTime = Time.time;
            lastPosition = MainVRCamera.transform.position;
            stateHistoryQueue.Clear();

            InitializeHardwareReplication();
            InitializeJobBuffers();
            InitializeCsvFile();

            isRecording = true;
            StartCoroutine(LockedSamplingLoop());
        }

        public void StopRecording()
        {
            if (!isRecording) return;

            isRecording = false;
            StopAllCoroutines();

            if (csvWriter != null)
            {
                csvWriter.Flush();
                csvWriter.Close();
                csvWriter = null;
                
                // --------------------------------------------------------------------------
                // LIFECYCLE AUTOMATION TRIGGER HOOK
                // --------------------------------------------------------------------------
                // The file is officially saved and closed. We can now pass it safely 
                // to our local Python pipeline script without data collisions!
                TriggerPythonPipeline(currentCsvPath);
            }

            if (rayCommands.IsCreated) rayCommands.Dispose();
            if (rayResults.IsCreated) rayResults.Dispose();

            if (hiddenCaptureCamera != null) Destroy(hiddenCaptureCamera.gameObject);
            if (colorRenderTexture != null) colorRenderTexture.Release();
            if (colorReadbackTex != null) Destroy(colorReadbackTex);
        }

        /// <summary>
        /// Spawns an asynchronous background worker thread to execute the Python script.
        /// This prevents your VR scene or active agent simulation from freezing during training.
        /// </summary>
        private void TriggerPythonPipeline(string savedCsvPath)
        {
            // Define your project file structure configurations here
            string pythonExe = "python"; // or path to your custom miniconda/venv binary
            string pythonScript = "C:/Project/ML/incremental_train.py";
            string targetOnnxOutput = "C:/Project/Assets/Models/ImitationAgentModel.onnx";

            Debug.Log($"[Lifecycle Engine] Human data finalized. Launching training pipeline for: {savedCsvPath}");

            System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.FileName = pythonExe;
                    // Passes the explicit, newly generated file string as an argument
                    startInfo.Arguments = $"\"{pythonScript}\" --csv_input \"{savedCsvPath}\" --model_output \"{targetOnnxOutput}\"";
                    startInfo.UseShellExecute = false;
                    startInfo.RedirectStandardOutput = true;
                    startInfo.RedirectStandardError = true;
                    startInfo.CreateNoWindow = true;

                    using (System.Diagnostics.Process process = System.Diagnostics.Process.Start(startInfo))
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        string error = process.StandardError.ReadToEnd();
                        process.WaitForExit();

                        if (process.ExitCode == 0)
                        {
                            Debug.Log($"[Lifecycle Engine] SUCCESS: Model fine-tuned and exported back to: {targetOnnxOutput}");
                        }
                        else
                        {
                            Debug.LogError($"[Lifecycle Engine] Python Error: {error}");
                        }
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"[Lifecycle Engine] Pipeline Exception: {ex.Message}");
                }
            });
        }

        private void InitializeHardwareReplication()
        {
            GameObject camGO = new GameObject("Hidden_BC_CaptureCamera");
            camGO.transform.SetParent(MainVRCamera.transform, false);
            hiddenCaptureCamera = camGO.AddComponent<Camera>();
            
            hiddenCaptureCamera.CopyFrom(MainVRCamera);
            hiddenCaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            hiddenCaptureCamera.backgroundColor = Color.black;

            colorRenderTexture = new RenderTexture(ImageWidth, ImageHeight, 24, RenderTextureFormat.ARGB32);
            colorRenderTexture.Create();
            hiddenCaptureCamera.targetTexture = colorRenderTexture;

            colorReadbackTex = new Texture2D(ImageWidth, ImageHeight, TextureFormat.RGB24, false);
        }

        private void InitializeJobBuffers()
        {
            rayCommands = new NativeArray<RaycastCommand>(TotalPixels, Allocator.Persistent);
            rayResults = new NativeArray<RaycastHit>(TotalPixels, Allocator.Persistent);
            currentRayBuffer = new float[TotalPixels];
        }

        private void InitializeCsvFile()
        {
            string dirPath = Path.Combine(Application.dataPath, SaveDirectoryName);
            if (!Directory.Exists(dirPath)) Directory.CreateDirectory(dirPath);

            string filename = $"BC_Run_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv";
            currentCsvPath = Path.Combine(dirPath, filename);
            csvWriter = new StreamWriter(currentCsvPath, false, Encoding.UTF8);

            StringBuilder header = new StringBuilder();
            header.Append("Timestamp,Pos_X,Pos_Y,Pos_Z,Goal_Dir_X,Goal_Dir_Y,Goal_Dir_Z,");
            header.Append("Action_Vel_X,Action_Vel_Y,Action_Vel_Z,Action_DeltaRot_X,Action_DeltaRot_Y,Action_DeltaRot_Z,");

            for (int w = 0; w < TemporalWindowSize; w++)
            {
                for (int p = 0; p < TotalPixels; p++)
                {
                    header.Append($"F{w}_P{p}_R,F{w}_P{p}_G,F{w}_P{p}_B,F{w}_P{p}_Depth,");
                }
            }
            header.Length--; 
            csvWriter.WriteLine(header.ToString());
        }

        private IEnumerator LockedSamplingLoop()
        {
            while (isRecording)
            {
                yield return new WaitForSeconds(SamplingInterval);

                if (MainVRCamera == null) continue;

                Transform hTransform = MainVRCamera.transform;
                Vector3 currentPosition = hTransform.position;
                float timeStamp = Time.time - startTime;

                // --- GOAL SECTOR ---
                Vector3 localGoalVector = Vector3.zero;
                if (CurrentPOITarget != null)
                {
                    Vector3 worldGoalDir = (CurrentPOITarget.position - currentPosition).normalized;
                    localGoalVector = hTransform.InverseTransformDirection(worldGoalDir); 
                }

                // --- IMAGE CAPTURE ---
                hiddenCaptureCamera.Render();
                RenderTexture.active = colorRenderTexture;
                colorReadbackTex.ReadPixels(new Rect(0, 0, ImageWidth, ImageHeight), 0, 0);
                colorReadbackTex.Apply();
                Color32[] colorPixels = colorReadbackTex.GetPixels32();

                // --- TRIGONOMETRIC MATHEMATICAL CONE CONSTRUCT ---
                Vector3 origin = hTransform.position + hTransform.forward * 0.1f;
                int index = 0;

                // Convert overridden parameters directly to radians safely
                float fovRadH = HorizonalFovOverride * Mathf.Deg2Rad;
                float fovRadV = VerticalFovOverride * Mathf.Deg2Rad;

                // Track extreme index coordinates for custom boundary debugging outputs
                int indexTopLeft = 0;
                int indexTopRight = ImageWidth - 1;
                int indexBottomLeft = (ImageHeight - 1) * ImageWidth;
                int indexBottomRight = TotalPixels - 1;
                int indexCenter = ((ImageHeight / 2) * ImageWidth) + (ImageWidth / 2);

                for (int y = 0; y < ImageHeight; y++)
                {
                    float vFactor = (ImageHeight > 1) ? ((float)y / (ImageHeight - 1)) - 0.5f : 0f;
                    float vAngle = vFactor * fovRadV;

                    for (int x = 0; x < ImageWidth; x++)
                    {
                        float hFactor = (ImageWidth > 1) ? ((float)x / (ImageWidth - 1)) - 0.5f : 0f;
                        float hAngle = hFactor * fovRadH;

                        Vector3 localDir = new Vector3(Mathf.Tan(hAngle), Mathf.Tan(vAngle), 1.0f).normalized;
                        Vector3 worldDir = hTransform.TransformDirection(localDir);

                        rayCommands[index] = new RaycastCommand(
                            origin, 
                            worldDir, 
                            new QueryParameters(ArchitectureLayerMask, true, QueryTriggerInteraction.Collide, false), 
                            MaxRayDistance
                        );
                        index++;
                    }
                }

                JobHandle handle = RaycastCommand.ScheduleBatch(rayCommands, rayResults, 16);
                handle.Complete();

                for (int i = 0; i < rayResults.Length; i++)
                {
                    float dist = rayResults[i].distance;
                    bool hasHit = dist > 0;
                    currentRayBuffer[i] = hasHit ? dist / MaxRayDistance : 1.0f;

                    // --- ISOLATED OUTSIDE BOUNDARY FRUSTUM DEBUGGER ---
                    if (EnableDebugRays)
                    {
                        if (i == indexTopLeft || i == indexTopRight || i == indexBottomLeft || i == indexBottomRight || i == indexCenter)
                        {
                            float drawDist = hasHit ? dist : MaxRayDistance;
                            Color debugColor = (i == indexCenter) ? Color.blue : (hasHit ? Color.green : Color.red);
                            
                            // Draws a thick, persistent visual pointer along the frame border bounds
                            Debug.DrawRay(origin, rayCommands[i].direction * drawDist, debugColor, SamplingInterval);
                        }
                    }
                }

                // --- ACTION GENERATION ---
                Vector3 worldVelocity = (currentPosition - lastPosition) / SamplingInterval;
                Vector3 localVelocityAction = hTransform.InverseTransformDirection(worldVelocity);
                lastPosition = currentPosition;

                Vector3 currentForward = hTransform.forward;
                Math3D.CartesianToSpherical(currentForward, out float azimuth, out float elevation, out _);
                Vector3 deltaRotationAction = new Vector3(azimuth - Azimuth, elevation - Elevation, 0f);
                Azimuth = azimuth;
                Elevation = elevation;

                // --- SERIALIZE FRAME ---
                StringBuilder frameChannelData = new StringBuilder();
                for (int i = 0; i < TotalPixels; i++)
                {
                    frameChannelData.Append($"{colorPixels[i].r},{colorPixels[i].g},{colorPixels[i].b},{currentRayBuffer[i]:F4},");
                }

                stateHistoryQueue.Enqueue(frameChannelData.ToString());
                if (stateHistoryQueue.Count > TemporalWindowSize)
                {
                    stateHistoryQueue.Dequeue();
                }

                if (stateHistoryQueue.Count == TemporalWindowSize)
                {
                    StringBuilder rowBuilder = new StringBuilder();
                    rowBuilder.Append($"{timeStamp:F3},{currentPosition.x:F3},{currentPosition.y:F3},{currentPosition.z:F3},");
                    rowBuilder.Append($"{localGoalVector.x:F4},{localGoalVector.y:F4},{localGoalVector.z:F4},");
                    rowBuilder.Append($"{localVelocityAction.x:F4},{localVelocityAction.y:F4},{localVelocityAction.z:F4},");
                    rowBuilder.Append($"{deltaRotationAction.x:F4},{deltaRotationAction.y:F4},{deltaRotationAction.z:F4},");

                    foreach (string historicalFrame in stateHistoryQueue)
                    {
                        rowBuilder.Append(historicalFrame);
                    }

                    rowBuilder.Length--; 
                    csvWriter.WriteLine(rowBuilder.ToString());
                }
            }
        }

        private void OnDestroy()
        {
            StopRecording();
        }
    }
}