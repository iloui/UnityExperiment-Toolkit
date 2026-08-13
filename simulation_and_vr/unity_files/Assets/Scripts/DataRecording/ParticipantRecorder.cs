using UnityEngine;
using System.IO;
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
        private string currentEpisodeDir;
        private string currentManifestPath;
        private int episodeFrameCount;

        private BinaryWriter rgbWriter;
        private BinaryWriter depthWriter;
        private BinaryWriter goalWriter;
        private BinaryWriter poseWriter;
        private BinaryWriter actionVelWriter;
        private BinaryWriter actionRotWriter;
        private BinaryWriter timestampWriter;

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
            episodeFrameCount = 0;

            InitializeHardwareReplication();
            InitializeJobBuffers();
            InitializeEpisodeStorage();

            isRecording = true;
            StartCoroutine(LockedSamplingLoop());
        }

        public void StopRecording()
        {
            if (!isRecording) return;

            isRecording = false;
            StopAllCoroutines();

            CloseBinaryWriters();

            if (!string.IsNullOrEmpty(currentEpisodeDir))
            {
                WriteEpisodeManifest();
                TriggerPythonPipeline(currentEpisodeDir);
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
        private void TriggerPythonPipeline(string savedEpisodeDir)
        {
            // Traverses up from 'unity-files/Assets/' to 'simulation_and_vr/'
            string rootDir = Path.GetFullPath(Path.Combine(Application.dataPath, "../../"));
            string trainingDir = Path.Combine(rootDir, "imitationagent_training");
             
            string pipelineDir = Path.Combine(trainingDir, "training_pipeline");
            string localEnvPackages = Path.Combine(trainingDir, "python_embedded");
             
            string bootstrapScript = Path.Combine(pipelineDir, "bootstrap_env.py");
            string pythonScript = Path.Combine(pipelineDir, "incremental_train.py");
            string targetOnnxOutput = Path.Combine(rootDir, "unity_files", "Assets", "ImitationModel", "ImitationAgentModel.onnx");

            // --------------------------------------------------------------------------
            // CROSS-PLATFORM PYTHON RESOLUTION ENGINE
            // --------------------------------------------------------------------------
            string nativePythonCmd = "python3";
            if (Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer)
            {
                nativePythonCmd = "python";
            }

            Debug.Log($"[Lifecycle Engine] Activating background pipeline on platform: {Application.platform}");

            System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    // Step A: Silently trigger the bootstrap script to guarantee packages exist
                    System.Diagnostics.ProcessStartInfo bootstrapInfo = new System.Diagnostics.ProcessStartInfo();
                    bootstrapInfo.FileName = nativePythonCmd;
                    bootstrapInfo.Arguments = $"\"{bootstrapScript}\"";
                    bootstrapInfo.UseShellExecute = false;
                    bootstrapInfo.CreateNoWindow = true;
                    
                    using (System.Diagnostics.Process bootstrapProcess = System.Diagnostics.Process.Start(bootstrapInfo))
                    {
                        bootstrapProcess.WaitForExit();
                        if (bootstrapProcess.ExitCode != 0)
                        {
                            Debug.LogError("[Lifecycle Engine] Critical Error: Environment Bootstrapping verification failed.");
                            return;
                        }
                    }

                    // Step B: Trigger the training pipeline with the recorded episode directory.
                    System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo();
                    startInfo.FileName = nativePythonCmd;
                    startInfo.Arguments = $"\"{pythonScript}\" --episode_dir \"{savedEpisodeDir}\" --model_output \"{targetOnnxOutput}\"";
                    startInfo.UseShellExecute = false;
                    startInfo.RedirectStandardOutput = true;
                    startInfo.RedirectStandardError = true;
                    startInfo.CreateNoWindow = true;
                    
                    // Injecting python_embedded directory directly into the Python import path environment variable
                    startInfo.EnvironmentVariables["PYTHONPATH"] = localEnvPackages;

                    using (System.Diagnostics.Process process = System.Diagnostics.Process.Start(startInfo))
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        string error = process.StandardError.ReadToEnd();
                        process.WaitForExit();

                        if (process.ExitCode == 0)
                        {
                            Debug.Log($"[Lifecycle Engine] PYTHON PIPELINE SUCCESS:\n{output}");
                        }
                        else
                        {
                            Debug.LogError($"[Lifecycle Engine] Python Pipeline Error (Exit Code {process.ExitCode}): {error}");
                        }
                    }
                }
                catch (System.Exception ex)
                {
                    Debug.LogError($"[Lifecycle Engine] Cross-Platform Execution Exception: {ex.Message}");
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

        private void InitializeEpisodeStorage()
        {
            try
            {
                string rootDir = Path.GetFullPath(Path.Combine(Application.dataPath, "../../"));
                string dirPath = Path.Combine(rootDir, "imitationagent_training", "zarr_recordings");
                if (!Directory.Exists(dirPath))
                {
                    Directory.CreateDirectory(dirPath);
                }

                string episodeId = $"episode_{System.DateTime.Now:yyyyMMdd_HHmmss_fff}";
                currentEpisodeDir = Path.Combine(dirPath, episodeId);
                Directory.CreateDirectory(currentEpisodeDir);

                currentManifestPath = Path.Combine(currentEpisodeDir, ZarrDataContract.ManifestFileName);
                episodeFrameCount = 0;

                rgbWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "rgb.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
                depthWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "depth.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
                goalWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "goal.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
                poseWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "pose.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
                actionVelWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "action_vel.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
                actionRotWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "action_rot.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
                timestampWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "timestamp.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));

                Debug.Log($"[Lifecycle Engine] Binary episode channel established at: {currentEpisodeDir}");
            }
            catch (IOException ex)
            {
                Debug.LogError($"[Lifecycle Engine] Failed to initialize binary episode target: {ex.Message}");
            }
        }

        private void CloseBinaryWriters()
        {
            if (rgbWriter != null)
            {
                rgbWriter.Flush();
                rgbWriter.Close();
                rgbWriter = null;
            }

            if (depthWriter != null)
            {
                depthWriter.Flush();
                depthWriter.Close();
                depthWriter = null;
            }

            if (goalWriter != null)
            {
                goalWriter.Flush();
                goalWriter.Close();
                goalWriter = null;
            }

            if (poseWriter != null)
            {
                poseWriter.Flush();
                poseWriter.Close();
                poseWriter = null;
            }

            if (actionVelWriter != null)
            {
                actionVelWriter.Flush();
                actionVelWriter.Close();
                actionVelWriter = null;
            }

            if (actionRotWriter != null)
            {
                actionRotWriter.Flush();
                actionRotWriter.Close();
                actionRotWriter = null;
            }

            if (timestampWriter != null)
            {
                timestampWriter.Flush();
                timestampWriter.Close();
                timestampWriter = null;
            }
        }

        private void WriteEpisodeManifest()
        {
            if (string.IsNullOrEmpty(currentEpisodeDir)) return;

            var manifest = new ZarrDataContract.EpisodeManifest
            {
                episode_id = Path.GetFileName(currentEpisodeDir),
                participant_id = "unknown",
                session_id = "unknown",
                frame_count = episodeFrameCount,
                start_time = startTime,
                end_time = Time.time,
                arrays = new ZarrDataContract.ArrayEntry[]
                {
                    new ZarrDataContract.ArrayEntry { name = "rgb", shape = new int[] { 0, ZarrDataContract.ImageHeight, ZarrDataContract.ImageWidth, 3 }, dtype = ZarrDataContract.ArrayDtypes.Rgb },
                    new ZarrDataContract.ArrayEntry { name = "depth", shape = new int[] { 0, ZarrDataContract.ImageHeight, ZarrDataContract.ImageWidth }, dtype = ZarrDataContract.ArrayDtypes.Depth },
                    new ZarrDataContract.ArrayEntry { name = "goal", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "pose", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "action_vel", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "action_rot", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "timestamp", shape = new int[] { 0 }, dtype = ZarrDataContract.ArrayDtypes.Timestamp }
                }
            };

            string json = JsonUtility.ToJson(manifest, true);
            File.WriteAllText(currentManifestPath, json);
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

                // --- SERIALIZE FRAME AS FIXED-DIMENSION BINARY SHARDS ---
                WriteFrameBinary(
                    colorPixels,
                    currentRayBuffer,
                    currentPosition,
                    localGoalVector,
                    localVelocityAction,
                    deltaRotationAction,
                    timeStamp
                );
            }
        }

        private void WriteFrameBinary(Color32[] colorPixels, float[] depthBuffer, Vector3 pose, Vector3 goal, Vector3 actionVel, Vector3 actionRot, float timeStamp)
        {
            if (rgbWriter == null || depthWriter == null || goalWriter == null || poseWriter == null || actionVelWriter == null || actionRotWriter == null || timestampWriter == null)
            {
                return;
            }

            for (int i = 0; i < TotalPixels; i++)
            {
                Color32 pixel = colorPixels[i];
                rgbWriter.Write(pixel.r);
                rgbWriter.Write(pixel.g);
                rgbWriter.Write(pixel.b);
            }

            for (int i = 0; i < depthBuffer.Length; i++)
            {
                depthWriter.Write(depthBuffer[i]);
            }

            goalWriter.Write(goal.x);
            goalWriter.Write(goal.y);
            goalWriter.Write(goal.z);

            poseWriter.Write(pose.x);
            poseWriter.Write(pose.y);
            poseWriter.Write(pose.z);

            actionVelWriter.Write(actionVel.x);
            actionVelWriter.Write(actionVel.y);
            actionVelWriter.Write(actionVel.z);

            actionRotWriter.Write(actionRot.x);
            actionRotWriter.Write(actionRot.y);
            actionRotWriter.Write(actionRot.z);

            timestampWriter.Write(timeStamp);
            episodeFrameCount++;
        }

        private void OnDestroy()
        {
            StopRecording();
        }
    }
}