using UnityEngine;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Unity.Collections;
using Unity.Jobs;
using UnityTools.Core;

namespace Assets.Scripts.DataRecording
{
    public enum RecordingOutputFormat
    {
        Binary,
        Csv
    }

    /// <summary>
    /// Supplies the action label for the observation just captured, instead of deriving it from the
    /// camera's own motion. Used by DaggerHarness: the trained policy drives, and the NavMesh expert
    /// says what it should have done from each visited state.
    /// </summary>
    public interface IActionLabelSource
    {
        /// <param name="actionVel">Local (camera-frame) velocity label, m/s - z = forward, same as action_vel.</param>
        /// <param name="actionRot">(yaw, pitch, 0) delta in degrees per sampling interval, same as action_rot.</param>
        void GetActionLabel(out Vector3 actionVel, out Vector3 actionRot);
    }

    public class ParticipantRecorder : MonoBehaviour
    {
        [Header("Linked Framework Components")]
        [Tooltip("Defines this scene's start/POI(s)/end locations and selection rules. The same " +
                 "TaskScript that ImitationAgent uses for this scene, so both pipelines walk the " +
                 "identical POI progression via PoiProgressTracker.")]
        public TaskScript task;
        [Tooltip("Must equal exactly LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName) - " +
                 "recording refuses to start otherwise, since any broader/narrower mask silently corrupts " +
                 "every recorded depth value.")]
        public LayerMask ArchitectureLayerMask;

        [Header("Storage Format")]
        public RecordingOutputFormat OutputFormat = RecordingOutputFormat.Binary;

        [Header("Sampling & Storage Settings")]
        public string SaveDirectoryName = "VR_Recordings";
        public bool EnableDebugRays = true; // Enabled by default for edge testing
        public bool AutoStartForTesting = false;

        [Header("Episode Metadata (manifest.json)")]
        [Tooltip("'human' for participants. The synthetic harnesses set 'synthetic'.")]
        public string source = ZarrDataContract.EpisodeSources.Human;
        [Tooltip("Required: recording refuses to start while this is empty or 'unknown'. Set the participant's ID before each " +
                 "participant; the synthetic harnesses set their bot name (e.g. 'synthetic_rrw_v3').")]
        public string participantId = "";
        [Tooltip("Groups the episodes of one sitting. Left empty, it is set to the start time of the first episode " +
                 "(yyyyMMdd_HHmmss) and kept for the rest of Play mode.")]
        public string sessionId = "";
        [Tooltip("repetition_index written to the NEXT episode; incremented after every episode. Reset to 0 for a new " +
                 "participant, or set it when resuming a participant in a new Play session.")]
        public int repetitionIndex = 0;
        [Tooltip("How the mover was controlled. Only 'desktop' (keyboard + mouse) is in use.")]
        public string inputMode = ZarrDataContract.InputModeDesktop;
        /// <summary>start_jitter_heading_deg written to the NEXT episode. Set per episode by the synthetic harnesses; humans keep 0.</summary>
        [HideInInspector] public float startJitterHeadingDeg;
        /// <summary>synthetic_task_kind / synthetic_backtrack_m written to the NEXT episode. Set per episode by RecordRandomWalksHarness.</summary>
        [HideInInspector] public string syntheticTaskKind = "";
        [HideInInspector] public float syntheticBacktrackMeters;
        /// <summary>Design labels written to the NEXT episode's manifest ('trial'). Set per trial by ParticipantSession; empty otherwise.</summary>
        [HideInInspector] public ZarrDataContract.TrialInfo trialInfo = new ZarrDataContract.TrialInfo();

        /// <summary>Record from this camera instead of the scene's CanonicalRecordingCamera (DaggerHarness: the policy agent's head). null = unchanged behaviour.</summary>
        [HideInInspector] public Camera recordingCameraOverride;

        /// <summary>
        /// If set, action_vel/action_rot are taken from this source instead of the camera's motion. The label
        /// queried at sample i is written at sample i+1, which is exactly where the training loader looks for
        /// the action taken FROM observation i (see SingleEpisodeDataset: label = action at idx + window).
        /// </summary>
        [HideInInspector] public IActionLabelSource actionLabelSource;
        private Vector3 pendingLabelVel;
        private Vector3 pendingLabelRot;

        // All hardware/sampling constants below are read directly from ZarrDataContract - the
        // same contract ParticipantRecorder's manifest is written against and ImitationAgent
        // reads its live-inference constants from - rather than being separately-tunable
        // Inspector fields, so recording can never silently drift out of sync with the format
        // the model is trained on (see ImitationAgent.cs for the matching inference-side fix).
        private const int ImageWidth = ZarrDataContract.ImageWidth;
        private const int ImageHeight = ZarrDataContract.ImageHeight;
        private const int TotalPixels = ImageWidth * ImageHeight;

        private bool isRecording;
        private float startTime;
        private Vector3 lastPosition;
        private string currentEpisodeDir;
        private string currentManifestPath;
        private string currentCsvPath;
        private StreamWriter csvWriter;
        private int episodeFrameCount;

        private BinaryWriter rgbWriter;
        private BinaryWriter depthWriter;
        private BinaryWriter goalWriter;
        private BinaryWriter goalDistanceWriter;
        private BinaryWriter poseWriter;
        private BinaryWriter actionVelWriter;
        private BinaryWriter actionRotWriter;
        private BinaryWriter timestampWriter;

        private Camera recordingCamera;
        private Camera hiddenCaptureCamera;
        private RenderTexture colorRenderTexture;
        private Texture2D colorReadbackTex;

        private NativeArray<RaycastCommand> rayCommands;
        private NativeArray<RaycastHit> rayResults;
        private float[] currentRayBuffer;

        private PoiProgressTracker poiTracker;
        private float Azimuth;
        private float Elevation;

        // Manifest metadata captured at StartRecording (see WriteEpisodeManifest).
        private LayoutInfo layoutInfo;
        private ZarrDataContract.TaskPositions episodeTask;
        private float measuredCameraHeight;

        /// <summary>True between StartRecording() and StopRecording() (including the automatic stop on reaching the end target).</summary>
        public bool IsRecording => isRecording;

        /// <summary>Directory of the current/last binary episode (empty until a binary episode has been started).</summary>
        public string CurrentEpisodeDirectory => currentEpisodeDir;

        /// <summary>POIs fulfilled so far by this recorder's own tracker (the one that drives the recorded goal vector).</summary>
        public int NeedsFulfilled => poiTracker != null ? poiTracker.NeedsFulfilled : 0;

        /// <summary>Target the recorded goal vector currently points at (null when not recording).</summary>
        public Transform CurrentTarget => poiTracker != null ? poiTracker.CurrentTarget : null;

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

            // --- Dummy-proof preconditions: refuse to record rather than silently produce
            // --- garbage training data. ---
            if (task == null)
            {
                Debug.LogError("[ParticipantRecorder] CRITICAL: 'TaskScript task' is not assigned! Recording refused.");
                throw new System.InvalidOperationException("[ParticipantRecorder] Cannot start recording: 'task' (TaskScript) is not assigned.");
            }

            LayerMask requiredMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);
            if (ArchitectureLayerMask.value != requiredMask.value)
            {
                Debug.LogError($"[ParticipantRecorder] CRITICAL: ArchitectureLayerMask must equal exactly the " +
                    $"'{ZarrDataContract.ArchitectureObstacleLayerName}' layer (mask value {requiredMask.value}), " +
                    $"but is currently set to mask value {ArchitectureLayerMask.value}. Recording refused - a wrong " +
                    $"mask (including 'Everything') silently corrupts every recorded depth value.");
                throw new System.InvalidOperationException(
                    $"[ParticipantRecorder] Cannot start recording: ArchitectureLayerMask ({ArchitectureLayerMask.value}) " +
                    $"does not exactly match required layer '{ZarrDataContract.ArchitectureObstacleLayerName}' ({requiredMask.value}).");
            }

            // Episode metadata: training splits by participant and groups by layout from the manifest alone, so an
            // episode without them is unusable.
            if (source != ZarrDataContract.EpisodeSources.Human && source != ZarrDataContract.EpisodeSources.Synthetic)
            {
                throw new System.InvalidOperationException(
                    $"[ParticipantRecorder] Cannot start recording: source is '{source}', expected " +
                    $"'{ZarrDataContract.EpisodeSources.Human}' or '{ZarrDataContract.EpisodeSources.Synthetic}'.");
            }
            if (string.IsNullOrWhiteSpace(participantId) || participantId.Trim() == "unknown")
            {
                Debug.LogError("[ParticipantRecorder] CRITICAL: participantId is not set. Enter the participant's ID on the " +
                    "ParticipantRecorder (FPSController) before recording. Recording refused.");
                throw new System.InvalidOperationException("[ParticipantRecorder] Cannot start recording: participantId is not set.");
            }
            if (string.IsNullOrWhiteSpace(inputMode))
            {
                throw new System.InvalidOperationException("[ParticipantRecorder] Cannot start recording: inputMode is empty.");
            }
            // Throws unless the scene has exactly one LayoutInfo (added by Tools > Generative Design > Generate Layout).
            layoutInfo = LayoutInfo.ResolveInScene();
            if (string.IsNullOrWhiteSpace(sessionId))
            {
                sessionId = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            }

            // Throws if the scene doesn't have exactly one CanonicalRecordingCamera - no
            // fallback to any other camera is allowed (unless a harness explicitly overrides it).
            recordingCamera = recordingCameraOverride != null ? recordingCameraOverride : CanonicalRecordingCamera.ResolveInScene();

            startTime = Time.time;
            lastPosition = recordingCamera.transform.position;
            episodeFrameCount = 0;
            currentEpisodeDir = null;
            pendingLabelVel = Vector3.zero;
            pendingLabelRot = Vector3.zero;
            // Fresh episode = fresh rotation baseline, so the first action_rot delta is identical to a
            // freshly-added recorder's regardless of how many episodes this component recorded before.
            Azimuth = 0f;
            Elevation = 0f;

            poiTracker = new PoiProgressTracker(task);
            episodeTask = CaptureTaskPositions();
            measuredCameraHeight = float.NaN;

            InitializeHardwareReplication();
            InitializeJobBuffers();

            if (OutputFormat == RecordingOutputFormat.Csv)
            {
                InitializeCsvFile();
            }
            else
            {
                InitializeEpisodeStorage();
            }

            isRecording = true;
            StartCoroutine(LockedSamplingLoop());
        }

        public void StopRecording()
        {
            if (!isRecording) return;

            isRecording = false;
            StopAllCoroutines();

            if (OutputFormat == RecordingOutputFormat.Csv)
            {
                if (csvWriter != null)
                {
                    csvWriter.Flush();
                    csvWriter.Close();
                    csvWriter = null;
                }
            }
            else
            {
                CloseBinaryWriters();

                if (!string.IsNullOrEmpty(currentEpisodeDir))
                {
                    WriteEpisodeManifest();
                }
            }

            if (rayCommands.IsCreated) rayCommands.Dispose();
            if (rayResults.IsCreated) rayResults.Dispose();

            if (hiddenCaptureCamera != null) Destroy(hiddenCaptureCamera.gameObject);
            if (colorRenderTexture != null) colorRenderTexture.Release();
            if (colorReadbackTex != null) Destroy(colorReadbackTex);
        }

        private void InitializeHardwareReplication()
        {
            GameObject camGO = new GameObject("Hidden_BC_CaptureCamera");
            camGO.transform.SetParent(recordingCamera.transform, false);
            hiddenCaptureCamera = camGO.AddComponent<Camera>();

            hiddenCaptureCamera.CopyFrom(recordingCamera);
            hiddenCaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            hiddenCaptureCamera.backgroundColor = Color.black;

            colorRenderTexture = new RenderTexture(ImageWidth, ImageHeight, 24, RenderTextureFormat.ARGB32);
            colorRenderTexture.Create();
            hiddenCaptureCamera.targetTexture = colorRenderTexture;
            ContractCamera.Configure(hiddenCaptureCamera);

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
            try
            {
                string rootDir = Path.GetFullPath(Path.Combine(Application.dataPath, "../../"));
                string dirPath = Path.Combine(rootDir, "imitationagent_training", "vr_recordings");
                if (!Directory.Exists(dirPath))
                {
                    Directory.CreateDirectory(dirPath);
                }

                string filename = $"BC_Run_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv";
                currentCsvPath = Path.Combine(dirPath, filename);

                csvWriter = new StreamWriter(currentCsvPath, false, Encoding.UTF8);

                StringBuilder headerBuilder = new StringBuilder();
                headerBuilder.Append("Timestamp,Pos_X,Pos_Y,Pos_Z,Goal_Dir_X,Goal_Dir_Y,Goal_Dir_Z,Goal_Dist,Action_Vel_X,Action_Vel_Y,Action_Vel_Z,Action_Rot_X,Action_Rot_Y,Action_Rot_Z,");

                for (int t = 0; t < ZarrDataContract.TemporalWindowSize; t++)
                {
                    for (int i = 0; i < TotalPixels; i++)
                    {
                        headerBuilder.Append($"F{t}_P{i}_R,F{t}_P{i}_G,F{t}_P{i}_B,F{t}_P{i}_Depth,");
                    }
                }

                headerBuilder.Length--;
                csvWriter.WriteLine(headerBuilder.ToString());

                Debug.Log($"[Lifecycle Engine] Legacy CSV channel established at: {currentCsvPath}");
            }
            catch (IOException ex)
            {
                Debug.LogError($"[Lifecycle Engine] Failed to initialize legacy CSV target: {ex.Message}");
            }
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
                goalDistanceWriter = new BinaryWriter(new FileStream(Path.Combine(currentEpisodeDir, "goal_distance.bin"), FileMode.Create, FileAccess.Write, FileShare.Read));
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

            if (goalDistanceWriter != null)
            {
                goalDistanceWriter.Flush();
                goalDistanceWriter.Close();
                goalDistanceWriter = null;
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

            // Episodes shorter than CameraHeightCheckFrame were never measured; measure now (the rig has not moved vertically).
            if (float.IsNaN(measuredCameraHeight) && recordingCamera != null) CheckCameraHeight(recordingCamera.transform.position);
            if (float.IsNaN(measuredCameraHeight))
            {
                Debug.LogError("[ParticipantRecorder] Could not measure the camera height above the NavMesh (no NavMesh under the " +
                    "camera?). camera_height_m is written as -1 and training will refuse this episode.");
            }

            var manifest = new ZarrDataContract.EpisodeManifest
            {
                episode_id = Path.GetFileName(currentEpisodeDir),
                scene_name = gameObject.scene.name,
                layout_typology = layoutInfo.typology,
                layout_seed = layoutInfo.seed,
                source = source,
                participant_id = participantId,
                session_id = sessionId,
                repetition_index = repetitionIndex,
                input_mode = inputMode,
                task = episodeTask,
                start_jitter_heading_deg = startJitterHeadingDeg,
                synthetic_task_kind = syntheticTaskKind ?? "",
                synthetic_backtrack_m = syntheticBacktrackMeters,
                trial = trialInfo ?? new ZarrDataContract.TrialInfo(),
                camera_height_m = float.IsNaN(measuredCameraHeight) ? -1f : measuredCameraHeight,
                camera_fov_deg = hiddenCaptureCamera != null ? hiddenCaptureCamera.fieldOfView : -1f,
                frame_count = episodeFrameCount,
                start_time = startTime,
                end_time = Time.time,
                arrays = new ZarrDataContract.ArrayEntry[]
                {
                    new ZarrDataContract.ArrayEntry { name = "rgb", shape = new int[] { 0, ZarrDataContract.ImageHeight, ZarrDataContract.ImageWidth, 3 }, dtype = ZarrDataContract.ArrayDtypes.Rgb },
                    new ZarrDataContract.ArrayEntry { name = "depth", shape = new int[] { 0, ZarrDataContract.ImageHeight, ZarrDataContract.ImageWidth }, dtype = ZarrDataContract.ArrayDtypes.Depth },
                    new ZarrDataContract.ArrayEntry { name = "goal", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "goal_distance", shape = new int[] { 0 }, dtype = ZarrDataContract.ArrayDtypes.Scalar },
                    new ZarrDataContract.ArrayEntry { name = "pose", shape = new int[] { 0, ZarrDataContract.PoseColumns.Length }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "action_vel", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "action_rot", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "timestamp", shape = new int[] { 0 }, dtype = ZarrDataContract.ArrayDtypes.Timestamp }
                }
            };

            string json = JsonUtility.ToJson(manifest, true);
            File.WriteAllText(currentManifestPath, json);
            repetitionIndex++;
        }

        /// <summary>
        /// The episode's task in world coordinates. POIs and end come from the tracker's planned route (the targets the
        /// recorded goal vector will point at). Start is the task start marker nearest to the mover now: harnesses and
        /// generated scenes may use the mover itself or a moving marker as the start, so it is read at the start of recording.
        /// </summary>
        private ZarrDataContract.TaskPositions CaptureTaskPositions()
        {
            if (task.start == null || task.start.Length == 0)
            {
                throw new System.InvalidOperationException("[ParticipantRecorder] Cannot start recording: the TaskScript has no start marker.");
            }
            Vector3 start = task.start[0].transform.position;
            float best = float.PositiveInfinity;
            foreach (var marker in task.start)
            {
                if (marker == null) continue;
                float d = Vector3.Distance(marker.transform.position, transform.position);
                if (d < best) { best = d; start = marker.transform.position; }
            }

            var pois = new Vector3[poiTracker.PlannedPois.Count];
            for (int i = 0; i < pois.Length; i++) pois[i] = poiTracker.PlannedPois[i].position;

            return new ZarrDataContract.TaskPositions { start = start, pois = pois, end = poiTracker.PlannedEnd.position };
        }

        private IEnumerator LockedSamplingLoop()
        {
            while (isRecording)
            {
                yield return new WaitForSeconds(ZarrDataContract.SamplingIntervalSeconds);

                if (recordingCamera == null) continue;

                Transform hTransform = recordingCamera.transform;
                Vector3 currentPosition = hTransform.position;
                float timeStamp = Time.time - startTime;

                // --- GOAL SECTOR ---
                Vector3 localGoalVector = Vector3.zero;
                float goalDistanceMeters = 0f;
                if (poiTracker.CurrentTarget != null)
                {
                    Vector3 worldGoalOffset = poiTracker.CurrentTarget.position - currentPosition;
                    localGoalVector = hTransform.InverseTransformDirection(worldGoalOffset.normalized);
                    goalDistanceMeters = worldGoalOffset.magnitude;
                }

                // --- IMAGE CAPTURE ---
                hiddenCaptureCamera.Render();
                RenderTexture.active = colorRenderTexture;
                colorReadbackTex.ReadPixels(new Rect(0, 0, ImageWidth, ImageHeight), 0, 0);
                colorReadbackTex.Apply();
                Color32[] colorPixels = colorReadbackTex.GetPixels32();

                // --- DEPTH: one ray per RGB pixel centre, from the same camera (pixel-aligned with colorPixels) ---
                ContractCamera.FillDepthRays(hiddenCaptureCamera, rayCommands, ArchitectureLayerMask);
                Vector3 origin = hiddenCaptureCamera.transform.position;

                // Track extreme index coordinates for custom boundary debugging outputs (row 0 is the bottom row)
                int indexBottomLeft = 0;
                int indexBottomRight = ImageWidth - 1;
                int indexTopLeft = (ImageHeight - 1) * ImageWidth;
                int indexTopRight = TotalPixels - 1;
                int indexCenter = ((ImageHeight / 2) * ImageWidth) + (ImageWidth / 2);

                JobHandle handle = RaycastCommand.ScheduleBatch(rayCommands, rayResults, 16);
                handle.Complete();

                for (int i = 0; i < rayResults.Length; i++)
                {
                    float dist = rayResults[i].distance;
                    bool hasHit = dist > 0;
                    currentRayBuffer[i] = hasHit ? dist / ZarrDataContract.MaxRayDistance : 1.0f;

                    // --- ISOLATED OUTSIDE BOUNDARY FRUSTUM DEBUGGER ---
                    if (EnableDebugRays)
                    {
                        if (i == indexTopLeft || i == indexTopRight || i == indexBottomLeft || i == indexBottomRight || i == indexCenter)
                        {
                            float drawDist = hasHit ? dist : ZarrDataContract.MaxRayDistance;
                            Color debugColor = (i == indexCenter) ? Color.blue : (hasHit ? Color.green : Color.red);

                            // Draws a thick, persistent visual pointer along the frame border bounds
                            Debug.DrawRay(origin, rayCommands[i].direction * drawDist, debugColor, ZarrDataContract.SamplingIntervalSeconds);
                        }
                    }
                }

                // --- ACTION GENERATION ---
                Vector3 worldVelocity = (currentPosition - lastPosition) / ZarrDataContract.SamplingIntervalSeconds;
                Vector3 localVelocityAction = hTransform.InverseTransformDirection(worldVelocity);
                lastPosition = currentPosition;

                Vector3 currentForward = hTransform.forward;
                Math3D.CartesianToSpherical(currentForward, out float azimuth, out float elevation, out _);
                Vector3 deltaRotationAction = new Vector3(azimuth - Azimuth, elevation - Elevation, 0f);
                Azimuth = azimuth;
                Elevation = elevation;

                if (actionLabelSource != null)
                {
                    localVelocityAction = pendingLabelVel;
                    deltaRotationAction = pendingLabelRot;
                    actionLabelSource.GetActionLabel(out pendingLabelVel, out pendingLabelRot);
                }

                if (OutputFormat == RecordingOutputFormat.Csv)
                {
                    WriteLegacyCsvRow(currentPosition, localGoalVector, goalDistanceMeters, localVelocityAction, deltaRotationAction, timeStamp, colorPixels, currentRayBuffer);
                }
                else
                {
                    if (episodeFrameCount == CameraHeightCheckFrame) CheckCameraHeight(currentPosition);
                    ZarrDataContract.HeadingPitch(hTransform, out float headingDeg, out float pitchDeg);
                    WriteFrameBinary(
                        colorPixels,
                        currentRayBuffer,
                        currentPosition,
                        headingDeg,
                        pitchDeg,
                        localGoalVector,
                        goalDistanceMeters,
                        localVelocityAction,
                        deltaRotationAction,
                        timeStamp
                    );
                }

                // --- POI PROGRESSION (shared with ImitationAgent via PoiProgressTracker) ---
                // Deliberately NOT currentPosition (the camera/head position, ~1.7-2m above the
                // floor-level POI markers) - that made the vertical gap alone exceed
                // ArrivalRadiusMeters, so arrival could never trigger even standing right on top
                // of a marker. Use this component's own transform instead: ParticipantRecorder
                // sits on the FPSController body (near ground level), exactly mirroring
                // ImitationAgent.Update() ticking with its own body transform.position rather
                // than its head anchor.
                poiTracker.Tick(transform.position);
                if (poiTracker.IsFinished)
                {
                    Debug.Log("[ParticipantRecorder] PoiProgressTracker reports all POIs fulfilled and end reached. Stopping recording automatically.");
                    StopRecording();
                    yield break;
                }
            }
        }

        // ~1 s in, so a rig that spawns in the air (the FPSController starts at y=1.05 and falls) has landed.
        private const int CameraHeightCheckFrame = 10;

        private void CheckCameraHeight(Vector3 cameraPosition)
        {
            if (!RecordingRig.TryNavMeshHeight(cameraPosition, out float navMeshY)) return;
            float height = cameraPosition.y - navMeshY;
            measuredCameraHeight = height;
            if (Mathf.Abs(height - ZarrDataContract.CameraHeightAboveNavMeshMeters) > ZarrDataContract.CameraHeightToleranceMeters)
            {
                Debug.LogWarning($"[ParticipantRecorder] Recording camera is {height:F2} m above the NavMesh, but the contract (and ImitationAgent) " +
                    $"uses {ZarrDataContract.CameraHeightAboveNavMeshMeters:F2} m - this episode will not match the other recordings. " +
                    "Fix the FPSController with Tools > Imitation Learning Agent > Advanced > Apply Contract Camera Height.");
            }
        }

        private void WriteLegacyCsvRow(Vector3 pose, Vector3 goal, float goalDistance, Vector3 actionVel, Vector3 actionRot, float timeStamp, Color32[] colorPixels, float[] depthBuffer)
        {
            if (csvWriter == null)
            {
                return;
            }

            StringBuilder rowBuilder = new StringBuilder();
            rowBuilder.Append($"{timeStamp:F3},{pose.x:F3},{pose.y:F3},{pose.z:F3},");
            rowBuilder.Append($"{goal.x:F4},{goal.y:F4},{goal.z:F4},{goalDistance:F4},");
            rowBuilder.Append($"{actionVel.x:F4},{actionVel.y:F4},{actionVel.z:F4},");
            rowBuilder.Append($"{actionRot.x:F4},{actionRot.y:F4},{actionRot.z:F4},");

            for (int i = 0; i < TotalPixels; i++)
            {
                rowBuilder.Append($"{colorPixels[i].r},{colorPixels[i].g},{colorPixels[i].b},{depthBuffer[i]:F4},");
            }

            rowBuilder.Length--;
            csvWriter.WriteLine(rowBuilder.ToString());
        }

        private void WriteFrameBinary(Color32[] colorPixels, float[] depthBuffer, Vector3 pose, float headingDeg, float pitchDeg, Vector3 goal, float goalDistance, Vector3 actionVel, Vector3 actionRot, double timeStamp)
        {
            if (rgbWriter == null || depthWriter == null || goalWriter == null || goalDistanceWriter == null || poseWriter == null || actionVelWriter == null || actionRotWriter == null || timestampWriter == null)
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

            goalDistanceWriter.Write(goalDistance);

            poseWriter.Write(pose.x);
            poseWriter.Write(pose.y);
            poseWriter.Write(pose.z);
            poseWriter.Write(headingDeg);
            poseWriter.Write(pitchDeg);

            actionVelWriter.Write(actionVel.x);
            actionVelWriter.Write(actionVel.y);
            actionVelWriter.Write(actionVel.z);

            actionRotWriter.Write(actionRot.x);
            actionRotWriter.Write(actionRot.y);
            actionRotWriter.Write(actionRot.z);

            timestampWriter.Write((double) timeStamp);
            episodeFrameCount++;
        }

        private void OnDestroy()
        {
            StopRecording();
        }
    }
}
