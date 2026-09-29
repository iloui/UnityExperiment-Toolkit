using System;
using UnityEngine;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Canonical binary dataset contract for the Unity-to-Python Zarr pipeline.
    /// This contract is intentionally fixed before implementation to avoid format drift.
    /// </summary>
    public static class ZarrDataContract
    {
        public const string SchemaVersion = "zarr-v3";
        public const string ManifestFileName = "manifest.json";

        public const int ImageWidth = 64;
        public const int ImageHeight = 64;
        public const int TotalPixels = ImageWidth * ImageHeight;
        public const int TemporalWindowSize = 1;

        public const float SamplingIntervalSeconds = 0.1f;
        public const float MaxRayDistance = 50f;
        /// <summary>
        /// Field of view of the capture camera, in degrees. The image is square, so this is both the vertical and the horizontal
        /// FOV. RGB and depth come from the same camera (see ContractCamera): depth pixel (x, y) is the distance along the ray
        /// through the centre of RGB pixel (x, y).
        /// </summary>
        public const float CameraFovDegrees = 60f;

        /// <summary>
        /// The single Unity layer name depth raycasts must be restricted to. Both
        /// ParticipantRecorder and ImitationAgent hard-fail at startup unless their
        /// ArchitectureLayerMask equals exactly LayerMask.GetMask(ArchitectureObstacleLayerName) -
        /// no other mask (including "Everything") is accepted, since a too-broad mask silently
        /// corrupts every recorded/inferred depth value.
        /// </summary>
        public const string ArchitectureObstacleLayerName = "Architecture_Obstacle";

        /// <summary>
        /// Distance (meters) within which a mover is considered to have arrived at its
        /// current POI/end target. Shared by PoiProgressTracker so recording and inference
        /// use identical arrival semantics.
        /// </summary>
        public const float ArrivalRadiusMeters = 1.5f;

        /// <summary>
        /// Recording/inference eye height: the camera sits this far above the NavMesh surface under the mover.
        /// ImitationAgent stands on the NavMesh, so this is its head height; the FPSController (human and synthetic
        /// recordings) stands on the floor collider, so its FirstPersonCharacter offset is set to hit the same world
        /// height (see RecordingRig.ApplyContractCameraHeight). The 2026-09 human episodes were recorded at ~1.70 m
        /// above the NavMesh (1.78 m above the floor) and do not match this.
        /// </summary>
        public const float CameraHeightAboveNavMeshMeters = 1.5f;
        /// <summary>ParticipantRecorder warns when the recording camera is further than this from CameraHeightAboveNavMeshMeters.</summary>
        public const float CameraHeightToleranceMeters = 0.03f;

        /// <summary>
        /// The recording body: FPSController.prefab's CharacterController and FirstPersonController walk settings, plus the
        /// InputManager "Vertical" axis ramp the humans' walk key (and the synthetic bot) go through. ImitationAgent moves
        /// with exactly these so the states it visits (wall clearance, sliding along walls, speed ramp) are the recorded ones.
        /// </summary>
        public static class RecordingBody
        {
            public const float Radius = 0.25f;
            public const float Height = 1.5f;
            public const float StepOffset = 0.3f;
            public const float SlopeLimitDegrees = 45f;
            public const float SkinWidth = 0.08f;
            public const float WalkSpeedMetersPerSecond = 1.3f;
            public const float StickToGroundForce = 10f;
            public const float GravityMultiplier = 2f;
            /// <summary>Vertical axis ramp-up per second while the walk key is held (1/3 s to full speed).</summary>
            public const float WalkAxisSensitivity = 3f;
            /// <summary>Vertical axis fall-off per second after release (effectively instant).</summary>
            public const float WalkAxisGravity = 100f;
        }

        /// <summary>
        /// Output contract of the discrete-head model (train_2DCNN_discrete.py; must match data_contract.py). 'motor_actions' is
        /// [walk_logit, yaw_logit_0 .. yaw_logit_6, pitch_deg_per_sample]. Yaw classes run strong-left .. straight .. strong-right;
        /// class c has signed magnitude sign(c - Straight) * YawBinCentersDeg[|c - Straight|], in degrees per sampling interval.
        /// </summary>
        public static class DiscreteHead
        {
            public static readonly float[] YawBinCentersDeg = { 0f, 1.5f, 5f, 12f };
            public const int YawClassCount = 7;
            public const int StraightClass = 3;
            public const int OutputSize = 1 + YawClassCount + 1;
        }

        /// <summary>
        /// Optional own-motion input appended after the pixels in 'sensory_inputs' (data_contract.PROPRIO_*), measured on the
        /// body the way ParticipantRecorder labels action_vel.z and action_rot.x. ImitationAgent detects the layout from the
        /// model's input size. v2 (Size): [speed / SpeedScaleMps, previous sample's speed / SpeedScaleMps, wrapped mean yaw of
        /// the last YawWindowSamples samples (deg/sample) / YawScaleDeg, signed heading change since the body last walked
        /// (speed above WalkThresholdMps) / TurnScaleDeg clipped to +-TurnClip, available = 1].
        /// v1 (LegacySize, the ImitationAgent_A_proprio_s* models of 2026-09-28): [speed, yaw, available].
        /// </summary>
        public static class Proprio
        {
            public const int Size = 5;
            public const int LegacySize = 3;
            public const int YawWindowSamples = 5;
            public const float SpeedScaleMps = 1.3f;
            public const float YawScaleDeg = 10f;
            public const float TurnScaleDeg = 180f;
            public const float TurnClip = 2f;
            /// <summary>data_contract.WALK_THRESHOLD_MPS: a sample faster than this counts as walking.</summary>
            public const float WalkThresholdMps = 0.65f;
        }

        /// <summary>
        /// Columns of the 'pose' array (data_contract.POSE_COLUMNS): camera world position, heading (eulerAngles.y, degrees in
        /// [0, 360), clockwise from above, 0 = +z) and pitch (eulerAngles.x wrapped to (-180, 180], + = looking down).
        /// </summary>
        public static readonly string[] PoseColumns = { "x", "y", "z", "heading_deg", "pitch_deg" };

        /// <summary>Heading and pitch of a camera/eye transform in the 'pose' and trajectory convention.</summary>
        public static void HeadingPitch(Transform eye, out float headingDeg, out float pitchDeg)
        {
            Vector3 euler = eye.eulerAngles;
            headingDeg = euler.y;
            pitchDeg = euler.x > 180f ? euler.x - 360f : euler.x;
        }

        /// <summary>
        /// The unified trajectory every evaluation source is compared in (data_contract.TRAJECTORY_COLUMNS; EvalHarness writes it
        /// as traj_*.csv, eval_tools/trajectories.py derives it for recorded episodes). One row per SamplingIntervalSeconds of
        /// sim time, t = 0 at the first row. x, y, z = eye position; heading/pitch as in PoseColumns; speed_mps = horizontal
        /// distance since the previous row / dt; yaw_rate_deg_s = wrapped heading change since the previous row / dt (both 0 in
        /// the first row); phase 0 = walking to a POI, 1 = walking to the end.
        /// </summary>
        public static class Trajectory
        {
            public static readonly string[] Columns =
                { "t", "x", "y", "z", "heading_deg", "pitch_deg", "speed_mps", "yaw_rate_deg_s", "phase", "pois_reached" };
            public const string CsvDelimiter = ";";
            public const int PhaseToPoi = 0;
            public const int PhaseToEnd = 1;
        }

        public static readonly string[] ArrayNames = new[]
        {
            "rgb",
            "depth",
            "goal",
            "goal_distance",
            "pose",
            "action_vel",
            "action_rot",
            "timestamp"
        };

        public static int ExpectedRgbLength
        {
            get { return ImageWidth * ImageHeight * 3; }
        }

        public static class ArrayDtypes
        {
            public const string Rgb = "uint8";
            public const string Depth = "float32";
            public const string Vector = "float32";
            public const string Scalar = "float32";
            public const string Timestamp = "float64";
        }

        /// <summary>
        /// Written to every manifest as recorder_version. Bump it whenever ParticipantRecorder changes what or how it
        /// records, so episodes from different recorder behaviour can be told apart without the scene.
        /// </summary>
        public const string RecorderVersion = "participant_recorder_v3";

        /// <summary>Values of the manifest's 'source' field (data_contract.EPISODE_SOURCES).</summary>
        public static class EpisodeSources
        {
            public const string Human = "human";
            public const string Synthetic = "synthetic";
        }

        /// <summary>Value of the manifest's 'input_mode' field. Desktop (keyboard + mouse) is the only mode in use.</summary>
        public const string InputModeDesktop = "desktop";

        /// <summary>
        /// Everything Python needs to place an episode without opening the scene: which layout (scene, typology, seed),
        /// who produced it (source, participant, session, repetition), the task in world coordinates, and the camera it
        /// was recorded with. data_contract.check_manifest refuses an episode that lacks any of these.
        /// </summary>
        [Serializable]
        public sealed class EpisodeManifest
        {
            public string schema_version = SchemaVersion;
            public string recorder_version = RecorderVersion;
            public string episode_id;

            public string scene_name;
            public string layout_typology;
            public int layout_seed;

            public string source;
            public string participant_id;
            public string session_id;
            public int repetition_index;
            public string input_mode = InputModeDesktop;

            public TaskPositions task;
            /// <summary>
            /// Random offset (degrees, + = clockwise from above) added to the start heading, which otherwise faces the first
            /// target. Set by the synthetic harnesses; 0 for humans, who start with the scene's FPSController heading.
            /// </summary>
            public float start_jitter_heading_deg;

            public int frame_count;
            public float start_time;
            public float end_time;
            public int image_width = ImageWidth;
            public int image_height = ImageHeight;
            public int total_pixels = TotalPixels;
            public int temporal_window_size = TemporalWindowSize;
            public float sampling_interval_s = SamplingIntervalSeconds;
            public float max_ray_distance = MaxRayDistance;
            /// <summary>Measured height of the recording camera above the NavMesh (contract: CameraHeightAboveNavMeshMeters).</summary>
            public float camera_height_m;
            public float camera_fov_deg = CameraFovDegrees;
            public ArrayEntry[] arrays;
        }

        /// <summary>The episode's task in world coordinates: start marker, POIs in visiting order, end marker.</summary>
        [Serializable]
        public sealed class TaskPositions
        {
            public Vector3 start;
            public Vector3[] pois;
            public Vector3 end;
        }

        [Serializable]
        public sealed class ArrayEntry
        {
            public string name;
            public int[] shape;
            public string dtype;
        }
    }
}
