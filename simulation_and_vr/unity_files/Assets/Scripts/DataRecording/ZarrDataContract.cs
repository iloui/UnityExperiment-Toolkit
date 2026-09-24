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
        public const string SchemaVersion = "zarr-v1";
        public const string ManifestFileName = "manifest.json";

        public const int ImageWidth = 64;
        public const int ImageHeight = 64;
        public const int TotalPixels = ImageWidth * ImageHeight;
        public const int TemporalWindowSize = 1;

        public const float SamplingIntervalSeconds = 0.1f;
        public const float MaxRayDistance = 50f;
        public const float HorizontalFovDegrees = 90f;
        public const float VerticalFovDegrees = 90f;

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

        [Serializable]
        public sealed class EpisodeManifest
        {
            public string schema_version = SchemaVersion;
            public string episode_id;
            public string participant_id;
            public string session_id;
            public int frame_count;
            public float start_time;
            public float end_time;
            public int image_width = ImageWidth;
            public int image_height = ImageHeight;
            public int total_pixels = TotalPixels;
            public int temporal_window_size = TemporalWindowSize;
            public float sampling_interval_s = SamplingIntervalSeconds;
            public float max_ray_distance = MaxRayDistance;
            public float horizontal_fov_deg = HorizontalFovDegrees;
            public float vertical_fov_deg = VerticalFovDegrees;
            public ArrayEntry[] arrays;
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
