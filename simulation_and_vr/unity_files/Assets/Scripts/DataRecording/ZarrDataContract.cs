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
