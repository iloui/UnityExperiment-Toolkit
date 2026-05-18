using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityTools.Core;

namespace Assets.Scripts.DataRecording
{
    public class ParticipantRecorder : MonoBehaviour
    {
        public Camera Camera;

        [Header("Raycast Settings")]
        public LayerMask ArchitectureLayerMask;
        public float MaxRayDistance = 50f;
        public float HorizontalFov = 180f;
        public float VerticalFov = 90f;
        public float DegreeStep = 5f;
        public bool EnableDebugRays = false;
        public bool AutoStartForTesting = false;

        public float Azimuth;

        public float Elevation;

        private float startTime;

        private bool isRecording;

        private NativeArray<RaycastCommand> commands;
        private NativeArray<RaycastHit> results;
        private Vector3[] localDirections;
        private float[] rayDataBuffer;

        private void Start()
        {
            if (AutoStartForTesting)
            {
                StartRecording();
            }
        }

        private void PrepareJobBuffers()
        {
            int hSteps = Mathf.CeilToInt(HorizontalFov / DegreeStep);
            int vSteps = Mathf.CeilToInt(VerticalFov / DegreeStep);
            int totalRays = hSteps * vSteps;

            if (commands.IsCreated) commands.Dispose();
            if (results.IsCreated) results.Dispose();

            commands = new NativeArray<RaycastCommand>(totalRays, Allocator.Persistent);
            results = new NativeArray<RaycastHit>(totalRays, Allocator.Persistent);
            localDirections = new Vector3[totalRays];
            rayDataBuffer = new float[totalRays];

            int index = 0;
            for (int v = 0; v < vSteps; v++)
            {
                float vAngle = -(VerticalFov / 2) + (v * DegreeStep);
                for (int h = 0; h < hSteps; h++)
                {
                    float hAngle = -(HorizontalFov / 2) + (h * DegreeStep);
                    // Point 15: Relative to camera forward
                    localDirections[index++] = Quaternion.Euler(vAngle, hAngle, 0) * Vector3.forward;
                }
            }
        }

        public void StartRecording()
        {
            startTime = Time.time;
            isRecording = true;
            PrepareJobBuffers();
        }

        public void StopRecording()
        {
            isRecording = false;
            if (commands.IsCreated) commands.Dispose();
            if (results.IsCreated) results.Dispose();
        }

        private void OnDestroy()
        {
            if (commands.IsCreated) commands.Dispose();
            if (results.IsCreated) results.Dispose();
        }

        private void Update()
        {
            if (!isRecording)
            {
                return;
            }

            var ct = Camera.transform;
            var ea = Camera.transform.rotation * Vector3.forward;

            Math3D.CartesianToSpherical(ea, out Azimuth, out Elevation, out _);

            // Point 5: Origin slightly forward
            Vector3 origin = ct.position + ct.forward * 0.1f;

            // Point 7: Job System for performance
            for (int i = 0; i < localDirections.Length; i++)
            {
                Vector3 direction = ct.TransformDirection(localDirections[i]);
                // Point 13: QueryTriggerInteraction.Collide for triggers
                commands[i] = new RaycastCommand(origin, direction, new QueryParameters(ArchitectureLayerMask, true, QueryTriggerInteraction.Collide, false), MaxRayDistance);
            }

            JobHandle handle = RaycastCommand.ScheduleBatch(commands, results, 1);
            handle.Complete();

            // Point 11: 1.0 if no hit, else normalized distance
            for (int i = 0; i < results.Length; i++)
            {
                float dist = results[i].distance;
                bool hasHit = dist > 0;
                rayDataBuffer[i] = hasHit ? dist / MaxRayDistance : 1.0f;

                // NEU: Debug-Strahlen hier zeichnen, nachdem die Ergebnisse vorliegen
                if (EnableDebugRays)
                {
                    Vector3 direction = ct.TransformDirection(localDirections[i]);
                    float drawDist = hasHit ? dist : MaxRayDistance;
                    // Grün bei Treffer, Rot wenn der Strahl ins Leere geht
                    Debug.DrawRay(origin, direction * drawDist, hasHit ? Color.green : Color.red);
                }
            }

            Database.CurrentTrial.AddTrackingData(new TrackingEntry
            {
                Time = Time.time - startTime,
                Position = ct.position,
                ViewAzimuth = Azimuth,
                ViewElevation = Elevation,
                ObservationSpace = (float[])rayDataBuffer.Clone() // Clone to avoid reference issues since we reuse rayDataBuffer
            });
        }
    }
}
