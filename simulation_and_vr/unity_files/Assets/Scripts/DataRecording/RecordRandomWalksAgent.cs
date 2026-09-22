using System;
using UnityEngine;
using UnityEngine.AI;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Synthetic "expert" walker for RecordRandomWalksHarness. A NavMeshAgent walks a pre-planned
    /// route (start -> one random intermediate waypoint -> end); the recorded body/head is that
    /// path position plus a temporally-correlated (Ornstein-Uhlenbeck / AR(1)) lateral offset,
    /// clamped so it never leaves the NavMesh or comes closer than `edgeClearanceMeters` to a wall.
    ///
    /// Tangent and look-ahead are derived every frame from the NavMeshAgent's *live*
    /// `Nav.path.corners` (never from a separately-precomputed polyline) - this is what
    /// guarantees the deviation model can never drift out of sync with where the agent is
    /// actually steering (e.g. right after the waypoint -> end re-path), which was the root
    /// cause of the offset occasionally being computed in the wrong direction and pushing the
    /// recorded body toward/through a wall. As a second, independent safety net, ApplyRig()
    /// hard-snaps the final body X/Z (not just Y) back onto the NavMesh via
    /// NavMesh.SamplePosition, so a recorded position off the walkable surface is not possible
    /// regardless of any upstream computation.
    ///
    /// The agent only drives transforms. ParticipantRecorder derives pose, action_vel and
    /// action_rot from the camera's motion between samples exactly as it does for a human, so the
    /// episode format cannot drift from the human recordings:
    ///   Root (NavMeshAgent + this)          - on the NavMesh, Unity's own live-steered position
    ///     Body (ParticipantRecorder)        - path position + lateral offset, snapped onto the NavMesh
    ///       Head (Camera + CanonicalRecordingCamera) - headHeightMeters above Body; yaw leads the body,
    ///                                                   small OU pitch, never roll
    /// </summary>
    [RequireComponent(typeof(NavMeshAgent))]
    public class RecordRandomWalksAgent : MonoBehaviour
    {
        [Header("Rig")]
        [Tooltip("Camera height above the walked surface. 1.78 = mean camera height of the human episodes.")]
        public float headHeightMeters = 1.78f;

        [Header("Deviation from planned path (OU / AR(1) lateral offset)")]
        [Tooltip("Stationary standard deviation of the lateral offset, meters.")]
        public float lateralSigmaMeters = 0.3f;
        [Tooltip("Correlation time of the lateral offset, seconds.")]
        public float lateralTauSeconds = 1.5f;
        [Tooltip("The offset is clamped so the body stays at least this far from the NavMesh boundary (walls).")]
        public float edgeClearanceMeters = 0.3f;
        [Tooltip("Hard cap on |offset| in units of sigma.")]
        public float maxOffsetSigmas = 3f;

        [Header("Head / body")]
        [Tooltip("Head looks at the point this far ahead on the live NavMeshAgent path.")]
        public float lookAheadMeters = 3f;
        [Tooltip("Head yaw rate limit, degrees/second.")]
        public float headYawRateDegPerSec = 180f;
        [Tooltip("Body yaw follows the head with this rate limit, degrees/second.")]
        public float bodyTurnRateDegPerSec = 90f;
        [Tooltip("Pitch is OU noise around 0 (never roll). Stationary sigma, degrees.")]
        public float pitchSigmaDegrees = 5f;
        public float pitchTauSeconds = 1f;
        public float pitchMaxDegrees = 15f;

        [Header("Routing")]
        [Tooltip("The agent switches from the waypoint leg to the end leg when this close to the waypoint.")]
        public float waypointSwitchDistanceMeters = 0.5f;

        public Transform Body { get; private set; }
        public Transform Head { get; private set; }
        public ParticipantRecorder Recorder { get; private set; }
        public NavMeshAgent Nav { get; private set; }
        public bool IsWalking { get; private set; }

        private System.Random rng;
        private Vector3 waypoint;
        private Vector3 endPoint;
        private int leg;

        private float lateralOffset;
        private float pitchDegrees;
        private float headYaw;
        private float bodyYaw;

        /// <summary>Builds the Body/Head rig and the ParticipantRecorder. Call once, right after AddComponent.</summary>
        public void BuildRig(LayerMask architectureMask, TaskScript task)
        {
            Nav = GetComponent<NavMeshAgent>();
            Nav.updateRotation = false;
            Nav.autoBraking = false;
            Nav.stoppingDistance = 0f;
            Nav.acceleration = 100f;
            Nav.obstacleAvoidanceType = ObstacleAvoidanceType.NoObstacleAvoidance;
            // Matches this project's own convention (see AgentScript.cs: navMeshAgent.radius = agentRadius) -
            // otherwise this would silently sit at Unity's NavMeshAgent default (0.5m) instead of task.agentRadius.
            // Note this does NOT change which NavMesh polygons exist - that's fixed at bake time - it only affects
            // this agent's own clearance queries.
            if (task.agentRadius > 0f) Nav.radius = task.agentRadius;

            var bodyGO = new GameObject("RRW_Body");
            bodyGO.transform.SetParent(transform, false);
            Body = bodyGO.transform;

            var headGO = new GameObject("RRW_Head");
            headGO.transform.SetParent(Body, false);
            headGO.transform.localPosition = new Vector3(0f, headHeightMeters, 0f);
            Head = headGO.transform;
            Camera cam = headGO.AddComponent<Camera>();
            cam.fieldOfView = ZarrDataContract.VerticalFovDegrees;
            headGO.AddComponent<CanonicalRecordingCamera>();

            Recorder = bodyGO.AddComponent<ParticipantRecorder>();
            Recorder.task = task;
            Recorder.ArchitectureLayerMask = architectureMask;
            Recorder.OutputFormat = RecordingOutputFormat.Binary;
            Recorder.EnableDebugRays = false;
            Recorder.AutoStartForTesting = false;
            Recorder.suppressPythonPipeline = true;
        }

        /// <summary>Warps the agent to `startPos` and begins walking start -> waypointPos -> endPos.</summary>
        public void BeginWalk(Vector3 startPos, Vector3 waypointPos, Vector3 endPos, float speed, System.Random random)
        {
            rng = random;
            waypoint = waypointPos;
            endPoint = endPos;

            Nav.speed = speed;
            Nav.Warp(startPos);
            leg = 0;
            Nav.SetDestination(waypoint);

            float sigma = lateralSigmaMeters;
            lateralOffset = Mathf.Clamp(Gaussian() * sigma, -maxOffsetSigmas * sigma, maxOffsetSigmas * sigma);
            pitchDegrees = Mathf.Clamp(Gaussian() * pitchSigmaDegrees, -pitchMaxDegrees, pitchMaxDegrees);

            // Nav.path isn't ready yet the instant after SetDestination (pathPending), so seed the initial
            // heading from the raw direction to the waypoint instead of the (not-yet-computed) live corners.
            Vector3 tangent = waypointPos - startPos;
            tangent.y = 0f;
            tangent = tangent.sqrMagnitude > 1e-6f ? tangent.normalized : Vector3.forward;
            headYaw = Mathf.Atan2(tangent.x, tangent.z) * Mathf.Rad2Deg;
            bodyYaw = headYaw;

            IsWalking = true;
            ApplyRig(startPos, tangent, 0f);
        }

        public void StopWalk()
        {
            IsWalking = false;
            if (Nav != null && Nav.isOnNavMesh) Nav.ResetPath();
        }

        private void LateUpdate()
        {
            if (!IsWalking) return;
            float dt = Time.deltaTime;
            if (dt <= 0f) return;

            if (leg == 0 && !Nav.pathPending && Nav.remainingDistance <= waypointSwitchDistanceMeters)
            {
                leg = 1;
                Nav.SetDestination(endPoint);
            }

            Vector3 pos = transform.position; // the NavMeshAgent's own, always-on-mesh live position
            Vector3[] corners = (Nav.hasPath && !Nav.pathPending) ? Nav.path.corners : null;
            Vector3 tangent = ComputeTangent(corners, pos);
            Vector3 lateralDir = Vector3.Cross(Vector3.up, tangent).normalized;

            // OU / AR(1): x' = x*exp(-dt/tau) + sigma*sqrt(1-exp(-2dt/tau))*N(0,1)
            lateralOffset = StepOu(lateralOffset, lateralSigmaMeters, lateralTauSeconds, dt);
            float maxOffset = maxOffsetSigmas * lateralSigmaMeters;
            float allowedPositive = AllowedAlong(pos, lateralDir, maxOffset);
            float allowedNegative = AllowedAlong(pos, -lateralDir, maxOffset);
            lateralOffset = Mathf.Clamp(lateralOffset, -allowedNegative, allowedPositive);

            pitchDegrees = Mathf.Clamp(StepOu(pitchDegrees, pitchSigmaDegrees, pitchTauSeconds, dt), -pitchMaxDegrees, pitchMaxDegrees);

            Vector3 lookTarget = corners != null && corners.Length >= 2
                ? PointAheadOnCorners(corners, pos, lookAheadMeters)
                : pos + tangent * lookAheadMeters;
            Vector3 look = lookTarget - pos;
            look.y = 0f;
            if (look.sqrMagnitude > 1e-4f)
            {
                float desiredYaw = Mathf.Atan2(look.x, look.z) * Mathf.Rad2Deg;
                headYaw = Mathf.MoveTowardsAngle(headYaw, desiredYaw, headYawRateDegPerSec * dt);
            }
            bodyYaw = Mathf.MoveTowardsAngle(bodyYaw, headYaw, bodyTurnRateDegPerSec * dt);

            ApplyRig(pos, tangent, lateralOffset);
        }

        private void ApplyRig(Vector3 pathPos, Vector3 tangent, float offset)
        {
            Vector3 lateralDir = Vector3.Cross(Vector3.up, tangent).normalized;
            Vector3 candidate = pathPos + lateralDir * offset;

            // Hard guarantee, independent of the clamp above: the recorded body's X/Z (not just Y) is
            // always snapped onto the NavMesh. Search radius is intentionally tight (edgeClearanceMeters)
            // so this never "teleports" the body across a thin wall to a nearby room's mesh - it should
            // only ever correct small residual error, since AllowedAlong() already keeps `candidate` on
            // the correct side of the nearest boundary.
            Vector3 bodyPos = NavMesh.SamplePosition(candidate, out NavMeshHit hit, Mathf.Max(edgeClearanceMeters, 0.05f), NavMesh.AllAreas)
                ? hit.position
                : pathPos;

            Body.position = bodyPos;
            Body.rotation = Quaternion.Euler(0f, bodyYaw, 0f);
            Head.position = bodyPos + Vector3.up * headHeightMeters;
            Head.rotation = Quaternion.Euler(pitchDegrees, headYaw, 0f);
        }

        // ------------------------------------------------------------------
        // Deviation helpers
        // ------------------------------------------------------------------

        private float StepOu(float x, float sigma, float tau, float dt)
        {
            float decay = Mathf.Exp(-dt / Mathf.Max(1e-3f, tau));
            return x * decay + sigma * Mathf.Sqrt(Mathf.Max(0f, 1f - decay * decay)) * Gaussian();
        }

        /// <summary>Free lateral room from `pos` along `dir` before the NavMesh boundary (minus the clearance), capped at `max`.</summary>
        private float AllowedAlong(Vector3 pos, Vector3 dir, float max)
        {
            Vector3 target = pos + dir * (max + edgeClearanceMeters);
            if (NavMesh.Raycast(pos, target, out NavMeshHit hit, NavMesh.AllAreas))
            {
                return Mathf.Clamp(hit.distance - edgeClearanceMeters, 0f, max);
            }
            return max;
        }

        private float Gaussian()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        // ------------------------------------------------------------------
        // Live-path helpers (all driven off Nav.path.corners, refreshed every frame - never a
        // separately-tracked polyline, so this can never drift out of sync with Unity's own
        // steering, e.g. across the waypoint -> end re-path).
        // ------------------------------------------------------------------

        private Vector3 ComputeTangent(Vector3[] corners, Vector3 pos)
        {
            if (corners != null && corners.Length >= 2)
            {
                int seg = NearestSegment(corners, pos);
                Vector3 t = corners[seg + 1] - corners[seg];
                t.y = 0f;
                if (t.sqrMagnitude > 1e-8f) return t.normalized;
            }
            Vector3 v = Nav.desiredVelocity;
            v.y = 0f;
            if (v.sqrMagnitude > 1e-8f) return v.normalized;
            return transform.forward; // last resort: only hit for a stray frame before any path exists
        }

        private static int NearestSegment(Vector3[] corners, Vector3 pos)
        {
            int best = 0;
            float bestD = float.MaxValue;
            for (int i = 0; i < corners.Length - 1; i++)
            {
                Vector3 a = corners[i], b = corners[i + 1];
                Vector3 ab = b - a;
                float len2 = ab.sqrMagnitude;
                float t = len2 > 1e-8f ? Mathf.Clamp01(Vector3.Dot(pos - a, ab) / len2) : 0f;
                float d = (a + ab * t - pos).sqrMagnitude;
                if (d < bestD)
                {
                    bestD = d;
                    best = i;
                }
            }
            return best;
        }

        private static Vector3 PointAheadOnCorners(Vector3[] corners, Vector3 pos, float lookAhead)
        {
            int seg = NearestSegment(corners, pos);
            Vector3 a = corners[seg], b = corners[seg + 1];
            Vector3 ab = b - a;
            float len2 = ab.sqrMagnitude;
            float t = len2 > 1e-8f ? Mathf.Clamp01(Vector3.Dot(pos - a, ab) / len2) : 0f;
            Vector3 from = Vector3.Lerp(a, b, t);

            float need = lookAhead;
            for (int i = seg; i < corners.Length - 1; i++)
            {
                Vector3 segFrom = i == seg ? from : corners[i];
                float segLen = Vector3.Distance(segFrom, corners[i + 1]);
                if (segLen >= need) return Vector3.Lerp(segFrom, corners[i + 1], segLen > 1e-6f ? need / segLen : 0f);
                need -= segLen;
            }
            return corners[corners.Length - 1];
        }
    }
}
