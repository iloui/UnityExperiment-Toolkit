using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using UnityStandardAssets.CrossPlatformInput;
using UnityStandardAssets.Characters.FirstPerson;

namespace Assets.Scripts.DataRecording
{
    /// <summary>Tuning for RecordRandomWalksAgent. Held by RecordRandomWalksHarness so it can be edited in the Inspector.</summary>
    [Serializable]
    public class RandomWalkBotSettings
    {
        [Header("Keyboard (defaults = ProjectSettings/InputManager 'Vertical')")]
        [Tooltip("Axis ramp-up per second while the key is held (InputManager sensitivity). 3 -> full walking speed after 0.33 s, " +
                 "which is where the human 1.2 m/s samples come from.")]
        public float verticalSensitivity = 3f;
        [Tooltip("Axis fall-off per second after release (InputManager gravity; 100 = effectively instant).")]
        public float verticalGravity = 100f;

        [Header("Mouse steering (humans: yaw is exactly 0 in ~48% of samples, turns come in bursts)")]
        [Tooltip("The bot aims at the point this far ahead on the NavMesh path to its current target.")]
        public float lookAheadMeters = 2.5f;
        [Tooltip("A mouse turn starts once the heading error exceeds this...")]
        public float yawDeadbandDegrees = 6f;
        [Tooltip("...and stops (mouse still, yaw label exactly 0) once it is below this.")]
        public float yawSettleDegrees = 1.5f;
        [Tooltip("Turn rate while turning = gain x remaining error, per second, clamped to [minYawRate, maxYawRate].")]
        public float yawGainPerSecond = 5f;
        public float minYawRateDegPerSec = 30f;
        [Tooltip("Human p99 is ~33 deg per 0.1 s sample.")]
        public float maxYawRateDegPerSec = 300f;
        [Tooltip("Slow random offset on the aim direction (steering imprecision), degrees, OU stationary sigma / correlation time.")]
        public float headingWanderSigmaDegrees = 4f;
        public float headingWanderTauSeconds = 4f;

        [Header("Route offset (walk beside the planned line, not on it)")]
        [Tooltip("The line the bot steers along is the NavMesh path shifted sideways by a normally distributed offset: " +
                 "stationary sigma in meters. Each walk starts with an offset drawn from N(0, sigma^2). 0 = follow the path exactly. " +
                 "The NavMesh path hugs every corner at the bake radius, so without this all walks take the same line.")]
        public float lateralOffsetSigmaMeters = 0.35f;
        [Tooltip("Correlation time of the offset, seconds (Ornstein-Uhlenbeck). Long on purpose: the offset is not observable " +
                 "by the model, so it should drift slowly rather than cause unexplained turns.")]
        public float lateralOffsetTauSeconds = 5f;
        [Tooltip("Hard cap on |offset| in units of sigma.")]
        public float lateralOffsetMaxSigmas = 2.5f;
        [Tooltip("The shifted aim point keeps at least this gap between the character capsule and the nearest wall to the side.")]
        public float lateralOffsetWallMarginMeters = 0.15f;
        [Tooltip("The offset fades to 0 over this distance to the current target (waypoint/end), so arrival stays on target.")]
        public float lateralOffsetFadeDistanceMeters = 3f;

        [Header("Mouse pitch")]
        [Tooltip("The bot's intended look-up/down angle wanders around level with this OU sigma (deg) / tau (s)...")]
        public float pitchSigmaDegrees = 4f;
        public float pitchTauSeconds = 3f;
        [Tooltip("...and is followed in the same deadband bursts as yaw.")]
        public float pitchDeadbandDegrees = 3f;
        public float pitchGainPerSecond = 4f;
        public float maxPitchRateDegPerSec = 90f;

        [Header("Walking key")]
        [Tooltip("Walk key goes down once the heading error is below this...")]
        public float walkStartDegrees = 25f;
        [Tooltip("...and up (stand and turn on the spot) once it exceeds this.")]
        public float turnInPlaceDegrees = 60f;
        [Tooltip("Every walk starts at the path heading plus a uniform random offset in +-this.")]
        public float startHeadingJitterDegrees = 180f;

        [Header("Look-around pauses")]
        [Tooltip("Mean pauses per minute of walking (Poisson). 0 disables.")]
        public float lookAroundRatePerMinute = 1.5f;
        public float lookAroundMinYawDegrees = 35f;
        public float lookAroundMaxYawDegrees = 90f;
        [Tooltip("Mean hold time per glance, seconds.")]
        public float lookAroundDwellSeconds = 0.5f;
        [Range(0f, 1f)] public float lookAroundSecondGlanceProbability = 0.5f;

        [Header("Wall excursions (walk up to a wall, stop, turn away)")]
        [Tooltip("Mean excursions per minute of walking (Poisson). 0 disables.")]
        public float wallExcursionRatePerMinute = 2f;
        public float wallExcursionMinAngleDegrees = 35f;
        public float wallExcursionMaxAngleDegrees = 85f;
        [Tooltip("Only walls this far away (from the rig's centre) are used.")]
        public float wallExcursionMinDistanceMeters = 1f;
        public float wallExcursionMaxDistanceMeters = 5f;
        [Tooltip("The walk key is released when the gap between the character capsule and the wall is inside this range (sampled per excursion).")]
        public float wallStopGapMinMeters = 0.05f;
        public float wallStopGapMaxMeters = 0.5f;
        [Tooltip("Mean time standing at the wall before turning away, seconds.")]
        public float wallDwellSeconds = 0.4f;

        [Header("Behaviour scheduling")]
        public float behaviourCooldownSeconds = 4f;
        [Tooltip("No pause/excursion when closer than this to the current target.")]
        public float behaviourMinTargetDistanceMeters = 3f;
        [Tooltip("The NavMesh path to the current target is recomputed this often, seconds.")]
        public float pathRefreshSeconds = 0.25f;

        [Header("Getting unstuck")]
        [Tooltip("Walk key held but the capsule moving slower than this (m/s)...")]
        public float stallSpeedMps = 0.15f;
        [Tooltip("...for this long counts as stuck: release the key and steer to a close, reachable path point.")]
        public float stallSeconds = 0.5f;
        [Tooltip("How long the short look-ahead is used after getting stuck, seconds.")]
        public float recoverySeconds = 1.5f;
        [Tooltip("Look-ahead used while recovering, meters.")]
        public float recoveryLookAheadMeters = 0.75f;
    }

    /// <summary>
    /// Synthetic "participant" for RecordRandomWalksHarness: plays the scene's FPSController exactly like a human does,
    /// through CrossPlatformInputManager's virtual "Vertical" (walk key), "Mouse X" and "Mouse Y" axes. Everything
    /// else - FirstPersonController (1.3 m/s, 50 Hz FixedUpdate, CharacterController radius 0.25 with wall sliding),
    /// MouseLook (yaw on the body, pitch on the camera), the FirstPersonCharacter camera and the ParticipantRecorder on
    /// the rig - is the unmodified human rig, so pose, images, goal vector and action labels are produced by the same
    /// code path as a human recording. The previous NavMeshAgent walker moved sideways relative to its camera (median
    /// 32 deg), jittered ~10 cm per sample and produced continuous speed/yaw labels that no human recording contains.
    ///
    /// Steering is pure pursuit on the NavMesh path to the recorder's current target (the waypoint POI, then the end):
    /// mouse turns start above a heading-error deadband and stop below a settle threshold, so yaw comes in bursts and is
    /// exactly 0 in between, like the humans'. The walk key is released for errors above turnInPlaceDegrees, so large
    /// reorientations (including every episode's random start heading) happen standing, then walking resumes.
    /// Look-around pauses and wall excursions (walk up to a wall, stop, turn away) add stop-and-turn situations.
    ///
    /// Runs before FirstPersonController (execution order) so this frame's mouse delta is used this frame.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    public class RecordRandomWalksAgent : MonoBehaviour
    {
        public RandomWalkBotSettings settings = new RandomWalkBotSettings();

        public ParticipantRecorder Recorder { get; private set; }
        public bool IsActive { get; private set; }

        /// <summary>Behaviour counters for the current walk (logged by the harness).</summary>
        public int TurnsInPlace { get; private set; }
        public int LookArounds { get; private set; }
        public int WallExcursions { get; private set; }
        public int StallRecoveries { get; private set; }

        private const string VerticalAxis = "Vertical";
        private const string HorizontalAxis = "Horizontal";
        private const string MouseXAxis = "Mouse X";
        private const string MouseYAxis = "Mouse Y";

        private enum Mode { Walk, LookAround, WallApproach, WallDwell }

        private FirstPersonController fps;
        private CharacterController cc;
        private Transform cam;
        private LayerMask architectureMask;
        private System.Random rng;

        private Mode mode;
        private bool keyDown;
        private float vertical;
        private bool yawTurning;
        private bool pitchTurning;
        private float headingWander;
        private float pitchTarget;
        private float behaviourCooldown;
        private float modeTimer;
        private float stallTimer;
        private float walkStallTimer;
        private float recoverUntil;
        private float lateralOffset;

        // Created in Configure(), not in a field initializer: Unity runs initializers in the constructor, where native
        // engine objects like NavMeshPath are not allocated (CalculatePath then throws a NullReferenceException).
        private NavMeshPath path;
        private Vector3[] corners;
        private float nextPathTime;
        private Transform pathTarget;

        private readonly List<float> lookYawTargets = new List<float>();
        private readonly List<float> lookDwellTimes = new List<float>();
        private int lookIndex;
        private Vector3 wallPoint;
        private float wallStopGap;

        public void Configure(FirstPersonController controller, Transform camera, ParticipantRecorder recorder, LayerMask architecture)
        {
            fps = controller;
            cc = controller.GetComponent<CharacterController>();
            cam = camera;
            Recorder = recorder;
            architectureMask = architecture;
            if (path == null) path = new NavMeshPath();
        }

        /// <summary>Starts playing. The rig must already stand at the start with its (random) start heading.</summary>
        public void Begin(System.Random random)
        {
            rng = random;
            mode = Mode.Walk;
            keyDown = false;
            vertical = 0f;
            yawTurning = pitchTurning = false;
            headingWander = 0f;
            pitchTarget = 0f;
            behaviourCooldown = settings.behaviourCooldownSeconds;
            stallTimer = 0f;
            walkStallTimer = 0f;
            recoverUntil = 0f;
            float maxOffset = settings.lateralOffsetMaxSigmas * settings.lateralOffsetSigmaMeters;
            lateralOffset = Mathf.Clamp(Gaussian() * settings.lateralOffsetSigmaMeters, -maxOffset, maxOffset); // N(0, sigma^2) at the start
            corners = null;
            pathTarget = null;
            nextPathTime = 0f;
            TurnsInPlace = LookArounds = WallExcursions = StallRecoveries = 0;
            IsActive = true;
        }

        /// <summary>Releases all keys / stops the mouse. Only valid while CrossPlatformInputManager is on virtual (Touch) input -
        /// SetAxis throws on hardware input.</summary>
        public void Stop()
        {
            IsActive = false;
            SetAxes(0f, 0f, 0f);
        }

        private void Update()
        {
            if (!IsActive || fps == null) return;
            float dt = Time.deltaTime;
            Transform target = Recorder != null ? Recorder.CurrentTarget : null;
            if (dt <= 0f || target == null)
            {
                SetAxes(0f, 0f, 0f);
                return;
            }

            Vector3 pos = transform.position;
            float yaw = transform.eulerAngles.y;
            UpdatePath(pos, target);

            headingWander = StepOu(headingWander, settings.headingWanderSigmaDegrees, settings.headingWanderTauSeconds, dt);
            float offsetCap = settings.lateralOffsetMaxSigmas * settings.lateralOffsetSigmaMeters;
            lateralOffset = Mathf.Clamp(StepOu(lateralOffset, settings.lateralOffsetSigmaMeters, settings.lateralOffsetTauSeconds, dt), -offsetCap, offsetCap);
            pitchTarget = StepOu(pitchTarget, settings.pitchSigmaDegrees, settings.pitchTauSeconds, dt);
            behaviourCooldown -= dt;

            // --- aim ---
            float aimYaw;
            switch (mode)
            {
                case Mode.LookAround:
                    aimYaw = lookYawTargets[lookIndex];
                    break;
                case Mode.WallApproach:
                case Mode.WallDwell:
                    aimYaw = YawTo(pos, wallPoint, yaw);
                    break;
                default:
                    aimYaw = YawTo(pos, LookAheadPoint(pos, target), yaw) + headingWander;
                    break;
            }
            float error = Mathf.DeltaAngle(yaw, aimYaw);

            // --- behaviours ---
            switch (mode)
            {
                case Mode.Walk:
                    // Pushing into a wall: the CharacterController stops dead when the push is head-on, and the aim point
                    // (computed from a position that no longer changes) never moves - release and re-aim close by.
                    walkStallTimer = keyDown && HorizontalSpeed() < settings.stallSpeedMps ? walkStallTimer + dt : 0f;
                    if (walkStallTimer > settings.stallSeconds)
                    {
                        walkStallTimer = 0f;
                        keyDown = false;
                        recoverUntil = Time.time + settings.recoverySeconds;
                        StallRecoveries++;
                    }
                    if (keyDown && behaviourCooldown <= 0f && HorizontalDistance(pos, target.position) > settings.behaviourMinTargetDistanceMeters)
                    {
                        if (Chance(settings.wallExcursionRatePerMinute, dt)) TryBeginWallExcursion(pos, yaw);
                        else if (Chance(settings.lookAroundRatePerMinute, dt)) BeginLookAround(yaw);
                    }
                    break;

                case Mode.LookAround:
                    if (!yawTurning && Mathf.Abs(error) <= settings.yawDeadbandDegrees)
                    {
                        lookDwellTimes[lookIndex] -= dt;
                        if (lookDwellTimes[lookIndex] <= 0f && ++lookIndex >= lookYawTargets.Count) EndBehaviour();
                    }
                    break;

                case Mode.WallApproach:
                    modeTimer -= dt;
                    stallTimer = keyDown && HorizontalSpeed() < settings.stallSpeedMps ? stallTimer + dt : 0f;
                    bool closeEnough = Physics.Raycast(pos, Flat(transform.forward), out RaycastHit hit, cc.radius + wallStopGap, architectureMask, QueryTriggerInteraction.Ignore);
                    if (closeEnough || stallTimer > 0.5f || modeTimer <= 0f)
                    {
                        mode = Mode.WallDwell;
                        modeTimer = settings.wallDwellSeconds * UniformRange(0.5f, 1.5f);
                    }
                    break;

                case Mode.WallDwell:
                    modeTimer -= dt;
                    if (modeTimer <= 0f) EndBehaviour();
                    break;
            }

            // --- walk key (FirstPersonController reads it in FixedUpdate) ---
            bool mayWalk = mode == Mode.Walk || mode == Mode.WallApproach;
            if (!mayWalk)
            {
                keyDown = false;
            }
            else if (keyDown && Mathf.Abs(error) > settings.turnInPlaceDegrees)
            {
                keyDown = false;
                TurnsInPlace++;
            }
            else if (!keyDown && Mathf.Abs(error) < settings.walkStartDegrees)
            {
                keyDown = true;
            }
            vertical = keyDown ? Mathf.Min(1f, vertical + settings.verticalSensitivity * dt) : Mathf.Max(0f, vertical - settings.verticalGravity * dt);

            // --- mouse (MouseLook applies GetAxis x sensitivity as degrees this frame) ---
            float yawRate = BurstRate(error, ref yawTurning, settings.yawDeadbandDegrees, settings.yawSettleDegrees,
                settings.yawGainPerSecond, settings.minYawRateDegPerSec, settings.maxYawRateDegPerSec);
            float pitch = -Mathf.DeltaAngle(0f, cam.localEulerAngles.x); // positive = looking up
            float pitchRate = BurstRate(pitchTarget - pitch, ref pitchTurning, settings.pitchDeadbandDegrees, settings.pitchDeadbandDegrees * 0.3f,
                settings.pitchGainPerSecond, 5f, settings.maxPitchRateDegPerSec);

            float xSensitivity = Mathf.Max(1e-4f, fps.MouseLook.XSensitivity);
            float ySensitivity = Mathf.Max(1e-4f, fps.MouseLook.YSensitivity);
            SetAxes(vertical, yawRate * dt / xSensitivity, pitchRate * dt / ySensitivity);
        }

        // ------------------------------------------------------------------
        // Steering
        // ------------------------------------------------------------------

        /// <summary>Signed rate (deg/s) that removes `error` in human-like bursts: 0 until |error| passes the deadband, then until it settles.</summary>
        private static float BurstRate(float error, ref bool turning, float deadband, float settle, float gain, float minRate, float maxRate)
        {
            float abs = Mathf.Abs(error);
            if (!turning && abs > deadband) turning = true;
            else if (turning && abs < settle) turning = false;
            if (!turning) return 0f;
            return Mathf.Sign(error) * Mathf.Clamp(gain * abs, minRate, maxRate);
        }

        private void UpdatePath(Vector3 pos, Transform target)
        {
            if (target == pathTarget && Time.time < nextPathTime) return;
            pathTarget = target;
            nextPathTime = Time.time + settings.pathRefreshSeconds;
            corners = null;
            if (!NavMesh.SamplePosition(pos, out NavMeshHit from, 2f, NavMesh.AllAreas)) return;
            if (!NavMesh.SamplePosition(target.position, out NavMeshHit to, 3f, NavMesh.AllAreas)) return;
            if (NavMesh.CalculatePath(from.position, to.position, NavMesh.AllAreas, path) && path.corners.Length >= 2) corners = path.corners;
        }

        /// <summary>
        /// The aim point: the furthest point up to lookAheadMeters along the path - shifted sideways by the current route
        /// offset - that the capsule can reach in a straight line. A plain fixed look-ahead cuts corners: around a wall end
        /// the line to it runs through the wall, and the rig walks into it.
        /// </summary>
        private Vector3 LookAheadPoint(Vector3 pos, Transform target)
        {
            if (corners == null) return target.position;
            Vector3 flatPos = new Vector3(pos.x, corners[0].y, pos.z);
            bool recovering = Time.time < recoverUntil;
            float lookAhead = recovering ? settings.recoveryLookAheadMeters : settings.lookAheadMeters;
            // The offset fades out near the current target (clean arrival) and is not applied while getting unstuck.
            float fade = Mathf.Clamp01((HorizontalDistance(pos, target.position) - 1f) / Mathf.Max(0.01f, settings.lateralOffsetFadeDistanceMeters));
            float offset = recovering ? 0f : lateralOffset * fade;

            for (float d = lookAhead; d > 0.3f; d -= 0.25f)
            {
                Vector3 p = OffsetPoint(PointAheadOnCorners(corners, flatPos, d), offset, pos.y);
                if (Reachable(pos, p)) return p;
            }
            return PointAheadOnCorners(corners, flatPos, 0.3f);
        }

        /// <summary>
        /// `pathPoint` moved sideways (perpendicular to the path there) by `offset`, but never so far that the capsule would
        /// come closer than lateralOffsetWallMarginMeters to a wall on that side.
        /// </summary>
        private Vector3 OffsetPoint(Vector3 pathPoint, float offset, float rigHeight)
        {
            if (Mathf.Abs(offset) < 1e-3f) return pathPoint;
            int seg = NearestSegment(corners, pathPoint);
            Vector3 tangent = corners[seg + 1] - corners[seg];
            tangent.y = 0f;
            if (tangent.sqrMagnitude < 1e-6f) return pathPoint;
            Vector3 side = Vector3.Cross(Vector3.up, tangent.normalized) * Mathf.Sign(offset);
            float clearance = cc.radius + settings.lateralOffsetWallMarginMeters;
            float wanted = Mathf.Abs(offset);
            Vector3 origin = new Vector3(pathPoint.x, rigHeight, pathPoint.z);
            if (Physics.Raycast(origin, side, out RaycastHit wall, wanted + clearance, architectureMask, QueryTriggerInteraction.Ignore))
            {
                wanted = Mathf.Max(0f, wall.distance - clearance);
            }
            return pathPoint + side * wanted;
        }

        private bool Reachable(Vector3 pos, Vector3 point)
        {
            Vector3 to = new Vector3(point.x, pos.y, point.z) - pos;
            float distance = to.magnitude;
            if (distance < 1e-3f) return true;
            // Slightly wider than the capsule. A cast that starts overlapping a wall ignores it, which is fine: the
            // CharacterController's skin keeps its centre at least radius + skin from any wall.
            return !Physics.SphereCast(pos, cc.radius + 0.05f, to / distance, out _, distance, architectureMask, QueryTriggerInteraction.Ignore);
        }

        private float HorizontalSpeed()
        {
            Vector3 v = cc.velocity;
            v.y = 0f;
            return v.magnitude;
        }

        private void BeginLookAround(float yaw)
        {
            float side = rng.NextDouble() < 0.5 ? -1f : 1f;
            lookYawTargets.Clear();
            lookDwellTimes.Clear();
            lookYawTargets.Add(yaw + side * UniformRange(settings.lookAroundMinYawDegrees, settings.lookAroundMaxYawDegrees));
            lookDwellTimes.Add(settings.lookAroundDwellSeconds * UniformRange(0.5f, 1.5f));
            if (rng.NextDouble() < settings.lookAroundSecondGlanceProbability)
            {
                lookYawTargets.Add(yaw - side * UniformRange(settings.lookAroundMinYawDegrees, settings.lookAroundMaxYawDegrees));
                lookDwellTimes.Add(settings.lookAroundDwellSeconds * UniformRange(0.5f, 1.5f));
            }
            lookIndex = 0;
            mode = Mode.LookAround;
            LookArounds++;
        }

        private void TryBeginWallExcursion(Vector3 pos, float yaw)
        {
            float side = rng.NextDouble() < 0.5 ? -1f : 1f;
            for (int attempt = 0; attempt < 2; attempt++, side = -side)
            {
                float angle = yaw + side * UniformRange(settings.wallExcursionMinAngleDegrees, settings.wallExcursionMaxAngleDegrees);
                Vector3 dir = Quaternion.Euler(0f, angle, 0f) * Vector3.forward;
                if (!Physics.Raycast(pos, dir, out RaycastHit hit, settings.wallExcursionMaxDistanceMeters, architectureMask, QueryTriggerInteraction.Ignore)) continue;
                if (hit.distance < settings.wallExcursionMinDistanceMeters) continue;
                // The capsule (radius 0.25) must fit along the way, not just the centre ray.
                if (Physics.SphereCast(pos, cc.radius, dir, out RaycastHit wide, hit.distance, architectureMask, QueryTriggerInteraction.Ignore)
                    && wide.distance < hit.distance - 0.3f) continue;

                wallPoint = hit.point;
                wallStopGap = UniformRange(settings.wallStopGapMinMeters, settings.wallStopGapMaxMeters);
                modeTimer = hit.distance / Mathf.Max(0.3f, fps.WalkSpeed) + 3f;
                stallTimer = 0f;
                mode = Mode.WallApproach;
                WallExcursions++;
                return;
            }
        }

        private void EndBehaviour()
        {
            mode = Mode.Walk;
            behaviourCooldown = settings.behaviourCooldownSeconds;
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        private static void SetAxes(float verticalValue, float mouseX, float mouseY)
        {
            CrossPlatformInputManager.SetAxis(VerticalAxis, verticalValue);
            CrossPlatformInputManager.SetAxis(HorizontalAxis, 0f);
            CrossPlatformInputManager.SetAxis(MouseXAxis, mouseX);
            CrossPlatformInputManager.SetAxis(MouseYAxis, mouseY);
        }

        private static float YawTo(Vector3 from, Vector3 to, float fallback)
        {
            Vector3 d = to - from;
            d.y = 0f;
            return d.sqrMagnitude > 1e-4f ? Mathf.Atan2(d.x, d.z) * Mathf.Rad2Deg : fallback;
        }

        private static Vector3 Flat(Vector3 v)
        {
            v.y = 0f;
            return v.sqrMagnitude > 1e-8f ? v.normalized : Vector3.forward;
        }

        private static float HorizontalDistance(Vector3 a, Vector3 b)
        {
            a.y = b.y = 0f;
            return Vector3.Distance(a, b);
        }

        private bool Chance(float ratePerMinute, float dt)
        {
            if (ratePerMinute <= 0f) return false;
            return rng.NextDouble() < 1.0 - Math.Exp(-ratePerMinute / 60.0 * dt);
        }

        private float UniformRange(float a, float b)
        {
            return a + (float)rng.NextDouble() * (b - a);
        }

        private float StepOu(float x, float sigma, float tau, float dt)
        {
            float decay = Mathf.Exp(-dt / Mathf.Max(1e-3f, tau));
            return x * decay + sigma * Mathf.Sqrt(Mathf.Max(0f, 1f - decay * decay)) * Gaussian();
        }

        private float Gaussian()
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        internal static int NearestSegment(Vector3[] corners, Vector3 pos)
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

        internal static Vector3 PointAheadOnCorners(Vector3[] corners, Vector3 pos, float lookAhead)
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
