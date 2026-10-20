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
        [Tooltip("The aim point is pushed away from walls on either side of the path until the capsule centre keeps this distance " +
                 "from them (centred where the passage is narrower). The NavMesh path hugs every wall at the bake radius, so the " +
                 "v3 bot walks ~0.5 m closer to walls than the participants. 0 = off (v3 behaviour).")]
        public float pathWallClearanceMeters = 0f;

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

        // v3 stop behaviours. Humans stand in 20.5% of samples (~4 stops per episode, median 2.0 s, 76% in open space,
        // 40% of standing samples turning); the v2 bot stood in 4.3% (median 0.4 s), so its data barely showed the model
        // when to stop and turn. All new fields, so existing scene components pick up these defaults.
        [Header("Stop: start orientation")]
        [Tooltip("Every walk starts standing for this long (uniform range, seconds) before the first turn / step.")]
        public float startPauseMinSeconds = 0.5f;
        public float startPauseMaxSeconds = 3f;
        [Tooltip("Chance of a glance to one side during the start pause.")]
        [Range(0f, 1f)] public float startGlanceProbability = 0.5f;

        [Header("Stop: orient-and-scan in open space")]
        [Tooltip("Mean scan stops per minute of walking (Poisson), on top of look-around pauses. 0 disables.")]
        public float scanStopRatePerMinute = 3f;
        [Tooltip("Standing still at the current heading before any glance (uniform range, seconds).")]
        public float scanHoldMinSeconds = 0.5f;
        public float scanHoldMaxSeconds = 2.5f;
        [Tooltip("Glances per scan stop: 0 with this probability, otherwise 1 or 2 (50/50).")]
        [Range(0f, 1f)] public float scanNoGlanceProbability = 0.35f;
        [Tooltip("Mean hold time per scan glance, seconds.")]
        public float scanGlanceDwellSeconds = 0.8f;

        [Header("Stop: turn in place at walls")]
        [Tooltip("While walking, a wall closer than this straight ahead (from the capsule centre)...")]
        public float wallStopDistanceMeters = 1.5f;
        [Tooltip("...with the path heading off by more than this makes the bot stop, stand, then turn on the spot toward the path.")]
        public float wallStopMinErrorDegrees = 20f;
        [Tooltip("Standing time facing the wall before turning (uniform range, seconds).")]
        public float wallStopDwellMinSeconds = 0.3f;
        public float wallStopDwellMaxSeconds = 1.2f;
        [Tooltip("No new wall stop within this many seconds of the last one.")]
        public float wallStopCooldownSeconds = 2f;

        [Header("Stop: pause at POI")]
        [Tooltip("On reaching a POI (the recorder switches to the next target) the bot stands for this long (uniform range, seconds)...")]
        public float poiPauseMinSeconds = 0.5f;
        public float poiPauseMaxSeconds = 2f;
        [Tooltip("...with a glance to one side with this probability, then turns toward the next target.")]
        [Range(0f, 1f)] public float poiGlanceProbability = 0.3f;

        // v5: the pocket loop. A memoryless policy cannot see which side the NavMesh path takes around a wall, so wall-stop
        // frames ("wall ahead, goal behind it -> stand, then turn left / right") teach it to stand at the wall but not
        // where to go. v5 turns while walking where there is room, stands at most once per spot, and - where both ways
        // around are about equally long - takes the side that looks more open, which a single frame does show.
        [Header("Pocket handling (v5; defaults = off)")]
        [Tooltip("At a wall stop with room towards the path side, keep walking and turn in a curve instead of standing and turning on the spot.")]
        public bool wallTurnWhileWalking = false;
        [Tooltip("Room = a capsule-wide cast 60 deg towards the path side is free for at least this far, meters.")]
        public float wallTurnMinSideClearanceMeters = 0.6f;
        [Tooltip("The walk key stays down through a walking turn for this long, even while the heading error exceeds turnInPlaceDegrees, seconds.")]
        public float wallTurnHoldSeconds = 1f;
        [Tooltip("At most this many standing wall stops per walk within wallStopRepeatRadiusMeters of each other; further ones turn while " +
                 "walking (or fall back to the plain walk-key rule if there is no room). 0 = no limit.")]
        [Min(0)] public int maxWallStopsPerSpot = 0;
        public float wallStopRepeatRadiusMeters = 2f;
        [Tooltip("Facing a wall (wallStopDistanceMeters) with the path off by more than wallStopMinErrorDegrees: if the routes around " +
                 "either side differ by less than this factor, walk towards the side that looks more open (longer free cast at " +
                 "+-sideProbeAngleDegrees) instead of the NavMesh side. 0 = off.")]
        public float sideCommitMaxLengthRatio = 0f;
        public float sideProbeAngleDegrees = 60f;
        public float sideProbeMaxDistanceMeters = 6f;
        [Tooltip("A side commitment ends within this distance of its aim point...")]
        public float sideCommitArrivalMeters = 1f;
        [Tooltip("...or after (distance / walk speed + this) seconds.")]
        public float sideCommitSlackSeconds = 4f;

        /// <summary>participant_id of the recalibrated bot (RecalibratedV4).</summary>
        public const string V4ParticipantId = "synthetic_rrw_v4";
        /// <summary>participant_id of the recalibrated bot with pocket handling and hidden-goal tasks (RecalibratedV5).</summary>
        public const string V5ParticipantId = "synthetic_rrw_v5";

        /// <summary>
        /// v4: the v3 bot refitted to the thesis participants P01-P08 (96 episodes) instead of the pilot. Measured with
        /// eval_tools/compare_bot_to_humans.py on the same 0.1 s grid and 0.65 m/s stop threshold as the evaluation:
        ///
        ///                                   P01-P08      v3 bot
        ///   stop share                      8.7%         22.5%
        ///   start pause (median)            0.7 s        2.4 s
        ///   mid-walk stop length (median)   0.8 s        2.0 s
        ///   mid-walk stops within 0.6 m of a wall  17%   41%
        ///   wall distance while walking (median)   1.58 m  1.09 m
        ///   yaw exactly 0                   39%          61%
        ///
        /// So the stops get shorter and rarer near walls (fewer excursions, wall stops only for walls that really block
        /// the way), the line moves away from walls (pathWallClearanceMeters, wider offset margin), and mouse turns start
        /// earlier. Everything not listed keeps its v3 value.
        /// </summary>
        public static RandomWalkBotSettings RecalibratedV4()
        {
            return new RandomWalkBotSettings
            {
                // yaw: humans turn more continuously than the v3 deadband allowed
                yawDeadbandDegrees = 4f,
                yawSettleDegrees = 1f,

                // wall clearance
                lateralOffsetWallMarginMeters = 0.5f,
                pathWallClearanceMeters = 0.9f,

                // short look-arounds
                lookAroundRatePerMinute = 1f,
                lookAroundDwellSeconds = 0.35f,
                lookAroundSecondGlanceProbability = 0.3f,

                // few wall excursions, short dwell
                wallExcursionRatePerMinute = 0.5f,
                wallDwellSeconds = 0.3f,

                // start: humans set off after 0.2-1.4 s (p10-p90)
                startPauseMinSeconds = 0.2f,
                startPauseMaxSeconds = 1.4f,
                startGlanceProbability = 0.2f,

                // scan stops: half as often, a third as long, mostly without a glance
                scanStopRatePerMinute = 1.5f,
                scanHoldMinSeconds = 0.3f,
                scanHoldMaxSeconds = 1.2f,
                scanNoGlanceProbability = 0.6f,
                scanGlanceDwellSeconds = 0.4f,

                // wall stops only for a wall that is close and far off the path
                wallStopDistanceMeters = 0.8f,
                wallStopMinErrorDegrees = 45f,
                wallStopDwellMinSeconds = 0.2f,
                wallStopDwellMaxSeconds = 0.6f,
                wallStopCooldownSeconds = 4f,

                // POI
                poiPauseMinSeconds = 0.2f,
                poiPauseMaxSeconds = 0.8f,
                poiGlanceProbability = 0.15f,
            };
        }

        /// <summary>
        /// v5 = v4 + pocket handling: walking turns at walls, one standing wall stop per spot, and visible-side commitment
        /// where the two ways around are within 25% of each other. Recorded together with hidden-goal tasks
        /// (RecordRandomWalksHarness.hiddenGoalShare, set by "Record Random Walks v5").
        /// </summary>
        public static RandomWalkBotSettings RecalibratedV5()
        {
            RandomWalkBotSettings s = RecalibratedV4();
            s.wallTurnWhileWalking = true;
            s.wallTurnMinSideClearanceMeters = 0.6f;
            s.wallTurnHoldSeconds = 1f;
            s.maxWallStopsPerSpot = 1;
            s.wallStopRepeatRadiusMeters = 2f;
            s.sideCommitMaxLengthRatio = 1.25f;
            s.sideProbeAngleDegrees = 60f;
            s.sideProbeMaxDistanceMeters = 6f;
            s.sideCommitArrivalMeters = 1f;
            s.sideCommitSlackSeconds = 4f;
            return s;
        }
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
    /// v3 adds stops the humans show and v2 lacked: a standing start, orient-and-scan stops in open space, stopping
    /// in front of a wall that blocks the path and turning on the spot toward it, and a pause at each POI. All of them
    /// are "stops": a LookAround sequence whose first entry holds the current heading.
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
        public int ScanStops { get; private set; }
        public int WallStops { get; private set; }
        public int PoiPauses { get; private set; }
        /// <summary>v5: wall stops replaced by a walking turn, wall stops over the per-spot limit, near-equal side decisions,
        /// and commitments to the visibly more open side against the NavMesh side.</summary>
        public int WalkingTurns { get; private set; }
        public int SuppressedWallStops { get; private set; }
        public int SideDecisions { get; private set; }
        public int SideCommits { get; private set; }

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
        private float nextWallStopTime;
        private Transform lastTarget;

        // v5 pocket handling
        private bool commitActive;
        private Vector3 commitPoint;
        private Transform commitTarget;
        private float commitUntil;
        private bool pathCommitted;
        private float nextSideDecisionTime;
        private float walkTurnUntil;
        private readonly List<Vector3> wallStopSpots = new List<Vector3>();
        private NavMeshPath probePath;

        public void Configure(FirstPersonController controller, Transform camera, ParticipantRecorder recorder, LayerMask architecture)
        {
            fps = controller;
            cc = controller.GetComponent<CharacterController>();
            cam = camera;
            Recorder = recorder;
            architectureMask = architecture;
            if (path == null) path = new NavMeshPath();
            if (probePath == null) probePath = new NavMeshPath();
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
            TurnsInPlace = LookArounds = WallExcursions = StallRecoveries = ScanStops = WallStops = PoiPauses = 0;
            WalkingTurns = SuppressedWallStops = SideDecisions = SideCommits = 0;
            nextWallStopTime = 0f;
            lastTarget = null;
            commitActive = pathCommitted = false;
            commitTarget = null;
            nextSideDecisionTime = 0f;
            walkTurnUntil = 0f;
            wallStopSpots.Clear();
            // Start orientation: stand (and maybe glance) before the first turn toward the path.
            BeginStop(transform.eulerAngles.y, UniformRange(settings.startPauseMinSeconds, settings.startPauseMaxSeconds),
                rng.NextDouble() < settings.startGlanceProbability ? 1 : 0, settings.lookAroundDwellSeconds);
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
            if (commitActive && (target != commitTarget || HorizontalDistance(pos, commitPoint) < settings.sideCommitArrivalMeters || Time.time > commitUntil))
            {
                commitActive = false;
            }
            UpdatePath(pos, target);

            // Pause at POI: the recorder switched to the next target, so stand (maybe glance) before turning toward it.
            if (lastTarget != null && target != lastTarget)
            {
                BeginStop(yaw, UniformRange(settings.poiPauseMinSeconds, settings.poiPauseMaxSeconds),
                    rng.NextDouble() < settings.poiGlanceProbability ? 1 : 0, settings.lookAroundDwellSeconds);
                PoiPauses++;
            }
            lastTarget = target;

            headingWander = StepOu(headingWander, settings.headingWanderSigmaDegrees, settings.headingWanderTauSeconds, dt);
            float offsetCap = settings.lateralOffsetMaxSigmas * settings.lateralOffsetSigmaMeters;
            lateralOffset = Mathf.Clamp(StepOu(lateralOffset, settings.lateralOffsetSigmaMeters, settings.lateralOffsetTauSeconds, dt), -offsetCap, offsetCap);
            pitchTarget = StepOu(pitchTarget, settings.pitchSigmaDegrees, settings.pitchTauSeconds, dt);
            behaviourCooldown -= dt;

            if (mode == Mode.Walk && settings.sideCommitMaxLengthRatio > 0f && !commitActive && Time.time >= nextSideDecisionTime
                && Time.time >= recoverUntil && HorizontalDistance(pos, target.position) > settings.behaviourMinTargetDistanceMeters)
            {
                MaybeCommitToVisibleSide(pos, yaw, target);
            }

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
                    // A wall close ahead while the path turns away: stop facing it, then turn on the spot (the walk key stays
                    // up until the error is below walkStartDegrees). This is the state M4 froze in - left/right both occur
                    // in the demonstrations, but each stop here commits to the path's side.
                    if (keyDown && Time.time >= nextWallStopTime && Mathf.Abs(error) > settings.wallStopMinErrorDegrees
                        && HorizontalDistance(pos, target.position) > ZarrDataContract.ArrivalRadiusMeters
                        && Physics.Raycast(pos, Flat(transform.forward), settings.wallStopDistanceMeters, architectureMask, QueryTriggerInteraction.Ignore))
                    {
                        nextWallStopTime = Time.time + settings.wallStopCooldownSeconds;
                        bool repeated = settings.maxWallStopsPerSpot > 0 && WallStopsNear(pos) >= settings.maxWallStopsPerSpot;
                        if (repeated) SuppressedWallStops++;
                        if ((settings.wallTurnWhileWalking || repeated) && SideOpen(pos, yaw, error))
                        {
                            walkTurnUntil = Time.time + settings.wallTurnHoldSeconds;
                            WalkingTurns++;
                            break;
                        }
                        if (repeated) break; // no room either: the walk-key rule below turns on the spot if it must
                        wallStopSpots.Add(pos);
                        BeginStop(yaw, UniformRange(settings.wallStopDwellMinSeconds, settings.wallStopDwellMaxSeconds), 0, 0f);
                        WallStops++;
                        break;
                    }
                    if (keyDown && behaviourCooldown <= 0f && HorizontalDistance(pos, target.position) > settings.behaviourMinTargetDistanceMeters)
                    {
                        if (Chance(settings.wallExcursionRatePerMinute, dt)) TryBeginWallExcursion(pos, yaw);
                        else if (Chance(settings.lookAroundRatePerMinute, dt)) BeginLookAround(yaw);
                        else if (Chance(settings.scanStopRatePerMinute, dt)) BeginScanStop(yaw);
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
            else if (keyDown && Mathf.Abs(error) > settings.turnInPlaceDegrees && Time.time >= walkTurnUntil)
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
            if (target == pathTarget && commitActive == pathCommitted && Time.time < nextPathTime) return;
            pathTarget = target;
            pathCommitted = commitActive;
            nextPathTime = Time.time + settings.pathRefreshSeconds;
            corners = null;
            if (!NavMesh.SamplePosition(pos, out NavMeshHit from, 2f, NavMesh.AllAreas)) return;
            if (!NavMesh.SamplePosition(commitActive ? commitPoint : target.position, out NavMeshHit to, 3f, NavMesh.AllAreas)) return;
            if (NavMesh.CalculatePath(from.position, to.position, NavMesh.AllAreas, path) && path.corners.Length >= 2) corners = path.corners;
        }

        /// <summary>
        /// The aim point: the furthest point up to lookAheadMeters along the path - shifted sideways by the current route
        /// offset - that the capsule can reach in a straight line. A plain fixed look-ahead cuts corners: around a wall end
        /// the line to it runs through the wall, and the rig walks into it.
        /// </summary>
        private Vector3 LookAheadPoint(Vector3 pos, Transform target)
        {
            if (corners == null) return commitActive ? commitPoint : target.position;
            Vector3 flatPos = new Vector3(pos.x, corners[0].y, pos.z);
            bool recovering = Time.time < recoverUntil;
            float lookAhead = recovering ? settings.recoveryLookAheadMeters : settings.lookAheadMeters;
            // The offset fades out near the current target (clean arrival) and is not applied while getting unstuck.
            float fade = Mathf.Clamp01((HorizontalDistance(pos, target.position) - 1f) / Mathf.Max(0.01f, settings.lateralOffsetFadeDistanceMeters));
            float offset = recovering ? 0f : lateralOffset * fade;

            for (float d = lookAhead; d > 0.3f; d -= 0.25f)
            {
                Vector3 p = OffsetPoint(PointAheadOnCorners(corners, flatPos, d), offset, pos.y);
                p = KeepClearOfWalls(p, pos.y, recovering ? 0f : fade);
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

        /// <summary>
        /// `point` moved perpendicular to the path until walls on both sides are at least pathWallClearanceMeters away
        /// (centred between them where the passage is narrower than twice that), scaled by `strength` (the arrival fade).
        /// </summary>
        private Vector3 KeepClearOfWalls(Vector3 point, float rigHeight, float strength)
        {
            float clearance = settings.pathWallClearanceMeters;
            if (clearance <= 0f || strength <= 0f) return point;
            int seg = NearestSegment(corners, point);
            Vector3 tangent = corners[seg + 1] - corners[seg];
            tangent.y = 0f;
            if (tangent.sqrMagnitude < 1e-6f) return point;
            Vector3 right = Vector3.Cross(Vector3.up, tangent.normalized);
            Vector3 origin = new Vector3(point.x, rigHeight, point.z);
            float range = 2f * clearance;
            float toRight = Physics.Raycast(origin, right, out RaycastHit r, range, architectureMask, QueryTriggerInteraction.Ignore) ? r.distance : range;
            float toLeft = Physics.Raycast(origin, -right, out RaycastHit l, range, architectureMask, QueryTriggerInteraction.Ignore) ? l.distance : range;

            float shift = 0f; // + = towards `right`
            if (toRight + toLeft < 2f * clearance) shift = 0.5f * (toRight - toLeft);
            else if (toRight < clearance) shift = toRight - clearance;
            else if (toLeft < clearance) shift = clearance - toLeft;
            return point + right * (shift * strength);
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

        // ------------------------------------------------------------------
        // v5 pocket handling
        // ------------------------------------------------------------------

        /// <summary>
        /// Facing a close wall with the path off to one side: probe both sides at +-sideProbeAngleDegrees, estimate the route
        /// through each (straight to a point short of the probe's hit, then NavMesh to the target), and if the two are within
        /// sideCommitMaxLengthRatio, commit to the side with the longer free probe - the side the camera sees open. Does
        /// nothing when that is the NavMesh side anyway, or when one way around is clearly shorter.
        /// </summary>
        private void MaybeCommitToVisibleSide(Vector3 pos, float yaw, Transform target)
        {
            float naturalError = Mathf.DeltaAngle(yaw, YawTo(pos, LookAheadPoint(pos, target), yaw));
            if (Mathf.Abs(naturalError) <= settings.wallStopMinErrorDegrees) return;
            if (!Physics.Raycast(pos, Flat(transform.forward), settings.wallStopDistanceMeters, architectureMask, QueryTriggerInteraction.Ignore)) return;
            nextSideDecisionTime = Time.time + settings.wallStopCooldownSeconds;
            bool against = ChooseVisibleSide(pos, yaw, naturalError, target.position, cc.radius, settings.sideProbeAngleDegrees,
                settings.sideProbeMaxDistanceMeters, settings.sideCommitMaxLengthRatio, architectureMask, probePath,
                out bool nearEqual, out Vector3 aimPoint);
            if (nearEqual) SideDecisions++;
            if (!against) return;

            commitActive = true;
            commitPoint = aimPoint;
            commitTarget = target;
            commitUntil = Time.time + HorizontalDistance(pos, commitPoint) / Mathf.Max(0.3f, fps.WalkSpeed) + settings.sideCommitSlackSeconds;
            SideCommits++;
            UpdatePath(pos, target); // re-plan to the commitment point now (pathCommitted differs)
        }

        /// <summary>
        /// The visible-side rule, shared with DaggerHarness's expert. From `pos` facing `yaw`, probe both sides at
        /// +-probeAngle with a body-wide cast, estimate the route through each (straight to a point short of the probe's hit,
        /// then NavMesh to `target`), and call the two near-equal when they are within maxLengthRatio. Returns true - with
        /// the aim point on that side - only when they are near-equal AND the visibly more open side is not the one the
        /// NavMesh path takes (`naturalError` > 0 = path to the right).
        /// </summary>
        internal static bool ChooseVisibleSide(Vector3 pos, float yaw, float naturalError, Vector3 target, float bodyRadius,
            float probeAngle, float probeMaxDistance, float maxLengthRatio, LayerMask mask, NavMeshPath scratch,
            out bool nearEqual, out Vector3 aimPoint)
        {
            nearEqual = false;
            aimPoint = Vector3.zero;
            if (!NavMesh.SamplePosition(target, out NavMeshHit targetHit, 3f, NavMesh.AllAreas)) return false;

            var open = new float[2];
            var length = new float[2];
            var aim = new Vector3[2];
            for (int i = 0; i < 2; i++)
            {
                float side = i == 0 ? -1f : 1f;
                Vector3 dir = Quaternion.Euler(0f, yaw + side * probeAngle, 0f) * Vector3.forward;
                float free = Physics.SphereCast(pos, bodyRadius, dir, out RaycastHit hit, probeMaxDistance, mask, QueryTriggerInteraction.Ignore)
                    ? hit.distance : probeMaxDistance;
                open[i] = free;
                length[i] = float.PositiveInfinity;
                float reach = free - 0.3f;
                if (reach < 1f) continue;
                if (!NavMesh.SamplePosition(pos + dir * reach, out NavMeshHit aimHit, 2f, NavMesh.AllAreas)) continue;
                if (!NavMesh.CalculatePath(aimHit.position, targetHit.position, NavMesh.AllAreas, scratch)
                    || scratch.status != NavMeshPathStatus.PathComplete || scratch.corners.Length < 2) continue;
                // The way on must lead further away from here, not back past the walker (that would be the other side).
                Vector3 onward = PointAheadOnCorners(scratch.corners, aimHit.position, 1.5f);
                if (HorizontalDistance(onward, pos) < HorizontalDistance(aimHit.position, pos)) continue;
                aim[i] = aimHit.position;
                length[i] = HorizontalDistance(pos, aimHit.position) + PathLength(scratch.corners);
            }
            if (float.IsInfinity(length[0]) || float.IsInfinity(length[1])) return false;
            if (Mathf.Max(length[0], length[1]) > maxLengthRatio * Mathf.Min(length[0], length[1])) return false;

            nearEqual = true;
            int chosen = open[1] >= open[0] ? 1 : 0;
            int naturalSide = naturalError > 0f ? 1 : 0; // + = clockwise = right
            if (chosen == naturalSide) return false;
            aimPoint = aim[chosen];
            return true;
        }

        /// <summary>A capsule-wide cast 60 deg towards the side of `error` is free for wallTurnMinSideClearanceMeters.</summary>
        private bool SideOpen(Vector3 pos, float yaw, float error)
        {
            Vector3 dir = Quaternion.Euler(0f, yaw + Mathf.Sign(error) * 60f, 0f) * Vector3.forward;
            return !Physics.SphereCast(pos, cc.radius, dir, out _, settings.wallTurnMinSideClearanceMeters, architectureMask, QueryTriggerInteraction.Ignore);
        }

        private int WallStopsNear(Vector3 pos)
        {
            int n = 0;
            foreach (Vector3 spot in wallStopSpots)
            {
                if (HorizontalDistance(spot, pos) < settings.wallStopRepeatRadiusMeters) n++;
            }
            return n;
        }

        private static float PathLength(Vector3[] pathCorners)
        {
            float len = 0f;
            for (int i = 1; i < pathCorners.Length; i++) len += Vector3.Distance(pathCorners[i - 1], pathCorners[i]);
            return len;
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

        /// <summary>
        /// A stop: stand at the current heading for holdSeconds, then `glances` look-arounds to alternating sides (each held
        /// ~glanceDwellSeconds), then back to walking - which turns on the spot first if the path is far off.
        /// </summary>
        private void BeginStop(float yaw, float holdSeconds, int glances, float glanceDwellSeconds)
        {
            lookYawTargets.Clear();
            lookDwellTimes.Clear();
            lookYawTargets.Add(yaw);
            lookDwellTimes.Add(Mathf.Max(0f, holdSeconds));
            float side = rng.NextDouble() < 0.5 ? -1f : 1f;
            for (int g = 0; g < glances; g++, side = -side)
            {
                lookYawTargets.Add(yaw + side * UniformRange(settings.lookAroundMinYawDegrees, settings.lookAroundMaxYawDegrees));
                lookDwellTimes.Add(glanceDwellSeconds * UniformRange(0.5f, 1.5f));
            }
            lookIndex = 0;
            keyDown = false;
            mode = Mode.LookAround;
        }

        private void BeginScanStop(float yaw)
        {
            int glances = rng.NextDouble() < settings.scanNoGlanceProbability ? 0 : (rng.NextDouble() < 0.5 ? 1 : 2);
            BeginStop(yaw, UniformRange(settings.scanHoldMinSeconds, settings.scanHoldMaxSeconds), glances, settings.scanGlanceDwellSeconds);
            ScanStops++;
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
