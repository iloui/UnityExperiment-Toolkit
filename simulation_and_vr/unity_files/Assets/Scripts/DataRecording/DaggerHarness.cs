using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;
using Unity.Sentis;
using Agent_Based_Modelling;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// DAgger-style data collection (Ross et al. 2011): the trained ImitationAgent drives, and every recorded frame is
    /// labelled with what a NavMesh expert would do from that exact state. This is the only way to get training data
    /// from the states the policy actually ends up in - pinned against a wall, facing the wrong way after a corner -
    /// which neither the humans nor the scripted random walker ever visit on their own.
    ///
    /// Per episode: a random start/end pair (plus, with waypointProbability, a random waypoint registered as the POI)
    /// is set on a runtime TaskScript, an ImitationAgent running `modelAsset` is spawned at the start with a random
    /// heading, and a ParticipantRecorder on the agent records from the agent's own head (same camera height/optics
    /// the policy sees at run time) into imitationagent_training/zarr_recordings, with participant_id "synthetic_dagger"
    /// and session_id = "dagger_" + run start time. Observations are the policy's; action labels come from this harness (it is the
    /// recorder's IActionLabelSource):
    ///   yaw     = heading error to a look-ahead point on the NavMesh path to the recorder's current target, x gain, rate-limited
    ///   forward = walking speed, reduced to 0 for large heading errors or when the agent's own wall check would block it
    ///             (i.e. "stop and turn on the spot")
    ///   pitch   = back towards level
    ///
    /// Execution is a beta-mixture as in DAgger: in each mixSegmentSeconds slice the expert executes with probability
    /// expertSegmentProbability, otherwise the policy does; and whenever the agent has moved less than
    /// stallDisplacementMeters within stallWindowSeconds the expert takes over for rescueSeconds, so a stuck episode
    /// yields recovery labels instead of a minute of identical frames. Labels are the expert's in every frame either way.
    /// Episodes that time out are kept (their stuck states are the most useful frames); only episodes shorter than
    /// minEpisodeSeconds are discarded.
    ///
    /// Scene requirements: baked NavMesh, Architecture_Obstacle geometry, exactly one CanonicalRecordingCamera (the
    /// ImitationAgent copies its optics from it, exactly as in EvalHarness). Owns its own runtime TaskScript + markers,
    /// so the scene's authored task is never modified. Do not run it together with EvalHarness/RecordRandomWalksHarness.
    ///
    /// Opt-in additions for the DAgger loop (Tools > Imitation Learning Agent > 3 · Train Model > Improve Model with DAgger);
    /// every default keeps the behaviour above:
    ///   - participantId: the recorded participant_id (one per DAgger round, e.g. synthetic_dagger_run1_r2)
    ///   - hiddenGoalShare: tasks whose goal is behind a close wall (same rule as RecordRandomWalksHarness v5)
    ///   - expertWalkingTurns / expertSideCommitMaxLengthRatio: the v5 pocket handling for the expert's labels - turn while
    ///     walking where there is room instead of "stand and turn" at every wall, and take the visibly more open side where
    ///     both ways around are about equally long
    ///   - evaluationEpisodes: before collecting, run this many policy-only episodes (no expert, no rescue) on a fixed task
    ///     sequence and discard their recordings - the success rate is comparable from round to round
    ///   - summaryPath: write the run's counts as JSON when done (read by the DAgger loop)
    /// </summary>
    public class DaggerHarness : MonoBehaviour, IActionLabelSource
    {
        [Header("Policy")]
        [Tooltip("The trained model to collect on-policy states with (normally the latest ImitationAgentModel).")]
        public ModelAsset modelAsset;

        [Header("Run Configuration")]
        [Min(1)] public int episodeCount = 25;
        [Tooltip("Seed for task sampling and execution mixing. 0 = time-based (the used seed is logged).")]
        public int seed = 0;
        [Min(1)] public int maxResampleAttempts = 15;
        [Tooltip("Sets Time.timeScale. Time.maximumDeltaTime is capped at the sampling interval so neither the agent's " +
                 "observation gate nor the recorder's 0.1s cadence ever skips a tick.")]
        [Min(0.01f)] public float simulationSpeed = 3f;
        public bool stopPlayModeWhenDone = true;
        [Tooltip("Give up on an episode if the agent hasn't run its first Update within this many real seconds.")]
        public float agentStartTimeoutRealSeconds = 30f;

        [Header("Tasks")]
        [Range(0f, 1f)] public float waypointProbability = 0.5f;
        public float waypointMinDistanceMeters = 3f;
        public float minPairPathLengthMeters = 6f;
        public float maxPathLengthFactor = 3f;
        public float navMeshSampleRadius = 2f;
        [Tooltip("The agent spawns facing its first target (ImitationAgent's own convention) and is then rotated by a uniform random angle in +-this.")]
        public float startHeadingJitterDegrees = 180f;

        [Header("Agent (copied onto the runtime TaskScript - match the eval runs)")]
        public float agentSpeed = 1.3f;
        public float agentSize = 1f;
        public float agentRadius = 0.5f;

        [Header("Episode")]
        public float minTimeoutSeconds = 30f;
        [Tooltip("Timeout = max(minTimeoutSeconds, timeoutFactor x shortest route / agentSpeed), simulated seconds. Timed-out episodes are kept.")]
        public float timeoutFactor = 3f;
        public float minEpisodeSeconds = 5f;

        [Header("Expert labels (NavMesh)")]
        [Tooltip("The expert steers towards the point this far ahead on the NavMesh path to the current target.")]
        public float expertLookAheadMeters = 2f;
        [Tooltip("Fraction of the heading error corrected per sample (0.1 s), before rate limiting.")]
        public float expertYawGain = 0.3f;
        [Tooltip("Rate limit on the expert's turn, degrees/second (human sharp turns reach ~60-120).")]
        public float expertMaxYawRateDegPerSec = 120f;
        [Tooltip("Full walking speed below this heading error, degrees...")]
        public float expertFullSpeedAngleDegrees = 25f;
        [Tooltip("...linearly down to standing (turn on the spot) at this heading error, degrees.")]
        public float expertStopAngleDegrees = 70f;
        [Tooltip("If ImitationAgent's own forward sphere cast would hit a wall within this distance, the expert label says: stand and turn.")]
        public float expertWallStopDistanceMeters = 0.6f;
        [Tooltip("Fraction of the head's pitch removed per sample.")]
        public float expertPitchGain = 0.2f;

        [Header("Expert pocket handling (v5; defaults = off)")]
        [Tooltip("Where the plain rule labels 'stand and turn' (wall ahead, or heading error past expertStopAngleDegrees) but a " +
                 "body-wide cast 60 deg towards the turn is free for expertWalkTurnMinSideClearanceMeters, label 'walk' while turning. " +
                 "Without it, every frame inside a pocket says 'stand' - the attractor the failing agents show.")]
        public bool expertWalkingTurns = false;
        public float expertWalkTurnMinSideClearanceMeters = 0.6f;
        [Tooltip("Facing a wall (expertWallStopDistanceMeters x 2) with the path off by more than expertFullSpeedAngleDegrees: if both " +
                 "ways around are within this factor, steer to the visibly more open side (RecordRandomWalksAgent.ChooseVisibleSide). 0 = off.")]
        public float expertSideCommitMaxLengthRatio = 0f;
        public float expertSideProbeAngleDegrees = 60f;
        public float expertSideProbeMaxDistanceMeters = 6f;
        public float expertSideCommitArrivalMeters = 1f;
        public float expertSideCommitSlackSeconds = 4f;
        public float expertSideDecisionCooldownSeconds = 2f;

        [Header("Hidden-goal tasks (share 0 = off)")]
        [Tooltip("Share of tasks whose goal lies behind a wall at most hiddenGoalMaxWallDistanceMeters ahead of the start and whose route " +
                 "walks at least hiddenGoalMinBacktrackMeters away from it; no waypoint, start heading at the goal +- hiddenGoalStartHeadingJitterDegrees.")]
        [Range(0f, 1f)] public float hiddenGoalShare = 0f;
        public float hiddenGoalMinBacktrackMeters = 2f;
        public float hiddenGoalMaxWallDistanceMeters = 3f;
        public float hiddenGoalStartHeadingJitterDegrees = 20f;
        [Min(1)] public int hiddenGoalMaxAttempts = 400;

        [Header("Recording / reporting")]
        [Tooltip("participant_id written to every manifest.")]
        public string participantId = "synthetic_dagger";
        [Tooltip("Policy-only episodes (no expert execution, no stall rescue) run before the collection on a fixed task sequence " +
                 "(evaluationSeed); their recordings are discarded. 0 = none.")]
        [Min(0)] public int evaluationEpisodes = 0;
        [Tooltip("Seed of the evaluation task sequence; keep it fixed across DAgger rounds so the tasks are the same.")]
        public int evaluationSeed = 12345;
        [Tooltip("If set, the run's counts are written here as JSON when it ends.")]
        public string summaryPath = "";

        [Header("Execution mixing (DAgger beta)")]
        [Tooltip("Probability that the expert (instead of the policy) executes a given mixSegmentSeconds slice. " +
                 "Lower it on later DAgger iterations, as the policy improves.")]
        [Range(0f, 1f)] public float expertSegmentProbability = 0.2f;
        public float mixSegmentSeconds = 2f;
        [Tooltip("Stall rescue: if the agent moved less than stallDisplacementMeters within stallWindowSeconds, the expert executes for rescueSeconds.")]
        public float stallWindowSeconds = 2.5f;
        public float stallDisplacementMeters = 0.25f;
        public float rescueSeconds = 3f;

        private System.Random rng;
        private LayerMask architectureMask;
        private Vector3[] triVerts;
        private int[] triIndices;
        private float[] triCumulativeArea;
        private float previousMaximumDeltaTime;
        private bool timeSettingsApplied;
        private bool fatal;
        private bool lastEpisodeOk;

        private TaskScript task;
        private Transform startMarker;
        private Transform endMarker;
        private Transform waypointMarker;
        private RecordingRig.MarkerTemplate startTemplate;
        private RecordingRig.MarkerTemplate poiTemplate;
        private RecordingRig.MarkerTemplate endTemplate;
        private Camera canonicalCamera;

        private ImitationAgent currentAgent;
        private ParticipantRecorder currentRecorder;
        // Created in Start(), not in a field initializer (native NavMeshPath objects can't be created in the constructor).
        private NavMeshPath expertPath;
        private float expertActiveUntil;
        private float nextSegmentAt;
        private float stallRefTime;
        private Vector3 stallRefPos;

        // v5 expert / hidden goal / evaluation state
        private NavMeshPath probePath;
        private bool expertCommitActive;
        private Vector3 expertCommitPoint;
        private Transform expertCommitTarget;
        private float expertCommitUntil;
        private float nextExpertSideDecision;
        private bool evaluating;
        private int evalEpisodes, evalReached, evalHidden, evalHiddenReached;
        private int hiddenRecorded, hiddenReached, expertSideCommits;
        private float lastBacktrack;

        private int episodesRecorded;
        private int episodesReachedGoal;
        private int episodesDiscarded;
        private string runSessionId;
        private int rescues;
        private float policyExecutedSeconds;
        private float expertExecutedSeconds;

        // OnEnable, not Awake: Awake also runs on a disabled harness, which would switch off the recorder of a human
        // session in the same scene. OnEnable still runs before every Start in the scene.
        void OnEnable()
        {
            RecordingRig.DisableSceneRecorders("DAgger");
        }

        IEnumerator Start()
        {
            if (!Validate()) yield break;
            expertPath = new NavMeshPath();

            int usedSeed = seed != 0 ? seed : Environment.TickCount;
            rng = new System.Random(usedSeed);
            Debug.Log($"[DAgger] Starting: {episodeCount} episodes with model '{modelAsset.name}', seed={usedSeed}, " +
                $"beta={expertSegmentProbability:F2}, waypointProbability={waypointProbability:F2}.");

            previousMaximumDeltaTime = Time.maximumDeltaTime;
            Time.maximumDeltaTime = ZarrDataContract.SamplingIntervalSeconds;
            Time.timeScale = simulationSpeed;
            timeSettingsApplied = true;

            BuildRuntimeObjects();
            yield return null; // let TaskScript.Start() run
            probePath = new NavMeshPath();

            if (evaluationEpisodes > 0)
            {
                // Policy only, on the same task sequence every round; recordings are discarded.
                System.Random collectionRng = rng;
                rng = new System.Random(evaluationSeed);
                evaluating = true;
                for (int ep = 0; ep < evaluationEpisodes && !fatal; ep++)
                {
                    bool ok = false;
                    for (int attempt = 1; attempt <= maxResampleAttempts && !fatal; attempt++)
                    {
                        yield return RunEpisode();
                        if (lastEpisodeOk) { ok = true; break; }
                    }
                    if (!ok && !fatal) Debug.LogWarning($"[DAgger] Evaluation slot {ep + 1}/{evaluationEpisodes}: {maxResampleAttempts} failed attempts - skipped.");
                }
                evaluating = false;
                rng = collectionRng;
                Debug.Log($"[DAgger] Evaluation (policy only): {evalReached}/{evalEpisodes} reached the goal" +
                    (evalHidden > 0 ? $", hidden-goal tasks {evalHiddenReached}/{evalHidden}." : "."));
            }

            for (int ep = 0; ep < episodeCount && !fatal; ep++)
            {
                bool ok = false;
                for (int attempt = 1; attempt <= maxResampleAttempts && !fatal; attempt++)
                {
                    yield return RunEpisode();
                    if (lastEpisodeOk) { ok = true; break; }
                    episodesDiscarded++;
                }
                if (!ok && !fatal) Debug.LogWarning($"[DAgger] Episode slot {ep + 1}/{episodeCount}: {maxResampleAttempts} failed attempts - skipped.");
            }

            float total = policyExecutedSeconds + expertExecutedSeconds;
            Debug.Log($"[DAgger] Done. Recorded {episodesRecorded} episodes (target {episodeCount}), {episodesReachedGoal} reached the goal, " +
                $"discarded {episodesDiscarded} attempts, {rescues} stall rescues, expert executed {(total > 0f ? 100f * expertExecutedSeconds / total : 0f):F0}% " +
                $"of the time{(fatal ? " - ABORTED by an error, see log" : "")}.");
            WriteSummary(total);
            RestoreTimeSettings();
#if UNITY_EDITOR
            if (stopPlayModeWhenDone) UnityEditor.EditorApplication.isPlaying = false;
#endif
        }

        // ------------------------------------------------------------------
        // Setup
        // ------------------------------------------------------------------

        private bool Validate()
        {
            if (modelAsset == null)
            {
                Debug.LogError("[DAgger] 'modelAsset' is not assigned. Refusing to start.");
                enabled = false;
                return false;
            }
            if (LayerMask.NameToLayer(ZarrDataContract.ArchitectureObstacleLayerName) < 0)
            {
                Debug.LogError($"[DAgger] Layer '{ZarrDataContract.ArchitectureObstacleLayerName}' does not exist. Refusing to start.");
                enabled = false;
                return false;
            }
            architectureMask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);

            try
            {
                canonicalCamera = CanonicalRecordingCamera.ResolveInScene();
            }
            catch (InvalidOperationException ex)
            {
                Debug.LogError($"[DAgger] {ex.Message} Refusing to start.");
                enabled = false;
                return false;
            }

            NavMeshTriangulation tri = NavMesh.CalculateTriangulation();
            if (tri.vertices == null || tri.vertices.Length == 0 || tri.indices == null || tri.indices.Length < 3)
            {
                Debug.LogError("[DAgger] NavMesh triangulation is empty - is the scene's NavMesh baked? Refusing to start.");
                enabled = false;
                return false;
            }
            triVerts = tri.vertices;
            triIndices = tri.indices;
            triCumulativeArea = new float[triIndices.Length / 3];
            float area = 0f;
            for (int t = 0; t < triCumulativeArea.Length; t++)
            {
                Vector3 a = triVerts[triIndices[3 * t]], b = triVerts[triIndices[3 * t + 1]], c = triVerts[triIndices[3 * t + 2]];
                area += 0.5f * Vector3.Cross(b - a, c - a).magnitude;
                triCumulativeArea[t] = area;
            }
            if (area <= 0f)
            {
                Debug.LogError("[DAgger] NavMesh has zero area. Refusing to start.");
                enabled = false;
                return false;
            }
            return true;
        }

        private void BuildRuntimeObjects()
        {
            // Markers look exactly like the ones humans see in this scene (copies; the scene's own are hidden).
            RecordingRig.FindSceneMarkerTemplates(out startTemplate, out poiTemplate, out endTemplate);
            endMarker = RecordingRig.CreateMarker(endTemplate, "DAgger_EndMarker", transform);
            waypointMarker = RecordingRig.CreateMarker(poiTemplate, "DAgger_WaypointMarker", transform);
            if (startTemplate != null)
            {
                startMarker = RecordingRig.CreateMarker(startTemplate, "DAgger_StartMarker", transform);
            }
            else
            {
                startMarker = new GameObject("DAgger_StartPoint").transform;
                startMarker.SetParent(transform, false);
            }
            RecordingRig.HideSceneTaskMarkers("DAgger");

            var taskGO = new GameObject("DAgger_Task");
            taskGO.transform.SetParent(transform, false);
            task = taskGO.AddComponent<TaskScript>();
            task.taskName = "DAgger";
            task.agentType = "DAgger";
            task.start = new[] { startMarker.gameObject };
            task.end = new[] { endMarker.gameObject };
            task.pointsOfInterest = Array.Empty<GameObject>();
            task.numberOfNeeds = 0;
            task.revisit = false;
            task.chooseNonDeterministically = false;
            task.agentSpeed = agentSpeed;
            task.agentSize = agentSize;
            task.agentRadius = agentRadius;
        }

        // ------------------------------------------------------------------
        // One episode
        // ------------------------------------------------------------------

        private IEnumerator RunEpisode()
        {
            lastEpisodeOk = false;

            // Short-circuit: with hiddenGoalShare 0 no extra random number is drawn, so existing runs stay reproducible per seed.
            bool hiddenGoal = hiddenGoalShare > 0f && rng.NextDouble() < hiddenGoalShare;
            Vector3 start, end, waypoint;
            bool useWaypoint;
            float routeLength;
            if (hiddenGoal)
            {
                if (!TrySampleHiddenGoalTask(out start, out end, out routeLength, out lastBacktrack)) yield break;
                useWaypoint = false;
                waypoint = end;
            }
            else if (!TrySampleTask(out start, out end, out useWaypoint, out waypoint, out routeLength)) yield break;

            RecordingRig.PlaceMarker(startMarker, startTemplate, start);
            RecordingRig.PlaceMarker(endMarker, endTemplate, end);
            RecordingRig.PlaceMarker(waypointMarker, poiTemplate, waypoint);
            waypointMarker.gameObject.SetActive(useWaypoint);
            task.pointsOfInterest = useWaypoint ? new[] { waypointMarker.gameObject } : Array.Empty<GameObject>();
            task.numberOfNeeds = useWaypoint ? 1 : 0;

            GameObject agentGO = SpawnAgent();
            float spawnRealtime = Time.realtimeSinceStartup;
            while (currentAgent.trajectory == null || currentAgent.trajectory.Count == 0)
            {
                if (Time.realtimeSinceStartup - spawnRealtime > agentStartTimeoutRealSeconds)
                {
                    Debug.LogError("[DAgger] ImitationAgent never started (see its errors above). Aborting the run.");
                    fatal = true;
                    Destroy(agentGO);
                    yield break;
                }
                yield return null;
            }
            // ImitationAgent.Start() has just spawned it facing its first target - randomise that before the first action.
            float jitterRange = hiddenGoal ? hiddenGoalStartHeadingJitterDegrees : startHeadingJitterDegrees;
            float headingJitter = UniformRange(-jitterRange, jitterRange);
            currentAgent.transform.Rotate(0f, headingJitter, 0f, Space.World);
            currentRecorder.startJitterHeadingDeg = headingJitter;
            currentRecorder.syntheticTaskKind = hiddenGoal ? "hidden_goal" : "random";
            currentRecorder.syntheticBacktrackMeters = hiddenGoal ? lastBacktrack : 0f;
            expertCommitActive = false;
            nextExpertSideDecision = 0f;

            try
            {
                currentRecorder.StartRecording();
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DAgger] ParticipantRecorder refused to start: {ex.Message}. Aborting the run.");
                fatal = true;
                Destroy(agentGO);
                yield break;
            }
            string episodeDir = currentRecorder.CurrentEpisodeDirectory;

            float t0 = Time.time;
            float timeout = Mathf.Max(minTimeoutSeconds, timeoutFactor * routeLength / Mathf.Max(0.1f, agentSpeed));
            expertActiveUntil = 0f;
            nextSegmentAt = t0;
            stallRefTime = t0;
            stallRefPos = currentAgent.transform.position;
            bool timedOut = false;

            while (currentRecorder.IsRecording)
            {
                if (Time.time - t0 > timeout)
                {
                    timedOut = true;
                    currentRecorder.StopRecording();
                    break;
                }
                UpdateExecutionSchedule();
                yield return null;
            }

            float duration = Time.time - t0;
            Destroy(agentGO);
            currentAgent = null;
            currentRecorder = null;
            yield return null;

            if (evaluating)
            {
                // Evaluation episodes only count; even a very short one is a valid success.
                RecordRandomWalksHarness.DiscardEpisode(episodeDir);
                evalEpisodes++;
                if (!timedOut) evalReached++;
                if (hiddenGoal) { evalHidden++; if (!timedOut) evalHiddenReached++; }
                Debug.Log($"[DAgger] Evaluation episode {evalEpisodes}/{evaluationEpisodes}: {duration:F1}s, " +
                    $"{(timedOut ? "timed out" : "reached the goal")}{(hiddenGoal ? ", hidden goal" : "")}.");
                lastEpisodeOk = true;
                yield break;
            }

            if (duration < minEpisodeSeconds)
            {
                Debug.LogWarning($"[DAgger] Episode lasted only {duration:F1}s (< minEpisodeSeconds {minEpisodeSeconds:F1}) - discarding.");
                RecordRandomWalksHarness.DiscardEpisode(episodeDir);
                yield break;
            }

            if (hiddenGoal) { hiddenRecorded++; if (!timedOut) hiddenReached++; }
            episodesRecorded++;
            if (!timedOut) episodesReachedGoal++;
            Debug.Log($"[DAgger] Episode {episodesRecorded}/{episodeCount} recorded: {duration:F1}s, {(timedOut ? "timed out (kept)" : "reached the goal")}, " +
                $"{(hiddenGoal ? "hidden goal" : useWaypoint ? "via waypoint (POI)" : "direct")}.");
            lastEpisodeOk = true;
        }

        private GameObject SpawnAgent()
        {
            // No visible body: the human FPSController has none, and a capsule would cast a shadow into the recorded RGB.
            var agentGO = new GameObject($"DAgger_Agent{episodesRecorded}");

            // Created here (same contract offset ImitationAgent.Start() would use) so the recorder can be given a camera on
            // it before the agent starts. The camera never renders on its own; the recorder copies it for its capture.
            var headGO = new GameObject("ImitationHeadAnchor");
            headGO.transform.SetParent(agentGO.transform, false);
            headGO.transform.localPosition = new Vector3(0f, ZarrDataContract.CameraHeightAboveNavMeshMeters / Mathf.Max(0.01f, agentSize), 0f);
            var viewGO = new GameObject("DAgger_RecordingView");
            viewGO.transform.SetParent(headGO.transform, false);
            Camera view = viewGO.AddComponent<Camera>();
            view.CopyFrom(canonicalCamera);
            view.targetTexture = null;
            viewGO.transform.localPosition = Vector3.zero;
            viewGO.transform.localRotation = Quaternion.identity;
            view.enabled = false;

            currentAgent = agentGO.AddComponent<ImitationAgent>();
            currentAgent.task = task;
            currentAgent.ArchitectureLayerMask = architectureMask;
            currentAgent.BaselineModelAsset = modelAsset;
            currentAgent.enableDeadlockNudge = false;
            currentAgent.AgentHeadAnchor = headGO.transform;
            currentAgent.ExecutedActionOverride = ExecuteMixed;

            currentRecorder = agentGO.AddComponent<ParticipantRecorder>();
            currentRecorder.task = task;
            currentRecorder.ArchitectureLayerMask = architectureMask;
            currentRecorder.OutputFormat = RecordingOutputFormat.Binary;
            currentRecorder.EnableDebugRays = false;
            currentRecorder.AutoStartForTesting = false;
            currentRecorder.recordingCameraOverride = view;
            currentRecorder.actionLabelSource = this;
            currentRecorder.source = ZarrDataContract.EpisodeSources.Synthetic;
            currentRecorder.participantId = string.IsNullOrWhiteSpace(participantId) ? "synthetic_dagger" : participantId.Trim();
            if (string.IsNullOrEmpty(runSessionId)) runSessionId = $"dagger_{DateTime.Now:yyyyMMdd_HHmmss}";
            currentRecorder.sessionId = runSessionId; // one session per harness run
            currentRecorder.repetitionIndex = episodesRecorded;
            currentRecorder.inputMode = ZarrDataContract.InputModeDesktop;
            return agentGO;
        }

        // ------------------------------------------------------------------
        // Expert + execution mixing
        // ------------------------------------------------------------------

        private void UpdateExecutionSchedule()
        {
            float now = Time.time;
            if (evaluating)
            {
                policyExecutedSeconds += Time.deltaTime;
                return;
            }
            if (now >= nextSegmentAt)
            {
                nextSegmentAt = now + mixSegmentSeconds;
                if (rng.NextDouble() < expertSegmentProbability) expertActiveUntil = Mathf.Max(expertActiveUntil, now + mixSegmentSeconds);
            }

            if (now - stallRefTime >= stallWindowSeconds)
            {
                Vector3 pos = currentAgent.transform.position;
                if (Vector3.Distance(pos, stallRefPos) < stallDisplacementMeters && now >= expertActiveUntil)
                {
                    expertActiveUntil = now + rescueSeconds;
                    rescues++;
                }
                stallRefTime = now;
                stallRefPos = pos;
            }

            if (now < expertActiveUntil) expertExecutedSeconds += Time.deltaTime;
            else policyExecutedSeconds += Time.deltaTime;
        }

        /// <summary>ImitationAgent.ExecutedActionOverride: the policy's rates, or the expert's while the expert is executing.</summary>
        private Vector3 ExecuteMixed(Vector3 policyRates)
        {
            if (currentRecorder == null || Time.time >= expertActiveUntil) return policyRates;
            ComputeExpertAction(out float forward, out float yawPerSample, out float pitchPerSample);
            float dt = ZarrDataContract.SamplingIntervalSeconds;
            return new Vector3(forward, yawPerSample / dt, pitchPerSample / dt);
        }

        /// <summary>IActionLabelSource: the expert's action from the state the recorder just captured.</summary>
        public void GetActionLabel(out Vector3 actionVel, out Vector3 actionRot)
        {
            ComputeExpertAction(out float forward, out float yawPerSample, out float pitchPerSample);
            actionVel = new Vector3(0f, 0f, forward);
            actionRot = new Vector3(yawPerSample, pitchPerSample, 0f);
        }

        /// <summary>
        /// The NavMesh expert, in the model's own units: forward in m/s, yaw/pitch in degrees per sampling interval
        /// (positive yaw = turn right, positive pitch = look up - the conventions ImitationAgent.MoveAgentLocomotion applies).
        /// </summary>
        private void ComputeExpertAction(out float forward, out float yawPerSample, out float pitchPerSample)
        {
            forward = yawPerSample = pitchPerSample = 0f;
            if (currentAgent == null) return;

            Transform body = currentAgent.transform;
            Vector3 pos = body.position;
            // Follow the recorder's target: that is what the recorded goal vector points at.
            Transform target = currentRecorder != null ? currentRecorder.CurrentTarget : null;
            Vector3 goal = target != null ? target.position : currentAgent.CurrentGoalPosition;

            Vector3 facing = body.forward;
            facing.y = 0f;
            if (expertSideCommitMaxLengthRatio > 0f) UpdateExpertCommitment(pos, facing, target, goal);
            Vector3 steerTo = expertCommitActive ? expertCommitPoint : goal;

            Vector3 ahead = steerTo;
            if (NavMesh.CalculatePath(pos, steerTo, NavMesh.AllAreas, expertPath) && expertPath.corners.Length >= 2)
            {
                ahead = RecordRandomWalksAgent.PointAheadOnCorners(expertPath.corners, pos, expertLookAheadMeters);
            }

            Vector3 toAhead = ahead - pos;
            toAhead.y = 0f;
            if (facing.sqrMagnitude < 1e-6f || toAhead.sqrMagnitude < 1e-6f) return;

            float headingError = Vector3.SignedAngle(facing, toAhead, Vector3.up);
            float maxYawPerSample = expertMaxYawRateDegPerSec * ZarrDataContract.SamplingIntervalSeconds;
            yawPerSample = Mathf.Clamp(headingError * expertYawGain, -maxYawPerSample, maxYawPerSample);

            float speedFactor = 1f - Mathf.InverseLerp(expertFullSpeedAngleDegrees, expertStopAngleDegrees, Mathf.Abs(headingError));
            // Same sphere cast ImitationAgent.MoveAgentLocomotion blocks on - if walking on would hit the wall, stand and turn.
            if (Physics.SphereCast(pos + Vector3.up * 0.5f, agentRadius * 0.8f, facing.normalized, out _, expertWallStopDistanceMeters, architectureMask))
            {
                speedFactor = 0f;
            }
            if (expertWalkingTurns && speedFactor < 1f && Mathf.Abs(headingError) > 1f)
            {
                // Room towards the turn: label "walk while turning". ImitationAgent's own wall check still blocks the step
                // until the turn has cleared the wall, so the executed motion stays safe.
                Vector3 turnDir = Quaternion.Euler(0f, Mathf.Sign(headingError) * 60f, 0f) * facing.normalized;
                if (!Physics.SphereCast(pos + Vector3.up * 0.5f, agentRadius * 0.8f, turnDir, out _, expertWalkTurnMinSideClearanceMeters, architectureMask))
                {
                    speedFactor = 1f;
                }
            }
            forward = agentSpeed * speedFactor;

            Transform head = currentAgent.AgentHeadAnchor;
            if (head != null)
            {
                float lookUpDegrees = -Mathf.DeltaAngle(0f, head.localEulerAngles.x);
                pitchPerSample = Mathf.Clamp(-lookUpDegrees * expertPitchGain, -3f, 3f);
            }
        }

        /// <summary>
        /// The expert's visible-side commitment (same rule as the v5 bot): ends at the aim point, on a target change or after
        /// its time budget; a new one is considered when a wall is close ahead and the path is off to one side.
        /// </summary>
        private void UpdateExpertCommitment(Vector3 pos, Vector3 facing, Transform target, Vector3 goal)
        {
            float now = Time.time;
            if (expertCommitActive && (target != expertCommitTarget || HorizontalDistance(pos, expertCommitPoint) < expertSideCommitArrivalMeters || now > expertCommitUntil))
            {
                expertCommitActive = false;
            }
            if (expertCommitActive || now < nextExpertSideDecision || facing.sqrMagnitude < 1e-6f) return;
            if (!Physics.SphereCast(pos + Vector3.up * 0.5f, agentRadius * 0.8f, facing.normalized, out _, 2f * expertWallStopDistanceMeters, architectureMask)) return;
            if (!NavMesh.CalculatePath(pos, goal, NavMesh.AllAreas, expertPath) || expertPath.corners.Length < 2) return;
            Vector3 ahead = RecordRandomWalksAgent.PointAheadOnCorners(expertPath.corners, pos, expertLookAheadMeters);
            Vector3 toAhead = ahead - pos;
            toAhead.y = 0f;
            if (toAhead.sqrMagnitude < 1e-6f) return;
            float naturalError = Vector3.SignedAngle(facing, toAhead, Vector3.up);
            if (Mathf.Abs(naturalError) <= expertFullSpeedAngleDegrees) return;

            nextExpertSideDecision = now + expertSideDecisionCooldownSeconds;
            float yaw = Mathf.Atan2(facing.x, facing.z) * Mathf.Rad2Deg;
            if (!RecordRandomWalksAgent.ChooseVisibleSide(pos + Vector3.up * 0.5f, yaw, naturalError, goal, agentRadius * 0.8f,
                    expertSideProbeAngleDegrees, expertSideProbeMaxDistanceMeters, expertSideCommitMaxLengthRatio, architectureMask,
                    probePath, out _, out Vector3 aimPoint)) return;
            expertCommitActive = true;
            expertCommitPoint = aimPoint;
            expertCommitTarget = target;
            expertCommitUntil = now + HorizontalDistance(pos, aimPoint) / Mathf.Max(0.3f, agentSpeed) + expertSideCommitSlackSeconds;
            expertSideCommits++;
        }

        private static float HorizontalDistance(Vector3 a, Vector3 b)
        {
            a.y = b.y = 0f;
            return Vector3.Distance(a, b);
        }

        [Serializable]
        private class RunSummary
        {
            public string participant_id;
            public string model;
            public string scene;
            public int episodes_target;
            public int episodes_recorded;
            public int episodes_reached_goal;
            public int episodes_discarded;
            public int hidden_goal_recorded;
            public int hidden_goal_reached;
            public int stall_rescues;
            public float expert_executed_share;
            public int expert_side_commits;
            public int eval_episodes;
            public int eval_reached;
            public int eval_hidden;
            public int eval_hidden_reached;
            public bool aborted;
            public string finished_utc;
        }

        private void WriteSummary(float totalExecutedSeconds)
        {
            if (string.IsNullOrWhiteSpace(summaryPath)) return;
            var summary = new RunSummary
            {
                participant_id = string.IsNullOrWhiteSpace(participantId) ? "synthetic_dagger" : participantId.Trim(),
                model = modelAsset != null ? modelAsset.name : "",
                scene = gameObject.scene.name,
                episodes_target = episodeCount,
                episodes_recorded = episodesRecorded,
                episodes_reached_goal = episodesReachedGoal,
                episodes_discarded = episodesDiscarded,
                hidden_goal_recorded = hiddenRecorded,
                hidden_goal_reached = hiddenReached,
                stall_rescues = rescues,
                expert_executed_share = totalExecutedSeconds > 0f ? expertExecutedSeconds / totalExecutedSeconds : 0f,
                expert_side_commits = expertSideCommits,
                eval_episodes = evalEpisodes,
                eval_reached = evalReached,
                eval_hidden = evalHidden,
                eval_hidden_reached = evalHiddenReached,
                aborted = fatal,
                finished_utc = DateTime.UtcNow.ToString("o"),
            };
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(summaryPath));
                string temp = summaryPath + ".tmp";
                File.WriteAllText(temp, JsonUtility.ToJson(summary, true));
                if (File.Exists(summaryPath)) File.Delete(summaryPath);
                File.Move(temp, summaryPath); // the DAgger loop never sees a half-written file
            }
            catch (Exception ex)
            {
                Debug.LogError($"[DAgger] Could not write the run summary to '{summaryPath}': {ex.Message}");
            }
        }

        // ------------------------------------------------------------------
        // Sampling
        // ------------------------------------------------------------------

        /// <summary>Hidden-goal pair (RecordRandomWalksHarness.TrySampleHiddenGoalPair's rule): goal behind a close wall, route walks away from it.</summary>
        private bool TrySampleHiddenGoalTask(out Vector3 start, out Vector3 end, out float routeLength, out float backtrack)
        {
            start = end = Vector3.zero;
            routeLength = backtrack = 0f;
            var path = new NavMeshPath();
            Vector3 eye = Vector3.up * ZarrDataContract.CameraHeightAboveNavMeshMeters;
            for (int attempt = 0; attempt < hiddenGoalMaxAttempts; attempt++)
            {
                if (!TrySampleNavMeshPoint(out start) || !TrySampleNavMeshPoint(out end)) continue;
                if (Vector3.Distance(start, end) <= ZarrDataContract.ArrivalRadiusMeters) continue;
                Vector3 toGoal = end - start;
                toGoal.y = 0f;
                if (toGoal.sqrMagnitude < 1e-6f) continue;
                if (!Physics.Raycast(start + eye, toGoal.normalized, Mathf.Min(hiddenGoalMaxWallDistanceMeters, toGoal.magnitude), architectureMask, QueryTriggerInteraction.Ignore)) continue;
                if (!NavMesh.CalculatePath(start, end, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete) continue;
                routeLength = PathLength(path.corners);
                if (routeLength < minPairPathLengthMeters) continue;
                backtrack = RecordRandomWalksHarness.BacktrackLength(path.corners, end);
                if (backtrack < hiddenGoalMinBacktrackMeters) continue;
                return true;
            }
            Debug.LogWarning($"[DAgger] No hidden-goal task after {hiddenGoalMaxAttempts} candidates.");
            return false;
        }

        private bool TrySampleTask(out Vector3 start, out Vector3 end, out bool useWaypoint, out Vector3 waypoint, out float routeLength)
        {
            start = end = waypoint = Vector3.zero;
            useWaypoint = false;
            routeLength = 0f;
            var path = new NavMeshPath();

            if (!TrySampleNavMeshPoint(out start) || !TrySampleNavMeshPoint(out end)) return false;
            if (Vector3.Distance(start, end) <= ZarrDataContract.ArrivalRadiusMeters) return false;
            if (!NavMesh.CalculatePath(start, end, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete) return false;
            float direct = PathLength(path.corners);
            if (direct < minPairPathLengthMeters) return false;
            routeLength = direct;
            waypoint = end;

            if (rng.NextDouble() >= waypointProbability) return true;

            useWaypoint = true;
            if (!TrySampleNavMeshPoint(out waypoint)) return false;
            if (Vector3.Distance(waypoint, start) < waypointMinDistanceMeters || Vector3.Distance(waypoint, end) < waypointMinDistanceMeters) return false;
            if (!NavMesh.CalculatePath(start, waypoint, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete) return false;
            float leg1 = PathLength(path.corners);
            if (!NavMesh.CalculatePath(waypoint, end, NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete) return false;
            routeLength = leg1 + PathLength(path.corners);
            return routeLength <= maxPathLengthFactor * direct;
        }

        private bool TrySampleNavMeshPoint(out Vector3 point)
        {
            point = Vector3.zero;
            float total = triCumulativeArea[triCumulativeArea.Length - 1];
            float pick = (float)(rng.NextDouble() * total);
            int lo = 0, hi = triCumulativeArea.Length - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (triCumulativeArea[mid] < pick) lo = mid + 1; else hi = mid;
            }

            Vector3 a = triVerts[triIndices[3 * lo]], b = triVerts[triIndices[3 * lo + 1]], c = triVerts[triIndices[3 * lo + 2]];
            float r1 = Mathf.Sqrt((float)rng.NextDouble());
            float r2 = (float)rng.NextDouble();
            Vector3 candidate = (1f - r1) * a + r1 * (1f - r2) * b + r1 * r2 * c;

            if (!NavMesh.SamplePosition(candidate, out NavMeshHit hit, navMeshSampleRadius, NavMesh.AllAreas)) return false;
            point = hit.position;
            return true;
        }

        private static float PathLength(Vector3[] corners)
        {
            float len = 0f;
            for (int i = 1; i < corners.Length; i++) len += Vector3.Distance(corners[i - 1], corners[i]);
            return len;
        }

        private float UniformRange(float a, float b)
        {
            return a + (float)rng.NextDouble() * (b - a);
        }

        // ------------------------------------------------------------------
        // Teardown
        // ------------------------------------------------------------------

        private void RestoreTimeSettings()
        {
            if (!timeSettingsApplied) return;
            Time.timeScale = 1f;
            if (previousMaximumDeltaTime > 0f) Time.maximumDeltaTime = previousMaximumDeltaTime;
            timeSettingsApplied = false;
        }

        void OnDestroy()
        {
            RestoreTimeSettings();
        }
    }
}
