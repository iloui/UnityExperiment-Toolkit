using System;
using System.Collections;
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

        private int episodesRecorded;
        private int episodesReachedGoal;
        private int episodesDiscarded;
        private string runSessionId;
        private int rescues;
        private float policyExecutedSeconds;
        private float expertExecutedSeconds;

        void Awake()
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

            if (!TrySampleTask(out Vector3 start, out Vector3 end, out bool useWaypoint, out Vector3 waypoint, out float routeLength)) yield break;

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
            float headingJitter = UniformRange(-startHeadingJitterDegrees, startHeadingJitterDegrees);
            currentAgent.transform.Rotate(0f, headingJitter, 0f, Space.World);
            currentRecorder.startJitterHeadingDeg = headingJitter;

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

            if (duration < minEpisodeSeconds)
            {
                Debug.LogWarning($"[DAgger] Episode lasted only {duration:F1}s (< minEpisodeSeconds {minEpisodeSeconds:F1}) - discarding.");
                RecordRandomWalksHarness.DiscardEpisode(episodeDir);
                yield break;
            }

            episodesRecorded++;
            if (!timedOut) episodesReachedGoal++;
            Debug.Log($"[DAgger] Episode {episodesRecorded}/{episodeCount} recorded: {duration:F1}s, {(timedOut ? "timed out (kept)" : "reached the goal")}, " +
                $"{(useWaypoint ? "via waypoint (POI)" : "direct")}.");
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
            currentRecorder.participantId = "synthetic_dagger";
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

            Vector3 ahead = goal;
            if (NavMesh.CalculatePath(pos, goal, NavMesh.AllAreas, expertPath) && expertPath.corners.Length >= 2)
            {
                ahead = RecordRandomWalksAgent.PointAheadOnCorners(expertPath.corners, pos, expertLookAheadMeters);
            }

            Vector3 facing = body.forward;
            facing.y = 0f;
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
            forward = agentSpeed * speedFactor;

            Transform head = currentAgent.AgentHeadAnchor;
            if (head != null)
            {
                float lookUpDegrees = -Mathf.DeltaAngle(0f, head.localEulerAngles.x);
                pitchPerSample = Mathf.Clamp(-lookUpDegrees * expertPitchGain, -3f, 3f);
            }
        }

        // ------------------------------------------------------------------
        // Sampling
        // ------------------------------------------------------------------

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
