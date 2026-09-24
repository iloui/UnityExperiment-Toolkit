namespace Agent_Based_Modelling
{
    using System.Collections.Generic;
    using UnityEngine;
    using UnityEngine.AI;
    using Assets.Scripts.DataRecording;

    /// <summary>
    /// NavMesh shortest-path baseline that EvalHarness runs with agentKind = ShortestPath. A NavMeshAgent walks the
    /// task, but the route (spawn marker, POI order, end marker) and the arrival rule come from PoiProgressTracker,
    /// exactly as for ImitationAgent and ParticipantRecorder, so the baseline finishes the same task under the same
    /// ArrivalRadiusMeters. It does not reuse AgentScript's state machine, which differs in three ways: it dwells
    /// task.poiTime at each POI, it counts a goal as reached within NavMeshAgent.stoppingDistance + 0.5 m of the goal
    /// (not ArrivalRadiusMeters), and it teleports agents it thinks are stuck on a real-time clock.
    ///
    /// The path's wall clearance is set by the NavMesh bake's agent radius, not by radiusMeters: NavMeshAgent.radius only
    /// affects avoidance between agents (turned off here). EvalHarness writes the bake radius to run_config.json.
    /// </summary>
    public class ShortestPathAgent : MonoBehaviour, IEvalAgent
    {
        public TaskScript task;
        public float speedMetersPerSecond = ZarrDataContract.RecordingBody.WalkSpeedMetersPerSecond;
        public float radiusMeters = ZarrDataContract.RecordingBody.Radius;
        public float accelerationMetersPerSecond2 = 50f;
        public float angularSpeedDegreesPerSecond = 120f;

        [HideInInspector] public List<Vector3> trajectory = new List<Vector3>();
        [HideInInspector] public bool destroyRequest = false;

        private PoiProgressTracker poiTracker;
        private NavMeshAgent navAgent;
        private Transform eye;
        private Transform currentDestination;

        public List<Vector3> Trajectory => trajectory;
        public bool DestroyRequest => destroyRequest;
        public int NeedsFulfilledCount => poiTracker != null ? poiTracker.NeedsFulfilled : 0;
        public IReadOnlyList<Transform> PlannedPois => poiTracker != null ? poiTracker.PlannedPois : null;
        public Transform PlannedEnd => poiTracker != null ? poiTracker.PlannedEnd : null;
        public Vector3 CurrentGoalPosition => (poiTracker != null && poiTracker.CurrentTarget != null) ? poiTracker.CurrentTarget.position : transform.position;
        public Transform EyeAnchor => eye;

        void Start()
        {
            poiTracker = new PoiProgressTracker(task);

            Vector3 spawn = poiTracker.ChooseSpawnPosition();
            if (NavMesh.SamplePosition(spawn, out NavMeshHit hit, 100f, NavMesh.AllAreas)) spawn = hit.position;
            else Debug.LogWarning($"[ShortestPathAgent] No NavMesh near start {spawn}; spawning there anyway.");
            transform.position = spawn;

            // The transform is scaled by agentSize like ImitationAgent's (capsule visual only), and NavMeshAgent radius/height
            // are in local units, so both are divided by it.
            float s = Mathf.Max(0.01f, task.agentSize);
            transform.localScale = new Vector3(s, s, s);

            var eyeGO = new GameObject("ShortestPathEye");
            eyeGO.transform.SetParent(transform, false);
            eyeGO.transform.localPosition = new Vector3(0f, ZarrDataContract.CameraHeightAboveNavMeshMeters / s, 0f);
            eye = eyeGO.transform;

            navAgent = gameObject.AddComponent<NavMeshAgent>();
            navAgent.baseOffset = 0f; // pivot on the NavMesh, like ImitationAgent
            navAgent.radius = radiusMeters / s;
            navAgent.height = ZarrDataContract.RecordingBody.Height / s;
            navAgent.speed = speedMetersPerSecond;
            navAgent.acceleration = accelerationMetersPerSecond2;
            navAgent.angularSpeed = angularSpeedDegreesPerSecond;
            navAgent.stoppingDistance = 0f;
            navAgent.autoBraking = false; // arrival is PoiProgressTracker's radius, reached before the goal itself
            navAgent.obstacleAvoidanceType = ObstacleAvoidanceType.NoObstacleAvoidance;
            navAgent.Warp(spawn);

            // Face the first goal at spawn, as ImitationAgent does.
            if (poiTracker.CurrentTarget != null)
            {
                Vector3 towardGoal = poiTracker.CurrentTarget.position - spawn;
                towardGoal.y = 0f;
                if (towardGoal.sqrMagnitude > 0.0001f) transform.rotation = Quaternion.LookRotation(towardGoal.normalized, Vector3.up);
            }

            UpdateDestination();
        }

        void Update()
        {
            trajectory.Add(transform.position);

            poiTracker.Tick(transform.position);
            if (poiTracker.IsFinished)
            {
                destroyRequest = true;
                if (navAgent.isOnNavMesh) navAgent.isStopped = true;
                return;
            }
            UpdateDestination();
        }

        private void UpdateDestination()
        {
            Transform target = poiTracker.CurrentTarget;
            if (target == null || target == currentDestination) return;
            currentDestination = target;

            Vector3 destination = target.position;
            if (NavMesh.SamplePosition(destination, out NavMeshHit hit, 2f, NavMesh.AllAreas)) destination = hit.position;
            if (!navAgent.SetDestination(destination))
            {
                Debug.LogError($"[ShortestPathAgent] SetDestination to '{target.name}' at {destination} failed; the episode will time out.");
            }
        }
    }
}
