using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Shared start -> POI(s) -> end progression logic used identically by ImitationAgent
    /// (autonomous movement) and ParticipantRecorder (live human recording), so POI
    /// selection/arrival semantics can never drift between the two pipelines.
    ///
    /// Wraps TaskScript's existing start/pointsOfInterest/end/numberOfNeeds/revisit/
    /// chooseNonDeterministically configuration. Deliberately does NOT read task.poiTime -
    /// other (non-imitation) agent types still depend on that field, and dwelling at a POI
    /// is intentionally not modelled here: arrival advances immediately, since any dwell
    /// behaviour is expected to be learned by the imitation model from the recorded label
    /// distribution instead of being hand-scripted identically on both sides.
    /// </summary>
    public class PoiProgressTracker
    {
        private enum State { ChoosingPoi, FindingPoi, TaskCompleted }

        private readonly TaskScript task;
        private readonly bool[] poiMask;

        private State state;
        private int currentPoiIndex = -1;

        public Transform CurrentTarget { get; private set; }
        public bool IsFinished { get; private set; }
        public int NeedsFulfilled { get; private set; }

        public PoiProgressTracker(TaskScript task)
        {
            this.task = task;
            poiMask = new bool[task.pointsOfInterest.Length];
            for (int i = 0; i < poiMask.Length; i++) poiMask[i] = true;

            NeedsFulfilled = 0;
            IsFinished = false;
            AdvanceChoosingState();
        }

        /// <summary>Picks a random start location, for components that spawn a mover (e.g. ImitationAgent).</summary>
        public Vector3 ChooseSpawnPosition()
        {
            return task.start[Random.Range(0, task.start.Length)].transform.position;
        }

        /// <summary>
        /// Advances the tracker's state based on the mover's current position. Call once per
        /// tick (every Update() for a continuously-moving agent, once per recorder sample for
        /// a periodically-sampled recording) - arrival radius and progression rules are
        /// identical either way.
        /// </summary>
        public void Tick(Vector3 currentPosition)
        {
            if (IsFinished || CurrentTarget == null) return;

            float distance = Vector3.Distance(currentPosition, CurrentTarget.position);
            if (distance > ZarrDataContract.ArrivalRadiusMeters) return;

            if (state == State.FindingPoi)
            {
                Debug.Log($"[PoiProgressTracker] Reached POI '{CurrentTarget.name}' (index {currentPoiIndex}) at distance {distance:F2}m. " +
                    $"{NeedsFulfilled + 1}/{task.numberOfNeeds} needs fulfilled.");

                NeedsFulfilled++;
                if (!task.revisit) poiMask[currentPoiIndex] = false;
                AdvanceChoosingState();
            }
            else if (state == State.TaskCompleted)
            {
                Debug.Log($"[PoiProgressTracker] Reached end target '{CurrentTarget.name}' at distance {distance:F2}m. Task complete.");
                IsFinished = true;
            }
        }

        private void AdvanceChoosingState()
        {
            if (NeedsFulfilled < task.numberOfNeeds)
            {
                currentPoiIndex = ChoosePoiIndex();
                CurrentTarget = task.pointsOfInterest[currentPoiIndex].transform;
                state = State.FindingPoi;
            }
            else
            {
                CurrentTarget = task.end[Random.Range(0, task.end.Length)].transform;
                state = State.TaskCompleted;
            }
        }

        private int ChoosePoiIndex()
        {
            if (task.chooseNonDeterministically)
            {
                var valid = new List<int>();
                for (int i = 0; i < poiMask.Length; i++) if (poiMask[i]) valid.Add(i);

                if (valid.Count == 0)
                {
                    Debug.LogError("[PoiProgressTracker] Non-deterministic POI choice failed! No remaining valid POIs available in poiMask.");
                    return 0;
                }

                return valid[Random.Range(0, valid.Count)];
            }

            for (int i = 0; i < poiMask.Length; i++) if (poiMask[i]) return i;

            Debug.LogWarning("[PoiProgressTracker] ChoosePoiIndex fallback to 0. All POIs may have been marked visited.");
            return 0;
        }
    }
}
