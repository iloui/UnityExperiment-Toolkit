using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// What EvalHarness reads from any agent it evaluates (ImitationAgent, ShortestPathAgent). Everything the harness
    /// derives - episode clock, path length, stalls, SPL, the per-tick CSV - comes from these members, so both agent kinds
    /// are measured by the same code.
    /// </summary>
    public interface IEvalAgent
    {
        /// <summary>One position per agent Update, spawn point first. Empty until the agent's first Update.</summary>
        List<Vector3> Trajectory { get; }
        /// <summary>True once the agent has reached the end marker (PoiProgressTracker.IsFinished).</summary>
        bool DestroyRequest { get; }
        int NeedsFulfilledCount { get; }
        IReadOnlyList<Transform> PlannedPois { get; }
        Transform PlannedEnd { get; }
        Vector3 CurrentGoalPosition { get; }
        /// <summary>The agent's eye (ZarrDataContract.CameraHeightAboveNavMeshMeters above the NavMesh). Its local x rotation is the head pitch.</summary>
        Transform EyeAnchor { get; }
    }
}
