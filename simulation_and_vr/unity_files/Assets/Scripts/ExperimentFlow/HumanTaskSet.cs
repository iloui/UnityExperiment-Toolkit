using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Assets.Scripts.ExperimentFlow
{
    /// <summary>
    /// The tasks humans walk in one layout scene (methodology 3.8.5): the scene's own Start -> POI_0 -> End task is the test
    /// task, plus three training tasks placed around it by Tools > Imitation Learning Agent > 2 · Record Human Walks > Generate Human Tasks (HumanTaskGenerator).
    /// Lives on Environment_Root and holds positions only - no extra markers or TaskScripts in the scene, so the bot, the
    /// eval agent and ScenePreparation never see the training tasks. ParticipantSession moves the scene's single set of
    /// markers onto the trial's task after loading the scene.
    /// </summary>
    public class HumanTaskSet : MonoBehaviour
    {
        public const string TestTaskId = "test";
        public const string TrainTaskIdPrefix = "train_";
        public const string RoleTrain = "train";
        public const string RoleTest = "test";

        [Serializable]
        public sealed class TaskDef
        {
            public string id;
            /// <summary>Marker positions as the scene's markers take them: Start on the NavMesh, POI and End sphere centres above it.</summary>
            public Vector3 start;
            public Vector3 poi;
            public Vector3 end;
            /// <summary>Rig yaw at the start: along the first NavMesh path segment towards the POI.</summary>
            public float startYawDeg;
            /// <summary>Shortest NavMesh route Start -> POI -> End.</summary>
            public float routeLengthM;
            /// <summary>Share of the test route within the coverage radius of this task's route (0 for the test task itself).</summary>
            public float testRouteOverlap;
            /// <summary>Smallest horizontal distance of the task's markers to an obstacle.</summary>
            public float minWallClearanceM;
            /// <summary>Smallest share of reference routes passing the task's markers (low = dead-end pocket).</summary>
            public float minThroughTraffic;
        }

        [Tooltip("Snapshot of the scene's Start/POI_0/End when the set was generated. The session refuses the scene if the markers moved since.")]
        public TaskDef test = new TaskDef();
        public TaskDef[] training = new TaskDef[0];
        [Tooltip("Index into 'training' of the task left out in condition C (smallest overlap with the test route), fixed before any recording.")]
        public int cExcludedIndex = -1;
        [Tooltip("Share of the test route covered by the union of the training routes.")]
        public float testRouteCoverage;
        [Tooltip("Height of the FPSController pivot above the Start marker (ScenePreparation.ControllerPivotAboveNavMesh).")]
        public float controllerPivotAboveStart = 1.05f;
        [Tooltip("Tools > Imitation Learning Agent > 2 · Record Human Walks > Re-roll Human Tasks increments it; the generator seed is layout seed + salt + reroll.")]
        public int reroll;
        public int generatorSeed;
        public string generatorVersion;
        [TextArea(3, 12)] public string generationReport;

        public int TrainingCount => training != null ? training.Length : 0;

        public static string TrainTaskId(int index) => TrainTaskIdPrefix + index;

        /// <summary>The task with this id ("test" or "train_i"), or null.</summary>
        public TaskDef Find(string id)
        {
            if (id == TestTaskId) return test;
            if (training == null) return null;
            foreach (TaskDef t in training) if (t != null && t.id == id) return t;
            return null;
        }

        /// <summary>Null if the scene's current Start/POI_0/End still sit where the set was generated, else the reason.</summary>
        public string CheckTestTaskUnchanged(TaskScript task, float tolerance = 0.1f)
        {
            if (task == null || task.start == null || task.start.Length != 1 || task.pointsOfInterest == null || task.pointsOfInterest.Length != 1 ||
                task.end == null || task.end.Length != 1)
            {
                return "the scene's TaskScript is not a single Start -> POI_0 -> End task";
            }
            if (Vector3.Distance(task.start[0].transform.position, test.start) > tolerance) return "Start moved since the human tasks were generated";
            if (Vector3.Distance(task.pointsOfInterest[0].transform.position, test.poi) > tolerance) return "POI_0 moved since the human tasks were generated";
            if (Vector3.Distance(task.end[0].transform.position, test.end) > tolerance) return "End moved since the human tasks were generated";
            return null;
        }

        /// <summary>Moves the scene task's markers onto 'def' (play mode, freshly loaded scene only).</summary>
        public static void ApplyToTask(TaskScript task, TaskDef def)
        {
            task.start[0].transform.position = def.start;
            task.pointsOfInterest[0].transform.position = def.poi;
            task.end[0].transform.position = def.end;
        }

        /// <summary>Puts the rig on the task start, facing along the path to the POI.</summary>
        public void PlaceRig(Transform rig, TaskDef def)
        {
            var body = rig.GetComponent<CharacterController>();
            bool bodyWasEnabled = body != null && body.enabled;
            if (body != null) body.enabled = false;
            rig.SetPositionAndRotation(def.start + Vector3.up * controllerPivotAboveStart, Quaternion.Euler(0f, def.startYawDeg, 0f));
            if (body != null) body.enabled = bodyWasEnabled;
            Physics.SyncTransforms();
        }

#if UNITY_EDITOR
        private static readonly Color TestColor = new Color(0.9f, 0.2f, 0.2f);
        private static readonly Color[] TrainColors = { new Color(0.2f, 0.6f, 1f), new Color(0.2f, 0.85f, 0.4f), new Color(1f, 0.75f, 0.1f), Color.magenta };

        private void OnDrawGizmosSelected()
        {
            DrawTask(test, TestColor, "test");
            if (training == null) return;
            for (int i = 0; i < training.Length; i++)
            {
                if (training[i] == null) continue;
                string label = training[i].id + (i == cExcludedIndex ? " (C-excluded)" : "");
                DrawTask(training[i], TrainColors[i % TrainColors.Length], label);
            }
        }

        private static void DrawTask(TaskDef t, Color color, string label)
        {
            if (t == null) return;
            Gizmos.color = color;
            Gizmos.DrawWireCube(t.start + Vector3.up * 0.5f, Vector3.one * 0.5f);
            Gizmos.DrawSphere(t.poi, 0.25f);
            Gizmos.DrawWireSphere(t.end, 0.35f);
            Gizmos.DrawLine(t.start + Vector3.up * 0.5f, t.poi);
            Gizmos.DrawLine(t.poi, t.end);
            Handles.color = color;
            Handles.Label(t.start + Vector3.up * 1.2f, $"{label} start");
            Handles.Label(t.poi + Vector3.up * 0.5f, $"{label} POI");
            Handles.Label(t.end + Vector3.up * 0.5f, $"{label} end");
        }
#endif
    }
}
