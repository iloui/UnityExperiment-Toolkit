using System;
using Assets.Scripts.DataRecording;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    internal static class AgentLinker
    {
        private const float AgentY = 1.05f;

        public static void SpawnAndWireParticipants(Transform root, LayoutGenerationSettings settings, LayoutGenerationData layoutData)
        {
            if (root == null) throw new ArgumentNullException(nameof(root));

            var controller = SpawnController(root, settings, layoutData.SpawnPosition, layoutData.SpawnRotation);
            var goal = SpawnGoal(root, layoutData.GoalPosition);
            WireRecorder(controller, goal, settings);
        }

        private static GameObject SpawnController(Transform root, LayoutGenerationSettings settings, Vector3 spawnPosition, Quaternion spawnRotation)
        {
            var prefab = settings.FpsControllerPrefab != null ? settings.FpsControllerPrefab : SceneExporter.FindDefaultControllerPrefab();
            if (prefab == null)
            {
                throw new InvalidOperationException("No FPSController prefab was assigned or found in the project.");
            }

            var controller = UnityEditor.PrefabUtility.InstantiatePrefab(prefab) as GameObject;
            if (controller == null)
            {
                throw new InvalidOperationException("Could not instantiate the FPSController prefab.");
            }

            controller.name = "FPSController";
            controller.transform.SetParent(root, false);
            controller.transform.SetPositionAndRotation(new Vector3(spawnPosition.x, AgentY, spawnPosition.z), spawnRotation);
            return controller;
        }

        private static GameObject SpawnGoal(Transform root, Vector3 goalPosition)
        {
            var goal = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            goal.name = "Goal_POI";
            goal.transform.SetParent(root, false);
            goal.transform.position = goalPosition;
            goal.transform.localScale = Vector3.one * 0.4f;
            var collider = goal.GetComponent<Collider>();
            if (collider != null)
            {
                collider.enabled = false;
            }

            return goal;
        }

        private static void WireRecorder(GameObject controller, GameObject goal, LayoutGenerationSettings settings)
        {
            var recorder = controller.GetComponent<ParticipantRecorder>();
            if (recorder == null)
            {
                recorder = controller.AddComponent<ParticipantRecorder>();
            }

            var cameraTransform = controller.transform.Find("FirstPersonCharacter");
            if (cameraTransform == null)
            {
                cameraTransform = controller.GetComponentInChildren<Camera>(true)?.transform;
            }

            if (cameraTransform == null)
            {
                throw new InvalidOperationException("Could not find the FirstPersonCharacter camera on the instantiated FPSController.");
            }

            var camera = cameraTransform.GetComponent<Camera>();
            if (camera == null)
            {
                throw new InvalidOperationException("The FirstPersonCharacter child does not contain a Camera component.");
            }

            // Both ParticipantRecorder and ImitationAgent now resolve their optical settings from
            // a single marked camera in the scene (CanonicalRecordingCamera.ResolveInScene()),
            // rather than a manually-assigned reference, so recording and inference can never be
            // pointed at two different cameras by accident.
            if (camera.GetComponent<CanonicalRecordingCamera>() == null)
            {
                camera.gameObject.AddComponent<CanonicalRecordingCamera>();
            }

            // PoiProgressTracker (shared by ParticipantRecorder and ImitationAgent) reads its
            // start/POI(s)/end configuration from a TaskScript rather than a directly-assigned
            // target Transform. This generator only ever produces a single goal marker, so that
            // maps onto a zero-POI task (numberOfNeeds = 0) whose sole "end" is that goal -
            // PoiProgressTracker treats that as the current target from the very first tick,
            // exactly matching the previous direct CurrentPOITarget = goal.transform behaviour.
            var taskScript = controller.GetComponent<TaskScript>();
            if (taskScript == null)
            {
                taskScript = controller.AddComponent<TaskScript>();
            }
            taskScript.taskName = "Generated_" + controller.scene.name;
            taskScript.agentType = "ImitationAgent";
            taskScript.start = new[] { controller };
            taskScript.pointsOfInterest = Array.Empty<GameObject>();
            taskScript.end = new[] { goal };
            taskScript.numberOfNeeds = 0;
            taskScript.revisit = false;
            taskScript.chooseNonDeterministically = false;

            recorder.task = taskScript;

            // Set (not OR) the mask to exactly the obstacle layer - both ParticipantRecorder and
            // ImitationAgent now hard-fail unless ArchitectureLayerMask equals exactly
            // LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName), so merging it into
            // whatever mask happened to be serialized already (the previous `|=` behaviour) could
            // still leave a too-broad mask that fails that check.
            var obstacleLayer = SceneExporter.EnsureLayer(ZarrDataContract.ArchitectureObstacleLayerName);
            recorder.ArchitectureLayerMask = 1 << obstacleLayer;

            UnityEditor.EditorUtility.SetDirty(recorder);
            UnityEditor.EditorUtility.SetDirty(taskScript);
        }
    }
}
