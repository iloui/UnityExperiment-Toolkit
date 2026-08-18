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

            recorder.MainVRCamera = camera;
            recorder.CurrentPOITarget = goal.transform;

            var obstacleLayer = SceneExporter.EnsureLayer("Architecture_Obstacle");
            var mask = recorder.ArchitectureLayerMask.value;
            mask |= 1 << obstacleLayer;
            recorder.ArchitectureLayerMask = mask;

            UnityEditor.EditorUtility.SetDirty(recorder);
        }
    }
}
