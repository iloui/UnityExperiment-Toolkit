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

            // The task (Start/POI_0/End + TaskScript) and the recorder's task reference are set by ScenePreparation.Prepare,
            // which also moves the controller onto the Start marker once the NavMesh is baked.
            var controller = SpawnController(root, settings, layoutData.SpawnPosition, layoutData.SpawnRotation);
            WireRecorder(controller);
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

        private static void WireRecorder(GameObject controller)
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

            // Set (not OR) the mask to exactly the obstacle layer - both ParticipantRecorder and
            // ImitationAgent now hard-fail unless ArchitectureLayerMask equals exactly
            // LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName), so merging it into
            // whatever mask happened to be serialized already (the previous `|=` behaviour) could
            // still leave a too-broad mask that fails that check.
            var obstacleLayer = SceneExporter.EnsureLayer(ZarrDataContract.ArchitectureObstacleLayerName);
            recorder.ArchitectureLayerMask = 1 << obstacleLayer;

            UnityEditor.EditorUtility.SetDirty(recorder);
        }
    }
}
