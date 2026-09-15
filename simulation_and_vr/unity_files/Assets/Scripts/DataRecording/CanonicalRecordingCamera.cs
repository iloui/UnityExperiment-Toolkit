using System;
using UnityEngine;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Marks the single camera in a scene that both ParticipantRecorder (human data
    /// collection) and ImitationAgent (live inference) must copy their optical settings
    /// (FOV, clip planes, HDR, culling mask, post-processing) from via Camera.CopyFrom, so
    /// the two pipelines can never be pointed at two different cameras by accident.
    ///
    /// Place this component on whichever camera ParticipantRecorder actually records from
    /// (e.g. FirstPersonCharacter under FPSController). Exactly one instance is expected per
    /// scene - both ResolveInScene() callers hard-fail rather than silently falling back to
    /// Camera.main or any other guess.
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class CanonicalRecordingCamera : MonoBehaviour
    {
        public Camera Camera { get; private set; }

        private void Awake()
        {
            Camera = GetComponent<Camera>();
        }

        public static Camera ResolveInScene()
        {
            var found = UnityEngine.Object.FindObjectsOfType<CanonicalRecordingCamera>();
            if (found.Length != 1)
            {
                throw new InvalidOperationException(
                    $"[CanonicalRecordingCamera] Expected exactly one CanonicalRecordingCamera in the scene, found {found.Length}. " +
                    "Both ParticipantRecorder and ImitationAgent require a single unambiguous camera to copy optical settings " +
                    "from - add the component to the intended recording camera (e.g. FirstPersonCharacter) and make sure no " +
                    "other object in the scene also carries it.");
            }

            Camera camera = found[0].Camera != null ? found[0].Camera : found[0].GetComponent<Camera>();
            if (camera == null)
            {
                throw new InvalidOperationException(
                    "[CanonicalRecordingCamera] The CanonicalRecordingCamera in the scene has no Camera component attached.");
            }

            return camera;
        }
    }
}
