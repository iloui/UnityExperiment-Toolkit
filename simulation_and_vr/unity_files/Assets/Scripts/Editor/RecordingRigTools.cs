using System.IO;
using Assets.Scripts.DataRecording;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityStandardAssets.Characters.FirstPerson;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Puts the human FPSController's camera at ZarrDataContract.CameraHeightAboveNavMeshMeters (1.5 m above the NavMesh,
    /// ImitationAgent's eye height) as a prefab-instance override in layout scenes - the shared FPSController.prefab is not
    /// touched, since the Zollverein experiment scenes use it too. RecordRandomWalksHarness applies the same height at run
    /// time, so after this human and synthetic recordings are taken from the same height.
    /// </summary>
    internal static class RecordingRigTools
    {
        private const string LayoutScenesFolder = "Assets/Scenes/TestLayouts";

        [MenuItem("Tools/Recording/Apply Contract Camera Height (Open Scene)")]
        public static void ApplyToOpenScene()
        {
            Scene scene = EditorSceneManager.GetActiveScene();
            if (ApplyInScene(scene) > 0)
            {
                EditorSceneManager.MarkSceneDirty(scene);
                Debug.Log($"[RecordingRigTools] Save '{scene.name}' to keep the change.");
            }
        }

        [MenuItem("Tools/Recording/Apply Contract Camera Height (All Layout Scenes)")]
        public static void ApplyToAllLayoutScenes()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            string previous = EditorSceneManager.GetActiveScene().path;
            int changedScenes = 0;
            foreach (string path in Directory.GetFiles(LayoutScenesFolder, "*.unity"))
            {
                Scene scene = EditorSceneManager.OpenScene(path.Replace('\\', '/'), OpenSceneMode.Single);
                if (ApplyInScene(scene) > 0 && EditorSceneManager.SaveScene(scene)) changedScenes++;
            }
            if (!string.IsNullOrEmpty(previous)) EditorSceneManager.OpenScene(previous, OpenSceneMode.Single);
            Debug.Log($"[RecordingRigTools] Camera height applied and saved in {changedScenes} scene(s) under {LayoutScenesFolder}.");
        }

        /// <summary>Applies the contract height to every FPSController in `scene`; returns how many were changed.</summary>
        public static int ApplyInScene(Scene scene)
        {
            int changed = 0;
            foreach (GameObject root in scene.GetRootGameObjects())
            {
                foreach (FirstPersonController fps in root.GetComponentsInChildren<FirstPersonController>(true))
                {
                    Transform cam = fps.transform.Find("FirstPersonCharacter");
                    if (cam == null) cam = fps.GetComponentInChildren<Camera>(true)?.transform;
                    if (cam == null)
                    {
                        Debug.LogWarning($"[RecordingRigTools] '{scene.name}/{fps.name}': no camera found - skipped.");
                        continue;
                    }

                    Undo.RecordObject(cam, "Apply contract camera height");
                    if (RecordingRig.ApplyContractCameraHeight(fps.transform, cam, out string message))
                    {
                        PrefabUtility.RecordPrefabInstancePropertyModifications(cam);
                        changed++;
                        Debug.Log($"[RecordingRigTools] '{scene.name}/{fps.name}': {message}");
                    }
                    else
                    {
                        Debug.LogWarning($"[RecordingRigTools] '{scene.name}/{fps.name}': {message}");
                    }
                }
            }
            return changed;
        }
    }
}
