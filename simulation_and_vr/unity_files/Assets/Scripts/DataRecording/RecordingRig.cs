using UnityEngine;
using UnityEngine.AI;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Shared helpers for the FPSController recording rig, used identically for human recordings (layout scenes,
    /// AgentLinker), the synthetic RecordRandomWalks bot and the harnesses, so the camera height and the scene rig's
    /// auto-recording are handled the same way everywhere.
    /// </summary>
    public static class RecordingRig
    {
        /// <summary>Height of the NavMesh surface below/near `position`.</summary>
        public static bool TryNavMeshHeight(Vector3 position, out float navMeshY)
        {
            if (NavMesh.SamplePosition(position, out NavMeshHit hit, 3f, NavMesh.AllAreas))
            {
                navMeshY = hit.position.y;
                return true;
            }
            navMeshY = 0f;
            return false;
        }

        /// <summary>Pivot height of `cc` once it rests on the floor collider below `position` (the rig spawns in the air and falls).</summary>
        public static bool TryGroundedPivotY(CharacterController cc, Vector3 position, out float pivotY)
        {
            pivotY = position.y;
            // The floor is at or just below the NavMesh surface; anything higher (e.g. a task-marker sphere next to the
            // spawn point) is not what the capsule will stand on.
            float maxFloorY = TryNavMeshHeight(position, out float navMeshY) ? navMeshY + 0.1f : float.MaxValue;
            float bestDistance = float.MaxValue;
            bool found = false;
            foreach (RaycastHit hit in Physics.RaycastAll(position + Vector3.up * 0.5f, Vector3.down, 6f, Physics.AllLayers, QueryTriggerInteraction.Ignore))
            {
                if (hit.collider == cc || hit.point.y > maxFloorY || hit.distance >= bestDistance) continue;
                bestDistance = hit.distance;
                float scale = cc.transform.lossyScale.y;
                pivotY = hit.point.y + (cc.height * 0.5f + cc.skinWidth - cc.center.y) * scale;
                found = true;
            }
            return found;
        }

        /// <summary>
        /// Sets `camera`'s local height so that, with the rig standing on the floor, the camera is
        /// ZarrDataContract.CameraHeightAboveNavMeshMeters above the NavMesh - the same eye height as ImitationAgent.
        /// Returns false (camera unchanged) when there is no NavMesh near the rig.
        /// </summary>
        public static bool ApplyContractCameraHeight(Transform rigRoot, Transform camera, out string message)
        {
            if (!TryNavMeshHeight(rigRoot.position, out float navMeshY))
            {
                message = $"no NavMesh near '{rigRoot.name}' - bake the NavMesh first; camera height left at {camera.localPosition.y:F3} (local).";
                return false;
            }

            float pivotY = rigRoot.position.y;
            var cc = rigRoot.GetComponent<CharacterController>();
            bool grounded = cc != null && TryGroundedPivotY(cc, rigRoot.position, out pivotY);

            float local = (navMeshY + ZarrDataContract.CameraHeightAboveNavMeshMeters - pivotY) / Mathf.Max(1e-4f, rigRoot.lossyScale.y);
            Vector3 p = camera.localPosition;
            camera.localPosition = new Vector3(p.x, local, p.z);
            message = $"camera local height {p.y:F3} -> {local:F3} ({ZarrDataContract.CameraHeightAboveNavMeshMeters:F2} m above the NavMesh)" +
                (grounded ? "." : " - WARNING: no floor collider found under the rig, its current (possibly airborne) height was used.");
            return true;
        }

        /// <summary>A scene task marker the synthetic harnesses copy, and how high its centre sits above the NavMesh.</summary>
        public sealed class MarkerTemplate
        {
            public GameObject source;          // null = fallback sphere (0.4 m, default material)
            public float heightAboveNavMesh;
        }

        /// <summary>
        /// Finds the marker look (mesh, material, scale, shadows, height) humans see in this scene: the first TaskScript
        /// whose end marker is rendered. start is null when the scene's start is not a visible marker (e.g. the
        /// an invisible empty marker, as in generated layouts) - then the synthetic runs show no start marker either.
        /// </summary>
        public static void FindSceneMarkerTemplates(out MarkerTemplate start, out MarkerTemplate poi, out MarkerTemplate end)
        {
            start = null;
            poi = end = new MarkerTemplate { source = null, heightAboveNavMesh = 0.2f };
            foreach (TaskScript task in Object.FindObjectsOfType<TaskScript>())
            {
                GameObject endGo = FirstRendered(task.end);
                if (endGo == null) continue;
                end = Template(endGo);
                GameObject poiGo = FirstRendered(task.pointsOfInterest);
                poi = poiGo != null ? Template(poiGo) : end;
                GameObject startGo = FirstRendered(task.start);
                start = startGo != null ? Template(startGo) : null;
                return;
            }
            Debug.LogWarning("[RecordingRig] No rendered task marker in this scene - synthetic markers fall back to a 0.4 m default-material sphere 0.2 m above the NavMesh.");
        }

        /// <summary>Hides (renderer only) every task marker already in the scene, so a harness's own markers are the only ones in the recorded RGB.</summary>
        public static void HideSceneTaskMarkers(string harnessName)
        {
            foreach (TaskScript task in Object.FindObjectsOfType<TaskScript>())
            {
                foreach (GameObject[] markers in new[] { task.start, task.pointsOfInterest, task.end })
                {
                    if (markers == null) continue;
                    foreach (GameObject m in markers)
                    {
                        if (m == null || m.GetComponent<CharacterController>() != null) continue;
                        foreach (Renderer r in m.GetComponentsInChildren<Renderer>()) r.enabled = false;
                    }
                }
            }
            Debug.Log($"[{harnessName}] Hid the scene's own task markers; the harness shows copies of them at its sampled task positions.");
        }

        /// <summary>A visual-only copy of `template` (no colliders, no scripts), parented to `parent`.</summary>
        public static Transform CreateMarker(MarkerTemplate template, string name, Transform parent)
        {
            GameObject go;
            if (template?.source != null)
            {
                go = Object.Instantiate(template.source);
                foreach (MonoBehaviour mb in go.GetComponentsInChildren<MonoBehaviour>()) Object.Destroy(mb);
                foreach (Renderer r in go.GetComponentsInChildren<Renderer>()) r.enabled = true;
            }
            else
            {
                go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.transform.localScale = Vector3.one * 0.4f;
            }
            foreach (Collider c in go.GetComponentsInChildren<Collider>()) c.enabled = false;
            go.name = name;
            go.SetActive(true);
            go.transform.SetParent(parent, true);
            return go.transform;
        }

        /// <summary>Puts `marker` at the NavMesh point `navMeshPoint`, raised by the template's height above the NavMesh.</summary>
        public static void PlaceMarker(Transform marker, MarkerTemplate template, Vector3 navMeshPoint)
        {
            marker.position = navMeshPoint + Vector3.up * (template != null ? template.heightAboveNavMesh : 0.2f);
        }

        private static GameObject FirstRendered(GameObject[] markers)
        {
            if (markers == null) return null;
            foreach (GameObject m in markers)
            {
                if (m != null && m.GetComponent<CharacterController>() == null && m.GetComponentInChildren<Renderer>() != null) return m;
            }
            return null;
        }

        private static MarkerTemplate Template(GameObject go)
        {
            float height = TryNavMeshHeight(go.transform.position, out float navMeshY) ? go.transform.position.y - navMeshY : 0.2f;
            return new MarkerTemplate { source = go, heightAboveNavMesh = height };
        }

        /// <summary>
        /// Stops every ParticipantRecorder already in the scene (e.g. a layout scene's human FPSController, which has
        /// AutoStartForTesting on) from recording during a harness run. Call from Awake, which runs before any recorder's Start.
        /// </summary>
        public static void DisableSceneRecorders(string harnessName)
        {
            foreach (ParticipantRecorder recorder in Object.FindObjectsOfType<ParticipantRecorder>(true))
            {
                if (!recorder.AutoStartForTesting) continue;
                recorder.AutoStartForTesting = false;
                Debug.Log($"[{harnessName}] Disabled auto-recording on the scene's ParticipantRecorder '{recorder.name}' for this run.");
            }
        }
    }
}
