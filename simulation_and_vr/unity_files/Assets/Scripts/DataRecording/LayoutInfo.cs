using UnityEngine;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Which generated layout this scene is. LayoutGeneratorWindow adds it to the environment root; ParticipantRecorder copies
    /// it into every episode manifest (layout_typology, layout_seed), so Python can group episodes by layout without the scene.
    /// Recording refuses to start in a scene without exactly one LayoutInfo.
    /// </summary>
    public class LayoutInfo : MonoBehaviour
    {
        [Tooltip("LayoutTopology name the scene was generated with (e.g. 'Comb').")]
        public string typology;
        [Tooltip("Internal RNG seed LayoutBuilder built the scene from.")]
        public int seed;
        [Tooltip("Full seed code from the Generate Layout window (typology, dimensions, comb settings, seed); regenerates the scene.")]
        public string seedCode;

        /// <summary>The scene's single LayoutInfo. Throws if there is none or more than one.</summary>
        public static LayoutInfo ResolveInScene()
        {
            var all = Object.FindObjectsOfType<LayoutInfo>(true);
            if (all.Length != 1)
            {
                throw new System.InvalidOperationException(
                    $"[LayoutInfo] Expected exactly one LayoutInfo in the scene, found {all.Length}. Generate the scene with " +
                    "Tools > Generate Layout (it adds one to the environment root), or add one by hand with the typology and seed.");
            }
            if (string.IsNullOrWhiteSpace(all[0].typology))
            {
                throw new System.InvalidOperationException($"[LayoutInfo] '{all[0].name}' has no typology set.");
            }
            return all[0];
        }
    }
}
