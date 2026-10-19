using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Assets.Scripts.ExperimentFlow
{
    /// <summary>
    /// Everything that defines a human recording session, so every participant gets the same procedure: which practice
    /// scene and layout scenes, how many training tasks per layout, and the on-screen instructions. ParticipantSession reads it; the
    /// layout order, the training-task order and the fold per participant are derived from the participant number (see
    /// SessionSchedule), never set by hand. The tasks themselves live in each layout scene's HumanTaskSet.
    /// Create it with Tools > Imitation Learning Agent > 2 · Record Human Walks > Create Participant Session Scene.
    /// </summary>
    [CreateAssetMenu(fileName = "ParticipantSessionConfig", menuName = "Recording/Participant Session Config")]
    public class ParticipantSessionConfig : ScriptableObject
    {
#if UNITY_EDITOR
        [Header("Scenes")]
        [Tooltip("Untracked practice run. Must be a layout scene (FPSController + ParticipantRecorder + TaskScript) that is NOT one of the layout scenes below.")]
        public SceneAsset practiceScene;
        [Tooltip("The layouts every participant walks: the training tasks of the scene's HumanTaskSet, then its test task. Layout order is counterbalanced per participant (balanced Latin square).")]
        public SceneAsset[] layoutScenes = new SceneAsset[3];
#endif
        [Tooltip("Filled from the scene fields above (a build has no SceneAsset).")]
        [SerializeField] private string practiceScenePath;
        [SerializeField] private string[] layoutScenePaths = new string[0];

        [Header("Procedure")]
        [Tooltip("Training tasks walked per layout before the test task (methodology 3.8.2). Every layout scene's HumanTaskSet must have exactly this many.")]
        [Min(1)] public int trainingTasksPerLayout = 3;
        [Tooltip("Participant IDs must match this regex. The trailing number selects the Latin-square row.")]
        public string participantIdPattern = @"^P\d{2,3}$";
        [Tooltip("Allow a session for an ID that already has completed episodes in the session log (e.g. to resume). " +
                 "Off: such an ID is refused, so a typo can't mix two people under one ID.")]
        public bool allowExistingParticipant = false;

        [Header("Experimenter keys")]
        [Tooltip("Aborts the running trial (or skips practice). An aborted episode is moved out of zarr_recordings and the trial is repeated.")]
        public KeyCode abortTrialKey = KeyCode.F12;

        [Header("Dry run")]
        [Tooltip("Checks the procedure without a participant: IDs are recorded as 'dryrun_<ID>', rows go to session_log_dryrun.csv, " +
                 "and the teleport key jumps the player to the current target. Delete the dry-run episodes afterwards.")]
        public bool dryRun = false;
        public KeyCode dryRunTeleportKey = KeyCode.T;

        [Header("Instructions")]
        [TextArea(6, 20)]
        public string instructions =
            "You will walk through three buildings, four runs in each (12 runs in total).\n\n" +
            "In every run:\n" +
            "1. Walk to the sphere labelled POI.\n" +
            "2. Then walk to the sphere labelled EXIT. The run ends when you reach it.\n\n" +
            "The spheres are in different places in every run. Walk the way you naturally would. There is no time limit.\n\n" +
            "Controls: W A S D to walk, mouse to look around.\n\n" +
            "First comes a practice run that is not recorded.";

        public string PracticeScenePath
        {
            get
            {
#if UNITY_EDITOR
                SyncScenePaths();
#endif
                return practiceScenePath;
            }
        }

        public string[] LayoutScenePaths
        {
            get
            {
#if UNITY_EDITOR
                SyncScenePaths();
#endif
                return layoutScenePaths;
            }
        }

#if UNITY_EDITOR
        private void OnValidate() => SyncScenePaths();

        private void SyncScenePaths()
        {
            practiceScenePath = practiceScene != null ? AssetDatabase.GetAssetPath(practiceScene) : "";
            layoutScenePaths = new string[layoutScenes != null ? layoutScenes.Length : 0];
            for (int i = 0; i < layoutScenePaths.Length; i++)
            {
                layoutScenePaths[i] = layoutScenes[i] != null ? AssetDatabase.GetAssetPath(layoutScenes[i]) : "";
            }
        }
#endif
    }
}
