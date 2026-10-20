#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.Sentis;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// Play-mode half of the DAgger loop (Editor/DaggerLoop.cs). The loop writes a PlayRequest to
    /// imitationagent_training/dagger/play_request.json and enters Play mode; on entering, this spawns a DaggerHarness
    /// configured from the request in the open scene and marks the request consumed (play_request.json.active). Other
    /// harnesses in the scene are disabled for that session. Nothing is added to the scene asset: the harness exists only
    /// for this Play session. Without a request file it does nothing, so normal Play sessions are unaffected.
    /// </summary>
    public static class DaggerRunBootstrap
    {
        [Serializable]
        public class PlayRequest
        {
            public string request_id;
            public string scene_path;
            public string model_asset;
            public string participant_id;
            public int episodes;
            public int evaluation_episodes;
            public int evaluation_seed;
            public int seed;
            public float hidden_goal_share;
            public float expert_segment_probability;
            public bool expert_walking_turns;
            public float expert_side_commit_ratio;
            public string summary_path;
        }

        private static string DaggerDir => Path.GetFullPath(Path.Combine(Application.dataPath, "../../imitationagent_training/dagger"));
        public static string RequestPath => Path.Combine(DaggerDir, "play_request.json");
        public static string ConsumedPath => RequestPath + ".active";

        public static void WriteRequest(PlayRequest request)
        {
            Directory.CreateDirectory(DaggerDir);
            if (File.Exists(ConsumedPath)) File.Delete(ConsumedPath);
            File.WriteAllText(RequestPath, JsonUtility.ToJson(request, true));
        }

        /// <summary>The request a Play session picked up, or null.</summary>
        public static PlayRequest ReadConsumed() => Read(ConsumedPath);
        public static PlayRequest ReadPending() => Read(RequestPath);

        public static void ClearRequest()
        {
            TryDelete(RequestPath);
            TryDelete(ConsumedPath);
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void OnEnterPlayMode()
        {
            PlayRequest request = ReadPending();
            if (request == null) return;
            string scenePath = SceneManager.GetActiveScene().path;
            if (!string.Equals(scenePath, request.scene_path, StringComparison.Ordinal))
            {
                Debug.LogWarning($"[DAgger loop] A DAgger request for '{request.scene_path}' is pending, but '{scenePath}' was played - not started here.");
                return;
            }
            File.Move(RequestPath, ConsumedPath); // consumed: the loop now waits for the summary

            foreach (var other in UnityEngine.Object.FindObjectsOfType<DaggerHarness>()) other.enabled = false;
            foreach (var other in UnityEngine.Object.FindObjectsOfType<EvalHarness>()) other.enabled = false;
            foreach (var other in UnityEngine.Object.FindObjectsOfType<RecordRandomWalksHarness>()) other.enabled = false;

            var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(request.model_asset);
            if (model == null)
            {
                Debug.LogError($"[DAgger loop] Model '{request.model_asset}' could not be loaded as a ModelAsset - stopping Play mode.");
                EditorApplication.isPlaying = false;
                return;
            }

            var go = new GameObject("DaggerLoop_Harness");
            var harness = go.AddComponent<DaggerHarness>();
            harness.modelAsset = model;
            harness.episodeCount = Mathf.Max(0, request.episodes);
            harness.seed = request.seed;
            harness.participantId = request.participant_id;
            harness.evaluationEpisodes = Mathf.Max(0, request.evaluation_episodes);
            harness.evaluationSeed = request.evaluation_seed;
            harness.hiddenGoalShare = request.hidden_goal_share;
            harness.expertSegmentProbability = request.expert_segment_probability;
            harness.expertWalkingTurns = request.expert_walking_turns;
            harness.expertSideCommitMaxLengthRatio = request.expert_side_commit_ratio;
            harness.summaryPath = request.summary_path;
            harness.stopPlayModeWhenDone = true;
            Debug.Log($"[DAgger loop] DaggerHarness started in '{scenePath}': model '{model.name}', {request.evaluation_episodes} evaluation + " +
                      $"{request.episodes} collection episodes as '{request.participant_id}'.");
        }

        private static PlayRequest Read(string path)
        {
            try
            {
                return File.Exists(path) ? JsonUtility.FromJson<PlayRequest>(File.ReadAllText(path)) : null;
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static void TryDelete(string path)
        {
            try { if (File.Exists(path)) File.Delete(path); }
            catch (IOException) { }
        }
    }
}
#endif
