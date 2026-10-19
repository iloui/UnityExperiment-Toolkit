using System.Collections.Generic;
using System.Linq;
using Assets.Scripts.DataRecording;
using Unity.Sentis;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// One-click setup for the thesis evaluation of the open layout scene. Runs ScenePreparation.Prepare first (idempotent:
    /// keeps the scene's hand-placed Start/POI_0/End, re-bakes the NavMesh and re-exports the occupancy grid that
    /// eval_tools reads), then points a fresh or existing EvalHarness at the scene's task, sets the canonical evaluation
    /// values (methodology 3.8.1) and fills the queue: ShortestPath first, then every ImitationAgent_&lt;A|B|C&gt;_proprio2*
    /// model in Assets/ImitationModel with its training sidecar, in condition / fold / seed order.
    /// </summary>
    internal static class SetupEvalScene
    {
        private const string ModelFolder = "Assets/ImitationModel";
        private const string ModelNamePrefix = "ImitationAgent_";
        private const string ModelTag = "_proprio2";

        [MenuItem(ToolsMenu.EvaluateDesign + "Set Up Evaluation Harness", priority = ToolsMenu.EvaluateDesignPriority)]
        public static void WireUpCurrentScene()
        {
            Scene targetScene = EditorSceneManager.GetActiveScene();
            if (!targetScene.IsValid())
            {
                Debug.LogError("[SetupEvalScene] No valid active scene - open a scene first.");
                return;
            }

            try
            {
                ScenePreparation.Prepare(targetScene);
            }
            catch (System.Exception ex)
            {
                Debug.LogException(ex);
                EditorUtility.DisplayDialog("Eval scene setup failed", ex.Message, "OK");
                return;
            }
            TaskScript task = ScenePreparation.FindTask(targetScene);

            GameObject harnessGO = GameObject.Find("EvalHarness") ?? new GameObject("EvalHarness");
            EvalHarness harness = harnessGO.GetComponent<EvalHarness>();
            if (harness == null) harness = harnessGO.AddComponent<EvalHarness>();

            List<EvalHarness.QueueEntry> queue = BuildQueue(out List<string> missingSidecars);

            harness.task = task;
            harness.queue = queue;
            // Canonical evaluation values, identical for every layout. Re-running this menu item resets them.
            harness.configurationCount = 20;
            harness.simulationSpeed = 3f;
            harness.episodeTimeoutSeconds = 90f;
            harness.startJitterPositionMeters = 0.5f;
            harness.startJitterHeadingDegrees = 15f;
            harness.randomizeMarkersPerConfiguration = false;
            harness.disableDeadlockNudge = true;
            harness.imitationStochasticWalk = true;
            harness.imitationMinWalkHoldSeconds = 1.5f;
            harness.maxPlausibleSpeedMps = 2.5f;
            harness.heatmapEveryNthEpisode = 0;
            EditorUtility.SetDirty(harness);
            EditorSceneManager.MarkSceneDirty(targetScene);

            string list = string.Join("\n", queue.Select((e, i) => $"  {i + 1,2}. {e.Label}"));
            string warning = missingSidecars.Count > 0
                ? $"\n\nSkipped (no sidecar .json next to the model): {string.Join(", ", missingSidecars)}"
                : "";
            Debug.Log($"[SetupEvalScene] '{targetScene.name}' wired up: queue '{harness.queueName}', {queue.Count} entries x " +
                $"{harness.configurationCount} episodes, base seed {harness.baseSeed}:\n{list}{warning}");
            EditorUtility.DisplayDialog("Eval scene wired up",
                $"'{targetScene.name}': {queue.Count} entries x {harness.configurationCount} episodes on the hand-placed task " +
                $"(jitter 0.5 m / 15 deg, 90 s, 3x speed). Occupancy grid re-exported.{warning}\n\nSave the scene (Cmd+S), then press Play.",
                "OK");
        }

        /// <summary>ShortestPath, then the proprio2 models sorted by condition, fold, seed; each with its same-named sidecar.</summary>
        private static List<EvalHarness.QueueEntry> BuildQueue(out List<string> missingSidecars)
        {
            missingSidecars = new List<string>();
            var models = new List<(string condition, string name, ModelAsset model, TextAsset sidecar)>();
            foreach (string guid in AssetDatabase.FindAssets("t:ModelAsset", new[] { ModelFolder }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                string name = System.IO.Path.GetFileNameWithoutExtension(path);
                if (!name.StartsWith(ModelNamePrefix) || !name.Contains(ModelTag)) continue;
                string condition = name.Substring(ModelNamePrefix.Length, 1);
                if (condition != "A" && condition != "B" && condition != "C") continue;
                if (name[ModelNamePrefix.Length + 1] != '_') continue;

                var sidecar = AssetDatabase.LoadAssetAtPath<TextAsset>(System.IO.Path.ChangeExtension(path, ".json"));
                if (sidecar == null)
                {
                    missingSidecars.Add(name);
                    continue;
                }
                models.Add((condition, name, AssetDatabase.LoadAssetAtPath<ModelAsset>(path), sidecar));
            }

            // Names sort as condition, then f<fold>, then s<seed> (A has no fold).
            var queue = new List<EvalHarness.QueueEntry> { new EvalHarness.QueueEntry { agentKind = EvalHarness.AgentKind.ShortestPath } };
            foreach (var m in models.OrderBy(m => m.condition).ThenBy(m => m.name, System.StringComparer.Ordinal))
            {
                queue.Add(new EvalHarness.QueueEntry { agentKind = EvalHarness.AgentKind.Imitation, modelAsset = m.model, sidecar = m.sidecar });
            }
            return queue;
        }
    }
}
