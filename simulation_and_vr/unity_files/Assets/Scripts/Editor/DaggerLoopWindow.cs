using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using Unity.Sentis;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Settings, progress and controls of the DAgger loop (DaggerLoop). The window is only a view: the loop runs from
    /// EditorApplication.update and its state file, so closing the window does not stop it.
    /// </summary>
    internal class DaggerLoopWindow : EditorWindow
    {
        private const string SettingsPrefKey = "Thesis.DaggerLoopSettings";

        private DaggerLoopState settings;
        private ModelAsset startModel;
        private readonly List<SceneAsset> scenes = new List<SceneAsset>();
        private string baseSourcesText = "";
        private bool allowHumanTrainedStart;
        private Vector2 scroll;

        [MenuItem(ToolsMenu.TrainModel + "Improve Model with DAgger…", priority = ToolsMenu.TrainModelPriority + 2)]
        private static void Open()
        {
            var window = GetWindow<DaggerLoopWindow>("DAgger");
            window.minSize = new Vector2(460, 420);
            window.Show();
        }

        private void OnEnable()
        {
            settings = LoadSettings();
            startModel = string.IsNullOrEmpty(settings.start_model) ? null : AssetDatabase.LoadAssetAtPath<ModelAsset>(settings.start_model);
            scenes.Clear();
            foreach (string path in settings.scenes)
            {
                var scene = AssetDatabase.LoadAssetAtPath<SceneAsset>(path);
                if (scene != null) scenes.Add(scene);
            }
            baseSourcesText = string.Join(", ", settings.base_sources);
        }

        private void OnInspectorUpdate() => Repaint();

        private void OnGUI()
        {
            DaggerLoopState run = DaggerLoop.LoadActive();
            scroll = EditorGUILayout.BeginScrollView(scroll);
            if (run == null) DrawSettings();
            else DrawRun(run);
            EditorGUILayout.EndScrollView();
        }

        // ------------------------------------------------------------------
        // Settings (no active run)
        // ------------------------------------------------------------------

        private void DrawSettings()
        {
            EditorGUILayout.HelpBox(
                "Each round: the current policy drives in every selected scene while the NavMesh expert labels each frame (DaggerHarness), " +
                "then a synthetic-only model (condition A) is trained on the base sources plus all DAgger rounds so far. " +
                "No human data is used, so the result can serve every fold and condition.", MessageType.None);

            EditorGUI.BeginChangeCheck();
            settings.run_name = EditorGUILayout.TextField(new GUIContent("Run name", "Folder imitationagent_training/dagger/<run>, participant ids synthetic_dagger_<run>_r<n>."), settings.run_name);
            startModel = (ModelAsset)EditorGUILayout.ObjectField(new GUIContent("Start model", "The round-1 policy (e.g. an A model trained on synthetic_rrw_v5)."), startModel, typeof(ModelAsset), false);
            baseSourcesText = EditorGUILayout.TextField(new GUIContent("Base sources", "Synthetic participant ids trained on in every round, comma-separated."), baseSourcesText);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Scenes", EditorStyles.boldLabel);
            for (int i = 0; i < scenes.Count; i++)
            {
                EditorGUILayout.BeginHorizontal();
                scenes[i] = (SceneAsset)EditorGUILayout.ObjectField(scenes[i], typeof(SceneAsset), false);
                if (GUILayout.Button("−", GUILayout.Width(24))) { scenes.RemoveAt(i); i--; }
                EditorGUILayout.EndHorizontal();
            }
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Add scene")) scenes.Add(null);
            if (GUILayout.Button("Add open scene"))
            {
                var open = AssetDatabase.LoadAssetAtPath<SceneAsset>(EditorSceneManager.GetActiveScene().path);
                if (open != null && !scenes.Contains(open)) scenes.Add(open);
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Per round", EditorStyles.boldLabel);
            settings.rounds = Mathf.Max(1, EditorGUILayout.IntField("Rounds", settings.rounds));
            settings.episodes_per_scene = Mathf.Max(1, EditorGUILayout.IntField(new GUIContent("Episodes per scene", "Recorded DAgger episodes per scene and round."), settings.episodes_per_scene));
            settings.evaluation_episodes = Mathf.Max(0, EditorGUILayout.IntField(new GUIContent("Evaluation episodes", "Policy-only episodes per scene before each collection, on the same tasks every round; not recorded. 0 = none."), settings.evaluation_episodes));
            settings.hidden_goal_share = EditorGUILayout.Slider(new GUIContent("Hidden-goal share", "Share of tasks whose goal is behind a close wall."), settings.hidden_goal_share, 0f, 1f);
            settings.expert_segment_probability = EditorGUILayout.Slider(new GUIContent("Expert executes (beta)", "Share of 2 s slices the expert drives instead of the policy (stall rescues come on top)."), settings.expert_segment_probability, 0f, 1f);
            settings.expert_walking_turns = EditorGUILayout.Toggle(new GUIContent("Expert: walking turns", "Label 'walk while turning' where there is room, instead of 'stand and turn' at every wall."), settings.expert_walking_turns);
            settings.expert_side_commit_ratio = EditorGUILayout.FloatField(new GUIContent("Expert: side commit ratio", "Take the visibly more open side when both ways around are within this factor. 0 = off."), settings.expert_side_commit_ratio);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Training", EditorStyles.boldLabel);
            settings.dagger_weight = EditorGUILayout.Slider(new GUIContent("DAgger share of batch", "Share of every batch drawn from this run's DAgger episodes (--source_weights)."), settings.dagger_weight, 0.05f, 0.95f);
            settings.epochs = Mathf.Max(1, EditorGUILayout.IntField("Epochs", settings.epochs));
            settings.seed = EditorGUILayout.IntField("Seed", settings.seed);
            settings.target_stop_share = EditorGUILayout.FloatField(new GUIContent("Target stop share", "--target_stop_share; P01-P08 stand in 8.7% of samples."), settings.target_stop_share);
            settings.warm_start = EditorGUILayout.Toggle(new GUIContent("Warm start", "Continue from the previous round's checkpoint (--init_from) instead of training from scratch."), settings.warm_start);
            settings.pause_after_round = EditorGUILayout.Toggle(new GUIContent("Pause after each round", "Stop after every trained round so you can check the numbers before going on."), settings.pause_after_round);
            settings.final_evaluation = EditorGUILayout.Toggle(new GUIContent("Final evaluation", "Evaluate the last model in every scene after the last round."), settings.final_evaluation);
            if (EditorGUI.EndChangeCheck()) StoreSettings();

            EditorGUILayout.Space();
            string problem = Validate(out string warning);
            if (warning != null)
            {
                EditorGUILayout.HelpBox(warning, MessageType.Warning);
                allowHumanTrainedStart = EditorGUILayout.ToggleLeft("Use it anyway (the DAgger data then carries that fold's human influence)", allowHumanTrainedStart);
                if (!allowHumanTrainedStart) problem = problem ?? "Confirm the start model above.";
            }
            if (problem != null) EditorGUILayout.HelpBox(problem, MessageType.Error);
            using (new EditorGUI.DisabledScope(problem != null || EditorApplication.isPlayingOrWillChangePlaymode))
            {
                if (GUILayout.Button("Start", GUILayout.Height(28)) && EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                {
                    StoreSettings();
                    DaggerLoop.Start(settings);
                }
            }
        }

        private string Validate(out string warning)
        {
            warning = null;
            if (string.IsNullOrWhiteSpace(settings.run_name) || !Regex.IsMatch(settings.run_name, @"^[A-Za-z0-9_\-]+$"))
                return "Run name: letters, digits, _ and - only.";
            if (Directory.Exists(DaggerLoop.RunDir(settings.run_name)))
                return $"A run named '{settings.run_name}' already exists (imitationagent_training/dagger/{settings.run_name}). Pick another name.";
            if (startModel == null) return "Pick a start model.";
            if (scenes.Count == 0 || scenes.Any(s => s == null)) return "Add at least one scene (and no empty slots).";
            List<string> sources = ParseSources();
            if (sources.Count == 0) return "Add at least one base source.";
            string bad = sources.FirstOrDefault(s => !s.StartsWith("synthetic"));
            if (bad != null) return $"Base source '{bad}' is not synthetic. The DAgger loop trains condition A only.";
            if (sources.Any(s => s.StartsWith("synthetic_dagger_" + settings.run_name)))
                return "The base sources must not include this run's own DAgger ids.";

            string sidecar = Path.ChangeExtension(AssetDatabase.GetAssetPath(startModel), ".json");
            if (File.Exists(sidecar))
            {
                Match m = Regex.Match(File.ReadAllText(sidecar), "\"condition\":\\s*\"([A-Z])\"");
                if (m.Success && m.Groups[1].Value != "A")
                    warning = $"The start model was trained under condition {m.Groups[1].Value}, i.e. with human data of one fold. " +
                              "Its DAgger states would then reach the other fold's models. An A model avoids that.";
            }
            return null;
        }

        private List<string> ParseSources() =>
            baseSourcesText.Split(new[] { ',', ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()).Where(s => s.Length > 0).Distinct().ToList();

        private void StoreSettings()
        {
            settings.start_model = startModel != null ? AssetDatabase.GetAssetPath(startModel) : "";
            settings.scenes = scenes.Where(s => s != null).Select(AssetDatabase.GetAssetPath).ToList();
            settings.base_sources = ParseSources();
            EditorPrefs.SetString(SettingsPrefKey, JsonUtility.ToJson(settings.CloneSettings()));
        }

        private static DaggerLoopState LoadSettings()
        {
            try
            {
                string json = EditorPrefs.GetString(SettingsPrefKey, "");
                if (!string.IsNullOrEmpty(json)) return JsonUtility.FromJson<DaggerLoopState>(json);
            }
            catch (Exception) { }
            return new DaggerLoopState();
        }

        // ------------------------------------------------------------------
        // Active run
        // ------------------------------------------------------------------

        private void DrawRun(DaggerLoopState run)
        {
            EditorGUILayout.LabelField($"Run '{run.run_name}'", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Status", StatusText(run));
            EditorGUILayout.LabelField("Current model", string.IsNullOrEmpty(run.current_model) ? "-" : Path.GetFileName(run.current_model));
            if (!string.IsNullOrEmpty(run.message))
            {
                MessageType type = run.status == DaggerLoop.Failed ? MessageType.Error : run.status == DaggerLoop.Paused ? MessageType.Warning : MessageType.Info;
                EditorGUILayout.HelpBox(run.message, type);
            }

            EditorGUILayout.Space();
            DrawHistory(run);

            EditorGUILayout.Space();
            EditorGUILayout.BeginHorizontal();
            bool active = DaggerLoop.IsActive(run);
            using (new EditorGUI.DisabledScope(!active || run.pause_requested))
            {
                if (GUILayout.Button("Pause after this step")) DaggerLoop.RequestPause();
            }
            using (new EditorGUI.DisabledScope(run.status != DaggerLoop.Paused && run.status != DaggerLoop.Failed))
            {
                if (GUILayout.Button("Resume")) DaggerLoop.Resume();
            }
            using (new EditorGUI.DisabledScope(!active && run.status != DaggerLoop.Paused && run.status != DaggerLoop.Failed))
            {
                if (GUILayout.Button("Abort") && EditorUtility.DisplayDialog("Abort DAgger run",
                        $"Abort run '{run.run_name}'? Recorded episodes and trained models stay; the run cannot be resumed.", "Abort", "Cancel"))
                {
                    DaggerLoop.Abort();
                }
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Open run folder")) EditorUtility.RevealInFinder(Path.Combine(DaggerLoop.RunDir(run.run_name), "state.json"));
            if (!string.IsNullOrEmpty(run.current_model) && GUILayout.Button("Select current model"))
            {
                var model = AssetDatabase.LoadAssetAtPath<ModelAsset>(run.current_model);
                if (model != null) { Selection.activeObject = model; EditorGUIUtility.PingObject(model); }
            }
            using (new EditorGUI.DisabledScope(active))
            {
                if (GUILayout.Button(new GUIContent("New run…", "Close this run (its folder stays) and configure a new one."))) DaggerLoop.Close();
            }
            EditorGUILayout.EndHorizontal();
        }

        private static string StatusText(DaggerLoopState run)
        {
            string where = run.final_eval_phase ? "final evaluation" : $"round {Mathf.Min(run.round, run.rounds)}/{run.rounds}";
            string scene = run.scene_index < run.scenes.Count ? Path.GetFileNameWithoutExtension(run.scenes[run.scene_index]) : "";
            switch (run.status)
            {
                case DaggerLoop.CollectScene:
                case DaggerLoop.EnteringPlay:
                case DaggerLoop.Playing: return $"{where} · collecting · {scene} ({run.scene_index + 1}/{run.scenes.Count})";
                case DaggerLoop.Train:
                case DaggerLoop.Training: return $"{where} · training";
                default: return $"{where} · {run.status}{(run.pause_requested ? " (pausing)" : "")}";
            }
        }

        private static void DrawHistory(DaggerLoopState run)
        {
            EditorGUILayout.LabelField("Rounds", EditorStyles.boldLabel);
            if (run.history.Count == 0)
            {
                EditorGUILayout.LabelField("  nothing finished yet");
                return;
            }
            EditorGUILayout.LabelField("Evaluation = policy alone, same tasks every round (hidden-goal tasks in brackets).", EditorStyles.miniLabel);
            foreach (DaggerRoundRecord r in run.history.OrderBy(h => h.round))
            {
                string title = r.final_evaluation ? "Final" : $"Round {r.round}";
                EditorGUILayout.LabelField($"{title}  ·  policy {Path.GetFileNameWithoutExtension(r.model)}" +
                                           (string.IsNullOrEmpty(r.trained_model) ? "" : $"  →  {Path.GetFileNameWithoutExtension(r.trained_model)}"),
                    EditorStyles.miniBoldLabel);
                foreach (DaggerSceneRecord sc in r.scenes)
                {
                    string eval = sc.eval_episodes > 0
                        ? $"eval {Percent(sc.eval_reached, sc.eval_episodes)}" + (sc.eval_hidden > 0 ? $" ({Percent(sc.eval_hidden_reached, sc.eval_hidden)})" : "")
                        : "eval -";
                    string collect = r.final_evaluation ? "" :
                        $"   ·   {sc.recorded} recorded, {sc.recorded - sc.reached_goal} timed out, {sc.stall_rescues} rescues, expert drove {sc.expert_executed_share:P0}";
                    EditorGUILayout.LabelField($"    {sc.scene}", eval + collect, EditorStyles.miniLabel);
                }
            }
        }

        private static string Percent(int k, int n) => n > 0 ? $"{100f * k / n:F0}% ({k}/{n})" : "-";
    }
}
