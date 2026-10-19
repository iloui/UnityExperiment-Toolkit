using Assets.Scripts.DataRecording;
using UnityEditor;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Manual control of RetrainPipeline, e.g. after a bot recording run (the harnesses never retrain; their data is picked up
    /// by the next session's retrain or this one).
    /// </summary>
    internal static class RetrainMenu
    {
        [MenuItem(ToolsMenu.TrainModel + "Train Model Now", priority = ToolsMenu.TrainModelPriority)]
        private static void RetrainNow()
        {
            RetrainPipeline.RequestFullRetrain("manual (Tools > Imitation Learning Agent > 3 · Train Model)");
        }

        [MenuItem(ToolsMenu.TrainModel + "Show Training Status", priority = ToolsMenu.TrainModelPriority + 1)]
        private static void ShowStatus()
        {
            string report = RetrainPipeline.StatusReport();
            Debug.Log("[Retrain] Status\n" + report);
            EditorUtility.DisplayDialog("Retrain Status", report, "OK");
        }
    }
}
