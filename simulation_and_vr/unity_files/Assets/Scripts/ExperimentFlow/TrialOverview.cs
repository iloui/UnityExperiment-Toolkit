using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Assets.Scripts.DataRecording;
using Random = UnityEngine.Random;

namespace Assets.Scripts
{
    public class TrialOverview : SetupPage
    {
        public int Repetitions = 3;
        
        public Text HeaderText;

        public Text DescriptionText;

        public Image DescriptionImage;

        public TMP_Text HintText;

        public Image HintImage;

        public PlayerMovement FPSController;

        /// <summary>
        /// Optional. When set, trial start/end are synchronised over the network
        /// so all participants in the session execute the same trial simultaneously.
        /// Leave null for single-participant experiments.
        /// </summary>
        public TrialSyncManager SyncManager;

        public ParticipantRecorder Recorder;

        public GameObject Spawnpoint;

        public string NextSceneName;

        private List<Target> tasks;
        
        private List<Material> materials;

        private int totalTasks;

        private Target currentTarget;

        private Material currentMaterial;

        private void Awake()
        {
            // In single-player mode seed by participant id so each participant gets
            // their own shuffled order. In multiplayer mode seed by the shared
            // ExperimentId so every client generates the identical trial sequence.
            var seed = SyncManager != null
                ? (Database.ExperimentId ?? string.Empty).GetHashCode()
                : Database.ParticipantId?.GetHashCode() ?? 0;
            Random.InitState(seed / 2 + SceneManager.GetActiveScene().name.GetHashCode() / 2);
            
            var allTargets = FindObjectsOfType<Target>();

            tasks = new List<Target>(allTargets.Length * Repetitions);
            for (var i = 0; i < Repetitions; i++)
            {
                RandomizeOrder(allTargets);
                tasks.AddRange(allTargets);
            }

            this.totalTasks = this.tasks.Count;
            
            var allMaterials = Resources.LoadAll<Material>("TargetMaterials/");
            materials = new List<Material>(allTargets.Length * Repetitions);

            if (allMaterials.Length == 0)
            {
                Debug.LogError("No target materials available.");
                this.enabled = false;
                Application.Quit(666);
                return;
            }

            while (materials.Count < tasks.Count)
            {
                RandomizeOrder(allMaterials);

                foreach (var material in allMaterials)
                {
                    materials.Add(material);

                    if (materials.Count >= tasks.Count)
                    {
                        break;
                    }
                }
            }
        }

        protected new void OnEnable()
        {
            base.OnEnable();

            Database.EndTrial();

            if (tasks.Count == 0)
            {
                SceneManager.LoadScene(NextSceneName, LoadSceneMode.Single);
                return;
            }

            FPSController.enabled = false;
            Recorder.StopRecording();
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;

            currentTarget = tasks[0];
            tasks.RemoveAt(0);

            currentMaterial = materials[0];
            materials.RemoveAt(0);
            var materialName = currentMaterial.name.ToLowerInvariant();

            var renderer = currentTarget.GetComponent<Renderer>();
            if (renderer == null)
            {
                Debug.LogError("The target has no renderer attached.");
                this.enabled = false;
                Application.Quit(666);
                return;
            }

            renderer.sharedMaterial = currentMaterial;

            HeaderText.text = $"Task Goal #{Database.TrialResults.Count + 1} of {this.totalTasks}";
            DescriptionText.text = $"Find the {currentTarget.Description} and go to the {materialName} ball that is placed there.";
            DescriptionImage.color = currentMaterial.color;
        }

        protected override void OnApplyPage()
        {
            if (SyncManager != null)
            {
                // Multiplayer: broadcast to all peers. The actual trial setup
                // happens in OnNetworkTrialStart() on every client.
                SyncManager.BroadcastTrialStart(currentTarget.Number, currentMaterial.name);
            }
            else
            {
                // Single-player: apply directly.
                OnNetworkTrialStart(currentTarget.Number, currentMaterial.name);
            }
        }

        /// <summary>
        /// Called on every client (including the one that clicked Start) by
        /// TrialSyncManager once the start signal has been broadcast.
        /// Contains all per-trial setup that must happen on every participant's machine.
        /// </summary>
        public void OnNetworkTrialStart(int targetId, string materialName)
        {
            if (HintText != null)
            {
                HintText.text = $"Destination: {currentTarget.Description}\nShape: Ball\nColor: {currentMaterial.name.ToLowerInvariant()}";
            }

            if (HintImage != null)
            {
                HintImage.color = currentMaterial.color;
            }

            PlaceFPSController();
            FPSController.enabled = true;
            Recorder.StartRecording();
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;

            Database.StartNewTrial(targetId, materialName);
        }

        protected override bool CanApplyPage()
        {
            return true;
        }

        private void PlaceFPSController()
        {
            var position = Spawnpoint.transform.position + Vector3.up * 0.9f;
            var rotation = Quaternion.Euler(0, 0, 0);

            FPSController.gameObject.PlaceObject(position, rotation);
        }

        private void RandomizeOrder<T>(IList<T> items)
        {
            var cnt = items.Count;
            for (var i = 0; i < cnt; i++)
            {
                var newIndex = Random.Range(0, cnt);
                var tmp = items[i];
                items[i] = items[newIndex];
                items[newIndex] = tmp;
            }
        }
    }
}
