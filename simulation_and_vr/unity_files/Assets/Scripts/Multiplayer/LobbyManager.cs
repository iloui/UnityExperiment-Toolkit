/*
DesignMind2: A Toolkit for Evidence-Based, Cognitively-Informed and Human-Centered Architectural Design
Copyright (C) 2023-2026  michal Gath-Morad, Christoph Hölscher, Raphaël Baur, Leonel Aguilar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
*/

// Uncomment once the Ubiq package is present in the project:
// #define UBIQ_PRESENT

using System;
using System.Collections;
using Assets.Scripts;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.SceneManagement;

// #if UBIQ_PRESENT
// using Ubiq.Messaging;
// using Ubiq.Rooms;
// using Ubiq.NetworkedBehaviour;
// #endif

/// <summary>
/// Manages the lobby scene that gates entry into the collaborative VR experiment.
///
/// On startup it:
///   1. Fetches the experiment config JSON (URL set in the Inspector).
///   2. Writes ExperimentId and DataCollectionServerURL into Database.
///   3. Waits until the Ubiq peer count reaches config.requiredParticipants.
///   4. The first client to see the threshold broadcasts a "countdown_start" signal.
///   5. All clients (including the sender) run a countdown and then load
///      config.nextSceneName simultaneously.
///
/// For single-participant experiments set requiredParticipants = 1 in the config;
/// the lobby transitions immediately without waiting for remote peers.
///
/// Scene setup:
///   - Add an Ubiq NetworkScene to the lobby scene.
///   - Attach this component to a persistent GameObject.
///   - Wire StatusText and set ConfigUrl in the Inspector.
/// </summary>
public class LobbyManager :
#if UBIQ_PRESENT
    NetworkedBehaviour
#else
    MonoBehaviour
#endif
{
    [Tooltip("URL of the experiment config JSON, relative to the streaming-assets root " +
             "or an absolute URL. E.g. 'experiment_1_config.json'.")]
    public string ConfigUrl = "experiment_1_config.json";

    [Tooltip("TMP_Text element used to display lobby status to the participant.")]
    public TMP_Text StatusText;

    // -----------------------------------------------------------------
    // Internal state
    // -----------------------------------------------------------------

    private VrExperimentConfig config;

#if UBIQ_PRESENT
    private NetworkScene networkScene;
#endif

    private bool countdownStarted;

    [Serializable]
    private struct LobbyMessage
    {
        public string type; // "countdown_start"
    }

    // -----------------------------------------------------------------
    // Unity lifecycle
    // -----------------------------------------------------------------

    private IEnumerator Start()
    {
        SetStatus("Loading experiment configuration\u2026");

        yield return FetchConfig();

        if (config == null)
        {
            SetStatus("Error: could not load experiment configuration.\nCheck the console and verify ConfigUrl.");
            yield break;
        }

        // Push shared state into the static Database so every subsequent scene
        // inherits the correct ExperimentId and server URL.
        Database.ExperimentId = config.experimentId;
        if (!string.IsNullOrWhiteSpace(config.dataAssemblyUrl))
        {
            Database.DataCollectionServerURL = config.dataAssemblyUrl;
        }

#if UBIQ_PRESENT
        networkScene = NetworkScene.Find(this);

        if (networkScene == null)
        {
            Debug.LogWarning("LobbyManager: No Ubiq NetworkScene found in scene. " +
                             "Treating as single-participant session.");
            StartCountdown();
            yield break;
        }

        // Subscribe to peer changes and do an immediate check in case the
        // required number of peers is already connected.
        networkScene.OnPeerAdded   += OnPeerCountChanged;
        networkScene.OnPeerRemoved += OnPeerCountChanged;
        CheckPeerCount();
#else
        // Ubiq not available: treat as single-participant, skip straight to countdown.
        StartCountdown();
#endif
    }

    // -----------------------------------------------------------------
    // Peer counting (Ubiq path)
    // -----------------------------------------------------------------

#if UBIQ_PRESENT
    private void OnPeerCountChanged(IPeer _) => CheckPeerCount();

    private void CheckPeerCount()
    {
        if (countdownStarted) return;

        var connected = networkScene.Peers.Count;
        var required  = config?.requiredParticipants ?? 1;

        SetStatus($"Waiting for participants\u2026\n{connected} / {required} connected");

        if (connected >= required)
        {
            // First peer to notice broadcasts so all clients get the same signal.
            var msg = new LobbyMessage { type = "countdown_start" };
            context.Send(JsonUtility.ToJson(msg));
            StartCountdown();
        }
    }

    // -----------------------------------------------------------------
    // Ubiq message handling
    // -----------------------------------------------------------------

    protected override void ProcessMessage(ReferenceCountedSceneGraphMessage message)
    {
        var msg = JsonUtility.FromJson<LobbyMessage>(message.ToString());
        if (msg.type == "countdown_start")
        {
            StartCountdown();
        }
    }
#endif

    // -----------------------------------------------------------------
    // Countdown and scene transition
    // -----------------------------------------------------------------

    private void StartCountdown()
    {
        if (countdownStarted) return;
        countdownStarted = true;
        StartCoroutine(CountdownCoroutine());
    }

    private IEnumerator CountdownCoroutine()
    {
        var seconds = Mathf.Max(1, Mathf.RoundToInt(config?.countdownSeconds ?? 3f));

        for (var i = seconds; i > 0; i--)
        {
            SetStatus($"All participants connected!\nStarting in {i}\u2026");
            yield return new WaitForSeconds(1f);
        }

        var scene = config?.nextSceneName;
        if (string.IsNullOrWhiteSpace(scene))
        {
            Debug.LogError("LobbyManager: nextSceneName is not set in the experiment config.");
            yield break;
        }

        SceneManager.LoadScene(scene, LoadSceneMode.Single);
    }

    // -----------------------------------------------------------------
    // Config fetching
    // -----------------------------------------------------------------

    private IEnumerator FetchConfig()
    {
        var url = ConfigUrl;

        // Relative URLs are resolved against StreamingAssets in standalone builds
        // and against Application.absoluteURL in WebGL builds.
        if (!url.StartsWith("http", StringComparison.OrdinalIgnoreCase) &&
            !url.StartsWith("file", StringComparison.OrdinalIgnoreCase))
        {
            if (Application.platform == RuntimePlatform.WebGLPlayer)
            {
                // Same origin as the WebGL page.
                var origin = Application.absoluteURL;
                var lastSlash = origin.LastIndexOf('/');
                url = (lastSlash >= 0 ? origin.Substring(0, lastSlash + 1) : origin) + url;
            }
            else
            {
                url = System.IO.Path.Combine(Application.streamingAssetsPath, url);
            }
        }

        using (var request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"LobbyManager: Failed to fetch config from '{url}'.\n{request.error}");
                yield break;
            }

            config = JsonUtility.FromJson<VrExperimentConfig>(request.downloadHandler.text);

            if (config == null)
            {
                Debug.LogError("LobbyManager: Config JSON parsed to null. Check the file format.");
            }
        }
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    private void SetStatus(string text)
    {
        if (StatusText != null)
            StatusText.text = text;

        Debug.Log($"[Lobby] {text}");
    }
}
