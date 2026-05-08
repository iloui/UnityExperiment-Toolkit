/*
DesignMind2: A Toolkit for Evidence-Based, Cognitively- Informed and Human-Centered Architectural Design
Copyright (C) 2023-2026  michal Gath-Morad, Christoph Hölscher, Raphaël Baur, Leonel Aguilar

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
*/

// Uncomment once the Ubiq package is present in the project:
// #define UBIQ_PRESENT

using UnityEngine;

// #if UBIQ_PRESENT
// using Ubiq.Messaging;
// #endif

/// <summary>
/// Synchronises a player's body position/rotation and camera orientation over
/// a Ubiq network.  Attach this component alongside PlayerMovement and
/// MouseTracker on the player prefab.
///
/// Ownership is determined by <see cref="isLocalPlayer"/>:
///   - true  → this instance belongs to the local peer; input components are
///             enabled and the transform is broadcast every frame.
///   - false → this instance represents a remote peer; input components are
///             disabled and the transform is driven by incoming messages.
///
/// In a typical Ubiq setup you would set isLocalPlayer from your avatar /
/// peer-manager spawning code, e.g.:
///   networkedPlayer.SetOwnership(peer == NetworkScene.Find(this).Me);
/// </summary>
public class UbiqNetworkedPlayer :
#if UBIQ_PRESENT
    NetworkedBehaviour
#else
    MonoBehaviour
#endif
{
    [Tooltip("True for the local player, false for remote peers. " +
             "Set this from your Ubiq peer/avatar manager after spawning.")]
    public bool isLocalPlayer = true;

    // -------------------------------------------------------------------------
    // Internal state
    // -------------------------------------------------------------------------

    private PlayerMovement playerMovement;
    private MouseTracker   mouseTracker;
    private Transform      cameraTransform;   // Child camera (head) transform.

    [System.Serializable]
    private struct PlayerState
    {
        // Body
        public float bodyPosX, bodyPosY, bodyPosZ;
        public float bodyRotY;              // Yaw only — body never pitches/rolls.
        // Head / camera
        public float headRotX;             // Pitch only — stored as local euler X.
    }

    // -------------------------------------------------------------------------
    // Unity lifecycle
    // -------------------------------------------------------------------------

    private void Start()
    {
        playerMovement = GetComponent<PlayerMovement>();
        mouseTracker   = GetComponentInChildren<MouseTracker>();

        // Find the camera child (same logic CaptureWalkthrough uses).
        for (int i = 0; i < transform.childCount; i++)
        {
            if (transform.GetChild(i).GetComponent<Camera>() != null)
            {
                cameraTransform = transform.GetChild(i);
                break;
            }
        }

        ApplyOwnership();
    }

    /// <summary>
    /// Call this after changing <see cref="isLocalPlayer"/> at runtime.
    /// </summary>
    public void SetOwnership(bool local)
    {
        isLocalPlayer = local;
        ApplyOwnership();
    }

    private void ApplyOwnership()
    {
        if (playerMovement != null) playerMovement.isLocalPlayer = isLocalPlayer;
        if (mouseTracker   != null) mouseTracker.isLocalPlayer   = isLocalPlayer;
    }

    // -------------------------------------------------------------------------
    // Networking
    // -------------------------------------------------------------------------

#if UBIQ_PRESENT
    private void Update()
    {
        if (!isLocalPlayer) return;

        var state = new PlayerState
        {
            bodyPosX = transform.position.x,
            bodyPosY = transform.position.y,
            bodyPosZ = transform.position.z,
            bodyRotY = transform.rotation.eulerAngles.y,
            headRotX = cameraTransform != null
                ? cameraTransform.localRotation.eulerAngles.x
                : 0f,
        };

        context.Send(JsonUtility.ToJson(state));
    }

    protected override void ProcessMessage(ReferenceCountedSceneGraphMessage message)
    {
        if (isLocalPlayer) return;  // Should not receive our own messages.

        var state = JsonUtility.FromJson<PlayerState>(message.ToString());

        transform.position = new Vector3(state.bodyPosX, state.bodyPosY, state.bodyPosZ);
        transform.rotation = Quaternion.Euler(0f, state.bodyRotY, 0f);

        if (cameraTransform != null)
        {
            cameraTransform.localRotation = Quaternion.Euler(state.headRotX, 0f, 0f);
        }
    }
#else
    // Ubiq not yet imported — stub so the project still compiles.
    private void Update() { }
#endif
}
