using Unity.Collections;
using UnityEngine;

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// The one camera model behind both the RGB and the depth channel. ParticipantRecorder and ImitationAgent configure their
    /// hidden capture camera with Configure and build the depth rays with FillDepthRays from that same camera, so depth pixel
    /// (x, y) is the distance along the ray through the centre of RGB pixel (x, y) and the 4-channel stack is pixel-aligned.
    /// </summary>
    public static class ContractCamera
    {
        /// <summary>Forces the contract FOV and a square aspect. Call after assigning the 64x64 targetTexture.</summary>
        public static void Configure(Camera camera)
        {
            camera.fieldOfView = ZarrDataContract.CameraFovDegrees;
            camera.aspect = (float)ZarrDataContract.ImageWidth / ZarrDataContract.ImageHeight;
        }

        /// <summary>
        /// Fills one ray per pixel, from the camera position through the pixel centre. Index y * ImageWidth + x with y = 0 the
        /// BOTTOM row, the same order as Texture2D.ReadPixels/GetPixels32 on the RGB frame. The hit distance along the ray,
        /// divided by MaxRayDistance, is the depth value.
        /// </summary>
        public static void FillDepthRays(Camera camera, NativeArray<RaycastCommand> commands, LayerMask layerMask)
        {
            var query = new QueryParameters(layerMask, true, QueryTriggerInteraction.Collide, false);
            Vector3 origin = camera.transform.position;
            int index = 0;
            for (int y = 0; y < ZarrDataContract.ImageHeight; y++)
            {
                for (int x = 0; x < ZarrDataContract.ImageWidth; x++)
                {
                    Ray ray = camera.ViewportPointToRay(new Vector3(
                        (x + 0.5f) / ZarrDataContract.ImageWidth,
                        (y + 0.5f) / ZarrDataContract.ImageHeight,
                        0f));
                    commands[index++] = new RaycastCommand(origin, ray.direction, query, ZarrDataContract.MaxRayDistance);
                }
            }
        }
    }
}
