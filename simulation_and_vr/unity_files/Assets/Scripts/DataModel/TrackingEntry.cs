using UnityEngine;

namespace Assets.Scripts
{
    public struct TrackingEntry
    {
        public float Time;

        public Vector3 Position;

        public float ViewAzimuth;

        public float ViewElevation;

        public float[] ObservationSpace;
    }
}
