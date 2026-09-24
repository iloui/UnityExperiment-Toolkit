using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Text;
using Assets.Scripts.DataRecording;
using Unity.AI.Navigation;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.SceneManagement;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Rasterises a layout's Architecture_Obstacle colliders into a top-down occupancy grid and writes it, with the task markers
    /// and the NavMesh shortest path, to S/eval_tools/layouts/&lt;scene_name&gt;.npz. Python (eval_tools/layout_metrics.py) derives
    /// wall distance and wall contacts from it, identically for human, synthetic, agent and NavMesh-baseline trajectories.
    ///
    /// Grid convention: occupancy[row, col] covers x in [origin_x + col * cell, origin_x + (col + 1) * cell) and
    /// z in [origin_z + row * cell, origin_z + (row + 1) * cell). A cell is occupied if any Architecture_Obstacle collider
    /// intersects its column between floor + FloorClearanceMeters and the contract eye height (so the ceiling never counts).
    /// ScenePreparation calls it at the end of Prepare Scene.
    /// </summary>
    internal static class OccupancyGridExporter
    {
        public const int FormatVersion = 1;
        public const float CellSizeMeters = 0.05f;
        private const float FloorClearanceMeters = 0.1f; // column starts above the floor so floor geometry never counts
        private const int PaddingCells = 4;
        private const float MarkerNavMeshSearchRadius = 1.5f; // POI/End spheres float 1 m above the NavMesh

        /// <summary>S/eval_tools/layouts, from the Unity project's Assets folder.</summary>
        public static string OutputDirectory =>
            Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "eval_tools", "layouts"));

        /// <summary>Writes the .npz for `scene` and returns its path. Needs a baked NavMesh and a complete task.</summary>
        public static string Export(Scene scene, TaskScript task, LayoutInfo layoutInfo, NavMeshSurface surface)
        {
            if (task == null || task.start.Length == 0 || task.end.Length == 0)
                throw new InvalidOperationException("Occupancy export needs a task with Start and End.");

            // Markers on the NavMesh: Start sits on it, POI/End spheres float above it.
            Vector3 start = SnapToNavMesh(task.start[0].transform.position, "Start");
            var route = new List<Vector3> { start };
            foreach (GameObject poi in task.pointsOfInterest) route.Add(SnapToNavMesh(poi.transform.position, poi.name));
            route.Add(SnapToNavMesh(task.end[0].transform.position, "End"));
            float floorY = start.y;

            int mask = LayerMask.GetMask(ZarrDataContract.ArchitectureObstacleLayerName);
            if (mask == 0) throw new InvalidOperationException($"Layer '{ZarrDataContract.ArchitectureObstacleLayerName}' does not exist.");

            // Grid extent: the xz bounds of every obstacle collider, plus a few cells of padding.
            bool any = false;
            Bounds bounds = default;
            foreach (GameObject rootGo in scene.GetRootGameObjects())
            {
                foreach (Collider c in rootGo.GetComponentsInChildren<Collider>(true))
                {
                    if (!c.enabled || c.isTrigger || (mask & (1 << c.gameObject.layer)) == 0) continue;
                    if (!any) { bounds = c.bounds; any = true; }
                    else bounds.Encapsulate(c.bounds);
                }
            }
            if (!any) throw new InvalidOperationException("The scene has no Architecture_Obstacle colliders.");

            float cell = CellSizeMeters;
            float originX = Mathf.Floor(bounds.min.x / cell) * cell - PaddingCells * cell;
            float originZ = Mathf.Floor(bounds.min.z / cell) * cell - PaddingCells * cell;
            int width = Mathf.CeilToInt((bounds.max.x - originX) / cell) + PaddingCells;
            int height = Mathf.CeilToInt((bounds.max.z - originZ) / cell) + PaddingCells;

            float bottom = floorY + FloorClearanceMeters;
            float top = floorY + ZarrDataContract.CameraHeightAboveNavMeshMeters;
            var halfExtents = new Vector3(cell * 0.5f, (top - bottom) * 0.5f, cell * 0.5f);
            float centerY = (top + bottom) * 0.5f;

            Physics.SyncTransforms();
            var occupancy = new bool[height * width];
            int occupiedCount = 0;
            for (int row = 0; row < height; row++)
            {
                float z = originZ + (row + 0.5f) * cell;
                for (int col = 0; col < width; col++)
                {
                    var center = new Vector3(originX + (col + 0.5f) * cell, centerY, z);
                    if (Physics.CheckBox(center, halfExtents, Quaternion.identity, mask, QueryTriggerInteraction.Ignore))
                    {
                        occupancy[row * width + col] = true;
                        occupiedCount++;
                    }
                }
            }

            // NavMesh shortest path per segment (Start -> POI_0 -> ... -> End), and the concatenated corners for plotting.
            var segmentLengths = new float[route.Count - 1];
            var pathCorners = new List<Vector3>();
            for (int i = 0; i < segmentLengths.Length; i++)
            {
                var path = new NavMeshPath();
                if (!NavMesh.CalculatePath(route[i], route[i + 1], NavMesh.AllAreas, path) || path.status != NavMeshPathStatus.PathComplete)
                    throw new InvalidOperationException($"No complete NavMesh path for task segment {i}.");
                for (int k = 0; k < path.corners.Length; k++)
                {
                    if (k > 0) segmentLengths[i] += Vector3.Distance(path.corners[k - 1], path.corners[k]);
                    if (k > 0 || i == 0) pathCorners.Add(path.corners[k]);
                }
            }

            float bakeRadius = surface != null ? NavMesh.GetSettingsByID(surface.agentTypeID).agentRadius : float.NaN;

            Directory.CreateDirectory(OutputDirectory);
            string outPath = Path.Combine(OutputDirectory, scene.name + ".npz");
            string tmpPath = outPath + ".tmp";
            using (var file = new FileStream(tmpPath, FileMode.Create, FileAccess.Write))
            using (var zip = new ZipArchive(file, ZipArchiveMode.Create))
            {
                Npy.WriteInt32(zip, "format_version", FormatVersion);
                Npy.WriteString(zip, "scene_name", scene.name);
                Npy.WriteString(zip, "layout_typology", layoutInfo != null ? layoutInfo.typology ?? "" : "");
                Npy.WriteInt32(zip, "layout_seed", layoutInfo != null ? layoutInfo.seed : 0);
                Npy.WriteBool2D(zip, "occupancy", occupancy, height, width);
                Npy.WriteFloat64(zip, "origin_xz", new double[] { originX, originZ }, 2);
                Npy.WriteFloat64(zip, "cell_size", new double[] { cell });
                Npy.WriteFloat64(zip, "floor_y", new double[] { floorY });
                Npy.WriteFloat64(zip, "slice_y", new double[] { bottom, top }, 2);
                Npy.WriteFloat64(zip, "navmesh_bake_radius", new double[] { bakeRadius });
                Npy.WriteFloat64(zip, "body_radius", new double[] { ZarrDataContract.RecordingBody.Radius });
                Npy.WriteFloat64(zip, "arrival_radius", new double[] { ZarrDataContract.ArrivalRadiusMeters });
                Npy.WriteVectors(zip, "start", new List<Vector3> { route[0] }, squeeze: true);
                Npy.WriteVectors(zip, "pois", route.GetRange(1, route.Count - 2), squeeze: false);
                Npy.WriteVectors(zip, "end", new List<Vector3> { route[route.Count - 1] }, squeeze: true);
                var lengths = new double[segmentLengths.Length];
                for (int i = 0; i < lengths.Length; i++) lengths[i] = segmentLengths[i];
                Npy.WriteFloat64(zip, "segment_lengths", lengths, lengths.Length);
                Npy.WriteVectors(zip, "shortest_path", pathCorners, squeeze: false);
            }
            if (File.Exists(outPath)) File.Delete(outPath);
            File.Move(tmpPath, outPath);

            float total = 0f;
            foreach (float l in segmentLengths) total += l;
            Debug.Log($"[OccupancyGridExporter] '{scene.name}': {width}x{height} cells at {cell * 100f:F0} cm " +
                $"({occupiedCount} occupied), shortest path {total:F1} m over {segmentLengths.Length} segment(s) -> {outPath}");
            return outPath;
        }

        private static Vector3 SnapToNavMesh(Vector3 position, string label)
        {
            if (!NavMesh.SamplePosition(position, out NavMeshHit hit, MarkerNavMeshSearchRadius, NavMesh.AllAreas))
                throw new InvalidOperationException($"Task marker '{label}' is not above the NavMesh.");
            return hit.position;
        }

        /// <summary>Minimal .npy (format 1.0) writer for the few dtypes the layout file needs; np.load reads the resulting .npz.</summary>
        private static class Npy
        {
            public static void WriteBool2D(ZipArchive zip, string name, bool[] data, int rows, int cols)
            {
                var bytes = new byte[data.Length];
                for (int i = 0; i < data.Length; i++) bytes[i] = data[i] ? (byte)1 : (byte)0;
                Write(zip, name, "|b1", $"({rows}, {cols})", bytes);
            }

            public static void WriteInt32(ZipArchive zip, string name, int value) =>
                Write(zip, name, "<i4", "()", BitConverter.GetBytes(value));

            /// <summary>1-D float64 array of length `length`, or a 0-d scalar when length is omitted.</summary>
            public static void WriteFloat64(ZipArchive zip, string name, double[] values, int length = -1)
            {
                var bytes = new byte[values.Length * 8];
                for (int i = 0; i < values.Length; i++) Buffer.BlockCopy(BitConverter.GetBytes(values[i]), 0, bytes, i * 8, 8);
                Write(zip, name, "<f8", length < 0 ? "()" : $"({length},)", bytes);
            }

            /// <summary>N x 3 float64 (x, y, z) world positions, or a single (3,) vector with squeeze.</summary>
            public static void WriteVectors(ZipArchive zip, string name, List<Vector3> vectors, bool squeeze)
            {
                var values = new double[vectors.Count * 3];
                for (int i = 0; i < vectors.Count; i++)
                {
                    values[3 * i] = vectors[i].x;
                    values[3 * i + 1] = vectors[i].y;
                    values[3 * i + 2] = vectors[i].z;
                }
                var bytes = new byte[values.Length * 8];
                for (int i = 0; i < values.Length; i++) Buffer.BlockCopy(BitConverter.GetBytes(values[i]), 0, bytes, i * 8, 8);
                Write(zip, name, "<f8", squeeze ? "(3,)" : $"({vectors.Count}, 3)", bytes);
            }

            /// <summary>0-d numpy unicode string (UTF-32LE).</summary>
            public static void WriteString(ZipArchive zip, string name, string value)
            {
                int length = Math.Max(1, value.Length);
                var bytes = new byte[length * 4];
                byte[] encoded = new UTF32Encoding(false, false).GetBytes(value);
                Buffer.BlockCopy(encoded, 0, bytes, 0, encoded.Length);
                Write(zip, name, $"<U{length}", "()", bytes);
            }

            private static void Write(ZipArchive zip, string name, string descr, string shape, byte[] data)
            {
                if (!BitConverter.IsLittleEndian) throw new NotSupportedException("The .npy writer assumes a little-endian host.");
                string header = string.Format(CultureInfo.InvariantCulture,
                    "{{'descr': '{0}', 'fortran_order': False, 'shape': {1}, }}", descr, shape);
                // Magic (6) + version (2) + header length (2) + header, padded with spaces and a '\n' to a multiple of 64.
                int unpadded = 10 + header.Length + 1;
                header = header.PadRight(header.Length + (64 - unpadded % 64) % 64) + "\n";

                ZipArchiveEntry entry = zip.CreateEntry(name + ".npy", System.IO.Compression.CompressionLevel.Optimal);
                using (Stream s = entry.Open())
                using (var w = new BinaryWriter(s))
                {
                    w.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0 });
                    w.Write((ushort)header.Length);
                    w.Write(Encoding.ASCII.GetBytes(header));
                    w.Write(data);
                }
            }
        }
    }
}
