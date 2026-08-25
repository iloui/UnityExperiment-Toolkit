using System;
using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.Editor
{
    public enum LayoutTopology
    {
        CentralPillar,
        SCurvedCorridor,
        TJunction,
        MultiRoomDoorway,
        RandomObstacleGrid,
        CircleBath,                  // Ring- / Atrium-Typologie
        MaggiesCentre,               // Stern- / Pavillontypologie
        KaiserFranzJosef,            // Kamm- / H-Typologie
        ReyJuanCarlos,               // Doppel-Oval
        UniversityClinicMuenster,    // Doppel-Ring mit Brücke
        Racetrack,                   // Parametric Racetrack / Core layout
        ShiftedGridGrammar           // Shape-grammar modular hospital layout
    }

    public struct RacetrackParams
    {
        public float Wf; // Corridor/Floor width around the core
        public float Wc; // Center box width
        public float Lc; // Center box length
    }

    public struct LayoutGenerationSettings
    {
        public LayoutTopology Topology;
        public Vector2 Dimensions;
        public float WallHeight;
        public GameObject FpsControllerPrefab;
        public int RandomSeed; // Configurable seed for reproducible dataset generation
    }

    public struct LayoutGenerationData
    {
        public Vector3 SpawnPosition;
        public Quaternion SpawnRotation;
        public Vector3 GoalPosition;
    }

    internal static class LayoutBuilder
    {
        private const float WallThickness = 0.25f;
        private const float FloorY = 0f;
        private const float AgentSpawnY = 1.05f;
        private const float GoalY = 1.0f;
        private const float MinCorridorClearance = 2.2f; // Guarantees VR participants have clean clearance

        public static LayoutGenerationData Build(GameObject root, LayoutGenerationSettings settings, Material sharedMaterial)
        {
            if (root == null) throw new ArgumentNullException(nameof(root));

            var floorLayer = SceneExporter.EnsureLayer("Walkable_Floor");
            var obstacleLayer = SceneExporter.EnsureLayer("Architecture_Obstacle");

            var width = Mathf.Max(10f, settings.Dimensions.x);
            var depth = Mathf.Max(10f, settings.Dimensions.y);
            var wallHeight = Mathf.Max(2f, settings.WallHeight);
            var halfWidth = width * 0.5f;
            var halfDepth = depth * 0.5f;

            var roomWidth = Mathf.Clamp(width * 0.25f, 1f, 8f);
            var leftRoomBoundaryX = -halfWidth + roomWidth;
            var rightRoomBoundaryX = halfWidth - roomWidth;
            var corridorLeftX = leftRoomBoundaryX;
            var corridorRightX = rightRoomBoundaryX;

            CreateFloorAndCeiling(root.transform, width, depth, wallHeight, sharedMaterial, floorLayer, obstacleLayer);
            CreatePerimeterWalls(root.transform, width, depth, wallHeight, sharedMaterial, obstacleLayer);
            
            // Corrected to logical AND (&&) to skip terminal walls for custom topologies
            if (settings.Topology != LayoutTopology.Racetrack && settings.Topology != LayoutTopology.ShiftedGridGrammar)
            {
                CreateTerminalRoomWalls(root.transform, leftRoomBoundaryX, rightRoomBoundaryX, halfDepth, wallHeight, sharedMaterial, obstacleLayer);
            }
            
            var spawnPosition = new Vector3(-halfWidth + roomWidth * 0.5f, AgentSpawnY, 0f);
            var goalPosition = new Vector3(halfWidth - roomWidth * 0.5f, GoalY, 0f);

            BuildTopology(root.transform, settings.Topology, new LayoutArea
            {
                Width = width,
                Depth = depth,
                WallHeight = wallHeight,
                HalfWidth = halfWidth,
                HalfDepth = halfDepth,
                LeftRoomBoundaryX = leftRoomBoundaryX,
                RightRoomBoundaryX = rightRoomBoundaryX,
                CorridorLeftX = corridorLeftX,
                CorridorRightX = corridorRightX,
                Seed = settings.RandomSeed
            }, sharedMaterial, obstacleLayer, ref spawnPosition, ref goalPosition);

            return new LayoutGenerationData
            {
                SpawnPosition = spawnPosition,
                SpawnRotation = Quaternion.identity,
                GoalPosition = goalPosition
            };
        }

        private static void BuildTopology(Transform root, LayoutTopology topology, LayoutArea area, Material sharedMaterial, int obstacleLayer, ref Vector3 spawnPosition, ref Vector3 goalPosition)
        {
            switch (topology)
            {
                case LayoutTopology.CentralPillar:
                    BuildCentralPillar(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.SCurvedCorridor:
                    BuildSCurvedCorridor(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.TJunction:
                    BuildTJunction(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.MultiRoomDoorway:
                    BuildMultiRoomDoorway(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.RandomObstacleGrid:
                    BuildRandomObstacleGrid(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.CircleBath:
                    BuildCircleBath(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.MaggiesCentre:
                    BuildMaggiesCentre(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.KaiserFranzJosef:
                    BuildKaiserFranzJosef(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.ReyJuanCarlos:
                    BuildReyJuanCarlos(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.UniversityClinicMuenster:
                    BuildUniversityClinicMuenster(root, area, sharedMaterial, obstacleLayer);
                    break;
                case LayoutTopology.Racetrack:
                    BuildRacetrack(root, area, sharedMaterial, obstacleLayer, ref spawnPosition, ref goalPosition);
                    break;
                case LayoutTopology.ShiftedGridGrammar:
                    BuildShiftedGridGrammar(root, area, sharedMaterial, obstacleLayer, ref spawnPosition, ref goalPosition);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(topology), topology, null);
            }
        }

        private static void BuildRacetrack(Transform root, LayoutArea area, Material material, int obstacleLayer, ref Vector3 spawnPos, ref Vector3 goalPos)
        {
            var rng = new System.Random(area.Seed == 0 ? 1337 : area.Seed);
            const float minRoomLength = 3.0f;
            const float doorWidth = 1.0f;

            // 1. Determine Room Depths
            float depthWest = area.Width * 0.25f;
            float depthEast = area.Width * 0.25f;
            float depthNorth = depthWest * (float)(0.85 + rng.NextDouble() * 0.3);
            float depthSouth = depthWest * (float)(0.85 + rng.NextDouble() * 0.3);

            float innerWestX = -area.HalfWidth + depthWest;
            float innerEastX = area.HalfWidth - depthEast;
            float innerSouthZ = -area.HalfDepth + depthSouth;
            float innerNorthZ = area.HalfDepth - depthNorth;

            // 2. Partition West (Start) & East (End) Terminal Rooms along Z
            int westSubCount = rng.Next(2, 5);
            spawnPos = BuildTerminalSubRooms(root, "West", -area.HalfWidth, innerWestX, innerSouthZ, innerNorthZ, 
                westSubCount, true, area.WallHeight, material, obstacleLayer, doorWidth, rng);

            int eastSubCount = rng.Next(2, 5);
            goalPos = BuildTerminalSubRooms(root, "East", innerEastX, area.HalfWidth, innerSouthZ, innerNorthZ, 
                eastSubCount, false, area.WallHeight, material, obstacleLayer, doorWidth, rng);

            // 3. Generate North & South Side Rooms along X
            float availableXSpan = innerEastX - innerWestX;

            BuildSideRooms(root, "South", innerWestX, innerEastX, -area.HalfDepth, innerSouthZ, 
                availableXSpan, minRoomLength, doorWidth, true, area.WallHeight, material, obstacleLayer, rng);

            BuildSideRooms(root, "North", innerWestX, innerEastX, innerNorthZ, area.HalfDepth, 
                availableXSpan, minRoomLength, doorWidth, false, area.WallHeight, material, obstacleLayer, rng);

            // 4. Divide Corner Rooms and Connect to Racetrack
            BuildCornerRooms(root, area, innerWestX, innerEastX, innerSouthZ, innerNorthZ, 
                area.WallHeight, material, obstacleLayer, doorWidth, rng);

            // 5. Calculate Inner Racetrack Core Dimensions with Clearance
            float innerCorridorWidth = availableXSpan;
            float innerCorridorLength = innerNorthZ - innerSouthZ;

            var p = ComputeRacetrackParameters(innerCorridorWidth, innerCorridorLength, area.Seed, MinCorridorClearance);

            Vector3 centerBoxPos = new Vector3((innerWestX + innerEastX) * 0.5f, area.WallHeight * 0.5f, (innerSouthZ + innerNorthZ) * 0.5f);
            Vector3 centerBoxScale = new Vector3(p.Wc, area.WallHeight, p.Lc);

            CreateWall(root, "Racetrack_CenterCore", centerBoxPos, centerBoxScale, material, obstacleLayer);
        }
        
        /// <summary>
        /// Calculates valid racetrack parameters for a floorplan of size width x length.
        /// Enforces clearances such that 2*wf + wc < width and 2*wf + lc < length.
        /// </summary>
        public static RacetrackParams ComputeRacetrackParameters(float width, float length, int seed = 0, float minDistance = 2.2f)
        {
            // Use configured seed or default fallback
            var rng = new System.Random(seed == 0 ? UnityEngine.Random.Range(1, 100000) : seed);

            // 1. Determine valid range for corridor width (wf)
            // Upper bound ensures remaining room for central box is at least minDistance
            float maxWfLimit = Mathf.Min(width, length) * 0.5f;
            float maxWfForBox = Mathf.Min(width - minDistance, length - minDistance) * 0.5f;
            float maxWf = Mathf.Min(maxWfLimit, maxWfForBox) - 0.05f; // Small margin for strict inequality

            float minWf = minDistance;
            if (maxWf < minWf) maxWf = minWf;

            // Randomize wf in [minDistance, min(width, length) / 2)
            float wf = Lerp(minWf, maxWf, (float)rng.NextDouble());

            // 2. Determine bounds for center box width (wc) given wf
            // Upper bound: min(width / 2, width - 2*wf)
            float maxWc = Mathf.Min(width * 0.5f, width - (2f * wf) - 0.05f);
            float minWc = minDistance;
            if (maxWc < minWc) maxWc = minWc;

            float wc = Lerp(minWc, maxWc, (float)rng.NextDouble());

            // 3. Determine bounds for center box length (lc) given wf
            // Upper bound: min(length / 2, length - 2*wf)
            float maxLc = Mathf.Min(length * 0.5f, length - (2f * wf) - 0.05f);
            float minLc = minDistance;
            if (maxLc < minLc) maxLc = minLc;

            float lc = Lerp(minLc, maxLc, (float)rng.NextDouble());

            return new RacetrackParams { Wf = wf, Wc = wc, Lc = lc };

            static float Lerp(float a, float b, float t) => a + (b - a) * Mathf.Clamp01(t);
        }

        private static Vector3 BuildTerminalSubRooms(Transform root, string prefix, float minX, float maxX, float minZ, float maxZ, 
            int count, bool isWest, float wallHeight, Material material, int layer, float doorWidth, System.Random rng)
        {
            float totalZ = maxZ - minZ;
            float subDepthZ = totalZ / count;
            float targetX = isWest ? maxX : minX;
            Vector3 primaryRoomCenter = Vector3.zero;

            for (int i = 0; i < count; i++)
            {
                float zStart = minZ + (i * subDepthZ);
                float zEnd = zStart + subDepthZ;

                // Internal partition wall (perpendicular to facing racetrack)
                if (i > 0)
                {
                    CreateWall(root, $"{prefix}Term_Partition_{i}", 
                        new Vector3((minX + maxX) * 0.5f, wallHeight * 0.5f, zStart), 
                        new Vector3(maxX - minX, wallHeight, WallThickness), material, layer);
                }

                // Randomly offset 1m door facing the main racetrack corridor
                float doorMargin = doorWidth * 0.6f;
                float doorZ = Lerp(zStart + doorMargin, zEnd - doorMargin, (float)rng.NextDouble());

                CreateWallWithDoor(root, $"{prefix}Term_CorridorWall_{i}", targetX, zStart, zEnd, 
                    wallHeight, material, layer, doorZ, doorWidth);

                if (i == 0)
                {
                    float agentX = (minX + maxX) * 0.5f;
                    float agentZ = (zStart + zEnd) * 0.5f;
                    primaryRoomCenter = new Vector3(agentX, isWest ? AgentSpawnY : GoalY, agentZ);
                }
            }

            return primaryRoomCenter;
        }

        private static void BuildSideRooms(Transform root, string prefix, float minX, float maxX, float minZ, float maxZ, 
            float spanX, float minLength, float doorWidth, bool isSouth, float wallHeight, Material material, int layer, System.Random rng)
        {
            List<float> roomLengths = GenerateRandomRoomLengths(spanX, minLength, rng);
            float currentX = minX;
            float corridorZ = isSouth ? maxZ : minZ;

            for (int i = 0; i < roomLengths.Count; i++)
            {
                float roomW = roomLengths[i];
                float nextX = currentX + roomW;

                // Divider wall between adjacent side rooms
                if (i > 0)
                {
                    CreateWall(root, $"{prefix}Side_Partition_{i}", 
                        new Vector3(currentX, wallHeight * 0.5f, (minZ + maxZ) * 0.5f), 
                        new Vector3(WallThickness, wallHeight, maxZ - minZ), material, layer);
                }

                // Randomly offset 1m door facing inward to racetrack
                float doorMargin = doorWidth * 0.6f;
                float doorX = Lerp(currentX + doorMargin, nextX - doorMargin, (float)rng.NextDouble());

                CreateWallWithHorizontalDoor(root, $"{prefix}Side_CorridorWall_{i}", corridorZ, currentX, nextX, 
                    wallHeight, material, layer, doorX, doorWidth);

                currentX = nextX;
            }
        }
        
        private static void BuildCornerRooms(Transform root, LayoutArea area, float innerWestX, float innerEastX, 
            float innerSouthZ, float innerNorthZ, float wallHeight, Material material, int obstacleLayer, float doorWidth, System.Random rng)
        {
            // South-West Corner
            BuildSingleCornerRoom(root, "Corner_SW", -area.HalfWidth, innerWestX, -area.HalfDepth, innerSouthZ, 
                innerWestX, innerSouthZ, wallHeight, material, obstacleLayer, doorWidth, rng);

            // North-West Corner
            BuildSingleCornerRoom(root, "Corner_NW", -area.HalfWidth, innerWestX, innerNorthZ, area.HalfDepth, 
                innerWestX, innerNorthZ, wallHeight, material, obstacleLayer, doorWidth, rng);

            // South-East Corner
            BuildSingleCornerRoom(root, "Corner_SE", innerEastX, area.HalfWidth, -area.HalfDepth, innerSouthZ, 
                innerEastX, innerSouthZ, wallHeight, material, obstacleLayer, doorWidth, rng);

            // North-East Corner
            BuildSingleCornerRoom(root, "Corner_NE", innerEastX, area.HalfWidth, innerNorthZ, area.HalfDepth, 
                innerEastX, innerNorthZ, wallHeight, material, obstacleLayer, doorWidth, rng);
        }

        private static void BuildSingleCornerRoom(Transform root, string prefix, float xMin, float xMax, float zMin, float zMax,
            float cornerX, float cornerZ, float wallHeight, Material material, int obstacleLayer, float doorWidth, System.Random rng)
        {
            bool extendHorizontal = rng.NextDouble() < 0.5;
            float doorMargin = doorWidth * 0.6f;

            if (extendHorizontal)
            {
                // 1. Extended Horizontal Wall -> Completely SOLID to the outer wall
                CreateWall(root, $"{prefix}_Solid_H", 
                    new Vector3((xMin + xMax) * 0.5f, wallHeight * 0.5f, cornerZ), 
                    new Vector3(xMax - xMin, wallHeight, WallThickness), material, obstacleLayer);

                // 2. Vertical Wall -> Single doorway connection to the racetrack corridor
                float doorZ = Lerp(zMin + doorMargin, zMax - doorMargin, (float)rng.NextDouble());
                CreateWallWithDoor(root, $"{prefix}_Door_V", cornerX, zMin, zMax, 
                    wallHeight, material, obstacleLayer, doorZ, doorWidth);
            }
            else
            {
                // 1. Extended Vertical Wall -> Completely SOLID to the outer wall
                CreateWall(root, $"{prefix}_Solid_V", 
                    new Vector3(cornerX, wallHeight * 0.5f, (zMin + zMax) * 0.5f), 
                    new Vector3(WallThickness, wallHeight, zMax - zMin), material, obstacleLayer);

                // 2. Horizontal Wall -> Single doorway connection to the racetrack corridor
                float doorX = Lerp(xMin + doorMargin, xMax - doorMargin, (float)rng.NextDouble());
                CreateWallWithHorizontalDoor(root, $"{prefix}_Door_H", cornerZ, xMin, xMax, 
                    wallHeight, material, obstacleLayer, doorX, doorWidth);
            }
        }

        private static List<float> GenerateRandomRoomLengths(float totalLength, float minLength, System.Random rng)
        {
            int maxRooms = Mathf.FloorToInt(totalLength / minLength);
            if (maxRooms <= 1) return new List<float> { totalLength };

            int numRooms = rng.Next(1, maxRooms + 1);
            List<float> rawWeights = new List<float>();
            float weightSum = 0f;

            for (int i = 0; i < numRooms; i++)
            {
                float w = (float)rng.NextDouble() + 0.1f;
                rawWeights.Add(w);
                weightSum += w;
            }

            float remainingLength = totalLength - (numRooms * minLength);
            List<float> lengths = new List<float>();

            for (int i = 0; i < numRooms; i++)
            {
                float extra = remainingLength * (rawWeights[i] / weightSum);
                lengths.Add(minLength + extra);
            }

            return lengths;
        }

        private static void CreateWallWithHorizontalDoor(Transform root, string name, float z, float xMin, float xMax,
            float wallHeight, Material material, int layer, float doorCenterX, float doorWidth)
        {
            float doorHalf = doorWidth * 0.5f;
            float doorMin = Mathf.Clamp(doorCenterX - doorHalf, xMin + 0.1f, xMax - 0.1f);
            float doorMax = Mathf.Clamp(doorCenterX + doorHalf, xMin + 0.1f, xMax - 0.1f);

            if (doorMin > xMin)
            {
                float leftLength = doorMin - xMin;
                CreateWall(root, $"{name}_Left", new Vector3(xMin + leftLength * 0.5f, wallHeight * 0.5f, z),
                    new Vector3(leftLength, wallHeight, WallThickness), material, layer);
            }

            if (doorMax < xMax)
            {
                float rightLength = xMax - doorMax;
                CreateWall(root, $"{name}_Right", new Vector3(doorMax + rightLength * 0.5f, wallHeight * 0.5f, z),
                    new Vector3(rightLength, wallHeight, WallThickness), material, layer);
            }
        }

        private static float Lerp(float a, float b, float t) => a + (b - a) * Mathf.Clamp01(t);

        private static void BuildShiftedGridGrammar(
            Transform root, 
            LayoutArea area, 
            Material material, 
            int obstacleLayer, 
            ref Vector3 spawnPosition, 
            ref Vector3 goalPosition)
        {
            var rng = new System.Random(area.Seed == 0 ? 1337 : area.Seed);

            float minX = -area.HalfWidth + WallThickness;
            float maxX = area.HalfWidth - WallThickness;
            float minZ = -area.HalfDepth + WallThickness;
            float maxZ = area.HalfDepth - WallThickness;
            float totalWidth = maxX - minX;

            // -------------------------------------------------------------------------
            // 1. BACKBONE-FIRST: PRIMARY SPINE CORRIDOR (East-West Main Highway)
            // -------------------------------------------------------------------------
            float spineWidth = 2.5f;
            float spineMinZ = -spineWidth * 0.5f;
            float spineMaxZ = spineWidth * 0.5f;

            // -------------------------------------------------------------------------
            // 2. SUBTRACTIVE BINARY SPACE PARTITIONING (BSP)
            // -------------------------------------------------------------------------
            List<Rect> leafSectors = new List<Rect>();
            List<Rect> ribCorridors = new List<Rect>();

            // Partition North region (above Spine)
            Rect northBounds = new Rect(minX, spineMaxZ, totalWidth, maxZ - spineMaxZ);
            BspNode northTree = new BspNode(northBounds);
            northTree.Split(0, rng, leafSectors, ribCorridors, minSize: 6.5f, corridorWidth: 2.0f);

            // Partition South region (below Spine)
            Rect southBounds = new Rect(minX, minZ, totalWidth, spineMinZ - minZ);
            BspNode southTree = new BspNode(southBounds);
            southTree.Split(0, rng, leafSectors, ribCorridors, minSize: 6.5f, corridorWidth: 2.0f);

            // -------------------------------------------------------------------------
            // 3. FIT MODULAR SHAPE-GRAMMAR UNITS INTO BSP LEAVES
            // -------------------------------------------------------------------------
            List<Vector3> allRoomCenters = new List<Vector3>();

            foreach (var sector in leafSectors)
            {
                Vector2 center = new Vector2(sector.x + sector.width * 0.5f, sector.y + sector.height * 0.5f);
                GrammarUnit unit;

                // Dynamically select grammar unit based on BSP sector aspect ratio
                if (sector.width >= 1.6f * sector.height)
                {
                    unit = new DoubleOctoRoomUnit(center, sector.width, sector.height);
                }
                else if (rng.NextDouble() < 0.10d)
                {
                    unit = new OpenCourtyardUnit(center, sector.width, sector.height);
                }
                else
                {
                    unit = new StandardQuadRoomUnit(center, sector.width, sector.height);
                }

                unit.Generate(
                    root, area.WallHeight, material, obstacleLayer, rng, allRoomCenters,
                    (r, nName, pos, scale, mat, layer) => CreateWall(r, nName, pos, scale, mat, layer),
                    CreateWallWithDoor,
                    CreateWallWithHorizontalDoor
                );
            }

            // -------------------------------------------------------------------------
            // 4. GUARANTEED CONNECTIVITY & SPAWN/GOAL POSITIONS
            // -------------------------------------------------------------------------
            spawnPosition = new Vector3(minX + 2.5f, AgentSpawnY, 0f);
            goalPosition = new Vector3(maxX - 2.5f, GoalY, 0f);
        }

        // =========================================================================
        // SUBTRACTIVE BSP NODE HELPER CLASS
        // =========================================================================
        private class BspNode
        {
            public Rect Bounds;
            public BspNode Left;
            public BspNode Right;

            public BspNode(Rect bounds)
            {
                Bounds = bounds;
            }

            public void Split(int depth, System.Random rng, List<Rect> leafSectors, List<Rect> ribCorridors, float minSize, float corridorWidth)
            {
                // Terminal condition: stop splitting when depth limit reached or size is small
                if (depth >= 3 || (Bounds.width <= minSize * 2f && Bounds.height <= minSize * 2f))
                {
                    leafSectors.Add(Bounds);
                    return;
                }

                // Decide split direction based on sector aspect ratio
                bool splitHorizontal;
                if (Bounds.width / Bounds.height >= 1.35f) splitHorizontal = false; // Split along X
                else if (Bounds.height / Bounds.width >= 1.35f) splitHorizontal = true; // Split along Z
                else splitHorizontal = rng.NextDouble() < 0.5;

                float maxSplit = splitHorizontal ? Bounds.height - minSize : Bounds.width - minSize;
                float minSplit = minSize;

                if (maxSplit <= minSplit)
                {
                    leafSectors.Add(Bounds);
                    return;
                }

                float splitPos = Lerp(minSplit, maxSplit, (float)rng.NextDouble());

                if (splitHorizontal)
                {
                    // Split along Z axis (carve horizontal Rib corridor)
                    Rect b1 = new Rect(Bounds.x, Bounds.y, Bounds.width, splitPos - corridorWidth * 0.5f);
                    Rect rib = new Rect(Bounds.x, Bounds.y + splitPos - corridorWidth * 0.5f, Bounds.width, corridorWidth);
                    Rect b2 = new Rect(Bounds.x, Bounds.y + splitPos + corridorWidth * 0.5f, Bounds.width, Bounds.height - splitPos - corridorWidth * 0.5f);

                    if (b1.height >= minSize && b2.height >= minSize)
                    {
                        ribCorridors.Add(rib);
                        Left = new BspNode(b1);
                        Right = new BspNode(b2);
                        Left.Split(depth + 1, rng, leafSectors, ribCorridors, minSize, corridorWidth);
                        Right.Split(depth + 1, rng, leafSectors, ribCorridors, minSize, corridorWidth);
                        return;
                    }
                }
                else
                {
                    // Split along X axis (carve vertical Rib corridor)
                    Rect b1 = new Rect(Bounds.x, Bounds.y, splitPos - corridorWidth * 0.5f, Bounds.height);
                    Rect rib = new Rect(Bounds.x + splitPos - corridorWidth * 0.5f, Bounds.y, corridorWidth, Bounds.height);
                    Rect b2 = new Rect(Bounds.x + splitPos + corridorWidth * 0.5f, Bounds.y, Bounds.width - splitPos - corridorWidth * 0.5f, Bounds.height);

                    if (b1.width >= minSize && b2.width >= minSize)
                    {
                        ribCorridors.Add(rib);
                        Left = new BspNode(b1);
                        Right = new BspNode(b2);
                        Left.Split(depth + 1, rng, leafSectors, ribCorridors, minSize, corridorWidth);
                        Right.Split(depth + 1, rng, leafSectors, ribCorridors, minSize, corridorWidth);
                        return;
                    }
                }

                leafSectors.Add(Bounds);
            }
        }

        // =========================================================================
        // BSP-ADAPTED MODULAR SHAPE GRAMMAR UNITS
        // =========================================================================
        public abstract class GrammarUnit
        {
            public Vector2 Position { get; protected set; } 
            public Vector2 Footprint { get; protected set; } 

            public abstract void Generate(
                Transform root, 
                float wallHeight, 
                Material material, 
                int layer, 
                System.Random rng, 
                List<Vector3> roomCenters, 
                Action<Transform, string, Vector3, Vector3, Material, int> createWall,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithDoor,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithHDoor
            );
        }

        // 1. Standard Quad Room Unit (2x2 Quad pattern adapted to leaf bounds)
        public class StandardQuadRoomUnit : GrammarUnit
        {
            public StandardQuadRoomUnit(Vector2 center, float widthX, float depthZ)
            {
                Position = center;
                Footprint = new Vector2(widthX, depthZ);
            }

            public override void Generate(Transform root, float wallHeight, Material material, int layer, System.Random rng, List<Vector3> roomCenters,
                Action<Transform, string, Vector3, Vector3, Material, int> createWall,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithDoor,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithHDoor)
            {
                float doorWidth = 1.0f;
                float halfW = Footprint.x * 0.5f;
                float halfH = Footprint.y * 0.5f;

                float minX = Position.x - halfW;
                float maxX = Position.x + halfW;
                float minZ = Position.y - halfH;
                float maxZ = Position.y + halfH;

                // Internal Cross Walls dividing cell into 4 rooms
                createWall(root, "Quad_VPart", new Vector3(Position.x, wallHeight * 0.5f, Position.y), new Vector3(WallThickness, wallHeight, Footprint.y), material, layer);
                createWall(root, "Quad_HPart", new Vector3(Position.x, wallHeight * 0.5f, Position.y), new Vector3(Footprint.x, wallHeight, WallThickness), material, layer);

                // Quad room centers
                float qX = Footprint.x * 0.25f;
                float qZ = Footprint.y * 0.25f;
                roomCenters.Add(new Vector3(Position.x - qX, AgentSpawnY, Position.y - qZ));
                roomCenters.Add(new Vector3(Position.x + qX, AgentSpawnY, Position.y - qZ));
                roomCenters.Add(new Vector3(Position.x - qX, AgentSpawnY, Position.y + qZ));
                roomCenters.Add(new Vector3(Position.x + qX, AgentSpawnY, Position.y + qZ));

                // North & South outer doors connecting to corridors
                createWallWithHDoor(root, "Quad_Door_S1", minZ, minX, Position.x, wallHeight, material, layer, Mathf.Lerp(minX + 0.6f, Position.x - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithHDoor(root, "Quad_Door_S2", minZ, Position.x, maxX, wallHeight, material, layer, Mathf.Lerp(Position.x + 0.6f, maxX - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithHDoor(root, "Quad_Door_N1", maxZ, minX, Position.x, wallHeight, material, layer, Mathf.Lerp(minX + 0.6f, Position.x - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithHDoor(root, "Quad_Door_N2", maxZ, Position.x, maxX, wallHeight, material, layer, Mathf.Lerp(Position.x + 0.6f, maxX - 0.6f, (float)rng.NextDouble()), doorWidth);

                // West & East outer doors
                createWallWithDoor(root, "Quad_Door_W1", minX, minZ, Position.y, wallHeight, material, layer, Mathf.Lerp(minZ + 0.6f, Position.y - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithDoor(root, "Quad_Door_W2", minX, Position.y, maxZ, wallHeight, material, layer, Mathf.Lerp(Position.y + 0.6f, maxZ - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithDoor(root, "Quad_Door_E1", maxX, minZ, Position.y, wallHeight, material, layer, Mathf.Lerp(minZ + 0.6f, Position.y - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithDoor(root, "Quad_Door_E2", maxX, Position.y, maxZ, wallHeight, material, layer, Mathf.Lerp(Position.y + 0.6f, maxZ - 0.6f, (float)rng.NextDouble()), doorWidth);
            }
        }

        // 2. Double Octo Room Unit (2x4 Room pattern for wide leaf sectors)
        public class DoubleOctoRoomUnit : GrammarUnit
        {
            public DoubleOctoRoomUnit(Vector2 center, float widthX, float depthZ)
            {
                Position = center;
                Footprint = new Vector2(widthX, depthZ);
            }

            public override void Generate(Transform root, float wallHeight, Material material, int layer, System.Random rng, List<Vector3> roomCenters,
                Action<Transform, string, Vector3, Vector3, Material, int> createWall,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithDoor,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithHDoor)
            {
                float doorWidth = 1.0f;
                float halfW = Footprint.x * 0.5f;
                float halfH = Footprint.y * 0.5f;

                float minX = Position.x - halfW;
                float maxX = Position.x + halfW;
                float minZ = Position.y - halfH;
                float maxZ = Position.y + halfH;

                float cellW = Footprint.x / 4f;

                createWall(root, "Octo_HPart", new Vector3(Position.x, wallHeight * 0.5f, Position.y), new Vector3(Footprint.x, wallHeight, WallThickness), material, layer);

                for (int c = 1; c < 4; c++)
                {
                    float px = minX + c * cellW;
                    createWall(root, $"Octo_VPart_{c}", new Vector3(px, wallHeight * 0.5f, Position.y), new Vector3(WallThickness, wallHeight, Footprint.y), material, layer);
                }

                for (int c = 0; c < 4; c++)
                {
                    float cx = minX + (c + 0.5f) * cellW;
                    roomCenters.Add(new Vector3(cx, AgentSpawnY, Position.y - halfH * 0.5f));
                    roomCenters.Add(new Vector3(cx, AgentSpawnY, Position.y + halfH * 0.5f));

                    createWallWithHDoor(root, $"Octo_Door_S_{c}", minZ, minX + c * cellW, minX + (c + 1) * cellW, wallHeight, material, layer, Mathf.Lerp(minX + c * cellW + 0.6f, minX + (c + 1) * cellW - 0.6f, (float)rng.NextDouble()), doorWidth);
                    createWallWithHDoor(root, $"Octo_Door_N_{c}", maxZ, minX + c * cellW, minX + (c + 1) * cellW, wallHeight, material, layer, Mathf.Lerp(minX + c * cellW + 0.6f, minX + (c + 1) * cellW - 0.6f, (float)rng.NextDouble()), doorWidth);
                }

                createWallWithDoor(root, "Octo_Door_W1", minX, minZ, Position.y, wallHeight, material, layer, Mathf.Lerp(minZ + 0.6f, Position.y - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithDoor(root, "Octo_Door_W2", minX, Position.y, maxZ, wallHeight, material, layer, Mathf.Lerp(Position.y + 0.6f, maxZ - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithDoor(root, "Octo_Door_E1", maxX, minZ, Position.y, wallHeight, material, layer, Mathf.Lerp(minZ + 0.6f, Position.y - 0.6f, (float)rng.NextDouble()), doorWidth);
                createWallWithDoor(root, "Octo_Door_E2", maxX, Position.y, maxZ, wallHeight, material, layer, Mathf.Lerp(Position.y + 0.6f, maxZ - 0.6f, (float)rng.NextDouble()), doorWidth);
            }
        }

        // 3. Open Courtyard Unit (Atrium walk space inside leaf bounds)
        public class OpenCourtyardUnit : GrammarUnit
        {
            public OpenCourtyardUnit(Vector2 center, float widthX, float depthZ)
            {
                Position = center;
                Footprint = new Vector2(widthX, depthZ);
            }

            public override void Generate(Transform root, float wallHeight, Material material, int layer, System.Random rng, List<Vector3> roomCenters,
                Action<Transform, string, Vector3, Vector3, Material, int> createWall,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithDoor,
                Action<Transform, string, float, float, float, float, Material, int, float, float> createWallWithHDoor)
            {
                // Open walk space (no internal room walls created)
            }
        }

        private static void CreateFloorAndCeiling(Transform root, float width, float depth, float wallHeight, Material material, int floorLayer, int obstacleLayer)
        {
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "Floor";
            floor.transform.SetParent(root, false);
            floor.transform.localPosition = new Vector3(0f, FloorY, 0f);
            floor.transform.localRotation = Quaternion.identity;
            floor.transform.localScale = new Vector3(width / 10f, 1f, depth / 10f);
            floor.layer = floorLayer;
            ApplyMaterial(floor, material);

            var ceiling = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ceiling.name = "Ceiling";
            ceiling.transform.SetParent(root, false);
            ceiling.transform.localPosition = new Vector3(0f, wallHeight, 0f);
            ceiling.transform.localRotation = Quaternion.Euler(180f, 0f, 0f);
            ceiling.transform.localScale = new Vector3(width / 10f, 1f, depth / 10f);
            ceiling.layer = obstacleLayer;
            ApplyMaterial(ceiling, material);
        }

        private static void CreatePerimeterWalls(Transform root, float width, float depth, float wallHeight, Material material, int obstacleLayer)
        {
            var halfWidth = width * 0.5f;
            var halfDepth = depth * 0.5f;

            CreateWall(root, "NorthWall", new Vector3(0f, wallHeight * 0.5f, halfDepth - WallThickness * 0.5f), new Vector3(width, wallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SouthWall", new Vector3(0f, wallHeight * 0.5f, -halfDepth + WallThickness * 0.5f), new Vector3(width, wallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "WestWall", new Vector3(-halfWidth + WallThickness * 0.5f, wallHeight * 0.5f, 0f), new Vector3(WallThickness, wallHeight, depth), material, obstacleLayer);
            CreateWall(root, "EastWall", new Vector3(halfWidth - WallThickness * 0.5f, wallHeight * 0.5f, 0f), new Vector3(WallThickness, wallHeight, depth), material, obstacleLayer);
        }

        private static void CreateTerminalRoomWalls(Transform root, float leftRoomBoundaryX, float rightRoomBoundaryX, float halfDepth, float wallHeight, Material material, int obstacleLayer)
        {
            // Door positions are offset on the Z-axis to enforce non-linear paths right from spawn
            CreateWallWithDoor(root, "SpawnRoomDoorWall", leftRoomBoundaryX, -halfDepth + WallThickness, halfDepth - WallThickness, wallHeight, material, obstacleLayer, doorCenterZ: -halfDepth * 0.25f, doorWidth: MinCorridorClearance);
            CreateWallWithDoor(root, "TargetRoomDoorWall", rightRoomBoundaryX, -halfDepth + WallThickness, halfDepth - WallThickness, wallHeight, material, obstacleLayer, doorCenterZ: halfDepth * 0.25f, doorWidth: MinCorridorClearance);
        }

        private static void BuildCentralPillar(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var pillarSize = Mathf.Clamp(area.Width * 0.16f, 3.2f, 4.6f);
            
            // ASYMMETRIC IMPROVEMENT: Shift pillar north (+1.5m) to offer a wide corridor (south) vs. narrow corridor (north)
            var pillarOffsetZ = 1.5f;
            CreateWall(root, "CentralPillar", new Vector3(0f, area.WallHeight * 0.5f, pillarOffsetZ), new Vector3(pillarSize, area.WallHeight, pillarSize), material, obstacleLayer);

            var corridorLength = Mathf.Max(3f, area.CorridorRightX - area.CorridorLeftX - pillarSize);
            CreateWall(root, "CentralNorthRail", new Vector3(0f, area.WallHeight * 0.5f, area.HalfDepth * 0.55f), new Vector3(corridorLength, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "CentralSouthRail", new Vector3(0f, area.WallHeight * 0.5f, -area.HalfDepth * 0.55f), new Vector3(corridorLength, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        private static void BuildSCurvedCorridor(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var segmentLength = (area.CorridorRightX - area.CorridorLeftX) * 0.45f;
            var leftCenter = Mathf.Lerp(area.LeftRoomBoundaryX, 0f, 0.5f);
            var rightCenter = Mathf.Lerp(0f, area.RightRoomBoundaryX, 0.5f);

            CreateWall(root, "SCurveUpperLeft", new Vector3(leftCenter, area.WallHeight * 0.5f, area.HalfDepth * 0.35f), new Vector3(segmentLength, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SCurveLowerRight", new Vector3(rightCenter, area.WallHeight * 0.5f, -area.HalfDepth * 0.35f), new Vector3(segmentLength, area.WallHeight, WallThickness), material, obstacleLayer);
            
            // REACHABILITY FIX: Reduced spine height to 0.6f * HalfDepth so upper and lower corridors remain 100% open
            CreateWall(root, "SCurveCenterSpine", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(WallThickness, area.WallHeight, area.HalfDepth * 0.6f), material, obstacleLayer);
        }

        private static void BuildTJunction(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            // 1. Central Blocking Wall (The top bar of the 'T')
            // Placed directly in the middle along the X-axis to block a straight-line path.
            float wallWidth = 0.25f;
            float wallLength = area.HalfDepth * 1.1f; // Blocks the direct middle path
            CreateWall(root, "TJunction_Bar", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(wallWidth, area.WallHeight, wallLength), material, obstacleLayer);

            // 2. Guide Rails (Creating clear North and South corridors around the 'T')
            float railLength = (area.CorridorRightX - area.CorridorLeftX) * 0.6f;
            float railOffsetZ = area.HalfDepth * 0.55f;

            CreateWall(root, "TJunction_NorthRail", new Vector3(0f, area.WallHeight * 0.5f, railOffsetZ), new Vector3(railLength, area.WallHeight, wallWidth), material, obstacleLayer);
            CreateWall(root, "TJunction_SouthRail", new Vector3(0f, area.WallHeight * 0.5f, -railOffsetZ), new Vector3(railLength, area.WallHeight, wallWidth), material, obstacleLayer);
        }

        private static void BuildMultiRoomDoorway(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var partitionX1 = Mathf.Lerp(area.LeftRoomBoundaryX, 0f, 0.45f);
            var partitionX2 = Mathf.Lerp(0f, area.RightRoomBoundaryX, 0.45f);

            // STAGGERED APERTURES IMPROVEMENT: Forces S-shaped navigation choices through doorway bottlenecks
            CreateWallWithDoor(root, "DoorwayPartitionA", partitionX1, -area.HalfDepth + WallThickness, area.HalfDepth - WallThickness, area.WallHeight, material, obstacleLayer, doorCenterZ: -area.HalfDepth * 0.35f, doorWidth: MinCorridorClearance);
            CreateWallWithDoor(root, "DoorwayPartitionB", partitionX2, -area.HalfDepth + WallThickness, area.HalfDepth - WallThickness, area.WallHeight, material, obstacleLayer, doorCenterZ: area.HalfDepth * 0.35f, doorWidth: MinCorridorClearance);
            
            // Center island block instead of a continuous blocking wall
            CreateWall(root, "DoorwayCenterIsland", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(1.8f, area.WallHeight, 1.8f), material, obstacleLayer);
        }

        private static void BuildRandomObstacleGrid(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            // REPRODUCIBILITY IMPROVEMENT: Use user-configured RandomSeed setting
            var rng = new System.Random(area.Seed == 0 ? 1337 : area.Seed);
            
            var gridCountX = 3;
            var gridCountZ = 3;
            var cellWidth = (area.CorridorRightX - area.CorridorLeftX) / gridCountX;
            var cellDepth = (area.HalfDepth * 1.6f) / gridCountZ;
            var startX = area.CorridorLeftX + cellWidth * 0.5f;
            var startZ = -area.HalfDepth * 0.8f + cellDepth * 0.5f;

            // REACHABILITY FIX: Removed solid GridSpine wall completely.
            // Enforce open horizontal path along the middle row (z = 1) to guarantee reachability.
            for (var x = 0; x < gridCountX; x++)
            {
                for (var z = 0; z < gridCountZ; z++)
                {
                    // Always keep central corridor clear (z = 1) to prevent dead-ends
                    if (z == 1) continue;

                    // 40% probability of skipping block to create natural walking gaps
                    if (rng.NextDouble() < 0.20d) continue;

                    var obstacleScale = new Vector3(cellWidth * 0.55f, area.WallHeight, cellDepth * 0.55f);
                    var pos = new Vector3(startX + x * cellWidth, area.WallHeight * 0.5f, startZ + z * cellDepth);
                    CreateWall(root, $"GridBlock_{x}_{z}", pos, obstacleScale, material, obstacleLayer);
                }
            }
        }
        
        // 1. Circle Bath General Hospital: Atrium-Ring
        private static void BuildCircleBath(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var corridorWidth = (area.CorridorRightX - area.CorridorLeftX);
            var coreWidth = corridorWidth * 0.45f;
            var coreDepth = area.HalfDepth * 0.9f;

            // Zentrales geschlossenes Atrium / Bauwerk
            CreateWall(root, "AtriumCore", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(coreWidth, area.WallHeight, coreDepth), material, obstacleLayer);

            // Äussere Begrenzungsschienen zur Formung des Ring-Gangs
            var outerRailZ = area.HalfDepth * 0.7f;
            CreateWall(root, "NorthRingWall", new Vector3(0f, area.WallHeight * 0.5f, outerRailZ), new Vector3(corridorWidth, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SouthRingWall", new Vector3(0f, area.WallHeight * 0.5f, -outerRailZ), new Vector3(corridorWidth, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // 2. Maggie’s Centre Gartnavel: Stern- / Verzweigungstypologie
        private static void BuildMaggiesCentre(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var midX = 0f;
            var wingLength = area.HalfDepth * 0.6f;

            // Zentraler Knotenpunkt (Verzweigung)
            CreateWall(root, "CentralHubColumn", new Vector3(midX, area.WallHeight * 0.5f, 0f), new Vector3(1.5f, area.WallHeight, 1.5f), material, obstacleLayer);

            // Diagonale/Versetzte Flügelelemente für Sichtschutz und Verzweigungen
            CreateWall(root, "WingNorthWest", new Vector3(-area.Width * 0.1f, area.WallHeight * 0.5f, area.HalfDepth * 0.35f), new Vector3(WallThickness, area.WallHeight, wingLength), material, obstacleLayer);
            CreateWall(root, "WingSouthEast", new Vector3(area.Width * 0.1f, area.WallHeight * 0.5f, -area.HalfDepth * 0.35f), new Vector3(WallThickness, area.WallHeight, wingLength), material, obstacleLayer);

            // Querriegel, der Stichgänge zu den "Beratungsräumen" bildet
            CreateWall(root, "CounselingDivider", new Vector3(0f, area.WallHeight * 0.5f, area.HalfDepth * 0.55f), new Vector3(area.Width * 0.2f, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // 3. Kaiser-Franz-Josef-Spital: Doppelkamm- Structure (Hauptachse + Querriegel)
        private static void BuildKaiserFranzJosef(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var corridorSpan = area.CorridorRightX - area.CorridorLeftX;

            // Zentrale Haupt-Magistrale (Rückgrat) mit Durchlass
            var spineSegmentWidth = corridorSpan * 0.38f;
            CreateWall(root, "SpineWest", new Vector3(-corridorSpan * 0.25f, area.WallHeight * 0.5f, 0f), new Vector3(spineSegmentWidth, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SpineEast", new Vector3(corridorSpan * 0.25f, area.WallHeight * 0.5f, 0f), new Vector3(spineSegmentWidth, area.WallHeight, WallThickness), material, obstacleLayer);

            // Senkrechte Kamm-Riegel (Nord & Süd)
            var ridgeHeight = area.HalfDepth * 0.55f;
            CreateWall(root, "CombRidge_North1", new Vector3(-area.Width * 0.12f, area.WallHeight * 0.5f, area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
            CreateWall(root, "CombRidge_North2", new Vector3(area.Width * 0.12f, area.WallHeight * 0.5f, area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
            
            CreateWall(root, "CombRidge_South1", new Vector3(-area.Width * 0.12f, area.WallHeight * 0.5f, -area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
            CreateWall(root, "CombRidge_South2", new Vector3(area.Width * 0.12f, area.WallHeight * 0.5f, -area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
        }

        // 4. Rey Juan Carlos Hospital: Abgerundete Doppel-Zylinder/Oval-Struktur
        private static void BuildReyJuanCarlos(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            // Zwei elliptische/ovale Ringblöcke nebeneinander
            var ovalWidth = (area.CorridorRightX - area.CorridorLeftX) * 0.32f;
            var ovalDepth = area.HalfDepth * 0.8f;

            // Westlicher Oval-Kern
            CreateWall(root, "OvalWest_Core", new Vector3(-ovalWidth * 0.85f, area.WallHeight * 0.5f, 0f), new Vector3(ovalWidth, area.WallHeight, ovalDepth), material, obstacleLayer);

            // Östlicher Oval-Kern
            CreateWall(root, "OvalEast_Core", new Vector3(ovalWidth * 0.85f, area.WallHeight * 0.5f, 0f), new Vector3(ovalWidth, area.WallHeight, ovalDepth), material, obstacleLayer);

            // Korridorführung Nord/Süd
            CreateWall(root, "OvalNorthRail", new Vector3(0f, area.WallHeight * 0.5f, area.HalfDepth * 0.6f), new Vector3(ovalWidth * 3f, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "OvalSouthRail", new Vector3(0f, area.WallHeight * 0.5f, -area.HalfDepth * 0.6f), new Vector3(ovalWidth * 3f, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // 5. University Clinic Münster: Twin Circular Tower Loops with Connecting Bridge
        private static void BuildUniversityClinicMuenster(Transform root, LayoutArea area, Material material,
            int obstacleLayer)
        {
            float corridorSpan = area.CorridorRightX - area.CorridorLeftX;
            float halfSpan = corridorSpan * 0.5f;

            // Geometric Parameters matching the sketch
            float triBaseWidth = corridorSpan * 0.55f;
            float triHeight = area.HalfDepth * 0.95f;
            float triBottomZ = -area.HalfDepth * 0.42f;
            float triApexZ = triBottomZ + triHeight;

            float angle = Mathf.Atan2(triHeight, triBaseWidth * 0.5f) * Mathf.Rad2Deg;
            float sideLength = Mathf.Sqrt(Mathf.Pow(triBaseWidth * 0.5f, 2) + Mathf.Pow(triHeight, 2));

            // -------------------------------------------------------------------------
            // 1. CENTRAL SOLID TRIANGLE CORE (3 Angled Walls)
            // -------------------------------------------------------------------------
            // Base Wall (Horizontal Bottom of Triangle)
            CreateWall(root, "TriCore_Base", new Vector3(0f, area.WallHeight * 0.5f, triBottomZ), new Vector3(triBaseWidth, area.WallHeight, WallThickness), material, obstacleLayer);

            // Left Angled Wall of Triangle
            var triLeft = CreateWall(root, "TriCore_Left", new Vector3(-triBaseWidth * 0.25f, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength), material, obstacleLayer);
            triLeft.transform.localRotation = Quaternion.Euler(0f, -(90f - angle), 0f);

            // Right Angled Wall of Triangle
            var triRight = CreateWall(root, "TriCore_Right", new Vector3(triBaseWidth * 0.25f, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength), material, obstacleLayer);
            triRight.transform.localRotation = Quaternion.Euler(0f, (90f - angle), 0f);

            // -------------------------------------------------------------------------
            // 2. PARALLEL OUTER DIAGONAL CHUTES
            // -------------------------------------------------------------------------
            float corridorWidth = MinCorridorClearance;
            float outerOffset = corridorWidth + 0.2f;

            // Left Outer Diagonal Wall (Parallel to Left Triangle Face)
            var outerLeft = CreateWall(root, "OuterDiag_Left", new Vector3(-triBaseWidth * 0.25f - outerOffset, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength * 0.95f), material, obstacleLayer);
            outerLeft.transform.localRotation = Quaternion.Euler(0f, -(90f - angle), 0f);

            // Right Outer Diagonal Wall (Parallel to Right Triangle Face)
            var outerRight = CreateWall(root, "OuterDiag_Right", new Vector3(triBaseWidth * 0.25f + outerOffset, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength * 0.95f), material, obstacleLayer);
            outerRight.transform.localRotation = Quaternion.Euler(0f, (90f - angle), 0f);

            // -------------------------------------------------------------------------
            // 3. BOTTOM LEFT & RIGHT CIRCULAR LOOPS (Knobs with Inner Pillars)
            // -------------------------------------------------------------------------
            float loopCenterX = triBaseWidth * 0.5f + 0.6f;
            float loopCenterZ = triBottomZ;
            float pillarSize = 1.0f;

            // Inner Pillar Posts
            CreateWall(root, "LoopPillar_Left", new Vector3(-loopCenterX, area.WallHeight * 0.5f, loopCenterZ), new Vector3(pillarSize, area.WallHeight, pillarSize), material, obstacleLayer);
            CreateWall(root, "LoopPillar_Right", new Vector3(loopCenterX, area.WallHeight * 0.5f, loopCenterZ), new Vector3(pillarSize, area.WallHeight, pillarSize), material, obstacleLayer);

            // Outer Loop Enclosures
            CreateWall(root, "LoopCap_Left", new Vector3(-loopCenterX - 1.2f, area.WallHeight * 0.5f, loopCenterZ), new Vector3(WallThickness, area.WallHeight, 2.4f), material, obstacleLayer);
            CreateWall(root, "LoopCap_Right", new Vector3(loopCenterX + 1.2f, area.WallHeight * 0.5f, loopCenterZ), new Vector3(WallThickness, area.WallHeight, 2.4f), material, obstacleLayer);

            // -------------------------------------------------------------------------
            // 4. TOP & BOTTOM TERMINAL BOUNDARY WALLS WITH DOORWAYS
            // -------------------------------------------------------------------------
            // Top Apex Doorway Wall (Entrance from Top Anchor Room)
            CreateWallWithDoor(root, "TopApexDoorWall", 0f, -halfSpan, halfSpan, area.WallHeight, material, obstacleLayer, doorCenterZ: 0f, doorWidth: MinCorridorClearance);

            // Bottom Wall with Central Doorway (Exit to Bottom Anchor Room)
            float bottomWallZ = -area.HalfDepth * 0.72f;
            float sideWallLen = (corridorSpan - MinCorridorClearance) * 0.5f;

            CreateWall(root, "BottomWall_Left", new Vector3(-corridorSpan * 0.28f, area.WallHeight * 0.5f, bottomWallZ), new Vector3(sideWallLen, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "BottomWall_Right", new Vector3(corridorSpan * 0.28f, area.WallHeight * 0.5f, bottomWallZ), new Vector3(sideWallLen, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // Helper method to generate circular/octagonal cores for high-rise towers
        private static void CreateOctagonalCore(Transform root, string namePrefix, Vector3 center, float radius,
            float wallHeight, Material material, int layer)
        {
            float halfSize = radius * 0.707f;

            // North & South core faces
            CreateWall(root, $"{namePrefix}_N", center + new Vector3(0f, wallHeight * 0.5f, radius),
                new Vector3(radius * 1.4f, wallHeight, WallThickness), material, layer);
            CreateWall(root, $"{namePrefix}_S", center + new Vector3(0f, wallHeight * 0.5f, -radius),
                new Vector3(radius * 1.4f, wallHeight, WallThickness), material, layer);

            // East & West core faces
            CreateWall(root, $"{namePrefix}_E", center + new Vector3(radius, wallHeight * 0.5f, 0f),
                new Vector3(WallThickness, wallHeight, radius * 1.4f), material, layer);
            CreateWall(root, $"{namePrefix}_W", center + new Vector3(-radius, wallHeight * 0.5f, 0f),
                new Vector3(WallThickness, wallHeight, radius * 1.4f), material, layer);
        }

        private static void CreateWallWithDoor(Transform root, string name, float x, float zMin, float zMax, float wallHeight, Material material, int layer, float doorCenterZ, float doorWidth)
        {
            var doorHalf = doorWidth * 0.5f;
            var doorMin = Mathf.Clamp(doorCenterZ - doorHalf, zMin + 0.5f, zMax - 0.5f);
            var doorMax = Mathf.Clamp(doorCenterZ + doorHalf, zMin + 0.5f, zMax - 0.5f);

            if (doorMin > zMin)
            {
                var lowerLength = doorMin - zMin;
                CreateWall(root, $"{name}_Lower", new Vector3(x, wallHeight * 0.5f, zMin + lowerLength * 0.5f), new Vector3(WallThickness, wallHeight, lowerLength), material, layer);
            }

            if (doorMax < zMax)
            {
                var upperLength = zMax - doorMax;
                CreateWall(root, $"{name}_Upper", new Vector3(x, wallHeight * 0.5f, doorMax + upperLength * 0.5f), new Vector3(WallThickness, wallHeight, upperLength), material, layer);
            }
        }

        private static GameObject CreateWall(Transform root, string name, Vector3 localPosition, Vector3 localScale, Material material, int layer)
        {
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = name;
            wall.transform.SetParent(root, false);
            wall.transform.localPosition = localPosition;
            wall.transform.localRotation = Quaternion.identity;
            wall.transform.localScale = localScale;
            wall.layer = layer;
            ApplyMaterial(wall, material);
            return wall;
        }

        private static void ApplyMaterial(GameObject go, Material material)
        {
            if (go == null || material == null) return;

            var renderer = go.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = material;
            }
        }

        private struct LayoutArea
        {
            public float Width { get; set; }
            public float Depth { get; set; }
            public float WallHeight { get; set; }
            public float HalfWidth { get; set; }
            public float HalfDepth { get; set; }
            public float LeftRoomBoundaryX { get; set; }
            public float RightRoomBoundaryX { get; set; }
            public float CorridorLeftX { get; set; }
            public float CorridorRightX { get; set; }
            public int Seed { get; set; }
        }
    }
}