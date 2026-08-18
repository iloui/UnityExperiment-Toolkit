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
        RandomObstacleGrid
    }

    public struct LayoutGenerationSettings
    {
        public LayoutTopology Topology;
        public Vector2 Dimensions;
        public float WallHeight;
        public GameObject FpsControllerPrefab;
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

            var roomWidth = Mathf.Clamp(width * 0.24f, 6f, 8f);
            var leftRoomBoundaryX = -halfWidth + roomWidth;
            var rightRoomBoundaryX = halfWidth - roomWidth;
            var corridorLeftX = leftRoomBoundaryX;
            var corridorRightX = rightRoomBoundaryX;

            CreateFloorAndCeiling(root.transform, width, depth, wallHeight, sharedMaterial, floorLayer, obstacleLayer);
            CreatePerimeterWalls(root.transform, width, depth, wallHeight, sharedMaterial, obstacleLayer);
            CreateTerminalRoomWalls(root.transform, leftRoomBoundaryX, rightRoomBoundaryX, halfDepth, wallHeight, sharedMaterial, obstacleLayer);

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
                CorridorRightX = corridorRightX
            }, sharedMaterial, obstacleLayer);

            return new LayoutGenerationData
            {
                SpawnPosition = spawnPosition,
                SpawnRotation = Quaternion.identity,
                GoalPosition = goalPosition
            };
        }

        private static void BuildTopology(Transform root, LayoutTopology topology, LayoutArea area, Material sharedMaterial, int obstacleLayer)
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
                default:
                    throw new ArgumentOutOfRangeException(nameof(topology), topology, null);
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
            CreateWallWithDoor(root, "SpawnRoomDoorWall", leftRoomBoundaryX, -halfDepth + WallThickness, halfDepth - WallThickness, wallHeight, material, obstacleLayer, doorCenterZ: 0f, doorWidth: 2.2f);
            CreateWallWithDoor(root, "TargetRoomDoorWall", rightRoomBoundaryX, -halfDepth + WallThickness, halfDepth - WallThickness, wallHeight, material, obstacleLayer, doorCenterZ: 0f, doorWidth: 2.2f);
        }

        private static void BuildCentralPillar(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var pillarSize = Mathf.Clamp(area.Width * 0.16f, 3.2f, 4.6f);
            CreateWall(root, "CentralPillar", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(pillarSize, area.WallHeight, pillarSize), material, obstacleLayer);

            var corridorWidth = 2.4f;
            var corridorLength = Mathf.Max(3f, area.CorridorRightX - area.CorridorLeftX - pillarSize);
            CreateWall(root, "CentralNorthRail", new Vector3(0f, area.WallHeight * 0.5f, area.HalfDepth * 0.28f), new Vector3(corridorLength, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "CentralSouthRail", new Vector3(0f, area.WallHeight * 0.5f, -area.HalfDepth * 0.28f), new Vector3(corridorLength, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "CentralWestRail", new Vector3(-pillarSize * 0.75f, area.WallHeight * 0.5f, 0f), new Vector3(WallThickness, area.WallHeight, corridorWidth), material, obstacleLayer);
            CreateWall(root, "CentralEastRail", new Vector3(pillarSize * 0.75f, area.WallHeight * 0.5f, 0f), new Vector3(WallThickness, area.WallHeight, corridorWidth), material, obstacleLayer);
        }

        private static void BuildSCurvedCorridor(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var segmentLength = (area.CorridorRightX - area.CorridorLeftX) * 0.5f;
            var leftCenter = Mathf.Lerp(area.LeftRoomBoundaryX, 0f, 0.5f);
            var rightCenter = Mathf.Lerp(0f, area.RightRoomBoundaryX, 0.5f);

            CreateWall(root, "SCurveUpperLeft", new Vector3(leftCenter, area.WallHeight * 0.5f, area.HalfDepth * 0.34f), new Vector3(segmentLength, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SCurveLowerRight", new Vector3(rightCenter, area.WallHeight * 0.5f, -area.HalfDepth * 0.34f), new Vector3(segmentLength, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SCurveCenterSpine", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(WallThickness, area.WallHeight, area.HalfDepth * 1.15f), material, obstacleLayer);
        }

        private static void BuildTJunction(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            CreateWall(root, "TJunctionStem", new Vector3(-area.Width * 0.05f, area.WallHeight * 0.5f, 0f), new Vector3(area.CorridorRightX - area.CorridorLeftX, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "TJunctionBranchLeft", new Vector3(-area.Width * 0.12f, area.WallHeight * 0.5f, area.HalfDepth * 0.42f), new Vector3(WallThickness, area.WallHeight, area.HalfDepth * 0.85f), material, obstacleLayer);
            CreateWall(root, "TJunctionBranchRight", new Vector3(area.Width * 0.12f, area.WallHeight * 0.5f, area.HalfDepth * 0.42f), new Vector3(WallThickness, area.WallHeight, area.HalfDepth * 0.85f), material, obstacleLayer);
        }

        private static void BuildMultiRoomDoorway(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var partitionX1 = Mathf.Lerp(area.LeftRoomBoundaryX, 0f, 0.45f);
            var partitionX2 = Mathf.Lerp(0f, area.RightRoomBoundaryX, 0.45f);
            CreateWallWithDoor(root, "DoorwayPartitionA", partitionX1, -area.HalfDepth + WallThickness, area.HalfDepth - WallThickness, area.WallHeight, material, obstacleLayer, doorCenterZ: -area.HalfDepth * 0.28f, doorWidth: 2f);
            CreateWallWithDoor(root, "DoorwayPartitionB", partitionX2, -area.HalfDepth + WallThickness, area.HalfDepth - WallThickness, area.WallHeight, material, obstacleLayer, doorCenterZ: area.HalfDepth * 0.28f, doorWidth: 2f);
            CreateWall(root, "DoorwayMiddleColumn", new Vector3(0f, area.WallHeight * 0.5f, -area.HalfDepth * 0.15f), new Vector3(WallThickness, area.WallHeight, area.HalfDepth * 0.95f), material, obstacleLayer);
        }

        private static void BuildRandomObstacleGrid(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var rng = new System.Random(area.Width.GetHashCode() ^ area.Depth.GetHashCode() ^ obstacleLayer);
            var gridCountX = 4;
            var gridCountZ = 4;
            var cellWidth = Mathf.Max(1.8f, (area.CorridorRightX - area.CorridorLeftX) / gridCountX);
            var cellDepth = Mathf.Max(1.8f, area.HalfDepth * 1.2f / gridCountZ);
            var startX = -cellWidth * 1.5f;
            var startZ = -cellDepth * 1.5f;

            for (var x = 0; x < gridCountX; x++)
            {
                for (var z = 0; z < gridCountZ; z++)
                {
                    if (x == 1 && z == 1)
                    {
                        continue;
                    }

                    if (x == 2 && z == 2)
                    {
                        continue;
                    }

                    if (rng.NextDouble() < 0.35d)
                    {
                        continue;
                    }

                    var obstacleScale = new Vector3(cellWidth * 0.72f, area.WallHeight, cellDepth * 0.72f);
                    var pos = new Vector3(startX + x * cellWidth, area.WallHeight * 0.5f, startZ + z * cellDepth);
                    CreateWall(root, $"GridBlock_{x}_{z}", pos, obstacleScale, material, obstacleLayer);
                }
            }

            CreateWall(root, "GridSpine", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(area.CorridorRightX - area.CorridorLeftX, area.WallHeight, WallThickness), material, obstacleLayer);
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
        }
    }
}
