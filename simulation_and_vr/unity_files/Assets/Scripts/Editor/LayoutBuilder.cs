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
        UniversityClinicMuenster     // Doppel-Ring mit Brücke
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

            var roomWidth = Mathf.Clamp(width * 0.24f, 6f, 8f);
            var leftRoomBoundaryX = -halfWidth + roomWidth;
            var rightRoomBoundaryX = halfWidth - roomWidth;
            var corridorLeftX = leftRoomBoundaryX;
            var corridorRightX = rightRoomBoundaryX;

            CreateFloorAndCeiling(root.transform, width, depth, wallHeight, sharedMaterial, floorLayer, obstacleLayer);
            CreatePerimeterWalls(root.transform, width, depth, wallHeight, sharedMaterial, obstacleLayer);
            
            // Staggered door Z-positions to break spatial symmetry and force natural human S-turn alignment
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
                CorridorRightX = corridorRightX,
                Seed = settings.RandomSeed
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
                    if (rng.NextDouble() < 0.40d) continue;

                    var obstacleScale = new Vector3(cellWidth * 0.55f, area.WallHeight, cellDepth * 0.55f);
                    var pos = new Vector3(startX + x * cellWidth, area.WallHeight * 0.5f, startZ + z * cellDepth);
                    CreateWall(root, $"GridBlock_{x}_{z}", pos, obstacleScale, material, obstacleLayer);
                }
            }
        }
        
        // 1. Circle Bath General Hospital: Atrium-Ring[cite: 5]
        private static void BuildCircleBath(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var corridorWidth = (area.CorridorRightX - area.CorridorLeftX);
            var coreWidth = corridorWidth * 0.45f;
            var coreDepth = area.HalfDepth * 0.9f;

            // Zentrales geschlossenes Atrium / Bauwerk[cite: 5]
            CreateWall(root, "AtriumCore", new Vector3(0f, area.WallHeight * 0.5f, 0f), new Vector3(coreWidth, area.WallHeight, coreDepth), material, obstacleLayer);

            // Äussere Begrenzungsschienen zur Formung des Ring-Gangs[cite: 5]
            var outerRailZ = area.HalfDepth * 0.7f;
            CreateWall(root, "NorthRingWall", new Vector3(0f, area.WallHeight * 0.5f, outerRailZ), new Vector3(corridorWidth, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SouthRingWall", new Vector3(0f, area.WallHeight * 0.5f, -outerRailZ), new Vector3(corridorWidth, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // 2. Maggie’s Centre Gartnavel: Stern- / Verzweigungstypologie[cite: 6]
        private static void BuildMaggiesCentre(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var midX = 0f;
            var wingLength = area.HalfDepth * 0.6f;

            // Zentraler Knotenpunkt (Verzweigung)[cite: 6]
            CreateWall(root, "CentralHubColumn", new Vector3(midX, area.WallHeight * 0.5f, 0f), new Vector3(1.5f, area.WallHeight, 1.5f), material, obstacleLayer);

            // Diagonale/Versetzte Flügelelemente für Sichtschutz und Verzweigungen[cite: 6]
            CreateWall(root, "WingNorthWest", new Vector3(-area.Width * 0.1f, area.WallHeight * 0.5f, area.HalfDepth * 0.35f), new Vector3(WallThickness, area.WallHeight, wingLength), material, obstacleLayer);
            CreateWall(root, "WingSouthEast", new Vector3(area.Width * 0.1f, area.WallHeight * 0.5f, -area.HalfDepth * 0.35f), new Vector3(WallThickness, area.WallHeight, wingLength), material, obstacleLayer);

            // Querriegel, der Stichgänge zu den "Beratungsräumen" bildet[cite: 6]
            CreateWall(root, "CounselingDivider", new Vector3(0f, area.WallHeight * 0.5f, area.HalfDepth * 0.55f), new Vector3(area.Width * 0.2f, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // 3. Kaiser-Franz-Josef-Spital: Doppelkamm- Structure (Hauptachse + Querriegel)[cite: 7]
        private static void BuildKaiserFranzJosef(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            var corridorSpan = area.CorridorRightX - area.CorridorLeftX;

            // Zentrale Haupt-Magistrale (Rückgrat) mit Durchlass[cite: 7]
            var spineSegmentWidth = corridorSpan * 0.38f;
            CreateWall(root, "SpineWest", new Vector3(-corridorSpan * 0.25f, area.WallHeight * 0.5f, 0f), new Vector3(spineSegmentWidth, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "SpineEast", new Vector3(corridorSpan * 0.25f, area.WallHeight * 0.5f, 0f), new Vector3(spineSegmentWidth, area.WallHeight, WallThickness), material, obstacleLayer);

            // Senkrechte Kamm-Riegel (Nord & Süd)[cite: 7]
            var ridgeHeight = area.HalfDepth * 0.55f;
            CreateWall(root, "CombRidge_North1", new Vector3(-area.Width * 0.12f, area.WallHeight * 0.5f, area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
            CreateWall(root, "CombRidge_North2", new Vector3(area.Width * 0.12f, area.WallHeight * 0.5f, area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
            
            CreateWall(root, "CombRidge_South1", new Vector3(-area.Width * 0.12f, area.WallHeight * 0.5f, -area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
            CreateWall(root, "CombRidge_South2", new Vector3(area.Width * 0.12f, area.WallHeight * 0.5f, -area.HalfDepth * 0.45f), new Vector3(WallThickness, area.WallHeight, ridgeHeight), material, obstacleLayer);
        }

        // 4. Rey Juan Carlos Hospital: Abgerundete Doppel-Zylinder/Oval-Struktur[cite: 8]
        private static void BuildReyJuanCarlos(Transform root, LayoutArea area, Material material, int obstacleLayer)
        {
            // Zwei elliptische/ovale Ringblöcke nebeneinander[cite: 8]
            var ovalWidth = (area.CorridorRightX - area.CorridorLeftX) * 0.32f;
            var ovalDepth = area.HalfDepth * 0.8f;

            // Westlicher Oval-Kern[cite: 8]
            CreateWall(root, "OvalWest_Core", new Vector3(-ovalWidth * 0.85f, area.WallHeight * 0.5f, 0f), new Vector3(ovalWidth, area.WallHeight, ovalDepth), material, obstacleLayer);

            // Östlicher Oval-Kern[cite: 8]
            CreateWall(root, "OvalEast_Core", new Vector3(ovalWidth * 0.85f, area.WallHeight * 0.5f, 0f), new Vector3(ovalWidth, area.WallHeight, ovalDepth), material, obstacleLayer);

            // Korridorführung Nord/Süd[cite: 8]
            CreateWall(root, "OvalNorthRail", new Vector3(0f, area.WallHeight * 0.5f, area.HalfDepth * 0.6f), new Vector3(ovalWidth * 3f, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "OvalSouthRail", new Vector3(0f, area.WallHeight * 0.5f, -area.HalfDepth * 0.6f), new Vector3(ovalWidth * 3f, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // 5. University Clinic Münster: Doppelturm-Ring mit Verbindungsbrücke[cite: 9]
        // 5. University Clinic Münster: Twin Circular Tower Loops with Connecting Bridge[cite: 9, 10]
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
            // 1. CENTRAL SOLID TRIANGLE CORE (3 Angled Walls)[cite: 4]
            // -------------------------------------------------------------------------
            // Base Wall (Horizontal Bottom of Triangle)[cite: 4]
            CreateWall(root, "TriCore_Base", new Vector3(0f, area.WallHeight * 0.5f, triBottomZ), new Vector3(triBaseWidth, area.WallHeight, WallThickness), material, obstacleLayer);

            // Left Angled Wall of Triangle[cite: 4]
            var triLeft = CreateWall(root, "TriCore_Left", new Vector3(-triBaseWidth * 0.25f, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength), material, obstacleLayer);
            triLeft.transform.localRotation = Quaternion.Euler(0f, -(90f - angle), 0f);

            // Right Angled Wall of Triangle[cite: 4]
            var triRight = CreateWall(root, "TriCore_Right", new Vector3(triBaseWidth * 0.25f, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength), material, obstacleLayer);
            triRight.transform.localRotation = Quaternion.Euler(0f, (90f - angle), 0f);

            // -------------------------------------------------------------------------
            // 2. PARALLEL OUTER DIAGONAL CHUTES[cite: 4]
            // -------------------------------------------------------------------------
            float corridorWidth = MinCorridorClearance;
            float outerOffset = corridorWidth + 0.2f;

            // Left Outer Diagonal Wall (Parallel to Left Triangle Face)[cite: 4]
            var outerLeft = CreateWall(root, "OuterDiag_Left", new Vector3(-triBaseWidth * 0.25f - outerOffset, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength * 0.95f), material, obstacleLayer);
            outerLeft.transform.localRotation = Quaternion.Euler(0f, -(90f - angle), 0f);

            // Right Outer Diagonal Wall (Parallel to Right Triangle Face)[cite: 4]
            var outerRight = CreateWall(root, "OuterDiag_Right", new Vector3(triBaseWidth * 0.25f + outerOffset, area.WallHeight * 0.5f, (triBottomZ + triApexZ) * 0.5f), new Vector3(WallThickness, area.WallHeight, sideLength * 0.95f), material, obstacleLayer);
            outerRight.transform.localRotation = Quaternion.Euler(0f, (90f - angle), 0f);

            // -------------------------------------------------------------------------
            // 3. BOTTOM LEFT & RIGHT CIRCULAR LOOPS (Knobs with Inner Pillars)[cite: 4]
            // -------------------------------------------------------------------------
            float loopCenterX = triBaseWidth * 0.5f + 0.6f;
            float loopCenterZ = triBottomZ;
            float pillarSize = 1.0f;

            // Inner Pillar Posts[cite: 4]
            CreateWall(root, "LoopPillar_Left", new Vector3(-loopCenterX, area.WallHeight * 0.5f, loopCenterZ), new Vector3(pillarSize, area.WallHeight, pillarSize), material, obstacleLayer);
            CreateWall(root, "LoopPillar_Right", new Vector3(loopCenterX, area.WallHeight * 0.5f, loopCenterZ), new Vector3(pillarSize, area.WallHeight, pillarSize), material, obstacleLayer);

            // Outer Loop Enclosures[cite: 4]
            CreateWall(root, "LoopCap_Left", new Vector3(-loopCenterX - 1.2f, area.WallHeight * 0.5f, loopCenterZ), new Vector3(WallThickness, area.WallHeight, 2.4f), material, obstacleLayer);
            CreateWall(root, "LoopCap_Right", new Vector3(loopCenterX + 1.2f, area.WallHeight * 0.5f, loopCenterZ), new Vector3(WallThickness, area.WallHeight, 2.4f), material, obstacleLayer);

            // -------------------------------------------------------------------------
            // 4. TOP & BOTTOM TERMINAL BOUNDARY WALLS WITH DOORWAYS[cite: 4]
            // -------------------------------------------------------------------------
            // Top Apex Doorway Wall (Entrance from Top Anchor Room)[cite: 4]
            CreateWallWithDoor(root, "TopApexDoorWall", 0f, -halfSpan, halfSpan, area.WallHeight, material, obstacleLayer, doorCenterZ: 0f, doorWidth: MinCorridorClearance);

            // Bottom Wall with Central Doorway (Exit to Bottom Anchor Room)[cite: 4]
            float bottomWallZ = -area.HalfDepth * 0.72f;
            float sideWallLen = (corridorSpan - MinCorridorClearance) * 0.5f;

            CreateWall(root, "BottomWall_Left", new Vector3(-corridorSpan * 0.28f, area.WallHeight * 0.5f, bottomWallZ), new Vector3(sideWallLen, area.WallHeight, WallThickness), material, obstacleLayer);
            CreateWall(root, "BottomWall_Right", new Vector3(corridorSpan * 0.28f, area.WallHeight * 0.5f, bottomWallZ), new Vector3(sideWallLen, area.WallHeight, WallThickness), material, obstacleLayer);
        }

        // Helper method to generate circular/octagonal cores for high-rise towers[cite: 9]
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