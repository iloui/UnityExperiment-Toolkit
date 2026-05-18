using System.Collections.Generic;
using UnityEngine;
using System.IO;

public static class AnalyticsManager
{
    public static void ExportHeatMap(List<List<Vector3>>[] allAgentPositions, List<GameObject> pois, string savePath)
    {
        // 1. Calculate Simulation Bounds
        Bounds bounds = CalculateBounds(allAgentPositions, pois);
        
        // 2. Setup Temporary Camera for Floorplan
        GameObject camObj = new GameObject("TempAnalyticsCamera");
        Camera cam = camObj.AddComponent<Camera>();
        cam.orthographic = true;
        cam.backgroundColor = Color.white;
        cam.clearFlags = CameraClearFlags.SolidColor;
        
        // Find agents' average Y height to set the cut plane
        float avgY = bounds.center.y;
        if (allAgentPositions.Length > 0 && allAgentPositions[0].Count > 0 && allAgentPositions[0][0].Count > 0) {
            avgY = allAgentPositions[0][0][0].y;
        }

        // Clip the view: Only see a small slice around the agents' height
        float clipBelow = 0.5f; 
        float clipAbove = 2.5f; 
        
        camObj.transform.position = new Vector3(bounds.center.x, avgY + clipAbove, bounds.center.z);
        camObj.transform.rotation = Quaternion.Euler(90, 0, 0);
        cam.nearClipPlane = 0.01f;
        cam.farClipPlane = clipAbove + clipBelow;

        // Set orthographic size to fit bounds
        float boundsAspect = bounds.size.x / bounds.size.z;
        if (boundsAspect > 1f)
        {
            // Width is larger, fit based on extents.x
            cam.orthographicSize = bounds.extents.x;
        }
        else
        {
            // Depth is larger, fit based on extents.z
            cam.orthographicSize = bounds.extents.z;
        }
        
        // 3. Render Floorplan to Texture
        int res = 2048;
        RenderTexture rt = new RenderTexture(res, res, 24);
        cam.targetTexture = rt;
        cam.Render();
        
        RenderTexture.active = rt;
        Texture2D compositeTex = new Texture2D(res, res, TextureFormat.RGB24, false);
        compositeTex.ReadPixels(new Rect(0, 0, res, res), 0, 0);
        
        // 4. Overlay Heatmap Logic (Pixel-based)
        Color[] pixels = compositeTex.GetPixels();
        
        // Identify edges to draw black lines for boundaries
        Color[] tempPixels = new Color[pixels.Length];
        System.Array.Copy(pixels, tempPixels, pixels.Length);

        for (int y = 1; y < res - 1; y++) {
            for (int x = 1; x < res - 1; x++) {
                int idx = y * res + x;
                float grayscale = tempPixels[idx].grayscale;

                // Threshold to check if we are on geometry (object) or empty space
                bool isGeometry = grayscale < 0.95f; 

                if (isGeometry) {
                    // Check neighbors to find edges
                    bool isEdge = false;
                    for (int ny = -1; ny <= 1; ny++) {
                        for (int nx = -1; nx <= 1; nx++) {
                            if (tempPixels[(y + ny) * res + (x + nx)].grayscale > 0.98f) {
                                isEdge = true;
                                break;
                            }
                        }
                    }
                    // Edge becomes black, inner area remains white (transparent look)
                    pixels[idx] = isEdge ? Color.black : Color.white;
                } else {
                    pixels[idx] = Color.white;
                }
            }
        }
        
        // 4b. Draw POIs, Starts, and Ends in Task Colors
        // Find tasks to get colors
        EngineScript engine = Object.FindObjectOfType<EngineScript>();
        if (engine != null)
        {
            TaskScript[] tasks = engine.GetComponents<TaskScript>();
            foreach (var task in tasks)
            {
                if (task == null) continue;
                Color markerColor = task.taskColor;

                foreach (var s in task.start) if (s != null) DrawPOIMarker(pixels, res, bounds, cam, s.transform.position, markerColor);
                foreach (var e in task.end) if (e != null) DrawPOIMarker(pixels, res, bounds, cam, e.transform.position, markerColor);
                foreach (var p in task.pointsOfInterest) if (p != null) DrawPOIMarker(pixels, res, bounds, cam, p.transform.position, markerColor);
            }
        }

        foreach (var taskList in allAgentPositions)
        {
            foreach (var trajectory in taskList)
            {
                if (trajectory.Count < 2) continue;

                // Pre-calculate dwell times for coloring (reusing logic from AgentScript)
                float dwellThreshold = 0.5f;
                float[] dwellTimes = new float[trajectory.Count];
                for (int i = 0; i < trajectory.Count; i++)
                {
                    int dwellCount = 0;
                    for (int j = 0; j < trajectory.Count; j++)
                    {
                        if (Vector3.Distance(trajectory[i], trajectory[j]) < dwellThreshold) dwellCount++;
                    }
                    dwellTimes[i] = (float)dwellCount / trajectory.Count;
                }

                float maxDwell = 0;
                foreach (float d in dwellTimes) if (d > maxDwell) maxDwell = d;
                if (maxDwell == 0) maxDwell = 1f;

                // Draw lines between points
                for (int i = 1; i < trajectory.Count; i++)
                {
                    Vector3 p1 = trajectory[i - 1];
                    Vector3 p2 = trajectory[i];
                    
                    float avgDwell = (dwellTimes[i - 1] + dwellTimes[i]) / 2.0f;
                    Color heatColor = GetHeatmapColor(avgDwell / maxDwell);

                    DrawLineOnPixels(pixels, res, bounds, cam, p1, p2, heatColor);
                }
            }
        }
        
        compositeTex.SetPixels(pixels);
        compositeTex.Apply();
        
        // 5. Save to File
        byte[] bytes = compositeTex.EncodeToPNG();
        File.WriteAllBytes(savePath, bytes);
        
        // Cleanup
        Object.DestroyImmediate(camObj);
        Object.DestroyImmediate(rt);
        Debug.Log($"Heatmap saved to: {savePath}");
    }

    private static void DrawLineOnPixels(Color[] pixels, int res, Bounds bounds, Camera cam, Vector3 p1, Vector3 p2, Color color)
    {
        // Convert world to pixel coords
        float camSize = cam.orthographicSize;
        float boundsAspect = bounds.size.x / bounds.size.z;
        Vector2 start = WorldToPixel(p1, bounds.center, camSize, res, boundsAspect);
        Vector2 end = WorldToPixel(p2, bounds.center, camSize, res, boundsAspect);

        // Simple Bresenham-like line drawing
        int steps = Mathf.CeilToInt(Vector2.Distance(start, end) * 2); // Over-sample for continuity
        for (int s = 0; s <= steps; s++)
        {
            float t = (float)s / (steps == 0 ? 1 : steps);
            Vector2 interp = Vector2.Lerp(start, end, t);
            int px = Mathf.RoundToInt(interp.x);
            int py = Mathf.RoundToInt(interp.y);

            // 3x3 Brush for visibility
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    int tx = px + dx;
                    int ty = py + dy;
                    if (tx >= 0 && tx < res && ty >= 0 && ty < res)
                    {
                        int idx = ty * res + tx;
                        pixels[idx] = Color.Lerp(pixels[idx], color, 0.4f);
                    }
                }
            }
        }
    }

    private static void DrawPOIMarker(Color[] pixels, int res, Bounds bounds, Camera cam, Vector3 pos, Color color)
    {
        float camSize = cam.orthographicSize;
        float boundsAspect = bounds.size.x / bounds.size.z;
        Vector2 pixelPos = WorldToPixel(pos, bounds.center, camSize, res, boundsAspect);
        int px = Mathf.RoundToInt(pixelPos.x);
        int py = Mathf.RoundToInt(pixelPos.y);

        // Draw a larger cross/diamond marker for POIs
        int size = 6;
        for (int x = -size; x <= size; x++)
        {
            for (int y = -size; y <= size; y++)
            {
                // Simple diamond shape
                if (Mathf.Abs(x) + Mathf.Abs(y) <= size)
                {
                    int tx = px + x;
                    int ty = py + y;
                    if (tx >= 0 && tx < res && ty >= 0 && ty < res)
                    {
                        pixels[ty * res + tx] = color;
                    }
                }
            }
        }
    }

    private static Vector2 WorldToPixel(Vector3 pos, Vector3 center, float orthSize, int res, float boundsAspect)
    {
        if (boundsAspect > 1f)
        {
            float x = (pos.x - (center.x - orthSize)) / (orthSize * 2);
            float z = (pos.z - (center.z - orthSize / boundsAspect)) / (orthSize * 2 / boundsAspect);
            return new Vector2(x * res, z * res);
        }
        else
        {
            float x = (pos.x - (center.x - orthSize * boundsAspect)) / (orthSize * 2 * boundsAspect);
            float z = (pos.z - (center.z - orthSize)) / (orthSize * 2);
            return new Vector2(x * res, z * res);
        }
    }

    private static Color GetHeatmapColor(float normDwell)
    {
        if (normDwell < 0.25f) return Color.Lerp(new Color(0, 0.5f, 1), new Color(0, 1, 1), normDwell * 4f);
        if (normDwell < 0.5f) return Color.Lerp(new Color(0, 1, 1), new Color(0, 1, 0), (normDwell - 0.25f) * 4f);
        if (normDwell < 0.75f) return Color.Lerp(new Color(0, 1, 0), new Color(1, 1, 0), (normDwell - 0.5f) * 4f);
        return Color.Lerp(new Color(1, 1, 0), new Color(1, 0, 0), (normDwell - 0.75f) * 4f);
    }

    private static Bounds CalculateBounds(List<List<Vector3>>[] agentPositions, List<GameObject> pois)
    {
        Bounds b = new Bounds();
        bool initialized = false;

        // First encapsulate all geometry in the scene to get the full building dimensions
        Renderer[] allRenderers = Object.FindObjectsOfType<Renderer>();
        foreach (Renderer r in allRenderers)
        {
            if (r.gameObject.activeInHierarchy)
            {
                if (!initialized) { b.center = r.bounds.center; b.size = r.bounds.size; initialized = true; }
                else b.Encapsulate(r.bounds);
            }
        }

        foreach (var taskList in agentPositions)
        {
            foreach (var trajectory in taskList)
            {
                foreach (Vector3 pos in trajectory)
                {
                    if (!initialized) { b.center = pos; initialized = true; }
                    b.Encapsulate(pos);
                }
            }
        }

        foreach (GameObject poi in pois)
        {
            if (!initialized) { b.center = poi.transform.position; initialized = true; }
            b.Encapsulate(poi.transform.position);
        }

        return b;
    }
}
