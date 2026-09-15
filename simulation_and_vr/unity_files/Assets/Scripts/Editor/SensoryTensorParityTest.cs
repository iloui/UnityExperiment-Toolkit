using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Agent_Based_Modelling;
using Assets.Scripts.DataRecording;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace Assets.Scripts.Editor
{
    /// <summary>
    /// Automated parity test: proves the recording -> training pipeline
    /// (ParticipantRecorder's binary write format -> ingest_to_zarr.py ->
    /// train_onAllRecordings_2DCNNModel.SingleEpisodeDataset) and the live-inference pipeline
    /// (ImitationAgent.BuildSensoryTensor) produce the byte-for-byte-equivalent "sensory_inputs"
    /// tensor from the same synthetic input.
    ///
    /// Run from the command line via Unity batch mode, e.g.:
    ///   Unity -batchmode -projectPath &lt;path to unity_files&gt; -quit \
    ///     -executeMethod Assets.Scripts.Editor.SensoryTensorParityTest.RunSensoryTensorParityTest
    /// Exits with code 0 on a pass and a non-zero code on any failure (mismatch, or the Python
    /// side failing to run), so it fails loudly in CI rather than silently skipping.
    ///
    /// Also runnable interactively via Tools/Imitation Agent/Run Sensory Tensor Parity Test.
    /// </summary>
    internal static class SensoryTensorParityTest
    {
        private const int Seed = 1234;
        private const float FloatTolerance = 1e-4f;

        [MenuItem("Tools/Imitation Agent/Run Sensory Tensor Parity Test")]
        public static void RunSensoryTensorParityTestMenuItem()
        {
            RunSensoryTensorParityTest();
        }

        public static void RunSensoryTensorParityTest()
        {
            string tempDir = Path.Combine(Path.GetTempPath(), "SensoryTensorParityTest_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);

            try
            {
                bool passed = RunInternal(tempDir, out string failureReason);

                if (passed)
                {
                    Debug.Log("[SensoryTensorParityTest] PASS: recording pipeline and inference pipeline " +
                        "produced an identical sensory_inputs tensor from the same synthetic input.");
                    ExitIfBatchMode(0);
                }
                else
                {
                    Debug.LogError($"[SensoryTensorParityTest] FAIL: {failureReason}");
                    ExitIfBatchMode(1);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SensoryTensorParityTest] FAIL: unhandled exception: {ex}");
                ExitIfBatchMode(1);
            }
            finally
            {
                try { Directory.Delete(tempDir, true); } catch { /* best-effort cleanup */ }
            }
        }

        private static void ExitIfBatchMode(int code)
        {
            if (Application.isBatchMode)
            {
                EditorApplication.Exit(code);
            }
        }

        private static bool RunInternal(string tempDir, out string failureReason)
        {
            const int frameCount = ZarrDataContract.TemporalWindowSize;
            const int totalPixels = ZarrDataContract.TotalPixels;

            // --- 1. Generate deterministic synthetic episode data --------------------------------
            var random = new System.Random(Seed);
            Vector3 goal = new Vector3(0.31f, -0.12f, 0.94f);

            var frames = new List<ImitationAgent.SerializableFrameData>();
            var rawColor = new Color32[frameCount][];
            var rawDepth = new float[frameCount][];

            for (int f = 0; f < frameCount; f++)
            {
                var colorMatrix = new Color32[totalPixels];
                var depthMatrix = new float[totalPixels];
                for (int i = 0; i < totalPixels; i++)
                {
                    colorMatrix[i] = new Color32(
                        (byte)random.Next(256),
                        (byte)random.Next(256),
                        (byte)random.Next(256),
                        255);
                    depthMatrix[i] = (float)random.NextDouble();
                }

                rawColor[f] = colorMatrix;
                rawDepth[f] = depthMatrix;
                frames.Add(new ImitationAgent.SerializableFrameData { colorMatrix = colorMatrix, depthMatrix = depthMatrix });
            }

            // --- 2. Build the C# (inference-side) tensor via the exact production method ---------
            float[] csharpTensor = ImitationAgent.BuildSensoryTensor(goal, frames);

            // --- 3. Write the same data through the recording pipeline's binary format -----------
            string episodeDir = Path.Combine(tempDir, "episode_parity_test");
            Directory.CreateDirectory(episodeDir);
            WriteEpisodeBinaries(episodeDir, rawColor, rawDepth, goal, frameCount, totalPixels);
            WriteManifest(episodeDir, frameCount);

            // --- 4. Run the real ingest_to_zarr.py + SingleEpisodeDataset code path --------------
            string rootDir = Path.GetFullPath(Path.Combine(Application.dataPath, "../../"));
            string trainingDir = Path.Combine(rootDir, "imitationagent_training");
            string pipelineDir = Path.Combine(trainingDir, "training_pipeline");
            string pythonEmbedded = Path.Combine(trainingDir, "python_embedded");
            string datasetZarr = Path.Combine(episodeDir, "dataset.zarr");
            string tensorOutPath = Path.Combine(tempDir, "python_tensor.txt");

            string pythonCmd = (Application.platform == RuntimePlatform.WindowsEditor) ? "python" : "python3";

            // Mirror ParticipantRecorder.TriggerPythonPipeline: make sure python_embedded
            // actually has zarr/numpy/torch/onnx before relying on them.
            if (!RunPython(pythonCmd, pipelineDir, pythonEmbedded,
                    $"\"{Path.Combine(pipelineDir, "bootstrap_env.py")}\"",
                    out string bootstrapOutput, out string bootstrapError))
            {
                failureReason = $"bootstrap_env.py failed.\nstdout:\n{bootstrapOutput}\nstderr:\n{bootstrapError}";
                return false;
            }

            if (!RunPython(pythonCmd, pipelineDir, pythonEmbedded,
                    $"\"{Path.Combine(pipelineDir, "ingest_to_zarr.py")}\" --episode_dir \"{episodeDir}\" --output_zarr \"{datasetZarr}\"",
                    out string ingestOutput, out string ingestError))
            {
                failureReason = $"ingest_to_zarr.py failed.\nstdout:\n{ingestOutput}\nstderr:\n{ingestError}";
                return false;
            }

            if (!RunPython(pythonCmd, pipelineDir, pythonEmbedded,
                    $"\"{Path.Combine(pipelineDir, "print_first_window_tensor.py")}\" --episode_dir \"{episodeDir}\" --out \"{tensorOutPath}\"",
                    out string printOutput, out string printError))
            {
                failureReason = $"print_first_window_tensor.py failed.\nstdout:\n{printOutput}\nstderr:\n{printError}";
                return false;
            }

            if (!File.Exists(tensorOutPath))
            {
                failureReason = $"Expected Python output tensor file was not created: {tensorOutPath}";
                return false;
            }

            float[] pythonTensor = ParseFloatFile(tensorOutPath);

            // --- 5. Compare -----------------------------------------------------------------------
            if (pythonTensor.Length != csharpTensor.Length)
            {
                failureReason = $"Tensor length mismatch: C# produced {csharpTensor.Length} values, " +
                    $"Python produced {pythonTensor.Length} values.";
                return false;
            }

            for (int i = 0; i < csharpTensor.Length; i++)
            {
                float diff = Math.Abs(csharpTensor[i] - pythonTensor[i]);
                if (diff > FloatTolerance)
                {
                    failureReason = $"Value mismatch at index {i}: C# = {csharpTensor[i]}, Python = {pythonTensor[i]}, " +
                        $"diff = {diff} (tolerance {FloatTolerance}).";
                    return false;
                }
            }

            failureReason = null;
            return true;
        }

        private static void WriteEpisodeBinaries(string episodeDir, Color32[][] rawColor, float[][] rawDepth, Vector3 goal, int frameCount, int totalPixels)
        {
            using (var rgbWriter = new BinaryWriter(new FileStream(Path.Combine(episodeDir, "rgb.bin"), FileMode.Create, FileAccess.Write)))
            using (var depthWriter = new BinaryWriter(new FileStream(Path.Combine(episodeDir, "depth.bin"), FileMode.Create, FileAccess.Write)))
            using (var goalWriter = new BinaryWriter(new FileStream(Path.Combine(episodeDir, "goal.bin"), FileMode.Create, FileAccess.Write)))
            using (var poseWriter = new BinaryWriter(new FileStream(Path.Combine(episodeDir, "pose.bin"), FileMode.Create, FileAccess.Write)))
            using (var actionVelWriter = new BinaryWriter(new FileStream(Path.Combine(episodeDir, "action_vel.bin"), FileMode.Create, FileAccess.Write)))
            using (var actionRotWriter = new BinaryWriter(new FileStream(Path.Combine(episodeDir, "action_rot.bin"), FileMode.Create, FileAccess.Write)))
            using (var timestampWriter = new BinaryWriter(new FileStream(Path.Combine(episodeDir, "timestamp.bin"), FileMode.Create, FileAccess.Write)))
            {
                for (int f = 0; f < frameCount; f++)
                {
                    // Same write order as ParticipantRecorder.WriteFrameBinary: per-pixel RGB
                    // triplets, then per-pixel depth, for each frame in chronological order.
                    for (int i = 0; i < totalPixels; i++)
                    {
                        Color32 pixel = rawColor[f][i];
                        rgbWriter.Write(pixel.r);
                        rgbWriter.Write(pixel.g);
                        rgbWriter.Write(pixel.b);
                    }

                    for (int i = 0; i < totalPixels; i++)
                    {
                        depthWriter.Write(rawDepth[f][i]);
                    }

                    goalWriter.Write(goal.x);
                    goalWriter.Write(goal.y);
                    goalWriter.Write(goal.z);

                    poseWriter.Write(0f);
                    poseWriter.Write(0f);
                    poseWriter.Write(0f);

                    actionVelWriter.Write(0f);
                    actionVelWriter.Write(0f);
                    actionVelWriter.Write(0f);

                    actionRotWriter.Write(0f);
                    actionRotWriter.Write(0f);
                    actionRotWriter.Write(0f);

                    timestampWriter.Write((double)(f * ZarrDataContract.SamplingIntervalSeconds));
                }
            }
        }

        private static void WriteManifest(string episodeDir, int frameCount)
        {
            var manifest = new ZarrDataContract.EpisodeManifest
            {
                episode_id = "episode_parity_test",
                participant_id = "parity_test",
                session_id = "parity_test",
                frame_count = frameCount,
                start_time = 0f,
                end_time = frameCount * ZarrDataContract.SamplingIntervalSeconds,
                arrays = new ZarrDataContract.ArrayEntry[]
                {
                    new ZarrDataContract.ArrayEntry { name = "rgb", shape = new int[] { 0, ZarrDataContract.ImageHeight, ZarrDataContract.ImageWidth, 3 }, dtype = ZarrDataContract.ArrayDtypes.Rgb },
                    new ZarrDataContract.ArrayEntry { name = "depth", shape = new int[] { 0, ZarrDataContract.ImageHeight, ZarrDataContract.ImageWidth }, dtype = ZarrDataContract.ArrayDtypes.Depth },
                    new ZarrDataContract.ArrayEntry { name = "goal", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "pose", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "action_vel", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "action_rot", shape = new int[] { 0, 3 }, dtype = ZarrDataContract.ArrayDtypes.Vector },
                    new ZarrDataContract.ArrayEntry { name = "timestamp", shape = new int[] { 0 }, dtype = ZarrDataContract.ArrayDtypes.Timestamp }
                }
            };

            string json = JsonUtility.ToJson(manifest, true);
            File.WriteAllText(Path.Combine(episodeDir, ZarrDataContract.ManifestFileName), json);
        }

        private static bool RunPython(string pythonCmd, string workingDir, string pythonEmbeddedDir, string arguments, out string output, out string error)
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = pythonCmd,
                Arguments = arguments,
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            startInfo.EnvironmentVariables["PYTHONPATH"] = pythonEmbeddedDir;

            using (var process = Process.Start(startInfo))
            {
                output = process.StandardOutput.ReadToEnd();
                error = process.StandardError.ReadToEnd();
                process.WaitForExit();
                return process.ExitCode == 0;
            }
        }

        private static float[] ParseFloatFile(string path)
        {
            string[] lines = File.ReadAllLines(path);
            var values = new List<float>(lines.Length);
            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                values.Add(float.Parse(line, CultureInfo.InvariantCulture));
            }
            return values.ToArray();
        }
    }
}
