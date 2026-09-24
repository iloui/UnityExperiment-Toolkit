using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;
using Debug = UnityEngine.Debug;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Assets.Scripts.DataRecording
{
    /// <summary>
    /// One full retrain of the discrete model (train_2DCNN_discrete.py, from scratch, on everything in zarr_recordings) in a
    /// background process. ParticipantSession requests it once a session is finished; Tools > Thesis > Retrain Discrete Model
    /// Now starts it by hand. Never two at once: imitationagent_training/retrain.lock holds the running process.
    ///
    /// The process is a small shell/cmd wrapper that runs the trainer with the interpreter and PYTHONPATH=python_embedded set
    /// up by bootstrap_env.py, writes all output to retrain_logs/&lt;timestamp&gt;.log and its exit code to &lt;timestamp&gt;.exit.
    /// Unity only tails those files (every line goes to the console) and finalizes from them, so a run survives a domain
    /// reload, leaving Play mode or an Editor restart: whoever loads next picks it up from the lock file.
    ///
    /// On success the timestamped ONNX + .json sidecar are copied to Assets/ImitationModel/ and to
    /// ImitationAgent_discrete_latest.onnx/.json, which ImitationAgent hot-reloads. On failure the previous model stays.
    /// </summary>
    public static class RetrainPipeline
    {
        /// <summary>EditorPrefs (Editor) / PlayerPrefs (player) key for an interpreter path that overrides the bootstrap marker's.</summary>
        public const string PythonOverridePrefKey = "Thesis.RetrainPythonPath";
        public const string LatestModelFileName = "ImitationAgent_discrete_latest.onnx";
        private const string ModelPrefix = "ImitationAgent_discrete_";
        private const double StaleLockHours = 6.0;
        private const double PollIntervalSeconds = 0.5;
        private const int FailureTailLines = 40;
        private const string LogTag = "[Retrain]";

        private static string SimulationRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "../../"));
        private static string TrainingDir => Path.Combine(SimulationRoot, "imitationagent_training");
        private static string PipelineDir => Path.Combine(TrainingDir, "training_pipeline");
        private static string EmbeddedDir => Path.Combine(TrainingDir, "python_embedded");
        private static string BootstrapMarkerPath => Path.Combine(EmbeddedDir, ".bootstrap.json");
        private static string RequirementsPath => Path.Combine(PipelineDir, "requirements.txt");
        private static string SplitsPath => Path.Combine(TrainingDir, "splits.json");
        private static string LockPath => Path.Combine(TrainingDir, "retrain.lock");
        public static string LogDirectory => Path.Combine(TrainingDir, "retrain_logs");
        private static string TrainOutputDir => Path.Combine(TrainingDir, "imitation_agent_model");
        private static string AssetsModelDir => Path.Combine(Application.dataPath, "ImitationModel");
        /// <summary>The file ImitationAgent watches; overwritten by every successful retrain.</summary>
        public static string LatestModelPath => Path.Combine(AssetsModelDir, LatestModelFileName);

        private static string SetupCommand =>
            $"cd \"{SimulationRoot}\" && /opt/homebrew/bin/python3 imitationagent_training/training_pipeline/bootstrap_env.py";

        [Serializable]
        private class LockInfo
        {
            public int pid;
            public string started_utc;
            public string reason;
            public string output_onnx;
            public string log_path;
            public string exit_path;
            public string script_path;
        }

        [Serializable]
        private class BootstrapMarker
        {
            public string python;
            public string python_version;
            public string requirements_sha256;
        }

        // Tail state of the run this domain is following (null = none). Reset by a domain reload; Poll re-attaches.
        private static LockInfo followed;
        private static long logOffset;
        private static string partialLine = "";
        private static DateTime nextPollUtc = DateTime.MinValue;

        // ------------------------------------------------------------------
        // Public API
        // ------------------------------------------------------------------

        /// <summary>
        /// Starts one full retrain in the background. Returns false (and logs why) if a retrain is already running or the
        /// Python environment is not set up. Call on the main thread.
        /// </summary>
        public static bool RequestFullRetrain(string reason)
        {
            LockInfo running = ReadLiveLock(out _);
            if (running != null)
            {
                Debug.LogWarning($"{LogTag} Retrain already running (pid {running.pid}, since {running.started_utc} UTC, " +
                                 $"'{running.reason}') - not starting another for '{reason}'.");
                return false;
            }

            string python = ResolvePython(out string envError);
            if (python == null)
            {
                Debug.LogError($"{LogTag} Not started ('{reason}'): {envError}");
                return false;
            }

            string stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            Directory.CreateDirectory(LogDirectory);
            Directory.CreateDirectory(TrainOutputDir);
            var info = new LockInfo
            {
                started_utc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                reason = reason,
                output_onnx = Path.Combine(TrainOutputDir, $"{ModelPrefix}{stamp}.onnx"),
                log_path = Path.Combine(LogDirectory, stamp + ".log"),
                exit_path = Path.Combine(LogDirectory, stamp + ".exit"),
            };

            var args = new List<string>
            {
                "train_2DCNN_discrete.py",
                "--recordings_dir", "../zarr_recordings",
                "--sources", "all",
                "--output_onnx", $"../imitation_agent_model/{ModelPrefix}{stamp}.onnx",
                "--reason", SafeArgument(reason),
            };
            if (File.Exists(SplitsPath))
            {
                args.AddRange(new[] { "--splits", "../splits.json", "--condition", "C" });
            }

            try
            {
                bool windows = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;
                info.script_path = Path.Combine(LogDirectory, stamp + (windows ? ".cmd" : ".sh"));
                File.WriteAllText(info.script_path, windows ? WindowsScript(python, args, info) : UnixScript(python, args, info));

                var startInfo = new ProcessStartInfo
                {
                    FileName = windows ? "cmd.exe" : "/bin/sh",
                    Arguments = windows ? $"/c \"{info.script_path}\"" : $"\"{info.script_path}\"",
                    WorkingDirectory = PipelineDir,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using (Process process = Process.Start(startInfo))
                {
                    info.pid = process.Id;
                    if (windows)
                    {
                        // Unix gets 'nice' in the script; on Windows the trainer inherits the wrapper's class if set in time.
                        try { process.PriorityClass = ProcessPriorityClass.BelowNormal; } catch (Exception) { }
                    }
                }
                File.WriteAllText(LockPath, JsonUtility.ToJson(info, true));
            }
            catch (Exception ex)
            {
                Debug.LogError($"{LogTag} Could not start the retrain process: {ex.Message}");
                return false;
            }

            Follow(info, 0);
            EnsurePump();
            Debug.Log($"{LogTag} Started full retrain (pid {info.pid}) - '{reason}'.\n" +
                      $"python: {python}\nargs: {string.Join(" ", args)}\nlog: {info.log_path}" +
                      (File.Exists(SplitsPath) ? "" : "\n(no splits.json - training on every human participant)"));
            return true;
        }

        public static bool IsRunning => ReadLiveLock(out _) != null;

        /// <summary>The interpreter and PYTHONPATH the retrain uses, for other Editor tools that run pipeline scripts.</summary>
        public static bool TryGetPythonEnvironment(out string python, out string pythonPath, out string error)
        {
            python = ResolvePython(out error);
            pythonPath = EmbeddedDir;
            return python != null;
        }

        /// <summary>Running/idle, the last log file and the last exported model, for Tools > Thesis > Show Retrain Status.</summary>
        public static string StatusReport()
        {
            var sb = new StringBuilder();
            LockInfo running = ReadLiveLock(out string staleNote);
            if (running != null)
            {
                TimeSpan elapsed = DateTime.UtcNow - ParseUtc(running.started_utc);
                sb.AppendLine($"RUNNING  pid {running.pid}, {elapsed:hh\\:mm\\:ss} elapsed, '{running.reason}'");
                sb.AppendLine($"  output: {running.output_onnx}");
            }
            else
            {
                sb.AppendLine("IDLE" + (staleNote != null ? $" ({staleNote})" : ""));
            }

            string lastLog = Directory.Exists(LogDirectory)
                ? Directory.GetFiles(LogDirectory, "*.log").OrderBy(f => f, StringComparer.Ordinal).LastOrDefault()
                : null;
            sb.AppendLine($"Last log: {lastLog ?? "-"}");

            string lastModel = Directory.Exists(AssetsModelDir)
                ? Directory.GetFiles(AssetsModelDir, ModelPrefix + "2*.onnx").OrderBy(f => f, StringComparer.Ordinal).LastOrDefault()
                : null;
            sb.AppendLine($"Last exported model: {(lastModel != null ? Path.GetFileName(lastModel) : "-")}");
            string sidecar = lastModel != null ? Path.ChangeExtension(lastModel, ".json") : null;
            if (sidecar != null && File.Exists(sidecar))
            {
                Match m = Regex.Match(File.ReadAllText(sidecar), "\"reason\":\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
                if (m.Success) sb.AppendLine($"  reason: {m.Groups[1].Value}");
            }
            sb.AppendLine($"Latest (hot-reloaded): {(File.Exists(LatestModelPath) ? File.GetLastWriteTime(LatestModelPath).ToString("yyyy-MM-dd HH:mm:ss") : "missing")}");

            string python = ResolvePython(out string envError);
            sb.Append(python != null ? $"Python: {python} + PYTHONPATH=python_embedded" : $"Python: NOT READY - {envError}");
            return sb.ToString();
        }

        // ------------------------------------------------------------------
        // Environment
        // ------------------------------------------------------------------

        /// <summary>
        /// The interpreter bootstrap_env.py installed python_embedded for (or the pref override), or null with the reason.
        /// Also refuses when python_embedded no longer matches requirements.txt. Never falls back to a system Python.
        /// </summary>
        private static string ResolvePython(out string error)
        {
            error = null;
            if (!File.Exists(BootstrapMarkerPath))
            {
                error = $"python_embedded is not set up ({BootstrapMarkerPath} missing). Run once:\n  {SetupCommand}";
                return null;
            }
            BootstrapMarker marker;
            try { marker = JsonUtility.FromJson<BootstrapMarker>(File.ReadAllText(BootstrapMarkerPath)); }
            catch (Exception ex)
            {
                error = $"{BootstrapMarkerPath} is unreadable ({ex.Message}). Re-run:\n  {SetupCommand} --force";
                return null;
            }
            if (marker == null || marker.requirements_sha256 != Sha256(RequirementsPath))
            {
                error = $"python_embedded is out of date with requirements.txt. Re-run:\n  {SetupCommand}";
                return null;
            }

            string overridePath = GetPref(PythonOverridePrefKey);
            string python = string.IsNullOrEmpty(overridePath) ? marker.python : overridePath;
            if (string.IsNullOrEmpty(python) || !File.Exists(python))
            {
                error = $"Python interpreter '{python}' does not exist" +
                        (string.IsNullOrEmpty(overridePath) ? $". Re-run:\n  {SetupCommand}" : $" (from the '{PythonOverridePrefKey}' pref).");
                return null;
            }
            return python;
        }

        private static string Sha256(string path)
        {
            if (!File.Exists(path)) return null;
            using (var sha = SHA256.Create())
            using (var fs = File.OpenRead(path))
            {
                return BitConverter.ToString(sha.ComputeHash(fs)).Replace("-", "").ToLowerInvariant();
            }
        }

        private static string GetPref(string key)
        {
#if UNITY_EDITOR
            return EditorPrefs.GetString(key, "");
#else
            return PlayerPrefs.GetString(key, "");
#endif
        }

        /// <summary>The reason goes through a shell script: keep it to characters that need no quoting anywhere.</summary>
        private static string SafeArgument(string value) => Regex.Replace(value ?? "", @"[^A-Za-z0-9 _.:,+\-]", "_");

        private static string UnixScript(string python, List<string> args, LockInfo info)
        {
            string Q(string s) => "'" + s.Replace("'", "'\\''") + "'";
            // nice: below-normal priority, so a following recording session keeps its frame rate.
            // The exit code is written via rename so the reader never sees a half-written file.
            return "#!/bin/sh\n" +
                   $"cd {Q(PipelineDir)} || exit 1\n" +
                   $"export PYTHONPATH={Q(EmbeddedDir)}\n" +
                   "export PYTHONUNBUFFERED=1\n" +
                   $"nice -n 10 {Q(python)} {string.Join(" ", args.Select(Q))} > {Q(info.log_path)} 2>&1\n" +
                   $"echo $? > {Q(info.exit_path + ".tmp")} && mv {Q(info.exit_path + ".tmp")} {Q(info.exit_path)}\n";
        }

        private static string WindowsScript(string python, List<string> args, LockInfo info)
        {
            string Q(string s) => "\"" + s + "\"";
            return "@echo off\r\n" +
                   $"cd /d {Q(PipelineDir)} || exit /b 1\r\n" +
                   $"set \"PYTHONPATH={EmbeddedDir}\"\r\n" +
                   "set PYTHONUNBUFFERED=1\r\n" +
                   $"{Q(python)} {string.Join(" ", args.Select(Q))} > {Q(info.log_path)} 2>&1\r\n" +
                   $"echo %ERRORLEVEL%> {Q(info.exit_path + ".tmp")}\r\n" +
                   $"move /y {Q(info.exit_path + ".tmp")} {Q(info.exit_path)} > nul\r\n";
        }

        // ------------------------------------------------------------------
        // Lock
        // ------------------------------------------------------------------

        /// <summary>The lock of a run that is still going, else null. A stale lock (dead pid, or older than 6 h) counts as none.</summary>
        private static LockInfo ReadLiveLock(out string staleNote)
        {
            staleNote = null;
            LockInfo info = ReadLock();
            if (info == null) return null;
            if (File.Exists(info.exit_path)) return null; // finished, not yet finalized
            if (!ProcessAlive(info.pid))
            {
                staleNote = $"stale lock: pid {info.pid} is gone";
                return null;
            }
            if ((DateTime.UtcNow - ParseUtc(info.started_utc)).TotalHours > StaleLockHours)
            {
                staleNote = $"stale lock: older than {StaleLockHours} h";
                return null;
            }
            return info;
        }

        private static LockInfo ReadLock()
        {
            try
            {
                return File.Exists(LockPath) ? JsonUtility.FromJson<LockInfo>(File.ReadAllText(LockPath)) : null;
            }
            catch (Exception)
            {
                return null;
            }
        }

        private static bool ProcessAlive(int pid)
        {
            try
            {
                using (Process p = Process.GetProcessById(pid)) return !p.HasExited;
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static DateTime ParseUtc(string iso) =>
            DateTime.TryParse(iso, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out DateTime t)
                ? t.ToUniversalTime() : DateTime.MinValue;

        // ------------------------------------------------------------------
        // Main-thread polling: tail the log, finalize on exit
        // ------------------------------------------------------------------

        private static void Follow(LockInfo info, long offset)
        {
            followed = info;
            logOffset = offset;
            partialLine = "";
        }

        /// <summary>Main thread only (EditorApplication.update in the Editor, RetrainPipelinePump in a player).</summary>
        internal static void Poll()
        {
            if (DateTime.UtcNow < nextPollUtc) return;
            nextPollUtc = DateTime.UtcNow.AddSeconds(PollIntervalSeconds);

            if (followed == null)
            {
                if (!File.Exists(LockPath)) return;
                LockInfo info = ReadLock();
                if (info == null) return;
                // Re-attach after a domain reload / Editor restart. Output from before is in the log file, not repeated here.
                long offset = File.Exists(info.log_path) ? new FileInfo(info.log_path).Length : 0;
                Follow(info, offset);
                Debug.Log($"{LogTag} Following retrain pid {info.pid} ('{info.reason}'), log {info.log_path}.");
            }

            TailLog(false);

            bool exited = File.Exists(followed.exit_path);
            if (!exited)
            {
                if (ProcessAlive(followed.pid)) return;
                // The wrapper writes the exit file last; give a just-finished wrapper one more poll to do so.
                System.Threading.Thread.Sleep(200);
                exited = File.Exists(followed.exit_path);
            }
            TailLog(true);
            FinishRun(followed, exited ? ReadExitCode(followed.exit_path) : (int?)null);
            followed = null;
        }

        private static void TailLog(bool flush)
        {
            if (!File.Exists(followed.log_path)) return;
            try
            {
                using (var fs = new FileStream(followed.log_path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
                {
                    if (fs.Length <= logOffset) return;
                    fs.Seek(logOffset, SeekOrigin.Begin);
                    var buffer = new byte[fs.Length - logOffset];
                    int read = fs.Read(buffer, 0, buffer.Length);
                    logOffset += read;
                    partialLine += Encoding.UTF8.GetString(buffer, 0, read);
                }
            }
            catch (IOException)
            {
                return;
            }

            string[] lines = partialLine.Replace("\r\n", "\n").Split('\n');
            partialLine = flush ? "" : lines[lines.Length - 1];
            int complete = flush ? lines.Length : lines.Length - 1;
            for (int i = 0; i < complete; i++)
            {
                if (lines[i].Length > 0) LogPythonLine(lines[i]);
            }
        }

        private static void LogPythonLine(string line)
        {
            string msg = $"{LogTag} {line}";
            if (line.Contains("[ERROR]") || line.StartsWith("Traceback") || Regex.IsMatch(line, @"^\w*(Error|Exception)\b"))
                Debug.LogError(msg);
            else if (line.Contains("WARNING") || line.Contains("Warning:"))
                Debug.LogWarning(msg);
            else
                Debug.Log(msg);
        }

        private static int? ReadExitCode(string path)
        {
            try
            {
                return int.TryParse(File.ReadAllText(path).Trim(), out int code) ? code : (int?)null;
            }
            catch (IOException)
            {
                return null;
            }
        }

        private static void FinishRun(LockInfo info, int? exitCode)
        {
            try
            {
                if (exitCode == 0 && File.Exists(info.output_onnx))
                {
                    InstallModel(info);
                }
                else
                {
                    string why = exitCode.HasValue ? $"exit code {exitCode}" : "the process ended without an exit code (killed?)";
                    if (exitCode == 0) why = $"exit code 0 but no {info.output_onnx}";
                    Debug.LogError($"{LogTag} Retrain '{info.reason}' FAILED ({why}); the previous model stays in use. " +
                                   $"Last lines of {info.log_path}:\n{Tail(info.log_path, FailureTailLines)}");
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"{LogTag} Finalizing retrain '{info.reason}' failed: {ex}");
            }
            finally
            {
                TryDelete(LockPath);
                TryDelete(info.exit_path);
                TryDelete(info.script_path);
            }
        }

        /// <summary>
        /// Copies the timestamped model + sidecar into Assets/ImitationModel/ (kept, so eval runs can point at a fixed file)
        /// and over ImitationAgent_discrete_latest.onnx/.json, then re-imports so a hot-reloading ImitationAgent picks it up.
        /// </summary>
        private static void InstallModel(LockInfo info)
        {
            Directory.CreateDirectory(AssetsModelDir);
            string name = Path.GetFileName(info.output_onnx);
            string sidecar = Path.ChangeExtension(info.output_onnx, ".json");
            File.Copy(info.output_onnx, Path.Combine(AssetsModelDir, name), true);
            if (File.Exists(sidecar)) File.Copy(sidecar, Path.Combine(AssetsModelDir, Path.GetFileName(sidecar)), true);

            ReplaceWith(info.output_onnx, LatestModelPath);
            if (File.Exists(sidecar)) ReplaceWith(sidecar, Path.ChangeExtension(LatestModelPath, ".json"));

#if UNITY_EDITOR
            AssetDatabase.Refresh(); // main thread: Poll runs from EditorApplication.update
#endif
            Debug.Log($"{LogTag} Retrain '{info.reason}' finished: Assets/ImitationModel/{name} (also as {LatestModelFileName}).");
        }

        /// <summary>Copy to a hidden temp file next to the target (Unity ignores dot files), then swap it in.</summary>
        private static void ReplaceWith(string source, string target)
        {
            string temp = Path.Combine(Path.GetDirectoryName(target), "." + Path.GetFileName(target) + ".tmp");
            File.Copy(source, temp, true);
            if (File.Exists(target)) File.Delete(target);
            File.Move(temp, target);
            File.SetLastWriteTime(target, DateTime.Now); // ImitationAgent compares write times; a copy may keep the source's
        }

        private static string Tail(string path, int lines)
        {
            try
            {
                if (!File.Exists(path)) return "(no log file)";
                string[] all;
                using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var reader = new StreamReader(fs))
                {
                    all = reader.ReadToEnd().Replace("\r\n", "\n").Split('\n');
                }
                return string.Join("\n", all.Skip(Math.Max(0, all.Length - lines)));
            }
            catch (IOException ex)
            {
                return $"(could not read log: {ex.Message})";
            }
        }

        private static void TryDelete(string path)
        {
            try
            {
                if (!string.IsNullOrEmpty(path) && File.Exists(path)) File.Delete(path);
            }
            catch (IOException) { }
        }

        // ------------------------------------------------------------------
        // Pump
        // ------------------------------------------------------------------

#if UNITY_EDITOR
        // The Editor polls from EditorApplication.update rather than a MonoBehaviour, because a retrain started at the end of
        // a session keeps running after Play mode ends. Runs after every domain reload, so it also re-attaches to a running
        // retrain and finalizes one that finished while nobody was watching.
        [InitializeOnLoadMethod]
        private static void InstallEditorPump()
        {
            EditorApplication.update -= Poll;
            EditorApplication.update += Poll;
        }

        private static void EnsurePump() { }
#else
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void EnsurePump()
        {
            if (UnityEngine.Object.FindObjectOfType<RetrainPipelinePump>() != null) return;
            var go = new GameObject("RetrainPipelinePump") { hideFlags = HideFlags.HideAndDontSave };
            UnityEngine.Object.DontDestroyOnLoad(go);
            go.AddComponent<RetrainPipelinePump>();
        }
#endif
    }

    /// <summary>Player builds only: calls RetrainPipeline.Poll on the main thread.</summary>
    public class RetrainPipelinePump : MonoBehaviour
    {
        private void Update() => RetrainPipeline.Poll();
    }
}
