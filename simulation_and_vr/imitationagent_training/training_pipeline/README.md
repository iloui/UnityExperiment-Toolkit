# Training pipeline

## Python environment (once, and whenever `requirements.txt` changes)

Unity trains with the packages in `imitationagent_training/python_embedded/` and never installs anything itself.
Set them up by hand with the interpreter the training should use (Python >= 3.11; the compiled packages only load in
that minor version):

```bash
cd simulation_and_vr
/opt/homebrew/bin/python3 imitationagent_training/training_pipeline/bootstrap_env.py
```

This does a clean `pip install --target python_embedded -r requirements.txt` and writes
`python_embedded/.bootstrap.json` (interpreter path + requirements hash). Unity refuses to start a retrain when that
marker is missing or `requirements.txt` changed since. `--force` reinstalls anyway. To use a different interpreter
than the recorded one, set the EditorPrefs key `Thesis.RetrainPythonPath`.

Running a script by hand with the same environment (from `training_pipeline/`):

```bash
PYTHONPATH=../python_embedded /opt/homebrew/bin/python3 train_2DCNN_discrete.py --sources all
```

## Automatic retrain (`RetrainPipeline.cs`)

- A finished `ParticipantSession` starts one full retrain from scratch (`ParticipantSessionConfig.retrainAfterSession`,
  on by default; an aborted session only with `retrainOnAbort`). Recording single episodes and the bot/eval harnesses
  never start Python.
- `Tools > Thesis > Retrain Discrete Model Now` starts one by hand (e.g. after a bot recording run);
  `Tools > Thesis > Show Retrain Status` shows whether one is running, the last log and the last exported model.
- Command: `train_2DCNN_discrete.py --recordings_dir ../zarr_recordings --sources all
  --output_onnx ../imitation_agent_model/ImitationAgent_discrete_<yyyyMMdd_HHmmss>.onnx --reason "<why>"`, plus
  `--splits ../splits.json --condition C` when `imitationagent_training/splits.json` exists, so held-out participants
  are never trained on. `dryrun_*` participants are always skipped.
- Only one at a time: `imitationagent_training/retrain.lock` (pid + start time; ignored when the pid is gone or it is
  older than 6 h). Runs at `nice 10`.
- Every output line is shown in the Unity console with a `[Retrain]` prefix and stored in
  `imitationagent_training/retrain_logs/<timestamp>.log`. A run keeps going through Play-mode changes, script reloads
  and Editor restarts; Unity re-attaches from the lock file.
- On success the timestamped ONNX + `.json` sidecar (sources, participants, reason, held-out list) are copied to
  `unity_files/Assets/ImitationModel/` and over `ImitationAgent_discrete_latest.onnx/.json`, which a running
  `ImitationAgent` hot-reloads (Editor only). EvalHarness turns hot-reload off and should point at a timestamped file.
  On failure the previous model stays and the log tail is printed.

## splits.json

```json
{ "held_out": ["P09", "P10"], "conditions": { "B": ["P01", "P02", "P03"] } }
```

`held_out` participants are never trained on. Without an entry in `conditions`, condition A uses no humans and C
every human not held out; B must be listed.
