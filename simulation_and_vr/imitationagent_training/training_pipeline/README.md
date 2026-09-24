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
PYTHONPATH=../python_embedded /opt/homebrew/bin/python3 train_2DCNN_discrete.py --condition C
```

Modules: `model.py` (CNN trunk, discrete head, ONNX export), `dataset.py` (ingest, manifest checks,
`SingleEpisodeDataset`), `train_2DCNN_discrete.py` (training), `make_splits.py` / `check_splits.py` (participant splits).

## Automatic retrain (`RetrainPipeline.cs`)

- A finished `ParticipantSession` starts one full retrain from scratch (`ParticipantSessionConfig.retrainAfterSession`,
  on by default; an aborted session only with `retrainOnAbort`). Recording single episodes and the bot/eval harnesses
  never start Python.
- `Tools > Thesis > Retrain Discrete Model Now` starts one by hand (e.g. after a bot recording run);
  `Tools > Thesis > Show Retrain Status` shows whether one is running, the last log and the last exported model.
- Command: `train_2DCNN_discrete.py --recordings_dir ../zarr_recordings --synthetic_sources all --splits ../splits.json
  --condition C --output_onnx ../imitation_agent_model/ImitationAgent_discrete_<yyyyMMdd_HHmmss>.onnx --reason "<why>"`.
  Not started while `imitationagent_training/splits.json` is missing, so held-out participants are never trained on.
- Only one at a time: `imitationagent_training/retrain.lock` (pid + start time; ignored when the pid is gone or it is
  older than 6 h). Runs at `nice 10`.
- Every output line is shown in the Unity console with a `[Retrain]` prefix and stored in
  `imitationagent_training/retrain_logs/<timestamp>.log`. A run keeps going through Play-mode changes, script reloads
  and Editor restarts; Unity re-attaches from the lock file.
- On success the timestamped ONNX + `.json` sidecar (sources, participants, reason, held-out list) are copied to
  `unity_files/Assets/ImitationModel/` and over `ImitationAgent_discrete_latest.onnx/.json`, which a running
  `ImitationAgent` hot-reloads (Editor only). EvalHarness turns hot-reload off and should point at a timestamped file.
  On failure the previous model stays and the log tail is printed.

## Experiment: participant splits and conditions A/B/C

Once every participant is recorded, write the splits once (from `training_pipeline/`):

```bash
.venv/bin/python make_splits.py --n_heldout 4 --n_subset_B 4 --seed 1
```

`../splits.json` = `{"heldout": [...], "train": [...], "subset_B": [...], "seed": 1, "episodes_per_participant": {...}}`.
Human participants are sorted, shuffled with `--seed` and cut by the counts, so the file is reproducible; `dryrun_*`
participants are ignored; an existing file is only overwritten with `--force`. `subset_B` is a subset of `train`;
`heldout` participants are never loaded for training or validation and are the human baseline in EvalHarness.

| condition | humans trained on | human share per batch |
|---|---|---|
| A | none (synthetic only, plain shuffling) | 0 |
| B | `subset_B` | `--human_share` (0.5) |
| C | `train` | `--human_share` (0.5) |

```bash
.venv/bin/python train_2DCNN_discrete.py --condition A --seed 1   # -> ../imitation_agent_model/ImitationAgent_A_s1.onnx
.venv/bin/python train_2DCNN_discrete.py --condition B --seed 1
.venv/bin/python train_2DCNN_discrete.py --condition C --seed 1
.venv/bin/python check_splits.py --seed 1 ../imitation_agent_model/ImitationAgent_*_s1.json
```

- Synthetic data: `--synthetic_sources` (default `synthetic_rrw_v3`; `all` = every `synthetic*` participant id).
- Validation: human episodes are split leave-k-participants-out (`--val_participants`, default 1) from the condition's own
  humans, so B trains on `n_subset_B - 1` participants; synthetic episodes by `--val_ratio` per source. Condition A has
  no human validation frames, so its checkpoint is selected on synthetic frames only.
- The sidecar `.json` next to each model records condition, seed, git revision, the held-out / condition / train /
  validation participants, episode and frame counts, and the exact train and validation episode lists.
- `check_splits.py` fails (exit 1) if splits.json misses a recorded participant, if any condition's selection contains a
  held-out participant, or if a given sidecar's episode lists do.

**B vs C.** The human share of every batch is fixed (`--human_share`), whatever the number of human frames. B and C
therefore give the human data the same total weight in the loss; C spreads it over more participants (more distinct
frames, each drawn less often). The B/C difference measures human *diversity*, not how strongly the model is pulled
towards human behaviour. The A vs B/C difference mixes both.
