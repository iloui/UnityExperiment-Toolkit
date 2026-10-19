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
PYTHONPATH=../python_embedded /opt/homebrew/bin/python3 train_2DCNN_discrete.py --condition C --fold 1
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
  --condition C --fold 1 --output_onnx ../imitation_agent_model/ImitationAgent_discrete_<yyyyMMdd_HHmmss>.onnx --reason "<why>"`.
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

## Experiment: folds x tasks, conditions A/B/C (methodology 3.8)

Every human manifest carries its design labels (`trial`: fold, task_id, task_role, c_excluded, attempt_index; written
by `ParticipantSession`). Once every participant is recorded, collect and check them once (from `training_pipeline/`):

```bash
.venv/bin/python make_splits.py
```

`../splits.json` = `{"design": "folds_x_tasks_v1", "folds": {"1": [odd IDs], "2": [even IDs]}, "c_excluded":
{scene: task_id}, "test_episodes": {"1": [...], "2": [...]}, "episodes_per_participant": {...}, "repeats": [...]}`.
It refuses to write if a participant is in the wrong fold, a task was walked 0 or 2+ times, or the episodes disagree
on (or with `eval_tools/layouts/*_human_tasks.json` about) the C-excluded task (`--allow_incomplete` overrides).
`dryrun_*` participants are ignored; an existing file is only overwritten with `--force`.

| condition | humans trained on (fold f) | human tasks per layout | human share per batch |
|---|---|---|---|
| A | none (synthetic only, plain shuffling, no fold) | - | 0 |
| B | fold f | the three training tasks | `--human_share` (0.5) |
| C | fold f | the test task + the two training tasks not `c_excluded` | `--human_share` (0.5) |

The other fold is never loaded; its test-task walks (`test_episodes`) are the model's human reference. B and C have the
same participants, the same validation participant (per seed) and the same number of episodes: they differ only in
whether the agent saw humans walk the evaluated route (A->B transfer, B->C memorisation).

```bash
.venv/bin/python train_2DCNN_discrete.py --condition B --fold 1 --seed 1   # -> ../imitation_agent_model/ImitationAgent_B_proprio2_f1_s1.onnx
.venv/bin/python train_2DCNN_discrete.py --condition C --fold 1 --seed 1   # -> ..._C_proprio2_f1_s1.onnx
.venv/bin/python check_splits.py --seed 1 ../imitation_agent_model/ImitationAgent_*_s1.json
```

- Synthetic data: `--synthetic_sources` (default `synthetic_rrw_v3`; `all` = every `synthetic*` participant id).
- Validation: human episodes are split leave-k-participants-out (`--val_participants`, default 1) from the fold's
  participants, so B and C train on 3 participants x 9 episodes and validate on 1 x 9; synthetic episodes by
  `--val_ratio` per source. Condition A has no human validation frames; its checkpoint is selected on synthetic frames.
- The sidecar `.json` next to each model records condition, fold, seed, git revision, held-out / train / validation
  participants, the C-excluded tasks, human episodes per layout:task, and the exact train and validation episode lists.
- `check_splits.py` fails (exit 1) if a recorded participant is in no fold, if any fold/condition selection (or a given
  sidecar's episode lists) contains the other fold, a test-task walk in B or an excluded task in C, or if B and C of a
  fold differ in participants, validation participant or human episode count.
