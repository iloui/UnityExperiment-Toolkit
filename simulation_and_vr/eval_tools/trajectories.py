#!/usr/bin/env python3
"""Every evaluation source as one trajectory table (data_contract.TRAJECTORY_COLUMNS + META_COLUMNS).

Sources: EvalHarness runs (ImitationAgent under conditions A/B/C, the ShortestPath baseline) and recorded episodes
(held-out humans; synthetic bot episodes load the same way). Each loader returns a pandas DataFrame with exactly
TRAJECTORY_COLUMNS followed by META_COLUMNS, the same dtypes, and one row per SAMPLING_INTERVAL_S of sim time:

  t                 seconds since the episode's first sample, exactly k * 0.1
  x, y, z           eye/camera world position (Unity metres, y up)
  heading_deg       yaw in [0, 360), clockwise from above, 0 = +z
  pitch_deg         + = looking down
  speed_mps         horizontal distance moved since the previous row / 0.1 s (0 in the first row)
  yaw_rate_deg_s    wrapped heading change since the previous row / 0.1 s, + = clockwise (0 in the first row)
  phase             0 = walking to a POI, 1 = walking to the end
  pois_reached      POIs reached so far

  source            "imitation" | "shortest_path" | "human" | "synthetic"
  condition         "A" | "B" | "C" for imitation (EvalHarness.condition), else None
  layout            Unity scene name (the same for an eval run and the episodes recorded in that scene)
  participant       manifest participant_id for recordings, else None
  episode           "config<C>_agent<A>" for eval runs, manifest episode_id for recordings

Both kinds of input are resampled onto the exact 0.1 s grid (EvalHarness writes a row on the first frame past each grid
point; ParticipantRecorder's WaitForSeconds sampling drifts by up to a frame per sample), and speed_mps / yaw_rate_deg_s
are then recomputed here from the resampled pose for every source, so they are one definition, not two. Positions and
pitch are interpolated linearly, heading along the shorter arc; phase and pois_reached hold the previous sample.
For recordings, pois_reached is derived from goal_distance: at every sample it equals the distance from the camera to
exactly one of the manifest's planned targets (POIs in order, then end), which identifies the current target.

Usage:
  python trajectories.py EVAL_RUN_DIR|EPISODE_DIR [...]   # prints one line per loaded episode
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imitationagent_training" / "training_pipeline"))
from data_contract import (  # noqa: E402
    PHASE_TO_END, PHASE_TO_POI, POSE_COLUMNS, SAMPLING_INTERVAL_S, TRAJECTORY_COLUMNS, TRAJECTORY_CSV_DELIMITER,
    check_manifest,
)

META_COLUMNS = ("source", "condition", "layout", "participant", "episode")
SOURCE_IMITATION = "imitation"
SOURCE_SHORTEST_PATH = "shortest_path"
_AGENT_KIND_TO_SOURCE = {"Imitation": SOURCE_IMITATION, "ShortestPath": SOURCE_SHORTEST_PATH}

_FLOAT_COLUMNS = ("t", "x", "y", "z", "heading_deg", "pitch_deg", "speed_mps", "yaw_rate_deg_s")
_INT_COLUMNS = ("phase", "pois_reached")
# A recorded goal_distance must match one planned target this closely (float32 positions, JSON round trip).
_TARGET_MATCH_TOLERANCE_M = 0.05


def _resampled(t, x, y, z, heading, pitch, pois_reached, poi_count: int, meta: dict) -> pd.DataFrame:
    """Resample one episode onto the k * SAMPLING_INTERVAL_S grid and derive speed, yaw rate and phase (module doc)."""
    t = np.asarray(t, dtype=np.float64)
    t = t - t[0]
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{meta['episode']}: timestamps are not strictly increasing")
    grid = np.arange(int(np.floor(t[-1] / SAMPLING_INTERVAL_S + 1e-6)) + 1) * SAMPLING_INTERVAL_S

    def lerp(v):
        return np.interp(grid, t, np.asarray(v, dtype=np.float64))

    heading_grid = np.mod(lerp(np.unwrap(np.asarray(heading, dtype=np.float64), period=360.0)), 360.0)
    x_grid, z_grid = lerp(x), lerp(z)
    held = np.searchsorted(t, grid + 1e-9, side="right") - 1
    pois_grid = np.asarray(pois_reached, dtype=np.int64)[held]

    speed = np.zeros_like(grid)
    yaw_rate = np.zeros_like(grid)
    speed[1:] = np.hypot(np.diff(x_grid), np.diff(z_grid)) / SAMPLING_INTERVAL_S
    yaw_rate[1:] = (np.mod(np.diff(heading_grid) + 180.0, 360.0) - 180.0) / SAMPLING_INTERVAL_S

    df = pd.DataFrame({
        "t": grid, "x": x_grid, "y": lerp(y), "z": z_grid, "heading_deg": heading_grid, "pitch_deg": lerp(pitch),
        "speed_mps": speed, "yaw_rate_deg_s": yaw_rate,
        "phase": np.where(pois_grid >= poi_count, PHASE_TO_END, PHASE_TO_POI).astype(np.int64),
        "pois_reached": pois_grid,
    })
    for key in META_COLUMNS:
        df[key] = pd.Series([meta[key]] * len(df), dtype=object)
    assert tuple(df.columns) == TRAJECTORY_COLUMNS + META_COLUMNS
    return df


def _empty() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=np.float64) for c in _FLOAT_COLUMNS})
    for c in _INT_COLUMNS:
        df[c] = pd.Series(dtype=np.int64)
    for c in META_COLUMNS:
        df[c] = pd.Series(dtype=object)
    return df[list(TRAJECTORY_COLUMNS + META_COLUMNS)]


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else _empty()


# ----------------------------------------------------------------------------------------------------------------------
# EvalHarness runs
# ----------------------------------------------------------------------------------------------------------------------

def load_eval_run(run_dir: str | Path) -> pd.DataFrame:
    """All finished episodes of one EvalHarness run (harness_version 7: traj_*.csv listed in episodes.csv), concatenated.
    Episodes with fewer than two trajectory rows (e.g. agent_never_started) are skipped. Version 6 runs are refused: they
    spawned the agent facing the goal in a straight line, not along the NavMesh path the participants started on."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    if cfg.get("harness_version") != "7":
        raise ValueError(f"{run_dir}: harness_version is {cfg.get('harness_version')!r}, expected '7'. Re-run the evaluation.")
    source = _AGENT_KIND_TO_SOURCE[cfg["agent_kind"]]
    condition = cfg.get("condition") or None
    if source == SOURCE_IMITATION and condition is None:
        raise ValueError(f"{run_dir}: Imitation run without a condition in run_config.json")

    frames = []
    with open(run_dir / "episodes.csv", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            traj = pd.read_csv(run_dir / row["trajectory_file"], sep=TRAJECTORY_CSV_DELIMITER)
            if list(traj.columns) != list(TRAJECTORY_COLUMNS):
                raise ValueError(f"{run_dir / row['trajectory_file']}: columns {list(traj.columns)}, expected {list(TRAJECTORY_COLUMNS)}")
            if len(traj) < 2:
                continue
            meta = {"source": source, "condition": condition, "layout": cfg["scene"], "participant": None,
                    "episode": f"config{row['config_id']}_agent{row['agent_id']}"}
            frames.append(_resampled(traj["t"], traj["x"], traj["y"], traj["z"], traj["heading_deg"], traj["pitch_deg"],
                                     traj["pois_reached"], int(row["num_pois_total"]), meta))
    return _concat(frames)


# ----------------------------------------------------------------------------------------------------------------------
# Recorded episodes (humans, synthetic bot)
# ----------------------------------------------------------------------------------------------------------------------

def _pois_reached_from_goal_distance(pose_xyz: np.ndarray, goal_distance: np.ndarray, task: dict, name: str) -> np.ndarray:
    """Index of the target each sample's goal_distance was measured to (POIs in order, then end), capped at the POI count."""
    targets = np.array([[p["x"], p["y"], p["z"]] for p in task["pois"] + [task["end"]]], dtype=np.float64)
    dist = np.linalg.norm(pose_xyz[:, None, :] - targets[None, :, :], axis=2)
    err = np.abs(dist - goal_distance[:, None])
    current = np.maximum.accumulate(np.argmin(err, axis=1))  # targets are visited in order, never revisited
    worst = err[np.arange(len(current)), current].max()
    if worst > _TARGET_MATCH_TOLERANCE_M:
        raise ValueError(f"{name}: goal_distance matches no planned target (off by {worst:.3f} m) - task in the manifest is "
                         "not the one that was recorded")
    return np.minimum(current, len(task["pois"]))


def load_human_episode(zarr_dir: str | Path) -> pd.DataFrame:
    """One recorded episode (an episode folder or its dataset.zarr). Works for synthetic episodes too; 'source' is the
    manifest's."""
    import zarr

    zarr_dir = Path(zarr_dir)
    if zarr_dir.name != "dataset.zarr":
        zarr_dir = zarr_dir / "dataset.zarr"
    root = zarr.open_group(str(zarr_dir), mode="r")
    manifest = json.loads(root.attrs["manifest"])
    check_manifest(manifest, str(zarr_dir))

    pose = np.asarray(root["pose"][:], dtype=np.float64)
    if pose.shape[1:] != (len(POSE_COLUMNS),):
        raise ValueError(f"{zarr_dir}: pose has shape {pose.shape}, expected (T, {len(POSE_COLUMNS)})")
    goal_distance = np.asarray(root["goal_distance"][:], dtype=np.float64)
    timestamp = np.asarray(root["timestamp"][:], dtype=np.float64)
    pois_reached = _pois_reached_from_goal_distance(pose[:, :3], goal_distance, manifest["task"], str(zarr_dir))

    meta = {"source": manifest["source"], "condition": None, "layout": manifest["scene_name"],
            "participant": manifest["participant_id"], "episode": manifest["episode_id"]}
    return _resampled(timestamp, pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3], pose[:, 4], pois_reached,
                      len(manifest["task"]["pois"]), meta)


def load_recordings(recordings_dir: str | Path, source: str | None = "human", layout: str | None = None,
                    participants: set[str] | None = None) -> pd.DataFrame:
    """Every ingested episode (<recordings_dir>/*/dataset.zarr), optionally filtered by manifest source, scene and
    participant (e.g. the held-out test participants), concatenated."""
    frames = []
    for zarr_dir in sorted(Path(recordings_dir).glob("*/dataset.zarr")):
        manifest = json.loads((zarr_dir.parent / "manifest.json").read_text(encoding="utf-8"))
        if ((source is not None and manifest.get("source") != source)
                or (layout is not None and manifest.get("scene_name") != layout)
                or (participants is not None and manifest.get("participant_id") not in participants)):
            continue
        frames.append(load_human_episode(zarr_dir))
    return _concat(frames)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        path = Path(arg)
        df = load_eval_run(path) if (path / "run_config.json").exists() else load_human_episode(path)
        for (source, condition, layout, participant, episode), ep in df.groupby(list(META_COLUMNS), dropna=False, sort=False):
            who = condition if source == SOURCE_IMITATION else participant
            print(f"{source:13s} {who or '-':10s} {layout}  {episode}: {len(ep)} rows, {ep['t'].iloc[-1]:.1f} s, "
                  f"mean speed {ep['speed_mps'].mean():.2f} m/s, pois_reached {ep['pois_reached'].iloc[-1]}, "
                  f"final phase {ep['phase'].iloc[-1]}")


if __name__ == "__main__":
    main()
