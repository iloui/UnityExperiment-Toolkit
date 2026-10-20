#!/usr/bin/env python3
"""Calibration check: a synthetic bot's recordings next to the recorded participants, on the statistics the bot is tuned
by (RandomWalkBotSettings.RecalibratedV4). Same 0.1 s grid (trajectories.load_human_episode), stop threshold
(0.65 m/s) and wall distance (layout_metrics, body centre to the nearest wall) as compare_conditions.py.

  stop share            share of samples below the stop speed
  start pause           length of the standing bout the episode starts with
  mid-walk stops        standing bouts after the start pause: rate per minute of walking, length, share within 0.6 m of a wall
  wall distance         while walking
  yaw exactly 0         share of samples without any heading change

Raw episodes (Unity's .bin files) are converted to dataset.zarr first, exactly as training does.
Humans are every participant_id matching --humans (default P + digits, dry runs excluded).

Usage (from simulation_and_vr/):
  imitationagent_training/training_pipeline/.venv/bin/python eval_tools/compare_bot_to_humans.py --bots synthetic_rrw_v3 synthetic_rrw_v4 synthetic_rrw_v5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from layout_metrics import LAYOUTS_DIR, load_layout
from trajectories import load_human_episode
from dataset import ensure_all_episodes_ingested  # trajectories put training_pipeline on sys.path

HERE = Path(__file__).resolve().parent
DEFAULT_RECORDINGS = HERE.parent / "imitationagent_training" / "zarr_recordings"
STOP_SPEED_MPS = 0.65  # compare_conditions.STOP_SPEED_MPS
NEAR_WALL_M = 0.6


def bouts(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of consecutive True runs."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def episode_stats(df, layout) -> dict:
    speed = df["speed_mps"].to_numpy()[1:]
    xz = df[["x", "z"]].to_numpy()[1:]
    yaw = df["yaw_rate_deg_s"].to_numpy()[1:]
    stopped = speed < STOP_SPEED_MPS
    wall = layout.wall_distance(xz) if layout is not None else np.full(len(xz), np.nan)
    runs = bouts(stopped)
    start = runs[0] if runs and runs[0][0] == 0 else None
    mid = [b for b in runs if b != start and b[1] < len(stopped)]  # the bout at the very end is the arrival
    walk_min = (~stopped).sum() * 0.1 / 60.0
    return {
        "stop_share": stopped.mean(),
        "start_pause_s": (start[1] - start[0]) * 0.1 if start else 0.0,
        "mid_stops_per_walk_min": len(mid) / walk_min if walk_min > 0 else np.nan,
        "mid_stop_s": [(b - a) * 0.1 for a, b in mid],
        "mid_stop_near_wall": [float(np.nanmin(wall[a:b])) < NEAR_WALL_M for a, b in mid],
        "wall_walking_m": wall[~stopped],
        "yaw_zero": np.abs(yaw) < 1e-3,
    }


def episode_stats_placeholder() -> dict:
    """One-sample stand-in so an empty group still yields the full row set."""
    return {"stop_share": 0.0, "start_pause_s": 0.0, "mid_stops_per_walk_min": 0.0, "mid_stop_s": [0.0],
            "mid_stop_near_wall": [False], "wall_walking_m": np.zeros(1), "yaw_zero": np.zeros(1, dtype=bool)}


def load_group(recordings: Path, keep, task_kind: str | None = None) -> tuple[list[dict], set[str]]:
    """Episodes whose participant_id passes `keep`; with task_kind, only synthetic episodes of that synthetic_task_kind
    ("random" / "hidden_goal"; episodes recorded before participant_recorder_v5 have none and are dropped)."""
    layouts, stats, scenes = {}, [], set()
    for episode in sorted(recordings.iterdir()):
        manifest = episode / "manifest.json"
        if not manifest.exists():
            continue
        m = json.loads(manifest.read_text())
        if not keep(m.get("participant_id", "")):
            continue
        if task_kind and m.get("source") != "human" and m.get("synthetic_task_kind", "") != task_kind:
            continue
        try:
            df = load_human_episode(episode)
        except Exception as e:  # noqa: BLE001 - report and carry on, one broken episode should not stop the check
            print(f"  skipped {episode.name}: {e}")
            continue
        scene = df["layout"].iloc[0]
        if scene not in layouts:
            layouts[scene] = load_layout(scene) if (LAYOUTS_DIR / f"{scene}.npz").exists() else None
        scenes.add(scene)
        stats.append(episode_stats(df, layouts[scene]))
    return stats, scenes


def summary(stats: list[dict]) -> dict[str, float]:
    if not stats:
        return {k: (0 if k == "episodes" else np.nan) for k in summary([episode_stats_placeholder()])}
    cat =lambda k: np.concatenate([np.asarray(s[k], dtype=float) for s in stats]) if stats else np.zeros(0)  # noqa: E731
    wall = cat("wall_walking_m")
    wall = wall[~np.isnan(wall)]
    stops = cat("mid_stop_s")
    return {
        "episodes": len(stats),
        "stop share": np.mean([s["stop_share"] for s in stats]),
        "start pause median (s)": np.median([s["start_pause_s"] for s in stats]),
        "mid-walk stops / walk-min (median)": np.nanmedian([s["mid_stops_per_walk_min"] for s in stats]),
        "mid-walk stop length median (s)": np.median(stops) if len(stops) else np.nan,
        f"mid-walk stops < {NEAR_WALL_M} m from wall": np.mean(cat("mid_stop_near_wall")) if len(stops) else np.nan,
        "wall distance walking median (m)": np.median(wall) if len(wall) else np.nan,
        "wall distance walking p10 (m)": np.percentile(wall, 10) if len(wall) else np.nan,
        "yaw exactly 0": np.mean(cat("yaw_zero")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings_dir", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--bots", nargs="+", default=["synthetic_rrw_v3", "synthetic_rrw_v4", "synthetic_rrw_v5"], help="participant_ids of the bots")
    parser.add_argument("--humans", default=r"^P\d{2,3}$", help="regex for human participant_ids")
    parser.add_argument("--task_kind", choices=("random", "hidden_goal"), default=None,
                        help="only bot episodes of this kind (v5 records both); humans are always all episodes")
    args = parser.parse_args()

    # Unity writes raw .bin episodes; training converts them to dataset.zarr on load. Do the same here, so freshly
    # recorded episodes are read instead of skipped.
    ensure_all_episodes_ingested(args.recordings_dir)
    human_re = re.compile(args.humans)
    groups = {"humans": load_group(args.recordings_dir, lambda pid: bool(human_re.match(pid)))}
    for bot in args.bots:
        groups[bot] = load_group(args.recordings_dir, lambda pid, bot=bot: pid == bot, args.task_kind)

    tables = {name: summary(stats) for name, (stats, _) in groups.items()}
    rows = list(tables["humans"].keys())
    width = max(len(r) for r in rows)
    print(f"\n{'':{width}}  " + "  ".join(f"{n:>17}" for n in tables))
    for row in rows:
        cells = []
        for t in tables.values():
            v = t[row]
            cells.append(f"{v:>17.0f}" if row == "episodes" else f"{v:>17.3f}")
        print(f"{row:{width}}  " + "  ".join(cells))
    for name, (_, scenes) in groups.items():
        print(f"{name}: {', '.join(sorted(scenes)) or '-'}")


if __name__ == "__main__":
    main()
