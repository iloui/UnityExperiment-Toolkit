"""Per-source label statistics of the recorded episodes (raw .bin files, no zarr needed).

Compares human and synthetic recordings on what the discrete model learns from: standing still, stop lengths,
turning while standing, per-sample yaw, the 7 yaw classes of the 0.5 s training label, and the wall-ahead frames
where M4 froze. Run after recording, before training:

    python summarize_recordings.py --recordings_dir ../zarr_recordings
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_contract import ACTION_HORIZON_SAMPLES, WALK_THRESHOLD_MPS, YAW_BIN_EDGES_DEG

STAND_MPS = 0.05


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def load(ep: Path, name: str, cols: int):
    return np.fromfile(ep / f"{name}.bin", dtype=np.float32).reshape(-1, cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings_dir", default="../zarr_recordings")
    args = parser.parse_args()

    stats = defaultdict(lambda: defaultdict(list))
    for ep in sorted(p for p in Path(args.recordings_dir).iterdir() if (p / "manifest.json").exists()):
        pid = json.loads((ep / "manifest.json").read_text()).get("participant_id", "")
        src = pid if pid.startswith("synthetic") else "human"
        vel, rot, goal = load(ep, "action_vel", 3), load(ep, "action_rot", 3), load(ep, "goal", 3)
        depth = np.fromfile(ep / "depth.bin", dtype=np.float32).reshape(-1, 64, 64)
        n = len(vel)
        if n < 2:
            continue
        speed = np.hypot(vel[1:, 0], vel[1:, 2])
        yaw = wrap(rot[1:, 0])  # sample 0 holds the absolute heading, not a turn
        standing = speed < STAND_MPS

        s = stats[src]
        s["episodes"].append(1)
        s["seconds"].append(n * 0.1)
        s["speed"] += list(speed)
        s["yaw"] += list(yaw)
        s["stand_turn"] += list(standing & (np.abs(yaw) > 2.0))
        runs, k = [], 0
        for x in standing:
            if x:
                k += 1
            elif k:
                runs.append(k * 0.1)
                k = 0
        if k:
            runs.append(k * 0.1)
        s["stops"] += runs
        s["stops_per_ep"].append(len(runs))
        bearing = np.degrees(np.arctan2(goal[:, 0], goal[:, 2]))
        s["poi"].append(bool(np.any(np.abs(wrap(np.diff(bearing))) > 40.0)))
        y_all = wrap(rot[:, 0])
        y_all[0] = 0.0
        for i in range(1, n):
            label = y_all[i:min(i + ACTION_HORIZON_SAMPLES, n)].mean()
            mag = int(np.searchsorted(YAW_BIN_EDGES_DEG, abs(label), side="right"))
            s["cls"].append(mag if label > 0 else -mag)
            front = depth[i - 1, 24:40, 28:36].mean() * 50.0
            s["wall_turn"].append(front < 1.5 and abs(bearing[i - 1]) < 20 and mag >= 2)
            s["wall_turn_sign"].append(np.sign(label) if (front < 1.5 and abs(bearing[i - 1]) < 20 and mag >= 2) else 0)

    for src, s in sorted(stats.items()):
        speed, yaw = np.array(s["speed"]), np.abs(np.array(s["yaw"]))
        stops, cls = np.array(s["stops"]), np.array(s["cls"])
        wall_sign = np.array(s["wall_turn_sign"])
        print(f"\n=== {src}: {len(s['episodes'])} episodes, {len(speed)} samples, median episode {np.median(s['seconds']):.1f} s, "
              f"goal switch seen (bearing jump > 40 deg, misses switches straight ahead) in {100 * np.mean(s['poi']):.0f}% ===")
        print(f"  standing {100 * np.mean(speed < STAND_MPS):.1f}% | walking (> {WALK_THRESHOLD_MPS} m/s) {100 * np.mean(speed > WALK_THRESHOLD_MPS):.1f}% "
              f"| turning while standing {100 * np.mean(s['stand_turn']):.1f}%")
        if len(stops):
            print(f"  stops per episode {np.mean(s['stops_per_ep']):.1f} | stop length median {np.median(stops):.1f} s, p90 {np.percentile(stops, 90):.1f} s")
        print(f"  |yaw| per sample: ==0 {100 * np.mean(yaw == 0):.1f}%  <2 {100 * np.mean((yaw > 0) & (yaw < 2)):.1f}%  "
              f"2-5 {100 * np.mean((yaw >= 2) & (yaw < 5)):.1f}%  >=5 {100 * np.mean(yaw >= 5):.1f}%  max {yaw.max():.1f}")
        shares = [100 * np.mean(cls == c) for c in range(-3, 4)]
        print("  yaw classes (0.5 s label, strong-L .. straight .. strong-R): " + " ".join(f"{v:.1f}" for v in shares))
        print(f"  wall < 1.5 m ahead, goal within 20 deg, turning >= {YAW_BIN_EDGES_DEG[1]} deg/sample: {int(np.sum(s['wall_turn']))} frames "
              f"(left {int(np.sum(wall_sign < 0))}, right {int(np.sum(wall_sign > 0))})")


if __name__ == "__main__":
    main()
