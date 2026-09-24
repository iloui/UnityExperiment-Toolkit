"""
Diagnoses where and why closed-loop agents stall, using the per-tick logs EvalHarness writes into an EvalRun folder:
traj_config<C>_agent<A>.csv (always) and modeldebug_config<C>_agent<A>.csv (ImitationAgent, EvalHarness.logModelDebugTicks).

For every agent it finds the stall (the first moment after which the agent never again moves
more than --stall_move_m within --stall_window_s), then prints what the model was outputting there
(raw and applied forward speed / yaw, wall-blocked fraction, heading vs. goal bearing) and what
the recorded human episodes did within --radius_m of that same spot. It also writes an SVG map
overlaying all human trajectories (grey), the agents (colored) and the start/POI/end markers.

Usage (needs numpy + zarr, from python_embedded - see training_pipeline/README.md; from imitationagent_training/):
    PYTHONPATH=python_embedded /opt/homebrew/bin/python3 training_pipeline/analyze_stall.py \\
        ../unity_files/SimulationLogs/EvalRun_<...> --recordings zarr_recordings
"""

import argparse
import glob
import os

import numpy as np
import zarr

TICK_DT = 0.1  # ZarrDataContract.SamplingIntervalSeconds


def load_csv_columns(path):
    with open(path) as f:
        header = f.readline().strip().split(";")
        # Empty cells (model columns before the first model call) -> NaN.
        rows = [[float(c) if c else np.nan for c in ln.strip().split(";")] for ln in f if ln.strip()]
    if not rows:
        return None
    arr = np.array(rows)
    return {name: arr[:, i] for i, name in enumerate(header)}


def load_ticks(traj_path):
    """Trajectory columns, plus the model-debug columns (goal_*, blocked, raw_*, applied_*) when that file exists.
    Both files get one row per tick with the same t, so they are joined row by row."""
    tk = load_csv_columns(traj_path)
    debug_path = os.path.join(os.path.dirname(traj_path), "modeldebug_" + os.path.basename(traj_path)[len("traj_"):])
    if tk is not None and os.path.exists(debug_path):
        md = load_csv_columns(debug_path)
        if md is not None:
            n = min(len(tk["t"]), len(md["t"]))  # a run stopped mid-episode can leave one file a row ahead
            tk = {k: v[:n] for k, v in tk.items()}
            tk.update({k: v[:n] for k, v in md.items() if k != "t"})
    return tk


def load_humans(recordings_dir):
    eps = []
    for p in sorted(glob.glob(os.path.join(recordings_dir, "*", "dataset.zarr"))):
        z = zarr.open(p, mode="r")
        eps.append({
            "name": os.path.basename(os.path.dirname(p)),
            "xz": np.array(z["pose"])[:, [0, 2]],
            "vel": np.array(z["action_vel"]),
            # action_rot[:,0] = azimuth delta = yaw (wrapped, e.g. 357.6 -> -2.4); [:,1] = pitch. Same as the training loader.
            "rot": (np.array(z["action_rot"])[:, 0] + 180.0) % 360.0 - 180.0,
            "goal": np.array(z["goal"]),
        })
    return eps


def find_stall(t, xz, move_m, window_s):
    """Index of the first tick after which the agent never moves > move_m within window_s again, else None."""
    w = max(1, int(round(window_s / TICK_DT)))
    n = len(t)
    if n <= w:
        return None
    moved = np.array([np.linalg.norm(xz[min(i + w, n - 1)] - xz[i]) for i in range(n)])
    active = np.where(moved > move_m)[0]
    first_stalled = 0 if len(active) == 0 else active[-1] + 1
    return first_stalled if first_stalled < n - w else None


def describe_agent(name, tk, humans, args):
    t = tk["t"]
    xz = np.stack([tk["x"], tk["z"]], axis=1)
    i0 = find_stall(t, xz, args.stall_move_m, args.stall_window_s)
    if i0 is None:
        print(f"\n[{name}] no stall found ({len(t)} ticks, final pos {xz[-1].round(2)}).")
        return None
    sl = slice(i0, None)
    pos = xz[i0]
    print(f"\n[{name}] stalls at x={pos[0]:.2f}, z={pos[1]:.2f}  (tick {i0}/{len(t)}, t={t[i0] - t[0]:.1f}s into the episode, "
          f"{(len(t) - i0) * TICK_DT:.0f}s spent stalled), pois_reached={int(tk['pois_reached'][i0])}")
    if "raw_fwd" not in tk:
        print("  (no modeldebug_*.csv for this agent - model outputs not available)")
        return pos
    bearing = np.degrees(np.arctan2(tk["goal_x"][sl], tk["goal_z"][sl]))  # 0 = goal straight ahead
    print(f"  goal: {tk['goal_dist_m'][i0]:.2f} m away, bearing {bearing.mean():+.0f} deg from heading (mean while stalled)")
    print(f"  model raw forward  : mean {np.nanmean(tk['raw_fwd'][sl]):+.3f} m/s  (min {np.nanmin(tk['raw_fwd'][sl]):+.3f}, max {np.nanmax(tk['raw_fwd'][sl]):+.3f})   <- negative/near 0 = model chose to stop")
    print(f"  applied forward    : mean {np.nanmean(tk['applied_fwd_mps'][sl]):.3f} m/s")
    print(f"  raw yaw (deg/samp) : mean {np.nanmean(tk['raw_yaw_deg_per_sample'][sl]):+.2f}, std {np.nanstd(tk['raw_yaw_deg_per_sample'][sl]):.2f}  <- large std/sign flips = oscillation")
    print(f"  wall-blocked ticks : {100 * np.nanmean(tk['blocked'][sl]):.0f}%  <- ~100% = pushing into a wall; ~0% = stationary by choice")
    print(f"  actual speed       : mean {tk['speed_mps'][sl].mean():.3f} m/s")

    # what did humans do near this spot?
    fw, yw, n_frames, n_eps = [], [], 0, 0
    for h in humans:
        d = np.linalg.norm(h["xz"] - pos, axis=1)
        m = d < args.radius_m
        if m.any():
            n_eps += 1
            n_frames += int(m.sum())
            fw.append(h["vel"][m, 2])
            yw.append(h["rot"][m])
    if n_frames:
        fw, yw = np.concatenate(fw), np.concatenate(yw)
        print(f"  humans within {args.radius_m:.1f} m: {n_eps}/{len(humans)} episodes, {n_frames} frames | "
              f"forward mean {fw.mean():.2f} m/s (p10 {np.percentile(fw, 10):.2f}, p90 {np.percentile(fw, 90):.2f}, "
              f"{100 * (fw < 0.1).mean():.0f}% ~stopped) | yaw mean {yw.mean():+.2f}, std {yw.std():.2f} deg/sample")
    else:
        print(f"  humans within {args.radius_m:.1f} m: NONE - the agent stalls in a region no recording visited (compounding error / off-distribution).")
    return pos


def markers_from_episodes_csv(run_dir):
    import csv
    p = os.path.join(run_dir, "episodes.csv")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        row = next(csv.DictReader(f, delimiter=";"), None)
    if row is None:
        return {}

    def xz(s):
        v = [float(a) for a in s.split(",")]
        return (v[0], v[2])

    out = {"start": xz(row["start_pos"]), "end": xz(row["end_pos"])}
    if row["poi_positions"]:
        out["poi"] = xz(row["poi_positions"].split("|")[0])
    return out


def write_svg(path, humans, agents, markers, stalls):
    allpts = np.concatenate([h["xz"] for h in humans] + [a[1] for a in agents] + [np.array(list(markers.values()))]) if humans or agents else np.zeros((1, 2))
    lo, hi = allpts.min(axis=0) - 1.5, allpts.max(axis=0) + 1.5
    scale = 900 / max(hi[0] - lo[0], hi[1] - lo[1])
    W, H = (hi[0] - lo[0]) * scale, (hi[1] - lo[1]) * scale

    def P(x, z):
        return f"{(x - lo[0]) * scale:.1f},{H - (z - lo[1]) * scale:.1f}"  # +z up, like the harness PNGs

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" style="background:#fff">']
    for h in humans:
        out.append(f'<polyline fill="none" stroke="#bbb" stroke-width="1.5" points="{" ".join(P(x, z) for x, z in h["xz"])}"/>')
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for k, (name, xz) in enumerate(agents):
        out.append(f'<polyline fill="none" stroke="{palette[k % len(palette)]}" stroke-width="2.5" points="{" ".join(P(x, z) for x, z in xz)}"><title>{name}</title></polyline>')
    for name, col in (("start", "#2a2"), ("poi", "#0aa"), ("end", "#d22")):
        if name in markers:
            cx, cy = P(*markers[name]).split(",")
            out.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{col}"/><text x="{float(cx) + 10}" y="{float(cy) + 4}" font-size="13">{name}</text>')
    for s in stalls:
        cx, cy = P(*s).split(",")
        out.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="none" stroke="#000" stroke-width="2"/>')
    out.append(f'<text x="8" y="18" font-size="13">grey = {len(humans)} human recordings, colored = eval agents, black ring = stall point</text></svg>')
    with open(path, "w") as f:
        f.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--recordings", default="zarr_recordings")
    ap.add_argument("--radius_m", type=float, default=1.5)
    ap.add_argument("--stall_move_m", type=float, default=0.5)
    ap.add_argument("--stall_window_s", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="SVG output path (default: <run_dir>/stall_map.svg)")
    args = ap.parse_args()

    humans = load_humans(args.recordings)
    print(f"Loaded {len(humans)} human episodes.")
    agents, stalls = [], []
    for p in sorted(glob.glob(os.path.join(args.run_dir, "traj_*.csv"))):
        tk = load_ticks(p)
        name = os.path.basename(p)[len("traj_"):-len(".csv")]
        if tk is None:
            print(f"\n[{name}] empty tick log.")
            continue
        agents.append((name, np.stack([tk["x"], tk["z"]], axis=1)))
        s = describe_agent(name, tk, humans, args)
        if s is not None:
            stalls.append(s)
    if not agents:
        print("\nNo traj_*.csv found (EvalRun folder from before harness_version 6?). Drawing the human overlay only.")

    out = args.out or os.path.join(args.run_dir, "stall_map.svg")
    write_svg(out, humans, agents, markers_from_episodes_csv(args.run_dir), stalls)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
