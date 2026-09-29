#!/usr/bin/env python3
"""Synthetic test input for compare_conditions.py: 3 layouts, EvalHarness runs for A/B/C/AgentScript and held-out human
recordings, in exactly the formats Unity writes (layout .npz, run_config.json + episodes.csv + traj_*.csv, episode
folders with manifest.json + dataset.zarr). Plus manifest.yaml pointing at them.

Each layout is a walled room split by one wall with a gap; the route goes start -> POI through the gap -> end. The
sources differ only in made-up noise, pausing, speed and a detour: A wanders most, C least, humans pause and cut corners.

Usage:
  python make_comparison_dummy.py OUT_DIR
  python compare_conditions.py OUT_DIR/manifest.yaml --out OUT_DIR/results
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imitationagent_training" / "training_pipeline"))
from data_contract import (  # noqa: E402
    CAMERA_HEIGHT_M, SAMPLING_INTERVAL_S, SOURCE_HUMAN, TRAJECTORY_COLUMNS, make_episode_manifest, write_manifest,
)

CELL = 0.05
ARRIVAL_RADIUS = 1.5
TIMEOUT_S = 90.0
# name -> (room w, room h, wall x, gap from z, start xz, poi xz, end xz)
LAYOUTS = {
    "Dummy_Comb_1": (20.0, 12.0, 10.0, 8.5, (2.0, 2.0), (16.0, 3.0), (18.0, 10.0)),
    "Dummy_Grid_2": (24.0, 14.0, 12.0, 10.0, (2.5, 3.0), (20.0, 2.5), (21.0, 12.0)),
    "Dummy_Race_3": (18.0, 16.0, 9.0, 12.0, (2.0, 13.0), (15.0, 3.0), (3.0, 3.0)),
}
# source -> (episodes per layout, lateral noise m, detour m, pause prob / s, speed m/s, success prob)
PROFILES = {
    "A": (12, 0.9, 2.5, 0.10, 1.1, 0.55),
    "B": (12, 0.6, 1.5, 0.08, 1.2, 0.7),
    "C": (12, 0.4, 0.8, 0.06, 1.25, 0.85),
    "AgentScript": (12, 0.0, 0.0, 0.0, 1.4, 1.0),
    "Human": (8, 0.35, 0.5, 0.05, 1.3, 1.0),
}


def build_layout(name, spec, out: Path):
    w, h, wall_x, gap_z, s, p, e = spec
    nx, nz = int(round(w / CELL)), int(round(h / CELL))
    occ = np.zeros((nz, nx), dtype=bool)
    t = int(round(0.2 / CELL))
    occ[:t], occ[-t:], occ[:, :t], occ[:, -t:] = True, True, True, True
    c = int(round(wall_x / CELL))
    occ[: int(round(gap_z / CELL)), c - t // 2: c + t // 2 + 1] = True
    gap = np.array([wall_x, (gap_z + h) / 2])
    start, poi, end = (np.array([v[0], 0.0, v[1]]) for v in (s, p, e))
    corners = [np.array(s), gap, np.array(p)] if (s[0] < wall_x) != (p[0] < wall_x) else [np.array(s), np.array(p)]
    corners += [gap, np.array(e)] if (p[0] < wall_x) != (e[0] < wall_x) else [np.array(e)]
    seg1 = sum(np.linalg.norm(b - a) for a, b in zip(corners, corners[1:]))
    i_poi = next(i for i, q in enumerate(corners) if np.allclose(q, p))
    l1 = sum(np.linalg.norm(b - a) for a, b in zip(corners[:i_poi], corners[1:i_poi + 1]))
    np.savez(out / f"{name}.npz", format_version=np.int32(1), scene_name=name, layout_typology=name.split("_")[1],
             layout_seed=np.int32(0), occupancy=occ, origin_xz=np.zeros(2), cell_size=np.float64(CELL),
             start=start, end=end, pois=poi[None], segment_lengths=np.array([l1, seg1 - l1]),
             shortest_path=np.array([[q[0], 0.0, q[1]] for q in corners]), floor_y=np.float64(0), slice_y=np.float64(0.1),
             navmesh_bake_radius=np.float64(0.3), body_radius=np.float64(0.25), arrival_radius=np.float64(ARRIVAL_RADIUS))
    return corners, poi, end


def walk(corners, poi, end, prof, rng, dt_jitter=0.0):
    """Eye positions (N,3), headings, times, pois_reached along the corner path with noise, detour and pauses."""
    _, noise, detour, pause_p, speed, p_success = prof
    pts = [np.asarray(q, dtype=np.float64) for q in corners]
    if detour:
        mid = (pts[0] + pts[1]) / 2 + rng.normal(0, detour, 2)
        pts.insert(1, mid)
    success = rng.random() < p_success
    route = np.array(pts)
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1] if success else cum[-1] * rng.uniform(0.3, 0.7)
    s, t, out_s, out_t, paused = 0.0, 0.0, [], [], 0.0
    while s < total and t < TIMEOUT_S + 20:
        out_s.append(s)
        out_t.append(t)
        step = SAMPLING_INTERVAL_S * (1 + dt_jitter * rng.uniform(-0.5, 0.5))
        if paused > 0:
            paused -= step
        elif rng.random() < pause_p * step:
            paused = rng.uniform(0.5, 2.5)
        else:
            s += speed * step * rng.uniform(0.85, 1.1)
        t += step
    if not success:  # stuck: dither in place until timeout
        while t < TIMEOUT_S + 1:
            out_s.append(total + rng.normal(0, 0.05))
            out_t.append(t)
            t += SAMPLING_INTERVAL_S
    out_s.append(min(s, total))
    out_t.append(t)
    sarr = np.clip(np.array(out_s), 0, cum[-1])
    xz = np.stack([np.interp(sarr, cum, route[:, 0]), np.interp(sarr, cum, route[:, 1])], axis=1)
    offset = np.cumsum(rng.normal(0, noise * 0.08, len(sarr)))
    offset = offset - np.linspace(offset[0], offset[-1], len(offset))  # pinned at both ends
    d = np.gradient(xz, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    xz = xz + n * np.clip(offset, -1.5, 1.5)[:, None]
    heading = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360
    eye = np.stack([xz[:, 0], np.full(len(xz), CAMERA_HEIGHT_M), xz[:, 1]], axis=1)
    body = eye - [0, CAMERA_HEIGHT_M, 0]
    near_poi = np.linalg.norm(body - poi, axis=1) <= ARRIVAL_RADIUS
    pois_reached = np.maximum.accumulate(near_poi).astype(int)
    if success:  # stop at the tracker's arrival
        at_end = (pois_reached >= 1) & (np.linalg.norm(body - end, axis=1) <= ARRIVAL_RADIUS)
        if at_end.any():
            k = int(np.argmax(at_end)) + 1
            eye, heading, pois_reached, out_t = eye[:k], heading[:k], pois_reached[:k], out_t[:k]
    return eye, heading, np.asarray(out_t), pois_reached


def fmt(v):
    return ",".join(f"{x:g}" for x in v)


def write_eval_run(root: Path, name, label, corners, poi, end, start, rng):
    kind = "ShortestPath" if label == "AgentScript" else "Imitation"
    run = root / "SimulationLogs" / f"EvalRun_{name}_{label}"
    run.mkdir(parents=True, exist_ok=True)
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}  # noqa: E731
    (run / "run_config.json").write_text(json.dumps({
        "harness_version": "6", "agent_kind": kind, "condition": "" if kind == "ShortestPath" else label, "scene": name,
        "randomize_markers": False, "episode_timeout_sim_s": TIMEOUT_S,
        "task_start_positions": [xyz(start)], "task_poi_positions": [xyz(poi)], "task_end_positions": [xyz(end)],
    }, indent=2))
    rows = ["config_id;agent_id;outcome;num_pois_total;poi_positions;end_pos;trajectory_file"]
    for i in range(PROFILES[label][0]):
        eye, heading, t, pr = walk(corners, poi, end, PROFILES[label], rng)
        t = np.round(t, 4)
        keep = t <= TIMEOUT_S
        eye, heading, t, pr = eye[keep], heading[keep], t[keep], pr[keep]
        fname = f"traj_config{i}_agent0.csv"
        with open(run / fname, "w") as f:
            f.write(";".join(TRAJECTORY_COLUMNS) + "\n")
            for k in range(len(t)):
                f.write(f"{t[k]};{eye[k,0]};{eye[k,1]};{eye[k,2]};{heading[k]};5;0;0;{int(pr[k] >= 1)};{pr[k]}\n")
        rows.append(f"{i};0;x;1;{fmt(poi)};{fmt(end)};{fname}")
    (run / "episodes.csv").write_text("\n".join(rows) + "\n")
    return run


def write_human(root: Path, name, participant, rep, corners, poi, end, start, rng):
    import zarr
    eye, heading, t, pr = walk(corners, poi, end, PROFILES["Human"], rng, dt_jitter=0.3)
    target = np.where(pr[:, None] >= 1, end, poi)
    goal_distance = np.linalg.norm(eye - target, axis=1)
    ep_id = f"{participant}_{name}_{rep}"
    ep_dir = root / "recordings" / ep_id
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}  # noqa: E731
    m = make_episode_manifest(episode_id=ep_id, scene_name=name, layout_typology=name.split("_")[1], layout_seed=0,
                              source=SOURCE_HUMAN, participant_id=participant, session_id="dummy", repetition_index=rep,
                              task={"start": xyz(start), "pois": [xyz(poi)], "end": xyz(end)}, frame_count=len(t),
                              start_time=0.0, end_time=float(t[-1]))
    write_manifest(ep_dir / "manifest.json", m)
    g = zarr.open_group(str(ep_dir / "dataset.zarr"), mode="w")
    g.attrs["manifest"] = json.dumps(m)
    pose = np.column_stack([eye, heading, np.full(len(t), 5.0)]).astype(np.float32)
    g.create_array("pose", data=pose, chunks=(len(t), 5))
    g.create_array("goal_distance", data=goal_distance.astype(np.float32), chunks=(len(t),))
    g.create_array("timestamp", data=(t + 100.0).astype(np.float64), chunks=(len(t),))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1]).resolve()
    (root / "layouts").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    runs = {label: [] for label in ("A", "B", "C", "AgentScript")}
    participants = ["P90", "P91", "P92", "P93"]
    for name, spec in LAYOUTS.items():
        corners, poi, end = build_layout(name, spec, root / "layouts")
        start = np.array([spec[4][0], 0.0, spec[4][1]])
        for label in runs:
            runs[label].append(str(write_eval_run(root, name, label, corners, poi, end, start, rng).relative_to(root)))
        for k in range(PROFILES["Human"][0]):
            write_human(root, name, participants[k % len(participants)], k // len(participants), corners, poi, end, start, rng)
    lines = [f"timeout_s: {TIMEOUT_S:g}", "n_boot: 200", "seed: 0", "layouts_dir: layouts", "conditions:"]
    for label, rs in runs.items():
        lines.append(f"  {label}:")
        lines += [f"    - {r}" for r in rs]
    lines += ["humans:", "  recordings_dir: recordings", f"  participants: [{', '.join(participants)}]"]
    (root / "manifest.yaml").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'manifest.yaml'}")


if __name__ == "__main__":
    main()
