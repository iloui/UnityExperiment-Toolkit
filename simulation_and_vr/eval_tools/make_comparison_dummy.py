#!/usr/bin/env python3
"""Synthetic test input for compare_conditions.py in exactly the formats Unity and the recorder write: 3 layouts (.npz +
route_gates.json), one EvalHarness v7 queue per layout (SimulationLogs/EvalQueue_<layout>_dummy/EvalRun_<entry>/ with
run_config.json, episodes.csv, traj_*.csv, run_complete.json) for ShortestPath, A s1-3, B and C f1/f2 s1-3, and human
recordings P01-P08 (fold = odd/even) with one test walk per layout plus a training-task walk that must be ignored.

Each layout is a walled room split by one wall with a gap; the route goes start -> POI through the gap -> end. Sources
differ only in made-up noise, pausing, speed and detours: A wanders most, C least, humans pause and cut corners.

Usage:
  python make_comparison_dummy.py OUT_DIR
  python compare_conditions.py --logs OUT_DIR/SimulationLogs --queue-name dummy --recordings OUT_DIR/recordings \
      --layouts-dir OUT_DIR/layouts --gates OUT_DIR/layouts/route_gates.json --out OUT_DIR/results --n-boot 200
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imitationagent_training" / "training_pipeline"))
from data_contract import (  # noqa: E402
    CAMERA_HEIGHT_M, SAMPLING_INTERVAL_S, SOURCE_HUMAN, SOURCE_SYNTHETIC, TRAJECTORY_COLUMNS, make_episode_manifest, write_manifest,
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
    "A": (20, 0.9, 2.5, 0.10, 1.1, 0.55),
    "B": (20, 0.6, 1.5, 0.08, 1.2, 0.7),
    "C": (20, 0.4, 0.8, 0.06, 1.25, 0.85),
    "ShortestPath": (20, 0.0, 0.0, 0.0, 1.3, 1.0),
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


def write_eval_run(root: Path, name, label, source, fold, seed, heldout, corners, poi, end, start, rng):
    kind = "ShortestPath" if source == "ShortestPath" else "Imitation"
    run = root / "SimulationLogs" / f"EvalQueue_{name}_dummy" / f"EvalRun_{label}"
    run.mkdir(parents=True, exist_ok=True)
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}  # noqa: E731
    (run / "run_config.json").write_text(json.dumps({
        "harness_version": "7", "agent_kind": kind, "condition": "" if kind == "ShortestPath" else source, "scene": name,
        "model_asset": "" if kind == "ShortestPath" else label, "model_fold": fold, "model_seed": seed,
        "heldout_participants": heldout, "queue_name": "dummy",
        "randomize_markers": False, "episode_timeout_sim_s": TIMEOUT_S,
        "task_start_positions": [xyz(start)], "task_poi_positions": [xyz(poi + [0, 1, 0])], "task_end_positions": [xyz(end + [0, 1, 0])],
    }, indent=2))
    optimal = sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(corners, corners[1:]))
    rows = ["config_id;agent_id;outcome;num_pois_total;optimal_path_length_m;poi_positions;end_pos;trajectory_file"]
    n_ok = 0
    for i in range(PROFILES[source][0]):
        eye, heading, t, pr = walk(corners, poi, end, PROFILES[source], rng)
        t = np.round(t, 4)
        keep = t <= TIMEOUT_S
        eye, heading, t, pr = eye[keep], heading[keep], t[keep], pr[keep]
        body = eye - [0, CAMERA_HEIGHT_M, 0]
        ok = bool(pr[-1] >= 1 and np.linalg.norm(body[-1] - end) <= ARRIVAL_RADIUS)
        n_ok += ok
        fname = f"traj_config{i}_agent0.csv"
        with open(run / fname, "w") as f:
            f.write(";".join(TRAJECTORY_COLUMNS) + "\n")
            for k in range(len(t)):
                f.write(f"{t[k]};{eye[k,0]};{eye[k,1]};{eye[k,2]};{heading[k]};5;0;0;{int(pr[k] >= 1)};{pr[k]}\n")
        outcome = "success" if ok else "timeout"
        if source == "A" and i == 0:
            outcome = "invalid_speed"  # must be left out with a warning
        rows.append(f"{i};0;{outcome};1;{optimal:.3f};{fmt(poi)};{fmt(end)};{fname}")
    (run / "episodes.csv").write_text("\n".join(rows) + "\n")
    (run / "run_complete.json").write_text(json.dumps({"episodes": PROFILES[source][0], "successes": n_ok}))
    return run


def write_human(root: Path, name, participant, fold, task_id, corners, poi, end, start, rng):
    import zarr
    eye, heading, t, pr = walk(corners, poi, end, PROFILES["Human"], rng, dt_jitter=0.3)
    target = np.where(pr[:, None] >= 1, end + [0, 1, 0], poi + [0, 1, 0])
    goal_distance = np.linalg.norm(eye - target, axis=1)
    ep_id = f"{participant}_{name}_{task_id}"
    ep_dir = root / "recordings" / ep_id
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}  # noqa: E731
    # make_episode_manifest refuses a human manifest without trial labels, so build it as synthetic and relabel.
    m = make_episode_manifest(episode_id=ep_id, scene_name=name, layout_typology=name.split("_")[1], layout_seed=0,
                              source=SOURCE_SYNTHETIC, participant_id=participant, session_id="dummy", repetition_index=0,
                              task={"start": xyz(start), "pois": [xyz(poi + [0, 1, 0])], "end": xyz(end + [0, 1, 0])},
                              frame_count=len(t), start_time=0.0, end_time=float(t[-1]))
    m["source"] = SOURCE_HUMAN
    m["trial"] = {"task_id": task_id, "task_role": "test" if task_id == "test" else "train", "fold": fold, "attempt_index": 0}
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
    participants = [f"P0{i}" for i in range(1, 9)]
    folds = {p: 1 if int(p[1:]) % 2 else 2 for p in participants}
    heldout = {1: [p for p in participants if folds[p] == 2], 2: [p for p in participants if folds[p] == 1]}
    entries = [("ShortestPath", "ShortestPath", 0, 0)] + [(f"ImitationAgent_A_proprio2_s{s}", "A", 0, s) for s in (1, 2, 3)]
    entries += [(f"ImitationAgent_{c}_proprio2_f{f}_s{s}", c, f, s) for c in "BC" for f in (1, 2) for s in (1, 2, 3)]
    gates = {}
    for name, spec in LAYOUTS.items():
        corners, poi, end = build_layout(name, spec, root / "layouts")
        w, h, wall_x, gap_z = spec[:4]
        gates[name] = {"leg": 0 if (spec[4][0] < wall_x) != (spec[5][0] < wall_x) else 1,
                       "gates": [{"side": "north", "a": [wall_x, gap_z], "b": [wall_x, h]},
                                 {"side": "south", "a": [wall_x, gap_z], "b": [wall_x, 0.0]}]}
        start = np.array([spec[4][0], 0.0, spec[4][1]])
        for label, source, fold, seed in entries:
            write_eval_run(root, name, label, source, fold, seed, heldout.get(fold, []), corners, poi, end, start, rng)
        for p in participants:
            write_human(root, name, p, folds[p], "test", corners, poi, end, start, rng)
        write_human(root, name, "P01", 1, "train_0", corners, poi, end, start, rng)
    (root / "layouts" / "route_gates.json").write_text(json.dumps(gates, indent=2))
    print(f"wrote dummy queues, recordings and layouts to {root}")


if __name__ == "__main__":
    main()
