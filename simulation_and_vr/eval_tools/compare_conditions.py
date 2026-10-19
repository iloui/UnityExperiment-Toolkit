#!/usr/bin/env python3
"""Thesis results (methodology 3.9): conditions A, B, C and the shortest-path agent compared with held-out humans.

Inputs
  EvalHarness v7 queues: <logs>/EvalQueue_<layout>_<queue-name>/EvalRun_<entry>/ (run_config.json, episodes.csv,
      traj_*.csv, run_complete.json). Unfinished runs (no run_complete.json) and *_incomplete_* folders are skipped.
      Condition, fold, seed and held-out participants come from run_config.json (the model's training sidecar).
  Human test walks: <recordings>/*/manifest.json with source human and trial.task_id "test" (highest attempt_index per
      participant and layout). The participant's fold is manifest trial.fold.
  Layouts: <layouts-dir>/<layout>.npz (occupancy grid + task, written by Prepare Scene) and route_gates.json.

Trajectory representation (3.9.1): trajectories.py puts every episode on one 0.1 s grid with speed and yaw rate from
the eye's motion. Body = eye - 1.5 m. An episode ends when the body comes within ARRIVAL_RADIUS_M horizontally of the End
after the POI was reached, or at TIMEOUT_S; later samples are dropped. Path-length reference: the NavMesh shortest route
from the episode's own spawn via the POI to the End (agents: EvalHarness optimal_path_length_m; humans start on the
nominal start: the layout's segment_lengths).

Episode metrics (3.9.2, 3.9.3)
  success            POI then End reached within TIMEOUT_S
  time_to_goal_s     arrival time (successful episodes only)
  path_efficiency    path length / shortest route (successful episodes only)
  stop_share         share of samples with speed < STOP_SPEED_MPS (0.65 m/s = the walk/stop label threshold)
  contacts_per_min   wall-contact events (wall distance < 0.30 m, new event after >= 1 s free) per minute
  route_side         side of the first gate crossed on the decision leg (route_gates.json), else "none"
Sample metrics, each episode's samples weighted to a total of 1:
  yaw_rate_deg_s     |yaw rate| averaged over a centred 5-sample (0.5 s) window, all samples
  wall_distance_m    body-to-obstacle distance on walking samples (speed >= 0.65 m/s)
A model's time_to_goal_s and path_efficiency on a layout are undefined if it has fewer than MIN_SUCCESSES successes there
(also inside each bootstrap replicate); a condition's value is the mean over its models where defined.

Distances (3.9.4): Wasserstein-1 per model and layout against the held-out humans of the matching fold (B/C: the
participants in run_config heldout_participants; A and the shortest-path agent: each fold separately, averaged).
Sample metrics are compared as histograms (yaw 0.5 deg/s, wall distance 0.01 m bins). Condition value = mean over its
models; seed spread = min..max over models. Noise floor = W(fold-1 humans, fold-2 humans).

Scores (3.9.5): S = 1 - W(X, humans) / W(shortest path, humans) per metric and layout. A metric-layout pair enters the
layout score only if W(shortest path) exceeds the noise floor AND S is defined for A, B and C (one common metric set per
layout, so condition scores are comparable). Layout score = mean of S over the four H1 metrics in that set; overall = mean
over layouts. Differences: B - A (transfer, H1), C - B (memorisation, H2); success change in percentage points.

Bootstrap: N_BOOT percentile replicates. Per replicate the held-out humans are resampled by participant within each fold
(one draw for all layouts and metrics), and agent episodes are resampled by configuration within each layout. EvalHarness
gives configuration i the same start for every model, so one configuration draw per layout is shared by all models
(paired resampling). Every number below is computed on the same replicates.

Outputs, in <out>/<YYYYmmdd_HHMMSS>/:
  episodes.csv      one row per episode with every metric
  success.csv       success per model (Wilson CI) and per condition (bootstrap CI)
  distances.csv     W per layout x metric x source, CI, seed spread, noise floor, score S with CI, exclusion reason
  scores.csv        layout and overall scores per condition (+ noise-floor score), CI, seed spread
  differences.csv   B-A and C-B for scores and success, CI, supported (CI excludes 0)
  route_side.csv    share of north passes per source, difference to humans, CI
  run_info.json     inputs, settings, metric sets, exclusions, warnings
  trajectories.png/.pdf, heatmaps.png/.pdf, heatmap_differences.png/.pdf, scores.png/.pdf, distances.png/.pdf

Usage:
  python compare_conditions.py                         # queues EvalQueue_*_final under unity_files/SimulationLogs
  python compare_conditions.py --queue-name test --n-boot 200
  python compare_conditions.py --queues DIR [DIR ...] --recordings DIR --layouts-dir DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from layout_metrics import CONTACT_DISTANCE_M, EVENT_GAP_S, LAYOUTS_DIR, Layout, contact_events, load_layout
from trajectories import load_eval_run, load_human_episode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imitationagent_training" / "training_pipeline"))
from data_contract import CAMERA_HEIGHT_M, SAMPLING_INTERVAL_S, SOURCE_HUMAN, TASK_ROLE_TEST  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_LOGS = HERE.parent / "unity_files" / "SimulationLogs"
DEFAULT_RECORDINGS = HERE.parent / "imitationagent_training" / "zarr_recordings"
DEFAULT_GATES = LAYOUTS_DIR / "route_gates.json"
DEFAULT_OUT = HERE / "results"
HARNESS_VERSION = "7"

SP, HUMAN = "ShortestPath", "Human"
CONDITIONS = ("A", "B", "C")
AGENT_SOURCES = (SP,) + CONDITIONS
SOURCES = AGENT_SOURCES + (HUMAN,)
DIFFERENCES = (("B - A", "B", "A", "transfer (H1)"), ("C - B", "C", "B", "memorisation (H2)"))

TIMEOUT_S = 90.0
ARRIVAL_RADIUS_M = 1.5
ARRIVAL_TOLERANCE_M = 0.05      # float positions on the 0.1 s grid
STOP_SPEED_MPS = 0.65
YAW_WINDOW_SAMPLES = 5
MIN_SUCCESSES = 5
TASK_MATCH_TOLERANCE_M = 0.05
N_BOOT = 2000
HEATMAP_CELL_M = 0.25

EPISODE_METRICS = ("stop_share", "path_efficiency", "time_to_goal_s", "contacts_per_min")
SAMPLE_METRICS = ("yaw_rate_deg_s", "wall_distance_m")
DIST_METRICS = ("stop_share", "yaw_rate_deg_s", "wall_distance_m", "path_efficiency", "time_to_goal_s", "contacts_per_min")
SCORE_METRICS = ("stop_share", "yaw_rate_deg_s", "wall_distance_m", "path_efficiency")  # H1
SUCCESS_ONLY = ("path_efficiency", "time_to_goal_s")
HIST_EDGES = {"yaw_rate_deg_s": np.arange(0.0, 360.0 + 1e-9, 0.5), "wall_distance_m": np.arange(0.0, 15.0 + 1e-9, 0.01)}
METRIC_TITLES = {
    "stop_share": "Stop share", "yaw_rate_deg_s": "|Yaw rate| 0.5 s (deg/s)", "wall_distance_m": "Wall distance, walking (m)",
    "path_efficiency": "Path efficiency", "time_to_goal_s": "Time to goal (s)", "contacts_per_min": "Wall contacts / min",
}
COLORS = {SP: "#52514e", "A": "#8fb8ea", "B": "#2a78d6", "C": "#0d3f80", HUMAN: "#eb6834"}
WALL_COLOR = "#2b2b2b"


# ----------------------------------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Model:
    """One evaluated agent on one layout: an imitation model or the shortest-path agent."""
    layout: str
    label: str
    source: str             # ShortestPath | A | B | C
    fold: int               # training fold (0 for A and ShortestPath)
    seed: int
    heldout: tuple          # held-out participants (B/C); empty = compare with both folds
    run_dir: Path


@dataclass
class Episode:
    layout: str
    source: str
    model: str              # model label, or "Human"
    participant: str | None
    fold: int               # model's training fold, or the participant's fold
    seed: int
    unit: int               # configuration id (agents) or participant index (humans): the bootstrap resampling unit
    episode: str
    traj: pd.DataFrame
    optimal_m: float
    metrics: dict = field(default_factory=dict)
    hists: dict = field(default_factory=dict)
    keep: pd.DataFrame | None = None   # trajectory rows inside the episode window


def _xz(d) -> np.ndarray:
    return np.array([d["x"], d["z"]], dtype=np.float64)


def _check_task(what: str, start, pois, end, layout: Layout):
    """The task an episode ran (hand-placed markers, start before jitter) must be the layout file's (x/z only: the npz
    stores the markers on the NavMesh, the scene floats POI and End 1 m above it)."""
    got = np.vstack([start] + list(pois) + [end])
    want = np.vstack([layout.start[[0, 2]][None], layout.pois[:, [0, 2]], layout.end[[0, 2]][None]])
    if got.shape != want.shape or np.abs(got - want).max() > TASK_MATCH_TOLERANCE_M:
        raise ValueError(f"{what}: task markers {np.round(got, 2).tolist()} differ from {layout.scene_name}.npz "
                         f"{np.round(want, 2).tolist()}. Re-run Tools > Eval Harness > Wire Up Current Scene For Eval "
                         "(it re-exports the occupancy grid), then copy nothing - the npz is written to eval_tools/layouts.")


def find_runs(queue_dirs: list[Path], warnings: list[str]) -> list[Path]:
    runs = []
    for q in queue_dirs:
        for r in sorted(q.glob("EvalRun_*")):
            if not r.is_dir() or "_incomplete_" in r.name:
                continue
            if not (r / "run_complete.json").exists():
                warnings.append(f"{r}: unfinished (no run_complete.json) - skipped")
                continue
            runs.append(r)
    return runs


def load_agents(runs: list[Path], layout_for, warnings: list[str]) -> tuple[list[Model], list[Episode]]:
    models, episodes, seen = [], [], set()
    for run in runs:
        cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        if cfg.get("harness_version") != HARNESS_VERSION:
            raise ValueError(f"{run}: harness_version {cfg.get('harness_version')!r}, expected {HARNESS_VERSION!r}")
        if cfg.get("randomize_markers"):
            raise ValueError(f"{run}: randomize_markers is on - the thesis evaluation uses the hand-placed task only")
        if float(cfg["episode_timeout_sim_s"]) < TIMEOUT_S:
            raise ValueError(f"{run}: timeout {cfg['episode_timeout_sim_s']} s < {TIMEOUT_S} s")
        sp = cfg["agent_kind"] == "ShortestPath"
        source = SP if sp else cfg.get("condition", "")
        if source not in AGENT_SOURCES:
            raise ValueError(f"{run}: condition {source!r} is not A, B or C")
        label = SP if sp else cfg["model_asset"]
        layout = layout_for(cfg["scene"])
        if (cfg["scene"], label) in seen:
            raise ValueError(f"{run}: '{label}' on '{cfg['scene']}' appears in more than one run - pass one queue per layout")
        seen.add((cfg["scene"], label))
        _check_task(str(run), _xz(cfg["task_start_positions"][0]), [_xz(p) for p in cfg["task_poi_positions"]],
                    _xz(cfg["task_end_positions"][0]), layout)
        heldout = tuple(sorted(cfg.get("heldout_participants") or []))
        if source in ("B", "C") and not heldout:
            raise ValueError(f"{run}: condition {source} run without heldout_participants")
        model = Model(cfg["scene"], label, source, int(cfg.get("model_fold") or 0), int(cfg.get("model_seed") or 0),
                      heldout, run)
        models.append(model)

        traj = load_eval_run(run)
        rows = pd.read_csv(run / "episodes.csv", sep=";", dtype=str, keep_default_na=False)
        dropped = 0
        for row in rows.itertuples():
            key = f"config{row.config_id}_agent{row.agent_id}"
            ep = traj[traj["episode"] == key]
            if row.outcome not in ("success", "timeout") or len(ep) < 2:
                dropped += 1
                continue
            episodes.append(Episode(cfg["scene"], source, label, None, model.fold, model.seed, int(row.config_id), key,
                                    ep.reset_index(drop=True), float(row.optimal_path_length_m)))
        if dropped:
            warnings.append(f"{run.name} ({cfg['scene']}): {dropped} episode(s) not valid (outcome other/invalid_speed) - left out")
    return models, episodes


def load_humans(recordings_dir: Path, layouts: set[str], layout_for, warnings: list[str]) -> list[Episode]:
    picked: dict[tuple[str, str], tuple[int, Path, dict]] = {}
    for mpath in sorted(recordings_dir.glob("*/manifest.json")):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        trial = m.get("trial") or {}
        if m.get("source") != SOURCE_HUMAN or m.get("scene_name") not in layouts or trial.get("task_id") != TASK_ROLE_TEST:
            continue
        key = (m["participant_id"], m["scene_name"])
        attempt = int(trial.get("attempt_index", 0))
        if key in picked:
            warnings.append(f"{key}: several test walks - using attempt {max(attempt, picked[key][0])}")
        if key not in picked or attempt > picked[key][0]:
            picked[key] = (attempt, mpath.parent, m)
    participants = sorted({p for p, _ in picked})
    episodes = []
    for (participant, scene), (_, ep_dir, m) in sorted(picked.items()):
        task = m["task"]
        layout = layout_for(scene)
        _check_task(str(ep_dir), _xz(task["start"]), [_xz(p) for p in task["pois"]], _xz(task["end"]), layout)
        episodes.append(Episode(scene, HUMAN, HUMAN, participant, int(m["trial"]["fold"]), 0, participants.index(participant),
                                m["episode_id"], load_human_episode(ep_dir), float(layout.segment_lengths.sum())))
    return episodes


# ----------------------------------------------------------------------------------------------------------------------
# Per-episode metrics
# ----------------------------------------------------------------------------------------------------------------------

def _first_crossing(xz: np.ndarray, gates: list[dict]) -> str:
    """Side of the first gate segment the polyline xz crosses, or 'none'."""
    if len(xz) < 2:
        return "none"
    p, r = xz[:-1], np.diff(xz, axis=0)
    best_i, best_side = None, "none"
    for g in gates:
        a, b = np.asarray(g["a"], float), np.asarray(g["b"], float)
        s = b - a
        denom = r[:, 0] * s[1] - r[:, 1] * s[0]
        qp = a - p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qp[:, 0] * s[1] - qp[:, 1] * s[0]) / denom
            u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / denom
        hit = np.flatnonzero((np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))
        if len(hit) and (best_i is None or hit[0] < best_i):
            best_i, best_side = int(hit[0]), g["side"]
    return best_side


def compute_metrics(ep: Episode, layout: Layout, gates: dict | None):
    df = ep.traj
    t = df["t"].to_numpy()
    eye = df[["x", "y", "z"]].to_numpy()
    body_xz = eye[:, [0, 2]]
    pois_reached = df["pois_reached"].to_numpy()
    speed = df["speed_mps"].to_numpy()

    horiz_to_end = np.linalg.norm(body_xz - layout.end[[0, 2]], axis=1)
    at_end = (pois_reached >= len(layout.pois)) & (horiz_to_end <= ARRIVAL_RADIUS_M + ARRIVAL_TOLERANCE_M)
    i_goal = int(np.argmax(at_end)) if at_end.any() else None
    success = i_goal is not None and t[i_goal] <= TIMEOUT_S + 1e-9
    n = (i_goal + 1) if success else int(np.searchsorted(t, TIMEOUT_S + 1e-9, side="right"))
    n = max(n, 2)

    steps = speed[1:n]
    path_len = float(steps.sum() * SAMPLING_INTERVAL_S)
    yaw_smooth = np.abs(pd.Series(df["yaw_rate_deg_s"].to_numpy()).rolling(YAW_WINDOW_SAMPLES, center=True, min_periods=1)
                        .mean().to_numpy())[1:n]
    wall = layout.wall_distance(body_xz[:n])
    walking = speed[:n] >= STOP_SPEED_MPS
    wall_walk = wall[1:][walking[1:]]
    duration = float(t[n - 1])
    contacts = len(contact_events(t[:n], wall < CONTACT_DISTANCE_M, EVENT_GAP_S))

    side = np.nan
    if gates is not None:
        leg = pois_reached[:n] == int(gates["leg"])
        side = _first_crossing(body_xz[:n][leg], gates["gates"])

    ep.metrics = {
        "success": float(success),
        "time_to_goal_s": float(t[i_goal]) if success else np.nan,
        "path_efficiency": path_len / ep.optimal_m if success and ep.optimal_m > 0 else np.nan,
        "stop_share": float(np.mean(steps < STOP_SPEED_MPS)),
        "contacts_per_min": contacts / (duration / 60.0) if duration > 0 else np.nan,
        "contact_events": contacts,
        "route_side": side,
        "duration_s": duration,
        "path_length_m": path_len,
        "optimal_path_m": ep.optimal_m,
        "pois_reached": int(pois_reached[n - 1]),
        "yaw_rate_mean_deg_s": float(yaw_smooth.mean()) if len(yaw_smooth) else np.nan,
        "wall_distance_mean_m": float(wall_walk.mean()) if len(wall_walk) else np.nan,
    }
    for metric, values in (("yaw_rate_deg_s", yaw_smooth), ("wall_distance_m", wall_walk)):
        edges = HIST_EDGES[metric]
        if len(values):
            h, _ = np.histogram(np.clip(values, edges[0], edges[-1] - 1e-9), bins=edges)
            ep.hists[metric] = h / h.sum()
        else:
            ep.hists[metric] = None
    ep.keep = df.iloc[:n]


# ----------------------------------------------------------------------------------------------------------------------
# Distances
# ----------------------------------------------------------------------------------------------------------------------

def w1(u: np.ndarray, v: np.ndarray, uw: np.ndarray, vw: np.ndarray) -> float:
    """Weighted Wasserstein-1 between two 1-D samples (as scipy.stats.wasserstein_distance, without its input checks)."""
    us, vs = np.argsort(u), np.argsort(v)
    allv = np.sort(np.concatenate([u, v]))
    deltas = np.diff(allv)
    ucum = np.concatenate([[0.0], np.cumsum(uw[us])])
    vcum = np.concatenate([[0.0], np.cumsum(vw[vs])])
    ucdf = ucum[np.searchsorted(u[us], allv[:-1], side="right")] / ucum[-1]
    vcdf = vcum[np.searchsorted(v[vs], allv[:-1], side="right")] / vcum[-1]
    return float(np.sum(np.abs(ucdf - vcdf) * deltas))


class Group:
    """The episodes of one model, or of one group of humans, on one layout, as arrays for fast reweighting."""

    def __init__(self, eps: list[Episode], kind: str):
        self.kind = kind  # "cfg" (resampled by configuration) or "hum" (by participant)
        self.unit = np.array([e.unit for e in eps], dtype=np.int64)
        self.values = {m: np.array([e.metrics[m] for e in eps], dtype=np.float64) for m in EPISODE_METRICS}
        self.success = np.array([e.metrics["success"] for e in eps], dtype=np.float64)
        self.north = np.array([e.metrics["route_side"] == "north" for e in eps], dtype=np.float64)
        self.crossed = np.array([e.metrics["route_side"] in ("north", "south") for e in eps], dtype=np.float64)
        self.hists, self.has = {}, {}
        for m in SAMPLE_METRICS:
            nb = len(HIST_EDGES[m]) - 1
            self.hists[m] = np.array([e.hists[m] if e.hists[m] is not None else np.zeros(nb) for e in eps]).reshape(len(eps), nb)
            self.has[m] = np.array([e.hists[m] is not None for e in eps], dtype=bool)

    def weights(self, counts: np.ndarray) -> np.ndarray:
        return counts[self.unit].astype(np.float64)

    def defined(self, metric: str, w: np.ndarray, threshold: bool) -> bool:
        return not (threshold and metric in SUCCESS_ONLY and float((w * self.success).sum()) < MIN_SUCCESSES)


def distance(a: Group, wa: np.ndarray, b: Group, wb: np.ndarray, metric: str) -> float:
    if metric in SAMPLE_METRICS:
        ma, mb = a.has[metric] & (wa > 0), b.has[metric] & (wb > 0)
        if not ma.any() or not mb.any():
            return np.nan
        pa = (wa[ma, None] * a.hists[metric][ma]).sum(0) / wa[ma].sum()
        pb = (wb[mb, None] * b.hists[metric][mb]).sum(0) / wb[mb].sum()
        width = HIST_EDGES[metric][1] - HIST_EDGES[metric][0]
        return float(np.abs(np.cumsum(pa) - np.cumsum(pb)).sum() * width)
    va, vb = a.values[metric], b.values[metric]
    ma, mb = np.isfinite(va) & (wa > 0), np.isfinite(vb) & (wb > 0)
    if not ma.any() or not mb.any():
        return np.nan
    return w1(va[ma], vb[mb], wa[ma], wb[mb])


class Study:
    """Everything one evaluation needs per layout: model groups, human groups per fold, and how they are compared."""

    def __init__(self, models: list[Model], episodes: list[Episode]):
        self.layouts = sorted({m.layout for m in models})
        humans = [e for e in episodes if e.source == HUMAN]
        self.participants = sorted({e.participant for e in humans})
        self.n_participants = len(self.participants)
        self.folds = sorted({e.fold for e in humans})
        if self.folds != [1, 2]:
            raise ValueError(f"human test walks cover folds {self.folds}, expected [1, 2]")
        self.fold_members = {f: sorted({e.participant for e in humans if e.fold == f}) for f in self.folds}
        self.fold_index = {f: np.array([self.participants.index(p) for p in self.fold_members[f]]) for f in self.folds}

        self.models = {l: [m for m in models if m.layout == l] for l in self.layouts}
        self.groups: dict[tuple, Group] = {}
        self.n_configs: dict[str, int] = {}
        for l in self.layouts:
            for m in self.models[l]:
                eps = [e for e in episodes if e.layout == l and e.model == m.label]
                if not eps:
                    raise ValueError(f"{m.run_dir}: no valid episodes")
                self.groups[(l, m.label)] = Group(eps, "cfg")
            self.n_configs[l] = 1 + max(e.unit for e in episodes if e.layout == l and e.source != HUMAN)
            for f in self.folds:
                eps = [e for e in humans if e.layout == l and e.fold == f]
                if not eps:
                    raise ValueError(f"no fold-{f} human test walks on {l}")
                self.groups[(l, HUMAN, f)] = Group(eps, "hum")
            self.groups[(l, HUMAN, 0)] = Group([e for e in humans if e.layout == l], "hum")

        # Which human fold(s) each model is compared with.
        self.refs: dict[tuple, list[int]] = {}
        for l in self.layouts:
            for m in self.models[l]:
                if m.heldout:
                    match = [f for f in self.folds if tuple(self.fold_members[f]) == m.heldout]
                    if not match:
                        raise ValueError(f"{m.run_dir}: heldout_participants {list(m.heldout)} is not one human fold "
                                         f"({self.fold_members})")
                    self.refs[(l, m.label)] = match
                else:
                    self.refs[(l, m.label)] = list(self.folds)
            if sum(m.source == SP for m in self.models[l]) != 1:
                raise ValueError(f"{l}: needs exactly one ShortestPath run")
            for c in CONDITIONS:
                if not any(m.source == c for m in self.models[l]):
                    raise ValueError(f"{l}: no condition {c} runs")

    def evaluate(self, cfg_counts: dict[str, np.ndarray], hum_counts: np.ndarray) -> dict:
        """Raw quantities for one set of resampling weights (all ones = the point estimate)."""
        out = {}
        for l in self.layouts:
            hum = {f: self.groups[(l, HUMAN, f)] for f in self.folds}
            hw = {f: hum[f].weights(hum_counts) for f in self.folds}
            for k in DIST_METRICS:
                out[("floor", l, k)] = distance(hum[1], hw[1], hum[2], hw[2], k)
            for m in self.models[l]:
                g = self.groups[(l, m.label)]
                w = g.weights(cfg_counts[l])
                out[("success", l, m.label)] = float((w * g.success).sum() / w.sum())
                out[("north", l, m.label)] = (float((w * g.north).sum()), float((w * g.crossed).sum()))
                for k in DIST_METRICS:
                    if not g.defined(k, w, threshold=True):
                        out[("W", l, m.label, k)] = np.nan
                        continue
                    vals = [distance(g, w, hum[f], hw[f], k) for f in self.refs[(l, m.label)]]
                    out[("W", l, m.label, k)] = float(np.mean(vals)) if np.all(np.isfinite(vals)) else np.nan
            hall = self.groups[(l, HUMAN, 0)]
            whall = hall.weights(hum_counts)
            out[("north", l, HUMAN)] = (float((whall * hall.north).sum()), float((whall * hall.crossed).sum()))
        return out

    def sp_label(self, l: str) -> str:
        return next(m.label for m in self.models[l] if m.source == SP)

    def labels(self, l: str, source: str) -> list[str]:
        return [m.label for m in self.models[l] if m.source == source]


def summarise(study: Study, raw: dict, metric_sets: dict[str, list[str]] | None) -> dict:
    """Condition-level numbers from raw quantities. metric_sets fixes each layout's common score metrics (taken from the
    point estimate); None = derive them here (point estimate only)."""
    s = {}
    derived_sets = {}
    for l in study.layouts:
        for k in DIST_METRICS:
            floor = raw[("floor", l, k)]
            w_sp = raw[("W", l, study.sp_label(l), k)]
            s[("floor", l, k)] = floor
            s[("W", l, SP, k)] = w_sp
            for c in CONDITIONS:
                vals = np.array([raw[("W", l, lab, k)] for lab in study.labels(l, c)])
                s[("W", l, c, k)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                s[("n_defined", l, c, k)] = int(np.isfinite(vals).sum())
            ok_scale = np.isfinite(w_sp) and np.isfinite(floor) and w_sp > floor
            for c in CONDITIONS:
                s[("S", l, c, k)] = 1.0 - s[("W", l, c, k)] / w_sp if ok_scale else np.nan
            s[("S_floor", l, k)] = 1.0 - floor / w_sp if ok_scale else np.nan
            if k in SCORE_METRICS:
                if not ok_scale:
                    s[("excluded", l, k)] = "W(shortest path) within the human noise floor"
                elif not all(np.isfinite(s[("S", l, c, k)]) for c in CONDITIONS):
                    s[("excluded", l, k)] = "undefined for " + ", ".join(c for c in CONDITIONS if not np.isfinite(s[("S", l, c, k)]))
                else:
                    s[("excluded", l, k)] = ""
        derived_sets[l] = [k for k in SCORE_METRICS if s[("excluded", l, k)] == ""]
    sets = metric_sets if metric_sets is not None else derived_sets
    s["metric_sets"] = sets

    for c in CONDITIONS + ("floor",):
        layout_scores = []
        for l in study.layouts:
            key = ("S_floor", l) if c == "floor" else ("S", l, c)
            vals = [s[key + (k,)] for k in sets[l]]
            ls = float(np.nanmean(vals)) if vals and np.isfinite(vals).any() else np.nan
            s[("LS", l, c)] = ls
            layout_scores.append(ls)
        s[("LS", "overall", c)] = float(np.nanmean(layout_scores)) if np.isfinite(layout_scores).any() else np.nan

    for l in study.layouts:
        for src in AGENT_SOURCES:
            vals = [raw[("success", l, lab)] for lab in study.labels(l, src)]
            s[("success", l, src)] = float(np.mean(vals))
        for src in AGENT_SOURCES + (HUMAN,):
            labs = [HUMAN] if src == HUMAN else study.labels(l, src)
            north = sum(raw[("north", l, lab)][0] for lab in labs)
            crossed = sum(raw[("north", l, lab)][1] for lab in labs)
            s[("north", l, src)] = north / crossed if crossed > 0 else np.nan
        for src in AGENT_SOURCES:
            s[("north_diff", l, src)] = s[("north", l, src)] - s[("north", l, HUMAN)]
    for src in AGENT_SOURCES:
        s[("success", "overall", src)] = float(np.mean([s[("success", l, src)] for l in study.layouts]))

    for name, x, y, _ in DIFFERENCES:
        for l in study.layouts + ["overall"]:
            s[("dLS", name, l)] = s[("LS", l, x)] - s[("LS", l, y)]
            s[("dSuccess", name, l)] = 100.0 * (s[("success", l, x)] - s[("success", l, y)])
    return s


def model_spread(study: Study, raw: dict, sets: dict[str, list[str]]) -> dict:
    """Seed spread (point estimate): min..max over a condition's models of W and of the per-model layout score."""
    out = {}
    for l in study.layouts:
        sp = study.sp_label(l)
        for c in CONDITIONS:
            labs = study.labels(l, c)
            for k in DIST_METRICS:
                v = np.array([raw[("W", l, lab, k)] for lab in labs])
                out[("W", l, c, k)] = (np.nanmin(v), np.nanmax(v)) if np.isfinite(v).any() else (np.nan, np.nan)
            per_model = []
            for lab in labs:
                vals = []
                for k in sets[l]:
                    w_sp = raw[("W", l, sp, k)]
                    vals.append(1.0 - raw[("W", l, lab, k)] / w_sp)
                per_model.append(np.nanmean(vals) if vals and np.isfinite(vals).any() else np.nan)
            per_model = np.array(per_model)
            out[("LS", l, c)] = (np.nanmin(per_model), np.nanmax(per_model)) if np.isfinite(per_model).any() else (np.nan, np.nan)
    return out


def bootstrap(study: Study, sets, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    reps = []
    for r in range(n_boot):
        cfg_counts = {l: np.bincount(rng.integers(0, study.n_configs[l], study.n_configs[l]), minlength=study.n_configs[l])
                      for l in study.layouts}
        hum_counts = np.zeros(study.n_participants, dtype=np.int64)
        for f in study.folds:
            idx = study.fold_index[f]
            hum_counts += np.bincount(idx[rng.integers(0, len(idx), len(idx))], minlength=study.n_participants)
        reps.append(summarise(study, study.evaluate(cfg_counts, hum_counts), sets))
        if (r + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {r + 1}/{n_boot}", flush=True)
    return reps


def ci(reps: list[dict], key) -> tuple[float, float]:
    v = np.array([r[key] for r in reps], dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < max(10, 0.5 * len(reps)):
        return (np.nan, np.nan)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ----------------------------------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------------------------------

def episodes_table(episodes: list[Episode]) -> pd.DataFrame:
    rows = []
    for e in episodes:
        rows.append({"layout": e.layout, "source": e.source, "model": e.model, "participant": e.participant, "fold": e.fold,
                     "seed": e.seed, "unit": e.unit, "episode": e.episode, **e.metrics})
    return pd.DataFrame(rows)


def tables(study: Study, episodes: list[Episode], point: dict, reps: list[dict], spread: dict):
    eps = episodes_table(episodes)

    success_rows = []
    for l in study.layouts:
        for m in study.models[l]:
            sel = eps[(eps.layout == l) & (eps.model == m.label)]
            k, n = float(sel.success.sum()), len(sel)
            lo, hi = wilson(k, n)
            success_rows.append({"layout": l, "source": m.source, "model": m.label, "fold": m.fold, "seed": m.seed,
                                 "episodes": n, "successes": int(k), "success_rate": k / n if n else np.nan,
                                 "ci_low": lo, "ci_high": hi, "ci_method": "wilson"})
        hsel = eps[(eps.layout == l) & (eps.source == HUMAN)]
        success_rows.append({"layout": l, "source": HUMAN, "model": HUMAN, "episodes": len(hsel),
                             "successes": int(hsel.success.sum()), "success_rate": hsel.success.mean(), "ci_method": ""})
    for l in study.layouts + ["overall"]:
        for src in AGENT_SOURCES:
            lo, hi = ci(reps, ("success", l, src))
            success_rows.append({"layout": l, "source": src, "model": "(condition mean)", "success_rate": point[("success", l, src)],
                                 "ci_low": lo, "ci_high": hi, "ci_method": "bootstrap"})
    success = pd.DataFrame(success_rows)

    dist_rows = []
    for l in study.layouts:
        for k in DIST_METRICS:
            floor = point[("floor", l, k)]
            flo, fhi = ci(reps, ("floor", l, k))
            for src in AGENT_SOURCES:
                lo, hi = ci(reps, ("W", l, src, k))
                row = {"layout": l, "metric": k, "source": src, "W_to_humans": point[("W", l, src, k)], "ci_low": lo, "ci_high": hi,
                       "noise_floor": floor, "noise_floor_ci_low": flo, "noise_floor_ci_high": fhi,
                       "W_shortest_path": point[("W", l, SP, k)]}
                if src in CONDITIONS:
                    smin, smax = spread[("W", l, src, k)]
                    slo, shi = ci(reps, ("S", l, src, k))
                    row.update({"models_defined": point[("n_defined", l, src, k)], "models": len(study.labels(l, src)),
                                "seed_min": smin, "seed_max": smax, "score": point[("S", l, src, k)], "score_ci_low": slo,
                                "score_ci_high": shi, "score_floor": point[("S_floor", l, k)],
                                "in_layout_score": k in point["metric_sets"][l],
                                "excluded_reason": point.get(("excluded", l, k), "not an H1 metric")})
                dist_rows.append(row)
    distances = pd.DataFrame(dist_rows)

    score_rows = []
    for l in study.layouts + ["overall"]:
        for c in CONDITIONS:
            lo, hi = ci(reps, ("LS", l, c))
            smin, smax = spread[("LS", l, c)] if l != "overall" else (np.nan, np.nan)
            score_rows.append({"layout": l, "source": c, "score": point[("LS", l, c)], "ci_low": lo, "ci_high": hi,
                               "seed_min": smin, "seed_max": smax,
                               "metrics": ",".join(point["metric_sets"][l]) if l != "overall" else "mean over layouts"})
        lo, hi = ci(reps, ("LS", l, "floor"))
        score_rows.append({"layout": l, "source": "human noise floor", "score": point[("LS", l, "floor")], "ci_low": lo,
                           "ci_high": hi, "metrics": ",".join(point["metric_sets"][l]) if l != "overall" else "mean over layouts"})
    scores = pd.DataFrame(score_rows)

    diff_rows = []
    for name, _, _, meaning in DIFFERENCES:
        for l in study.layouts + ["overall"]:
            for measure, key, unit in (("score", "dLS", ""), ("success", "dSuccess", "percentage points")):
                est = point[(key, name, l)]
                lo, hi = ci(reps, (key, name, l))
                diff_rows.append({"comparison": name, "meaning": meaning, "layout": l, "measure": measure, "unit": unit,
                                  "estimate": est, "ci_low": lo, "ci_high": hi,
                                  "supported": bool(np.isfinite(lo) and lo > 0) if measure == "score" else None,
                                  "ci_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
    differences = pd.DataFrame(diff_rows)

    route_rows = []
    for l in study.layouts:
        if eps[eps.layout == l].route_side.isna().all():
            continue
        for src in SOURCES:
            sel = eps[(eps.layout == l) & (eps.source == src)]
            row = {"layout": l, "source": src, "episodes": len(sel), "crossed": int(sel.route_side.isin(["north", "south"]).sum()),
                   "north": int((sel.route_side == "north").sum()), "south": int((sel.route_side == "south").sum()),
                   "share_north": point[("north", l, src)]}
            lo, hi = ci(reps, ("north", l, src))
            row.update({"ci_low": lo, "ci_high": hi})
            if src != HUMAN:
                dlo, dhi = ci(reps, ("north_diff", l, src))
                row.update({"diff_to_humans": point[("north_diff", l, src)], "diff_ci_low": dlo, "diff_ci_high": dhi})
            route_rows.append(row)
    route = pd.DataFrame(route_rows)
    return eps, success, distances, scores, differences, route


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, out: Path, name: str) -> list[Path]:
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def _draw_layout(ax, layout: Layout):
    ax.imshow(np.ma.masked_where(~layout.occupancy, layout.occupancy), origin="lower", extent=layout.extent,
              cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest", zorder=1)
    ax.set_xlim(layout.extent[0], layout.extent[1])
    ax.set_ylim(layout.extent[2], layout.extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_markers(ax, layout: Layout):
    ax.scatter(*layout.start[[0, 2]], marker="o", s=30, color="#1b9e77", zorder=6, edgecolors="white", linewidths=0.6)
    ax.scatter(layout.pois[:, 0], layout.pois[:, 2], marker="*", s=70, color="#d95f02", zorder=6, edgecolors="white", linewidths=0.5)
    ax.scatter(*layout.end[[0, 2]], marker="s", s=30, color="#7570b3", zorder=6, edgecolors="white", linewidths=0.6)


def figure_trajectories(episodes, layouts, study, out):
    plt = _plt()
    cols = list(SOURCES)
    fig, axes = plt.subplots(len(study.layouts), len(cols), figsize=(2.6 * len(cols), 2.7 * len(study.layouts)), squeeze=False,
                             layout="constrained")
    for i, l in enumerate(study.layouts):
        for j, src in enumerate(cols):
            ax = axes[i, j]
            _draw_layout(ax, layouts[l])
            eps = [e for e in episodes if e.layout == l and e.source == src]
            alpha = 0.7 if src in (HUMAN, SP) else 0.3
            for e in eps:
                k = e.keep
                ax.plot(k["x"], k["z"], color=COLORS[src], lw=0.7, alpha=alpha, zorder=3,
                        ls="-" if e.metrics["success"] else (0, (2, 1.5)))
            _draw_markers(ax, layouts[l])
            if i == 0:
                ax.set_title(f"{'Shortest path' if src == SP else ('Humans' if src == HUMAN else 'Condition ' + src)}")
            if j == 0:
                ax.set_ylabel(l.split("_")[0])
            n_ok = sum(e.metrics["success"] for e in eps)
            ax.text(0.02, 0.02, f"{n_ok:.0f}/{len(eps)}", transform=ax.transAxes, fontsize=7, color="#333", zorder=7)
    fig.suptitle("Trajectories on the test task (solid = reached the End, dashed = timeout); counts = successful/all")
    paths = _save(fig, out, "trajectories")
    plt.close(fig)
    return paths


def _time_share(eps: list[Episode], layout: Layout) -> np.ndarray:
    x0, x1, z0, z1 = layout.extent
    xe = np.arange(x0, x1 + 1e-9, HEATMAP_CELL_M)
    ze = np.arange(z0, z1 + 1e-9, HEATMAP_CELL_M)
    acc = np.zeros((len(ze) - 1, len(xe) - 1))
    for e in eps:
        h, _, _ = np.histogram2d(e.keep["z"], e.keep["x"], bins=[ze, xe])
        acc += h / max(h.sum(), 1)
    return acc / max(len(eps), 1)


def figure_heatmaps(episodes, layouts, study, out):
    plt = _plt()
    from matplotlib.colors import PowerNorm, TwoSlopeNorm
    cols = list(SOURCES)
    shares = {(l, s): _time_share([e for e in episodes if e.layout == l and e.source == s], layouts[l])
              for l in study.layouts for s in cols}
    paths = []

    fig, axes = plt.subplots(len(study.layouts), len(cols), figsize=(2.6 * len(cols), 2.7 * len(study.layouts)), squeeze=False,
                             layout="constrained")
    for i, l in enumerate(study.layouts):
        vmax = max(np.percentile(shares[(l, s)][shares[(l, s)] > 0], 99) if (shares[(l, s)] > 0).any() else 1e-6 for s in cols)
        for j, src in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(np.ma.masked_equal(100 * shares[(l, src)], 0), origin="lower", extent=layouts[l].extent, cmap="magma_r",
                      norm=PowerNorm(0.5, vmin=0, vmax=100 * vmax), interpolation="nearest", zorder=2)
            _draw_layout(ax, layouts[l])
            _draw_markers(ax, layouts[l])
            if i == 0:
                ax.set_title("Shortest path" if src == SP else ("Humans" if src == HUMAN else f"Condition {src}"))
            if j == 0:
                ax.set_ylabel(l.split("_")[0])
    fig.suptitle(f"Share of time per {HEATMAP_CELL_M} m cell (each episode weighted equally; square-root colour scale)")
    paths += _save(fig, out, "heatmaps")
    plt.close(fig)

    fig, axes = plt.subplots(len(study.layouts), len(CONDITIONS), figsize=(2.9 * len(CONDITIONS), 2.7 * len(study.layouts)),
                             squeeze=False, layout="constrained")
    for i, l in enumerate(study.layouts):
        diffs = {c: 100 * (shares[(l, c)] - shares[(l, HUMAN)]) for c in CONDITIONS}
        vmax = max(float(np.percentile(np.abs(d[d != 0]), 99)) if (d != 0).any() else 1e-6 for d in diffs.values())
        for j, c in enumerate(CONDITIONS):
            ax = axes[i, j]
            im = ax.imshow(np.ma.masked_equal(diffs[c], 0), origin="lower", extent=layouts[l].extent, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest", zorder=2)
            _draw_layout(ax, layouts[l])
            _draw_markers(ax, layouts[l])
            if i == 0:
                ax.set_title(f"{c} minus humans")
            if j == 0:
                ax.set_ylabel(l.split("_")[0])
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.8, label="time share difference (%)")
    fig.suptitle("Where agents spend more (red) or less (blue) time than the held-out humans")
    paths += _save(fig, out, "heatmap_differences")
    plt.close(fig)
    return paths


def figure_scores(scores: pd.DataFrame, study, out):
    plt = _plt()
    names = study.layouts + ["overall"]
    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 1, 3.4))
    offsets = {"A": -0.22, "B": 0.0, "C": 0.22}
    for i, l in enumerate(names):
        fl = scores[(scores.layout == l) & (scores.source == "human noise floor")]
        if len(fl) and np.isfinite(fl.score.iloc[0]):
            ax.hlines(fl.score.iloc[0], i - 0.38, i + 0.38, color=COLORS[HUMAN], lw=2, zorder=2,
                      label="human noise floor" if i == 0 else None)
        for c in CONDITIONS:
            row = scores[(scores.layout == l) & (scores.source == c)]
            if not len(row) or not np.isfinite(row.score.iloc[0]):
                continue
            r = row.iloc[0]
            yerr = [[r.score - r.ci_low], [r.ci_high - r.score]] if np.isfinite(r.ci_low) else None
            ax.errorbar(i + offsets[c], r.score, yerr=yerr, fmt="o", color=COLORS[c], capsize=3, zorder=3,
                        label=f"Condition {c}" if i == 0 else None)
            if np.isfinite(r.seed_min):
                ax.vlines(i + offsets[c] + 0.07, r.seed_min, r.seed_max, color=COLORS[c], lw=1, alpha=0.6, zorder=3)
    ax.axhline(0, color=COLORS[SP], lw=0.8, ls="--")
    ax.axhline(1, color="#999", lw=0.8, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split("_")[0] for n in names])
    ax.set_ylabel("Human-likeness score S\n(0 = shortest path, 1 = humans)")
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    paths = _save(fig, out, "scores")
    plt.close(fig)
    return paths


def figure_distances(distances: pd.DataFrame, study, out):
    plt = _plt()
    fig, axes = plt.subplots(len(DIST_METRICS), len(study.layouts), figsize=(3.2 * len(study.layouts), 1.9 * len(DIST_METRICS)),
                             squeeze=False)
    for i, k in enumerate(DIST_METRICS):
        for j, l in enumerate(study.layouts):
            ax = axes[i, j]
            sel = distances[(distances.layout == l) & (distances.metric == k)]
            for x, src in enumerate(AGENT_SOURCES):
                r = sel[sel.source == src]
                if not len(r) or not np.isfinite(r.W_to_humans.iloc[0]):
                    ax.text(x, 0, "n/a", ha="center", fontsize=6, color="#999")
                    continue
                r = r.iloc[0]
                yerr = [[r.W_to_humans - r.ci_low], [r.ci_high - r.W_to_humans]] if np.isfinite(r.ci_low) else None
                ax.bar(x, r.W_to_humans, color=COLORS[src], width=0.7)
                if yerr:
                    ax.errorbar(x, r.W_to_humans, yerr=yerr, fmt="none", ecolor="#333", capsize=2, lw=0.8)
            if len(sel) and np.isfinite(sel.noise_floor.iloc[0]):
                ax.axhline(sel.noise_floor.iloc[0], color=COLORS[HUMAN], lw=1.2, ls="--")
            ax.set_xticks(range(len(AGENT_SOURCES)))
            ax.set_xticklabels(["SP", "A", "B", "C"], fontsize=7)
            if j == 0:
                ax.set_ylabel(METRIC_TITLES[k], fontsize=7)
            if i == 0:
                ax.set_title(l.split("_")[0])
    fig.suptitle("Wasserstein distance to held-out humans (95% CI; dashed = human noise floor)")
    fig.tight_layout()
    paths = _save(fig, out, "distances")
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-name", default="final", help="EvalHarness queueName (default: final)")
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS, help="folder with the EvalQueue_* folders")
    ap.add_argument("--queues", type=Path, nargs="*", help="explicit queue folders (overrides --logs/--queue-name)")
    ap.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS, help="recorded episodes (human test walks)")
    ap.add_argument("--layouts-dir", type=Path, default=LAYOUTS_DIR)
    ap.add_argument("--gates", type=Path, default=DEFAULT_GATES, help="route_gates.json (route side, H3)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="parent folder of the timestamped results folder")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    queues = args.queues or sorted(p for p in args.logs.glob(f"EvalQueue_*_{args.queue_name}") if p.is_dir())
    if not queues:
        sys.exit(f"No queue folders EvalQueue_*_{args.queue_name} in {args.logs}")
    warnings: list[str] = []
    layouts: dict[str, Layout] = {}

    def layout_for(scene: str) -> Layout:
        if scene not in layouts:
            layouts[scene] = load_layout(args.layouts_dir / f"{scene}.npz")
        return layouts[scene]

    gates_all = json.loads(args.gates.read_text(encoding="utf-8")) if args.gates and args.gates.exists() else {}

    print(f"Queues: {', '.join(q.name for q in queues)}")
    runs = find_runs(queues, warnings)
    models, agent_eps = load_agents(runs, layout_for, warnings)
    human_eps = load_humans(args.recordings, {m.layout for m in models}, layout_for, warnings)
    episodes = agent_eps + human_eps
    print(f"Loaded {len(models)} runs ({len(agent_eps)} agent episodes) and {len(human_eps)} human test walks "
          f"on {len({m.layout for m in models})} layout(s).")

    for e in episodes:
        compute_metrics(e, layouts[e.layout], gates_all.get(e.layout))
    study = Study(models, episodes)

    raw_point = study.evaluate({l: np.ones(study.n_configs[l], dtype=np.int64) for l in study.layouts},
                               np.ones(study.n_participants, dtype=np.int64))
    point = summarise(study, raw_point, None)
    sets = point["metric_sets"]
    spread = model_spread(study, raw_point, sets)
    print(f"Score metrics per layout: {sets}")
    print(f"Bootstrap: {args.n_boot} replicates")
    reps = bootstrap(study, sets, args.n_boot, args.seed)

    out = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    eps, success, distances, scores, differences, route = tables(study, episodes, point, reps, spread)
    for name, df in (("episodes", eps), ("success", success), ("distances", distances), ("scores", scores),
                     ("differences", differences), ("route_side", route)):
        df.to_csv(out / f"{name}.csv", index=False)

    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "queues": queues, "runs": runs, "recordings": args.recordings, "layouts_dir": args.layouts_dir, "gates": args.gates,
        "human_folds": study.fold_members,
        "settings": {"timeout_s": TIMEOUT_S, "arrival_radius_m": ARRIVAL_RADIUS_M, "stop_speed_mps": STOP_SPEED_MPS,
                     "yaw_window_samples": YAW_WINDOW_SAMPLES, "contact_distance_m": CONTACT_DISTANCE_M,
                     "contact_gap_s": EVENT_GAP_S, "min_successes": MIN_SUCCESSES, "n_boot": args.n_boot, "seed": args.seed,
                     "hist_bin": {k: float(v[1] - v[0]) for k, v in HIST_EDGES.items()}},
        "score_metrics_per_layout": sets,
        "excluded_score_metrics": {f"{l}:{k}": point[("excluded", l, k)] for l in study.layouts for k in SCORE_METRICS
                                   if point[("excluded", l, k)]},
        "warnings": warnings,
    }
    (out / "run_info.json").write_text(json.dumps(_jsonable(info), indent=2), encoding="utf-8")

    if not args.no_figures:
        figure_trajectories(episodes, layouts, study, out)
        figure_heatmaps(episodes, layouts, study, out)
        figure_scores(scores, study, out)
        figure_distances(distances, study, out)

    pd.set_option("display.width", 160)
    print("\nLayout and overall scores (S: 0 = shortest path, 1 = humans):")
    print(scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nDifferences:")
    print(differences.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for w in warnings:
        print(f"[warn] {w}")
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
