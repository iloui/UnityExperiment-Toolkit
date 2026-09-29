#!/usr/bin/env python3
"""Thesis results: conditions A/B/C and the AgentScript baseline compared with held-out humans, per layout.

Inputs (a YAML or JSON manifest; relative paths are relative to the manifest file):

  timeout_s: 90                  # success = POI and end reached within this many seconds (EvalHarness canonical 90)
  n_boot: 1000                   # bootstrap replicates (optional, default 1000)
  seed: 0                        # bootstrap RNG seed (optional)
  layouts_dir: layouts           # <scene>.npz from OccupancyGridExporter (optional, default eval_tools/layouts)
  conditions:                    # EvalHarness run folders (harness_version 6), any number per key, any layouts
    A: [../unity_files/SimulationLogs/EvalRun_..., ...]
    B: [...]
    C: [...]
    AgentScript: [...]           # ShortestPath runs
  humans:
    recordings_dir: ../recordings        # ingested episodes (<dir>/*/dataset.zarr)
    participants: [P07, P08, P09]        # the held-out test participants (required: never the training ones)

Every source goes through trajectories.py (one 0.1 s grid, one speed / yaw-rate definition) and layout_metrics.py (one
wall-distance definition), and every metric below is computed from those trajectories, not from EvalHarness's own
episodes.csv columns, so agents and humans are measured the same way.

Per episode (samples after the goal is reached, or after timeout_s, are dropped):
  success              reached the POI and then the end within timeout_s. "Reached the end" = pois_reached == #POIs
                       and the body centre (eye - 1.5 m) within the layout's arrival radius (+ ARRIVAL_TOLERANCE_M for
                       the 0.1 s grid) of the end marker - PoiProgressTracker's rule.
  path_efficiency      horizontal path length / NavMesh shortest start -> POI -> end (>= ~1; successful episodes only)
  time_to_goal_s       time the end was reached (successful episodes only)
  stop_share           fraction of samples with speed < STOP_SPEED_MPS
  contact_events       layout_metrics.contact_events on the wall distance
  time_to_first_poi_s  first sample with pois_reached >= 1 (episodes that reached it)
Per sample (pooled over the episodes of a layout x source):
  yaw_rate_abs_deg_s   |yaw_rate_deg_s|
  wall_distance_m      layout_metrics wall distance of the body centre

Wasserstein distance (scipy.stats.wasserstein_distance) from each of A, B, C, AgentScript to Human, per layout and
metric: per-episode values for episode metrics, pooled samples for per-sample metrics. 95% CIs: percentile bootstrap,
resampling episodes (with replacement) of every source independently; one set of replicates feeds every metric and the
headline numbers, so their CIs are consistent.

Headline (for B and C, each vs A):
  relative W reduction  mean over (metric, layout) of 1 - W_X / W_A (pairs with W_A == 0 skipped); > 0 = closer to humans
  success change        mean over layouts of success_rate_X - success_rate_A

Outputs, in <out>/<YYYYmmdd_HHMMSS>/ (default out: eval_tools/results):
  episodes.csv, summary.csv, wasserstein.csv, headline.csv, manifest copy,
  trajectories.png/.pdf (one panel per layout, all sources), occupancy.png/.pdf (layout x source time-share heatmaps),
  wasserstein.png/.pdf (W to humans with CIs, one panel per metric).

Usage:
  python compare_conditions.py MANIFEST.yaml [--out DIR] [--n-boot N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from layout_metrics import CONTACT_DISTANCE_M, LAYOUTS_DIR, Layout, contact_events, load_layout
from trajectories import SOURCE_IMITATION, SOURCE_SHORTEST_PATH, load_eval_run, load_recordings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "imitationagent_training" / "training_pipeline"))
from data_contract import CAMERA_HEIGHT_M, SAMPLING_INTERVAL_S, SOURCE_HUMAN  # noqa: E402

CONDITIONS = ("A", "B", "C")
BASELINE = "AgentScript"
HUMAN = "Human"
COMPARED = CONDITIONS + (BASELINE,)
LABELS = COMPARED + (HUMAN,)
HEADLINE_VS = "A"

EPISODE_METRICS = ("success", "path_efficiency", "time_to_goal_s", "stop_share", "contact_events", "time_to_first_poi_s")
SAMPLE_METRICS = ("yaw_rate_abs_deg_s", "wall_distance_m")
METRICS = EPISODE_METRICS + SAMPLE_METRICS
METRIC_TITLES = {
    "success": "Success", "path_efficiency": "Path length / optimal", "time_to_goal_s": "Time to goal (s)",
    "stop_share": "Stop share", "contact_events": "Contact events", "time_to_first_poi_s": "Time to first POI (s)",
    "yaw_rate_abs_deg_s": "|Yaw rate| (deg/s)", "wall_distance_m": "Wall distance (m)",
}

STOP_SPEED_MPS = 0.1
# The tracker fires on the first frame inside the radius; the 0.1 s grid can sit up to one step (~0.15 m) before it.
ARRIVAL_TOLERANCE_M = 0.2
DEFAULT_TIMEOUT_S = 90.0
DEFAULT_N_BOOT = 1000
TASK_MATCH_TOLERANCE_M = 0.05
HEATMAP_BIN_M = 0.25

# Conditions A -> C add human data, so they share one blue ramp (light -> dark); baseline neutral, humans orange.
COLORS = {"A": "#8fb8ea", "B": "#2a78d6", "C": "#0d3f80", BASELINE: "#52514e", HUMAN: "#eb6834"}
LINESTYLES = {BASELINE: "--"}
WALL_COLOR = "#2b2b2b"


# ----------------------------------------------------------------------------------------------------------------------
# Manifest and loading
# ----------------------------------------------------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        man = yaml.safe_load(text)
    else:
        man = json.loads(text)
    base = path.parent

    def resolve(p):
        return (base / p).resolve()

    conditions = man.get("conditions") or {}
    unknown = set(conditions) - set(COMPARED)
    if unknown:
        raise ValueError(f"{path}: unknown condition keys {sorted(unknown)}, expected a subset of {list(COMPARED)}")
    humans = man.get("humans") or {}
    if not humans.get("recordings_dir") or not humans.get("participants"):
        raise ValueError(f"{path}: 'humans' needs recordings_dir and participants (the held-out test participants)")
    return {
        "timeout_s": float(man.get("timeout_s", DEFAULT_TIMEOUT_S)),
        "n_boot": int(man.get("n_boot", DEFAULT_N_BOOT)),
        "seed": int(man.get("seed", 0)),
        "layouts_dir": resolve(man["layouts_dir"]) if man.get("layouts_dir") else LAYOUTS_DIR,
        "conditions": {k: [resolve(r) for r in (v or [])] for k, v in conditions.items()},
        "recordings_dir": resolve(humans["recordings_dir"]),
        "participants": [str(p) for p in humans["participants"]],
    }


def _xyz(d) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float64)


def _check_task(what: str, start, pois, end, layout: Layout):
    """The route an episode ran must be the layout's (the optimal path length comes from the layout)."""
    got = np.vstack([start] + list(pois) + [end])
    want = np.vstack([layout.start[None], layout.pois, layout.end[None]])
    if got.shape != want.shape or np.abs(got - want).max() > TASK_MATCH_TOLERANCE_M:
        raise ValueError(f"{what}: task markers {np.round(got, 2).tolist()} differ from {layout.scene_name}.npz "
                         f"{np.round(want, 2).tolist()} - re-run Prepare Scene or re-record")


def _load_run(label: str, run_dir: Path, timeout_s: float, layout_for) -> pd.DataFrame:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    df = load_eval_run(run_dir)
    expected = SOURCE_SHORTEST_PATH if label == BASELINE else SOURCE_IMITATION
    if cfg["agent_kind"] != ("ShortestPath" if label == BASELINE else "Imitation"):
        raise ValueError(f"{run_dir}: agent_kind {cfg['agent_kind']!r} listed under {label}")
    if label in CONDITIONS and cfg.get("condition") != label:
        raise ValueError(f"{run_dir}: condition {cfg.get('condition')!r} listed under {label}")
    if cfg.get("randomize_markers"):
        raise ValueError(f"{run_dir}: randomize_markers is on; the comparison needs the layout's fixed task")
    if float(cfg["episode_timeout_sim_s"]) < timeout_s:
        raise ValueError(f"{run_dir}: EvalHarness timeout {cfg['episode_timeout_sim_s']} s < timeout_s {timeout_s} s")
    layout = layout_for(cfg["scene"])
    if len(cfg["task_start_positions"]) != 1:
        raise ValueError(f"{run_dir}: {len(cfg['task_start_positions'])} start markers, expected the layout's one")
    start = _xyz(cfg["task_start_positions"][0])
    # Per-episode route (episodes.csv start_pos is the jittered spawn, so the start comes from run_config).
    for row in pd.read_csv(run_dir / "episodes.csv", sep=";", dtype=str, keep_default_na=False).itertuples():
        if row.outcome == "other":
            continue
        pois = [np.array(p.split(","), dtype=np.float64) for p in row.poi_positions.split("|") if p]
        _check_task(f"{run_dir} config {row.config_id}", start, pois, np.array(row.end_pos.split(","), dtype=np.float64),
                    layout)
    assert df.empty or (df["source"] == expected).all()
    df["label"] = label
    df["episode"] = run_dir.name + "/" + df["episode"]
    return df


def _load_humans(recordings_dir: Path, participants: list[str], layout_for) -> pd.DataFrame:
    wanted = set(participants)
    df = load_recordings(recordings_dir, source=SOURCE_HUMAN, participants=wanted)
    missing = wanted - set(df["participant"].unique())
    if missing:
        raise ValueError(f"{recordings_dir}: no human episodes for held-out participant(s) {sorted(missing)}")
    for mpath in sorted(recordings_dir.glob("*/manifest.json")):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        if m.get("source") == SOURCE_HUMAN and m.get("participant_id") in wanted:
            task = m["task"]
            _check_task(str(mpath.parent), _xyz(task["start"]), [_xyz(p) for p in task["pois"]], _xyz(task["end"]),
                        layout_for(m["scene_name"]))
    df["label"] = HUMAN
    return df


def load_all(man: dict) -> tuple[pd.DataFrame, dict[str, Layout]]:
    layouts: dict[str, Layout] = {}

    def layout_for(scene: str) -> Layout:
        if scene not in layouts:
            layouts[scene] = load_layout(man["layouts_dir"] / f"{scene}.npz")
        return layouts[scene]

    frames = [_load_run(label, run, man["timeout_s"], layout_for)
              for label in COMPARED for run in man["conditions"].get(label, [])]
    frames.append(_load_humans(man["recordings_dir"], man["participants"], layout_for))
    return pd.concat(frames, ignore_index=True), layouts


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------

def episode_metrics(ep: pd.DataFrame, layout: Layout, timeout_s: float) -> dict:
    """Episode metrics plus the per-sample arrays (samples up to the goal / timeout) - module doc for definitions."""
    t = ep["t"].to_numpy()
    eye = ep[["x", "y", "z"]].to_numpy()
    body = eye - np.array([0.0, CAMERA_HEIGHT_M, 0.0])
    pois_reached = ep["pois_reached"].to_numpy()
    speed = ep["speed_mps"].to_numpy()

    at_end = (pois_reached >= len(layout.pois)) & \
             (np.linalg.norm(body - layout.end, axis=1) <= layout.arrival_radius + ARRIVAL_TOLERANCE_M)
    i_goal = int(np.argmax(at_end)) if at_end.any() else None
    success = i_goal is not None and t[i_goal] <= timeout_s + 1e-9
    n = (i_goal + 1) if success else int(np.searchsorted(t, timeout_s + 1e-9, side="right"))

    steps = speed[1:n]  # row 0 has speed 0 by definition, not a stop
    path_len = float(steps.sum() * SAMPLING_INTERVAL_S)
    wall = layout.wall_distance(eye[:n, [0, 2]])
    reached_poi = np.flatnonzero(pois_reached[:n] >= 1)
    return {
        "success": float(success),
        "path_efficiency": path_len / float(layout.segment_lengths.sum()) if success else np.nan,
        "time_to_goal_s": float(t[i_goal]) if success else np.nan,
        "stop_share": float(np.mean(steps < STOP_SPEED_MPS)) if len(steps) else np.nan,
        "contact_events": float(len(contact_events(t[:n], wall < CONTACT_DISTANCE_M))),
        "time_to_first_poi_s": float(t[reached_poi[0]]) if len(reached_poi) else np.nan,
        "path_length_m": path_len,
        "duration_s": float(t[n - 1]),
        "_yaw_rate_abs_deg_s": np.abs(ep["yaw_rate_deg_s"].to_numpy()[1:n]),
        "_wall_distance_m": wall,
    }


def compute_episodes(df: pd.DataFrame, layouts: dict[str, Layout], timeout_s: float) -> pd.DataFrame:
    rows = []
    for (label, layout, episode), ep in df.groupby(["label", "layout", "episode"], sort=False):
        m = episode_metrics(ep, layouts[layout], timeout_s)
        rows.append({"layout": layout, "source": label, "episode": episode,
                     "participant": ep["participant"].iloc[0], **m})
    return pd.DataFrame(rows)


def _metric_values(eps: pd.DataFrame, metric: str) -> np.ndarray:
    if metric in SAMPLE_METRICS:
        arrays = eps["_" + metric].tolist()
        return np.concatenate(arrays) if arrays else np.zeros(0)
    v = eps[metric].to_numpy(dtype=np.float64)
    return v[np.isfinite(v)]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summary_table(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (layout, source), eps in episodes.groupby(["layout", "source"], sort=False):
        for metric in METRICS:
            v = _metric_values(eps, metric)
            row = {"layout": layout, "source": source, "metric": metric, "episodes": len(eps), "n": len(v)}
            if len(v):
                q25, med, q75 = np.percentile(v, [25, 50, 75])
                row.update(mean=v.mean(), median=med, q25=q25, q75=q75, iqr=q75 - q25)
            if metric == "success":
                lo, hi = wilson(int(v.sum()), len(v))
                row.update(ci_low=lo, ci_high=hi, ci_method="wilson")
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Wasserstein distances + bootstrap
# ----------------------------------------------------------------------------------------------------------------------

def _w(a: np.ndarray, b: np.ndarray) -> float:
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else np.nan


class _Values:
    """Per-episode values of one layout x source, laid out so a bootstrap resample is cheap."""

    def __init__(self, eps: pd.DataFrame):
        self.n = len(eps)
        self.episode = {m: eps[m].to_numpy(dtype=np.float64) for m in EPISODE_METRICS}
        self.samples = {m: eps["_" + m].tolist() for m in SAMPLE_METRICS}

    def values(self, metric: str, idx: np.ndarray | None = None) -> np.ndarray:
        if metric in SAMPLE_METRICS:
            arrays = self.samples[metric] if idx is None else [self.samples[metric][i] for i in idx]
            return np.concatenate(arrays) if arrays else np.zeros(0)
        v = self.episode[metric] if idx is None else self.episode[metric][idx]
        return v[np.isfinite(v)]

    def success_rate(self, idx: np.ndarray | None = None) -> float:
        v = self.episode["success"] if idx is None else self.episode["success"][idx]
        return float(v.mean()) if len(v) else np.nan


def _distances(vals: dict[str, _Values], idx: dict[str, np.ndarray] | None) -> np.ndarray:
    """(len(COMPARED), len(METRICS)) W to Human."""
    out = np.full((len(COMPARED), len(METRICS)), np.nan)
    if HUMAN not in vals:
        return out
    human = {m: vals[HUMAN].values(m, None if idx is None else idx[HUMAN]) for m in METRICS}
    for i, label in enumerate(COMPARED):
        if label in vals:
            for j, m in enumerate(METRICS):
                out[i, j] = _w(vals[label].values(m, None if idx is None else idx[label]), human[m])
    return out


def _success_rates(vals: dict[str, _Values], idx: dict[str, np.ndarray] | None) -> np.ndarray:
    return np.array([vals[label].success_rate(None if idx is None else idx[label]) if label in vals else np.nan
                     for label in LABELS])


def bootstrap(episodes: pd.DataFrame, layout_names: list[str], n_boot: int, seed: int):
    """Point estimates and replicates: W (layouts, COMPARED, METRICS), success rates (layouts, LABELS)."""
    rng = np.random.default_rng(seed)
    w_point = np.full((len(layout_names), len(COMPARED), len(METRICS)), np.nan)
    w_boot = np.full((n_boot,) + w_point.shape, np.nan)
    s_point = np.full((len(layout_names), len(LABELS)), np.nan)
    s_boot = np.full((n_boot,) + s_point.shape, np.nan)
    for li, layout in enumerate(layout_names):
        vals = {label: _Values(eps) for label, eps in episodes[episodes["layout"] == layout].groupby("source")}
        w_point[li] = _distances(vals, None)
        s_point[li] = _success_rates(vals, None)
        for b in range(n_boot):
            idx = {label: rng.integers(0, v.n, v.n) for label, v in vals.items()}
            w_boot[b, li] = _distances(vals, idx)
            s_boot[b, li] = _success_rates(vals, idx)
    return w_point, w_boot, s_point, s_boot


def _ci(reps: np.ndarray) -> tuple[float, float]:
    reps = reps[np.isfinite(reps)]
    return tuple(np.percentile(reps, [2.5, 97.5])) if len(reps) else (np.nan, np.nan)


def wasserstein_table(layout_names, w_point, w_boot) -> pd.DataFrame:
    rows = []
    for li, layout in enumerate(layout_names):
        for i, label in enumerate(COMPARED):
            for j, m in enumerate(METRICS):
                lo, hi = _ci(w_boot[:, li, i, j])
                rows.append({"layout": layout, "source": label, "metric": m,
                             "kind": "per_sample" if m in SAMPLE_METRICS else "per_episode",
                             "wasserstein_to_human": w_point[li, i, j], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _relative_reduction(w: np.ndarray, x: int, ref: int) -> float:
    """mean over (layout, metric) of 1 - W_x / W_ref, skipping pairs with W_ref == 0 or either missing. w: (L, S, M)."""
    wx, wr = w[:, x, :], w[:, ref, :]
    ok = np.isfinite(wx) & np.isfinite(wr) & (wr > 0)
    return float(np.mean(1.0 - wx[ok] / wr[ok])) if ok.any() else np.nan


def headline_table(layout_names, w_point, w_boot, s_point, s_boot) -> pd.DataFrame:
    ref_w, ref_s = COMPARED.index(HEADLINE_VS), LABELS.index(HEADLINE_VS)
    rows = []
    for label in CONDITIONS:
        if label == HEADLINE_VS:
            continue
        xw, xs = COMPARED.index(label), LABELS.index(label)
        scopes = [("all", slice(None))] + [(name, slice(li, li + 1)) for li, name in enumerate(layout_names)]
        for scope, sl in scopes:
            red = _relative_reduction(w_point[sl], xw, ref_w)
            red_reps = np.array([_relative_reduction(w_boot[b, sl], xw, ref_w) for b in range(len(w_boot))])
            ds = float(np.nanmean(s_point[sl, xs] - s_point[sl, ref_s]))
            with np.errstate(all="ignore"):
                ds_reps = np.nanmean(s_boot[:, sl, xs] - s_boot[:, sl, ref_s], axis=1)
            rows.append({"comparison": f"{label} vs {HEADLINE_VS}", "layout": scope,
                         "relative_W_reduction": red, "rel_red_ci_low": _ci(red_reps)[0], "rel_red_ci_high": _ci(red_reps)[1],
                         "success_change": ds, "success_ci_low": _ci(ds_reps)[0], "success_ci_high": _ci(ds_reps)[1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "legend.frameon": False})
    return plt


def _draw_layout(ax, layout: Layout):
    ax.imshow(np.ma.masked_where(~layout.occupancy, layout.occupancy), origin="lower", extent=layout.extent,
              cmap=_single(WALL_COLOR), vmin=0, vmax=1, interpolation="nearest", zorder=3)
    ax.set_xlim(layout.extent[:2])
    ax.set_ylim(layout.extent[2:])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")


def _draw_markers(ax, layout: Layout):
    kw = dict(zorder=6, color="#0b0b0b", edgecolors="white", linewidths=1.0)
    ax.scatter(*layout.start[[0, 2]], marker="o", s=45, **kw)
    ax.scatter(layout.pois[:, 0], layout.pois[:, 2], marker="*", s=110, **kw)
    ax.scatter(*layout.end[[0, 2]], marker="s", s=40, **kw)
    for name, p in [("start", layout.start)] + [("POI", q) for q in layout.pois] + [("end", layout.end)]:
        ax.annotate(name, (p[0], p[2]), xytext=(5, 5), textcoords="offset points", fontsize=7, color="#2b2b2b", zorder=6)


def _single(color):
    from matplotlib.colors import ListedColormap
    return ListedColormap([color])


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def _panel_size(layout: Layout, width: float = 5.0) -> tuple[float, float]:
    x0, x1, z0, z1 = layout.extent
    return width, width * (z1 - z0) / max(1e-6, x1 - x0)


def figure_trajectories(df: pd.DataFrame, layouts: dict[str, Layout], names: list[str], out_dir: Path) -> list[Path]:
    from matplotlib.lines import Line2D
    plt = _plt()
    w, h = _panel_size(layouts[names[0]])
    fig, axes = plt.subplots(1, len(names), figsize=(w * len(names), h + 0.9), squeeze=False)
    present = [label for label in LABELS if (df["label"] == label).any()]
    for ax, name in zip(axes[0], names):
        layout = layouts[name]
        _draw_layout(ax, layout)
        sub = df[df["layout"] == name]
        for zi, label in enumerate(present):
            eps = sub[sub["label"] == label].groupby("episode", sort=False)
            alpha = float(np.clip(3.0 / max(1, eps.ngroups), 0.12, 0.8))
            for _, ep in eps:
                ax.plot(ep["x"], ep["z"], color=COLORS[label], ls=LINESTYLES.get(label, "-"), lw=1.0, alpha=alpha,
                        zorder=4 + zi * 0.01, solid_capstyle="round")
        _draw_markers(ax, layout)
        ax.set_title(name, fontsize=10, loc="left")
    handles = [Line2D([], [], color=COLORS[label], ls=LINESTYLES.get(label, "-"), lw=2, label=label) for label in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths = _save(fig, out_dir, "trajectories")
    plt.close(fig)
    return paths


def figure_occupancy(df: pd.DataFrame, layouts: dict[str, Layout], names: list[str], out_dir: Path) -> list[Path]:
    """Share of samples per HEATMAP_BIN_M cell, per layout (rows) x source (columns); one log colour scale per row."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    plt = _plt()
    present = [label for label in LABELS if (df["label"] == label).any()]
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#8fb8ea", "#2a78d6", "#0d3f80", "#061f40"])
    w, h = _panel_size(layouts[names[0]], 3.2)
    fig, axes = plt.subplots(len(names), len(present), figsize=(w * len(present) + 0.8, h * len(names) + 0.4),
                             squeeze=False)
    for r, name in enumerate(names):
        layout = layouts[name]
        x0, x1, z0, z1 = layout.extent
        bx = np.arange(x0, x1 + HEATMAP_BIN_M, HEATMAP_BIN_M)
        bz = np.arange(z0, z1 + HEATMAP_BIN_M, HEATMAP_BIN_M)
        hists = {}
        for label in present:
            sub = df[(df["layout"] == name) & (df["label"] == label)]
            H, _, _ = np.histogram2d(sub["z"], sub["x"], bins=[bz, bx])
            hists[label] = H / H.sum() if H.sum() else H
        positive = np.concatenate([H[H > 0] for H in hists.values()]) if hists else np.zeros(0)
        norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if len(positive) else None
        im = None
        for c, label in enumerate(present):
            ax = axes[r, c]
            H = np.ma.masked_where(hists[label] <= 0, hists[label])
            im = ax.imshow(H, origin="lower", extent=(bx[0], bx[-1], bz[0], bz[-1]), cmap=cmap, norm=norm,
                           interpolation="nearest")
            _draw_layout(ax, layout)
            n_eps = df[(df["layout"] == name) & (df["label"] == label)]["episode"].nunique()
            ax.set_title(f"{label}  (n={n_eps})" if r == 0 else f"n={n_eps}", fontsize=9, loc="left")
            if c:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"{name}\nz (m)")
            if r < len(names) - 1:
                ax.set_xlabel("")
        if im is not None:
            fig.colorbar(im, ax=list(axes[r]), shrink=0.8, pad=0.01, label="share of samples")
    paths = _save(fig, out_dir, "occupancy")
    plt.close(fig)
    return paths


def figure_wasserstein(wtab: pd.DataFrame, names: list[str], out_dir: Path) -> list[Path]:
    plt = _plt()
    markers = ["o", "s", "^", "D", "v", "P"]
    ncols = 4
    nrows = int(np.ceil(len(METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows + 0.5), squeeze=False)
    offsets = np.linspace(-0.2, 0.2, len(names)) if len(names) > 1 else np.zeros(1)
    for k, m in enumerate(METRICS):
        ax = axes.flat[k]
        sub = wtab[wtab["metric"] == m]
        for i, label in enumerate(COMPARED):
            for li, name in enumerate(names):
                row = sub[(sub["source"] == label) & (sub["layout"] == name)]
                if row.empty or not np.isfinite(row["wasserstein_to_human"].iloc[0]):
                    continue
                y = row["wasserstein_to_human"].iloc[0]
                lo, hi = row["ci_low"].iloc[0], row["ci_high"].iloc[0]
                ax.errorbar(i + offsets[li], y, yerr=[[max(0.0, y - lo)], [max(0.0, hi - y)]], fmt=markers[li % len(markers)],
                            color=COLORS[label], ms=5, lw=1.2, capsize=0, mec="white", mew=0.8)
        ax.set_xticks(range(len(COMPARED)), COMPARED)
        ax.set_title(METRIC_TITLES[m], fontsize=9, loc="left")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(METRICS):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#52514e", marker=markers[li % len(markers)], ls="", label=name)
               for li, name in enumerate(names)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), title="layout", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Wasserstein distance to held-out humans (lower = more human-like), 95% bootstrap CI",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    paths = _save(fig, out_dir, "wasserstein")
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path, help="YAML/JSON manifest (see above)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results",
                    help="parent folder of the timestamped results folder (default eval_tools/results)")
    ap.add_argument("--n-boot", type=int, help="bootstrap replicates (overrides the manifest)")
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    n_boot = args.n_boot if args.n_boot is not None else man["n_boot"]
    df, layouts = load_all(man)
    names = sorted(df["layout"].unique())
    for name in names:
        labels = set(df.loc[df["layout"] == name, "label"])
        missing = [label for label in LABELS if label not in labels]
        if missing:
            print(f"[warn] {name}: no episodes for {missing}")

    episodes = compute_episodes(df, layouts, man["timeout_s"])
    summary = summary_table(episodes)
    print(f"[compare] {len(episodes)} episodes, {len(names)} layout(s); bootstrap {n_boot} replicates ...", flush=True)
    w_point, w_boot, s_point, s_boot = bootstrap(episodes, names, n_boot, man["seed"])
    wtab = wasserstein_table(names, w_point, w_boot)
    headline = headline_table(names, w_point, w_boot, s_point, s_boot)

    out_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.manifest, out_dir / ("manifest" + args.manifest.suffix))
    episodes.drop(columns=[c for c in episodes.columns if c.startswith("_")]).to_csv(out_dir / "episodes.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    wtab.to_csv(out_dir / "wasserstein.csv", index=False)
    headline.to_csv(out_dir / "headline.csv", index=False)
    figure_trajectories(df, layouts, names, out_dir)
    figure_occupancy(df, layouts, names, out_dir)
    figure_wasserstein(wtab, names, out_dir)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\nSuccess (Wilson 95% CI):")
    print(summary[summary["metric"] == "success"][["layout", "source", "n", "mean", "ci_low", "ci_high"]]
          .rename(columns={"mean": "rate"}).to_string(index=False))
    print("\nWasserstein distance to Human (CIs in wasserstein.csv):")
    print(wtab.pivot_table(index=["layout", "metric"], columns="source", values="wasserstein_to_human", sort=False)
          .reindex(columns=[c for c in COMPARED if c in set(wtab["source"])]).to_string())
    print("\nHeadline:")
    print(headline.to_string(index=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
