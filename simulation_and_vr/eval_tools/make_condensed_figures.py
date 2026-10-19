#!/usr/bin/env python3
"""Merged results-chapter figures for the condensed thesis, from one compare_conditions.py results folder.

Same visual grammar as make_final_figures.py (conditions A/B/C light-to-dark blue, shortest path grey, humans orange,
layouts in a fixed order). Each figure merges several earlier ones into one page-saving grid:

  fig_c1_trajectories  all episodes per source (columns) and layout (rows); dashed: timeout, ×: end position of a
                       failed episode. Replaces fig_trajectories_<layout> (x3) and fig_f5_failures.
  fig_c2_behaviour     stop share, yaw rate and wall distance per episode (rows) and layout (columns).
                       Replaces fig_f2_stopping, fig_f2b_yaw and fig_f2c_walls.

The remaining figures (fig_f1_success, fig_f3b_metrics, fig_f4_effects, fig_f7_heatmap_differences) come unchanged
from make_final_figures.py.

run_info.json stores absolute paths from the machine the evaluation ran on. If such a path no longer exists, it is
rebased onto this repository (the part after 'bachelorthesis-iljamatthiasblass/').

Usage:
  python make_condensed_figures.py RESULTS_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import compare_conditions as cc
from make_final_figures import LAYOUT_ORDER, _layouts
from make_results_figures import short
from make_thesis_figures import INK, MUTED, _style, load_all

REPO_NAME = "bachelorthesis-iljamatthiasblass"
REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCES = [cc.SP, "A", "B", "C", cc.HUMAN]
NAMES = {cc.SP: "Shortest path", "A": "A", "B": "B", "C": "C", cc.HUMAN: "Humans"}


def _rebase(p: str) -> str:
    if not p or Path(p).exists() or f"{REPO_NAME}/" not in p:
        return p
    return str(REPO_ROOT / p.split(f"{REPO_NAME}/", 1)[1])


def _load_info(res: Path) -> dict:
    info = json.loads((res / "run_info.json").read_text())
    for k in ("recordings", "layouts_dir", "gates"):
        if k in info:
            info[k] = _rebase(info[k])
    info["runs"] = [_rebase(r) for r in info["runs"]]
    missing = [r for r in info["runs"] + [info["recordings"], info["layouts_dir"]] if not Path(r).exists()]
    if missing:
        sys.exit(f"missing input paths, e.g. {missing[0]}")
    return info


# ----------------------------------------------------------------------------------------------------------------------
# C1: trajectories and failure end positions
# ----------------------------------------------------------------------------------------------------------------------

def fig_trajectories(plt, episodes, layouts_obj: dict, out: Path):
    """All episodes per source and layout; failed episodes dashed with their end position marked."""
    layouts = sorted(layouts_obj, key=lambda l: LAYOUT_ORDER.index(short(l)) if short(l) in LAYOUT_ORDER else 99)
    fig, axes = plt.subplots(len(layouts), len(SOURCES), figsize=(6.3, 4.4), squeeze=False)
    for i, l in enumerate(layouts):
        L = layouts_obj[l]
        for j, src in enumerate(SOURCES):
            ax = axes[i, j]
            cc._draw_layout(ax, L)
            eps = [e for e in episodes if e.layout == l and e.source == src]
            alpha = 0.75 if src in (cc.HUMAN, cc.SP) else 0.28
            for e in eps:
                k = e.keep
                ax.plot(k["x"], k["z"], color=cc.COLORS[src], lw=0.5, alpha=alpha, zorder=3,
                        ls="-" if e.metrics["success"] else (0, (2, 1.5)))
            fails = [e for e in eps if not e.metrics["success"]]
            if fails:
                end = np.array([[e.keep["x"].iloc[-1], e.keep["z"].iloc[-1]] for e in fails])
                ax.scatter(end[:, 0], end[:, 1], marker="x", s=6, color=INK, lw=0.5, alpha=0.6, zorder=5)
            cc._draw_markers(ax, L)
            n_ok = int(sum(e.metrics["success"] for e in eps))
            ax.text(0.5, -0.03, f"{n_ok}/{len(eps)} successful", transform=ax.transAxes, fontsize=6,
                    ha="center", va="top", color=MUTED, zorder=7)
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(NAMES[src], loc="center")
            if j == 0:
                ax.set_ylabel(short(l), fontsize=7.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.04, hspace=0.16)
    return cc._save(fig, out, "fig_c1_trajectories")


# ----------------------------------------------------------------------------------------------------------------------
# C2: behavioural metrics
# ----------------------------------------------------------------------------------------------------------------------

BEHAVIOUR_ROWS = [("stop_share", 100, "Stop share\n(% of samples)", (-3, 100), [0, 25, 50, 75, 100]),
                  ("yaw_rate_mean_deg_s", 1, "Yaw rate (°/s)", (0, 76), [0, 25, 50, 75]),
                  ("wall_distance_mean_m", 1, "Wall distance,\nwalking (m)", (0, 2.2), [0, 0.5, 1.0, 1.5, 2.0])]


def fig_behaviour(plt, eps: pd.DataFrame, out: Path):
    """Per-episode values (dots) and mean (dash) per source; band: range of the human test walks.

    Same strips as fig_f2_stopping / fig_f2b_yaw / fig_f2c_walls (identical jitter), stacked as rows."""
    layouts = _layouts(eps)
    xnames = {cc.SP: "Shortest\npath", cc.HUMAN: "Humans"}
    fig, axes = plt.subplots(len(BEHAVIOUR_ROWS), len(layouts), figsize=(6.3, 6.2), squeeze=False, sharey="row")
    for r, (column, scale, ylabel, ylim, yticks) in enumerate(BEHAVIOUR_ROWS):
        rng = np.random.default_rng(0)
        for c, l in enumerate(layouts):
            ax = axes[r, c]
            human = scale * eps[(eps.layout == l) & (eps.source == cc.HUMAN)][column].to_numpy()
            ax.axhspan(human.min(), human.max(), color=cc.COLORS[cc.HUMAN], alpha=0.12, lw=0, zorder=0)
            for i, s in enumerate(SOURCES):
                v = scale * eps[(eps.layout == l) & (eps.source == s)][column].dropna().to_numpy()
                alpha, size = (0.9, 9) if s == cc.HUMAN else (0.25, 5)
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=size, color=cc.COLORS[s], alpha=alpha, lw=0, zorder=2)
                ax.hlines(v.mean(), i - 0.3, i + 0.3, color=cc.COLORS[s], lw=2.2, zorder=3)
            if r == 0:
                ax.set_title(short(l), loc="left")
            ax.set_xticks(range(len(SOURCES)))
            ax.set_xticklabels([xnames.get(s, s) for s in SOURCES] if r == len(BEHAVIOUR_ROWS) - 1 else [], fontsize=7)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.6, len(SOURCES) - 0.4)
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if c == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(wspace=0.08, hspace=0.22)
    return cc._save(fig, out, "fig_c2_behaviour")


# ----------------------------------------------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    plt = cc._plt()
    _style(plt)
    for p in fig_behaviour(plt, pd.read_csv(res / "episodes.csv"), out):
        print(f"  wrote {p}")
    print("loading episodes (trajectories) ...")
    _, episodes, layouts_obj, _ = load_all(_load_info(res))
    for p in fig_trajectories(plt, episodes, layouts_obj, out):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
