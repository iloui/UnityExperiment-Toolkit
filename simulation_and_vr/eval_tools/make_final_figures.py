#!/usr/bin/env python3
"""Final results-chapter figures, one core message each, from one compare_conditions.py results folder.

Visual grammar shared by all figures (same as the trajectory overlay): layouts side by side in a fixed order,
conditions A/B/C in light-to-dark blue, shortest path grey, humans orange.

Figures (PNG + PDF in OUT_DIR):
  fig_f1_success   success rate per condition and layout                                   (Section 4.2)
  fig_f2_stopping  stop share per episode: baseline never stops, agents stop far too often   (Section 4.4)
  fig_f2b_yaw      yaw rate per episode: human level in CentralPillar, far off in TJunction  (Section 4.4)
  fig_f2c_walls    wall distance per episode: agents walk closer to walls than everyone      (Section 4.4)
  fig_f3_scores    score A -> B -> C per layout: closer to humans, still below the baseline   (Section 4.5)
  fig_f3b_metrics  per metric: agents beat the baseline only on looking around (yaw rate)   (Sections 4.4/4.5)
  fig_f4_effects   ΔS with 95 % CI: transfer (B - A) detected, memorisation (C - B) not       (Section 4.5)
  fig_f5_failures  end positions of failed episodes on the floor plan: failures cluster      (Section 4.3)
  fig_f6_route_side  share passing north around the central obstacle vs. humans (H3)       (Section 4.7)
  fig_f7_heatmap_differences  time share condition minus humans per floor cell             (Section 4.2)

fig_f3_scores draws each model as a small marker if RESULTS_DIR/sensitivity/per_model_scores.csv exists
(written by sensitivity_checks.py).

Usage:
  python make_final_figures.py RESULTS_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import compare_conditions as cc
from make_results_figures import short
from make_thesis_figures import INK, MUTED, _style, load_all

LAYOUT_ORDER = ["CentralPillar", "MultiRoomDoorway", "TJunction"]


def _layouts(df: pd.DataFrame) -> list[str]:
    present = {l for l in df.layout if l != "overall"}
    return sorted(present, key=lambda l: LAYOUT_ORDER.index(short(l)) if short(l) in LAYOUT_ORDER else 99)


def _pct(v: float) -> str:
    return f"{100 * v:.1f} %" if 0 < v < 0.1 else f"{100 * v:.0f} %"


# ----------------------------------------------------------------------------------------------------------------------
# F1: success
# ----------------------------------------------------------------------------------------------------------------------

def fig_success(plt, success: pd.DataFrame, out: Path):
    """Message: human data raises success where agents can succeed at all; TJunction defeats every condition."""
    layouts = _layouts(success)
    mean = success[success.model == "(condition mean)"]
    refs = success[success.source.isin([cc.SP, cc.HUMAN]) & (success.model != "(condition mean)")]
    assert (refs.success_rate == 1.0).all(), "reference line assumes baseline and humans always succeed"

    fig, axes = plt.subplots(1, len(layouts), figsize=(6.3, 2.1), sharey=True)
    for ax, l in zip(axes, layouts):
        for i, c in enumerate(cc.CONDITIONS):
            r = mean[(mean.layout == l) & (mean.source == c)].iloc[0]
            ax.bar(i, 100 * r.success_rate, width=0.62, color=cc.COLORS[c], zorder=2)
            ax.vlines(i, 100 * r.ci_low, 100 * r.ci_high, color=INK, lw=0.8, zorder=3)
            ax.text(i, 100 * max(r.success_rate, r.ci_high) + 4, _pct(r.success_rate), ha="center", va="bottom",
                    fontsize=7, color=INK)
        ax.axhline(100, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_title(short(l), loc="left")
        ax.set_xticks(range(len(cc.CONDITIONS)))
        ax.set_xticklabels(cc.CONDITIONS)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(cc.CONDITIONS) - 0.4)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 50, 100])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Success (%)")
    axes[-1].text(len(cc.CONDITIONS) - 0.4, 98, "shortest path and humans", ha="right", va="top",
                  fontsize=6.5, color=MUTED)
    fig.subplots_adjust(wspace=0.12)
    return cc._save(fig, out, "fig_f1_success")


# ----------------------------------------------------------------------------------------------------------------------
# F2: stopping
# ----------------------------------------------------------------------------------------------------------------------

def _episode_strips(plt, eps: pd.DataFrame, out: Path, column: str, scale: float, ylabel: str, ylim, yticks,
                    name: str):
    """Per-episode values (dots) and mean (dash) per source and layout; band: range of the human test walks."""
    layouts = _layouts(eps)
    sources = [cc.SP, "A", "B", "C", cc.HUMAN]
    names = {cc.SP: "Shortest\npath", cc.HUMAN: "Humans"}
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(layouts), figsize=(6.3, 2.4), sharey=True)
    for ax, l in zip(axes, layouts):
        human = scale * eps[(eps.layout == l) & (eps.source == cc.HUMAN)][column].to_numpy()
        ax.axhspan(human.min(), human.max(), color=cc.COLORS[cc.HUMAN], alpha=0.12, lw=0, zorder=0)
        for i, s in enumerate(sources):
            v = scale * eps[(eps.layout == l) & (eps.source == s)][column].dropna().to_numpy()
            alpha, size = (0.9, 9) if s == cc.HUMAN else (0.25, 5)
            ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=size, color=cc.COLORS[s], alpha=alpha, lw=0, zorder=2)
            ax.hlines(v.mean(), i - 0.3, i + 0.3, color=cc.COLORS[s], lw=2.2, zorder=3)
        ax.set_title(short(l), loc="left")
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([names.get(s, s) for s in sources], fontsize=7)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(sources) - 0.4)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel(ylabel)
    fig.subplots_adjust(wspace=0.08)
    return cc._save(fig, out, name)


def fig_stopping(plt, eps: pd.DataFrame, out: Path):
    """Message: the baseline errs one way (never stops), the learned agents the other (stop far too often)."""
    return _episode_strips(plt, eps, out, "stop_share", 100, "Stop share (% of samples)", (-3, 100),
                           [0, 25, 50, 75, 100], "fig_f2_stopping")


def fig_yaw(plt, eps: pd.DataFrame, out: Path):
    """Message: human data brings looking around to human level where agents succeed; TJunction stays far off."""
    return _episode_strips(plt, eps, out, "yaw_rate_mean_deg_s", 1, "Yaw rate (°/s)", (0, 76),
                           [0, 25, 50, 75], "fig_f2b_yaw")


def fig_walls(plt, eps: pd.DataFrame, out: Path):
    """Message: agents walk closer to walls than humans and the baseline; human data widens the distance."""
    return _episode_strips(plt, eps, out, "wall_distance_mean_m", 1, "Wall distance, walking (m)", (0, 2.2),
                           [0, 0.5, 1.0, 1.5, 2.0], "fig_f2c_walls")


# ----------------------------------------------------------------------------------------------------------------------
# F3: score trajectory A -> B -> C
# ----------------------------------------------------------------------------------------------------------------------

FLOOR = "human noise floor"


def fig_scores(plt, scores: pd.DataFrame, out: Path, model_scores: pd.DataFrame | None = None):
    """Message: human data moves agents toward humans (A -> B), yet every condition stays below the baseline (S = 0).

    Point estimates only: the paired differences that test H1/H2 carry their intervals in F4. With model_scores
    (layout, source, fold, score; layout 'overall' = mean over layouts), each model is drawn as a small marker
    beside its condition mean: square = A, circle = trained on F1, triangle = trained on F2."""
    panels = _layouts(scores) + ["overall"]
    conds = list(cc.CONDITIONS)
    fold_marker = {0: "s", 1: "o", 2: "^"}
    fold_dx = {0: 0.2, 1: 0.16, 2: 0.26}

    fig, grid = plt.subplots(1, len(panels) + 1, figsize=(6.3, 2.6), sharey=True,
                             gridspec_kw={"width_ratios": [1] * (len(panels) - 1) + [0.05, 0.85], "wspace": 0.3})
    grid[-2].set_visible(False)                       # spacer: sets Overall apart from the layouts
    axes = list(grid[:-2]) + [grid[-1]]
    for ax, l in zip(axes, panels):
        sel = scores[scores.layout == l].set_index("source")
        y = [sel.score[c] for c in conds]
        ax.axhline(0, color=cc.COLORS[cc.SP], lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.axhline(sel.score[FLOOR], color=cc.COLORS[cc.HUMAN], lw=1.2, zorder=1)
        ax.plot(range(len(conds)), y, color=MUTED, lw=1.0, zorder=2)
        for i, c in enumerate(conds):
            ax.plot(i, y[i], marker="o", ms=6.5, color=cc.COLORS[c], mec="white", mew=1.2, ls="", zorder=3)
            if model_scores is not None:
                ms = model_scores[(model_scores.layout == (l if l == "overall" else short(l))) & (model_scores.source == c)]
                for _, r in ms.iterrows():
                    ax.plot(i + fold_dx[int(r.fold)], r.score, marker=fold_marker[int(r.fold)], ms=3.2, ls="",
                            mfc="white", mec=cc.COLORS[c], mew=0.9, zorder=3)
        ax.set_title("Overall" if l == "overall" else short(l), loc="center")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(conds) - 0.5 + (0.15 if model_scores is not None else 0))
        ax.set_ylim(-6.4 if model_scores is not None else -4.3, 1.0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_facecolor("#f4f4f4")
    axes[0].set_ylabel("Human-likeness score S")
    right = axes[-1]
    right.text(len(conds) - 0.55, scores[(scores.layout == "overall") & (scores.source == FLOOR)].score.iloc[0] + 0.1,
               "second human group", ha="right", va="bottom", fontsize=6.5, color=cc.COLORS[cc.HUMAN])
    right.text(len(conds) - 0.55, 0.1, "shortest path", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    if model_scores is not None:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", ls="", ms=6, color=MUTED, mec="white", label="condition mean")]
        handles += [Line2D([], [], marker=fold_marker[f], ls="", ms=3.5, mfc="white", mec=MUTED, mew=0.9, label=lab)
                    for f, lab in ((0, "model (A)"), (1, "model trained on F1"), (2, "model trained on F2"))]
        axes[0].legend(handles=handles, frameon=False, loc="lower left", fontsize=6.2, handletextpad=0.2,
                       borderaxespad=0.1, labelspacing=0.25)
    return cc._save(fig, out, "fig_f3_scores")


# ----------------------------------------------------------------------------------------------------------------------
# F3b: per-metric distance relative to the baseline
# ----------------------------------------------------------------------------------------------------------------------

F3B_METRICS = [("yaw_rate_deg_s", "Yaw rate"), ("path_efficiency", "Path efficiency"),
               ("stop_share", "Stop share"), ("wall_distance_m", "Wall distance")]


def fig_metric_ratios(plt, distances: pd.DataFrame, out: Path):
    """Message: agents beat the baseline only on looking around; stopping and wall distance keep them behind.

    x = W(condition, humans) / W(shortest path, humans) = 1 - S, log scale: < 1 closer to humans than the baseline.
    Only metric-layout pairs that enter the layout score are drawn."""
    layouts = _layouts(distances)
    conds = list(cc.CONDITIONS)
    offset = {"A": -0.2, "B": 0.0, "C": 0.2}
    d = distances[distances.source.isin(conds) & (distances.in_layout_score == True)]  # noqa: E712

    fig, axes = plt.subplots(1, len(layouts), figsize=(6.3, 2.2), sharey=True, sharex=True)
    for ax, l in zip(axes, layouts):
        ax.axvspan(0.1, 1.0, color="#f4f4f4", lw=0, zorder=0)
        ax.axvline(1.0, color=cc.COLORS[cc.SP], lw=0.9, ls=(0, (3, 2)), zorder=1)
        for row, (k, _) in enumerate(F3B_METRICS):
            sel = d[(d.layout == l) & (d.metric == k)].set_index("source")
            if sel.empty:
                ax.text(1.0, row, "not scored", ha="center", va="center", fontsize=6.5, color=MUTED,
                        bbox=dict(fc="white", ec="none", pad=1), zorder=2)
                continue
            r0 = sel.iloc[0]
            ax.plot(r0.noise_floor / r0.W_shortest_path, row, marker="|", ms=9, mew=1.6,
                    color=cc.COLORS[cc.HUMAN], zorder=2)
            for c in conds:
                ax.plot(sel.W_to_humans[c] / sel.W_shortest_path[c], row + offset[c], marker="o", ms=5,
                        color=cc.COLORS[c], mec="white", mew=0.8, ls="", zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.2, 12)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["¼×", "½×", "1×", "2×", "4×", "8×"])
        ax.minorticks_off()
        ax.set_title(short(l), loc="center")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(F3B_METRICS)))
    axes[0].set_yticklabels([lab for _, lab in F3B_METRICS])
    axes[0].set_ylim(len(F3B_METRICS) - 0.5, -0.9)
    axes[1].set_xlabel("Distance to humans relative to the shortest path")
    for ax in axes:
        ax.text(0.93, -0.62, "← closer", ha="right", va="center", fontsize=6.5, color=MUTED)
        ax.text(1.07, -0.62, "further →", ha="left", va="center", fontsize=6.5, color=MUTED)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=cc.COLORS[c], ms=5, label=c) for c in conds]
    handles.append(Line2D([], [], marker="|", ls="", color=cc.COLORS[cc.HUMAN], ms=9, mew=1.6,
                          label="second human group"))
    axes[-1].legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5,
                    handletextpad=0.3)
    fig.subplots_adjust(wspace=0.12)
    return cc._save(fig, out, "fig_f3b_metrics")


# ----------------------------------------------------------------------------------------------------------------------
# F4: effects (H1, H2)
# ----------------------------------------------------------------------------------------------------------------------

def fig_effects(plt, diffs: pd.DataFrame, out: Path):
    """Message: adding human data helps (B - A > 0); seeing the test route adds nothing detectable overall (C - B).

    Overall (the tested quantity) on top and emphasised; layouts below, smaller and descriptive."""
    d = diffs[diffs.measure == "score"]
    rows = ["overall"] + _layouts(d)
    tests = [("B - A", "H1  Transfer (B − A)", cc.COLORS["B"]), ("C - B", "H2  Memorization (C − B)", cc.COLORS["C"])]
    ypos = {"overall": 0.0, **{l: 1.4 + i for i, l in enumerate(rows[1:])}}

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9), sharey=True, sharex=True)
    for ax, (comp, title, color) in zip(axes, tests):
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        for l in rows:
            r = d[(d.comparison == comp) & (d.layout == l)].iloc[0]
            main = l == "overall"
            ax.hlines(ypos[l], r.ci_low, r.ci_high, color=color if main else MUTED, lw=2.2 if main else 1.2, zorder=2)
            ax.plot(r.estimate, ypos[l], marker="o", ms=8 if main else 5.5, ls="",
                    mfc=(color if main else MUTED) if r.ci_excludes_zero else "white",
                    mec=color if main else MUTED, mew=1.4, zorder=3)
        ax.set_title(title, loc="left")
        ax.set_xlim(-1.5, 2.0)
        ax.set_xticks([-1, 0, 1, 2])
        ax.set_xlabel("Δ human-likeness score")
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks([ypos[l] for l in rows])
    axes[0].set_yticklabels(["Overall" if l == "overall" else short(l) for l in rows])
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    axes[0].set_ylim(ypos[rows[-1]] + 0.6, -0.6)
    fig.subplots_adjust(wspace=0.1)
    return cc._save(fig, out, "fig_f4_effects")


# ----------------------------------------------------------------------------------------------------------------------
# F5: where agents fail
# ----------------------------------------------------------------------------------------------------------------------

def fig_failures(plt, episodes, layouts_obj: dict, out: Path):
    """Message: failures are not random; agents get stuck in a few specific places."""
    layouts = sorted(layouts_obj, key=lambda l: LAYOUT_ORDER.index(short(l)) if short(l) in LAYOUT_ORDER else 99)
    conds = list(cc.CONDITIONS)
    fig, axes = plt.subplots(len(layouts), len(conds), figsize=(6.3, 6.6), squeeze=False)
    for i, l in enumerate(layouts):
        for j, c in enumerate(conds):
            ax = axes[i, j]
            cc._draw_layout(ax, layouts_obj[l])
            eps = [e for e in episodes if e.layout == l and e.source == c]
            fails = [e for e in eps if not e.metrics["success"]]
            if fails:
                end = np.array([[e.keep["x"].iloc[-1], e.keep["z"].iloc[-1]] for e in fails])
                ax.scatter(end[:, 0], end[:, 1], s=9, color=cc.COLORS[c], alpha=0.55, lw=0, zorder=4)
            cc._draw_markers(ax, layouts_obj[l])
            ax.text(0.5, -0.03, f"{len(fails)} of {len(eps)} failed", transform=ax.transAxes, fontsize=6.5,
                    ha="center", va="top", color=MUTED, zorder=7)
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(c, loc="center")
            if j == 0:
                ax.set_ylabel(short(l), fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.04, hspace=0.14)
    return cc._save(fig, out, "fig_f5_failures")


# ----------------------------------------------------------------------------------------------------------------------
# F6: route side (H3)
# ----------------------------------------------------------------------------------------------------------------------

def fig_route_side(plt, route: pd.DataFrame, out: Path):
    """Message: in CentralPillar, B and C match the human side preference more closely than A and the baseline; in
    TJunction too few agent episodes reach the decision to say anything.

    x = share passing north among the episodes that crossed a gate, 95 % CI; humans as orange reference line."""
    layouts = [l for l in _layouts(route)]
    sources = [cc.SP, "A", "B", "C"]
    names = {cc.SP: "Shortest path", "A": "A", "B": "B", "C": "C"}
    fig, axes = plt.subplots(1, len(layouts), figsize=(6.3, 1.9), sharey=True)
    for ax, l in zip(np.atleast_1d(axes), layouts):
        sel = route[route.layout == l].set_index("source")
        hum = sel.loc[cc.HUMAN]
        ax.axvline(hum.share_north, color=cc.COLORS[cc.HUMAN], lw=1.4, zorder=1)
        ax.text(hum.share_north, -0.75, f"humans {int(hum.north)}/{int(hum.crossed)}", ha="center", va="bottom",
                fontsize=6.5, color=cc.COLORS[cc.HUMAN])
        for i, s in enumerate(sources):
            r = sel.loc[s]
            enough = r.crossed >= 5
            if enough and s != cc.SP:
                ax.hlines(i, r.ci_low, r.ci_high, color=cc.COLORS[s], lw=1.4, zorder=2)
            if r.crossed > 0:
                ax.plot(r.share_north, i, marker="o", ms=5.5, ls="", zorder=3, mec=cc.COLORS[s], mew=1.2,
                        mfc=cc.COLORS[s] if enough else "white")
            ax.text(1.04, i, f"n = {int(r.crossed)}", va="center", fontsize=6.5, color=MUTED, clip_on=False)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0 %", "50 %", "100 %"])
        ax.set_xlabel("Share passing north")
        ax.set_title(short(l), loc="center", pad=12)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(sources)))
    ax0.set_yticklabels([names[s] for s in sources])
    ax0.set_ylim(len(sources) - 0.5, -0.9)
    fig.subplots_adjust(wspace=0.35)
    return cc._save(fig, out, "fig_f6_route_side")


# ----------------------------------------------------------------------------------------------------------------------
# F7: where agents spend time compared with humans
# ----------------------------------------------------------------------------------------------------------------------

def fig_heatmap_differences(plt, episodes, layouts_obj: dict, out: Path):
    """Message: agents spend less time than humans on the human routes and more along walls and in dead ends.

    Time share per 0.25 m cell (each episode weighted equally, folds pooled), condition minus humans."""
    from matplotlib.colors import TwoSlopeNorm
    layouts = sorted(layouts_obj, key=lambda l: LAYOUT_ORDER.index(short(l)) if short(l) in LAYOUT_ORDER else 99)
    conds = list(cc.CONDITIONS)
    fig, axes = plt.subplots(len(layouts), len(conds), figsize=(6.3, 6.6), squeeze=False)
    for i, l in enumerate(layouts):
        L = layouts_obj[l]
        hum = cc._time_share([e for e in episodes if e.layout == l and e.source == cc.HUMAN], L)
        diffs = {c: 100 * (cc._time_share([e for e in episodes if e.layout == l and e.source == c], L) - hum) for c in conds}
        vmax = max(float(np.percentile(np.abs(d[d != 0]), 99)) for d in diffs.values())
        for j, c in enumerate(conds):
            ax = axes[i, j]
            im = ax.imshow(np.ma.masked_equal(diffs[c], 0), origin="lower", extent=L.extent, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest", zorder=2)
            cc._draw_layout(ax, L)
            cc._draw_markers(ax, L)
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(f"{c} − humans", loc="center")
            if j == 0:
                ax.set_ylabel(short(l), fontsize=8.5, fontweight="bold")
        cb = fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.8, pad=0.02)
        cb.set_label("Δ time share (pp)", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
    return cc._save(fig, out, "fig_f7_heatmap_differences")


# ----------------------------------------------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    success = pd.read_csv(res / "success.csv")
    eps = pd.read_csv(res / "episodes.csv")
    scores = pd.read_csv(res / "scores.csv")
    distances = pd.read_csv(res / "distances.csv")
    diffs = pd.read_csv(res / "differences.csv")
    route = pd.read_csv(res / "route_side.csv")
    pm_path = res / "sensitivity" / "per_model_scores.csv"
    model_scores = pd.read_csv(pm_path) if pm_path.exists() else None
    if model_scores is None:
        print(f"  ({pm_path} missing: run sensitivity_checks.py first for per-model markers in fig_f3_scores)")

    plt = cc._plt()
    _style(plt)
    for p in fig_success(plt, success, out) + fig_stopping(plt, eps, out) + fig_yaw(plt, eps, out) + fig_walls(plt, eps, out) \
            + fig_scores(plt, scores, out, model_scores) \
            + fig_metric_ratios(plt, distances, out) + fig_effects(plt, diffs, out) + fig_route_side(plt, route, out):
        print(f"  wrote {p}")
    print("loading episodes (failure positions, time shares) ...")
    _, episodes, layouts_obj, _ = load_all(json.loads((res / "run_info.json").read_text()))
    for p in fig_failures(plt, episodes, layouts_obj, out) + fig_heatmap_differences(plt, episodes, layouts_obj, out):
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
