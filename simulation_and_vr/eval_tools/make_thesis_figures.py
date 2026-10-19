#!/usr/bin/env python3
"""Results-chapter figures that replace the result tables, from one compare_conditions.py results folder.

Every number comes from the results folder (CSVs written by compare_conditions.py) or, where the CSVs hold no per-model
or per-episode position data, from re-loading the same runs with the same loaders and metric code. Nothing is
recomputed with different rules.

Figures (PNG + PDF in OUT_DIR), with the tables they replace:
  fig_k1_task_performance   success per model and condition, time to goal, path efficiency      (Tables 4.1, 4.2, A.1)
  fig_k2_behaviour          behavioural metrics per source against the human mean              (Table 4.3)
  fig_k3_distance_heatmap   distance to the held-out humans in multiples of the noise floor    (Tables 4.4, A.2)
  fig_k4_scores             layout/overall scores with per-model dots by fold, ΔS and ΔSuccess  (Tables 4.5, 4.6)
  fig_k5_score_heatmap      human-likeness score S per metric, layout and condition            (Table A.3)
  fig_k6_route_side         side passed around the central obstacle                             (Table 4.7)
  fig_k7_failures           where failed episodes ended                                         (Table 4.8)

Usage:
  python make_thesis_figures.py RESULTS_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import compare_conditions as cc
from make_results_figures import short

SOURCES = [cc.SP, "A", "B", "C", cc.HUMAN]
NAMES = {cc.SP: "Shortest path", "A": "A", "B": "B", "C": "C", cc.HUMAN: "Humans"}
FOLD_MARKER = {0: "s", 1: "o", 2: "^"}           # A (no fold), F1, F2
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
METRIC_LABELS = {"stop_share": "Stop share", "yaw_rate_deg_s": "Yaw rate", "wall_distance_m": "Wall distance",
                 "path_efficiency": "Path efficiency", "time_to_goal_s": "Time to goal", "contacts_per_min": "Wall contacts"}
SCORE_METRICS = list(cc.SCORE_METRICS)
ALL_METRICS = SCORE_METRICS + ["time_to_goal_s", "contacts_per_min"]

# End regions of failed episodes (x in m, layout frame of Figures 4.5-4.6). Upper bounds inclusive.
FAIL_REGIONS = {
    "MultiRoomDoorway": [("West of second room", -np.inf, 3.3), ("Second room", 3.3, 7.5), ("Start room", 7.5, np.inf)],
    "TJunction": [("Elsewhere (west)", -np.inf, 0.0), ("Pocket east of stem", 0.0, 3.3), ("Elsewhere (east)", 3.3, np.inf)],
}


def _style(plt):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "font.family": "sans-serif"})


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _legend_sources(ax, sources, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=cc.COLORS[s], ms=6, label=NAMES[s]) for s in sources]
    return ax.legend(handles=handles, frameon=False, **kw)


def _fold_legend(ax, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=FOLD_MARKER[f], ls="", mfc="white", mec=MUTED, ms=5, label=lab)
               for f, lab in ((0, "model (A)"), (1, "model trained on F1"), (2, "model trained on F2"))]
    return ax.legend(handles=handles, frameon=False, **kw)


# ----------------------------------------------------------------------------------------------------------------------
# Per-model scores (not in the CSVs: only the seed range is)
# ----------------------------------------------------------------------------------------------------------------------

def load_all(info: dict):
    """Models and episodes exactly as compare_conditions.py loaded them (paths from run_info.json), with metrics."""
    warnings: list[str] = []
    layouts: dict = {}

    def layout_for(scene):
        if scene not in layouts:
            layouts[scene] = cc.load_layout(Path(info["layouts_dir"]) / f"{scene}.npz")
        return layouts[scene]

    models, agent_eps = cc.load_agents([Path(r) for r in info["runs"]], layout_for, warnings)
    human_eps = cc.load_humans(Path(info["recordings"]), {m.layout for m in models}, layout_for, warnings)
    gates = json.loads(Path(info["gates"]).read_text()) if info.get("gates") and Path(info["gates"]).exists() else {}
    episodes = agent_eps + human_eps
    for e in episodes:
        cc.compute_metrics(e, layouts[e.layout], gates.get(e.layout))
    return models, episodes, layouts, sorted({m.layout for m in models})


def per_model_scores(models, episodes, distances: pd.DataFrame) -> pd.DataFrame:
    """Layout score of each model on the point estimate, with the layout's score metric set from distances.csv."""
    study = cc.Study(models, episodes)
    raw = study.evaluate({l: np.ones(study.n_configs[l], dtype=np.int64) for l in study.layouts},
                         np.ones(study.n_participants, dtype=np.int64))
    rows = []
    for l in study.layouts:
        d = distances[(distances.layout == l)]
        metric_set = sorted(set(d[d.in_layout_score == True].metric))  # noqa: E712
        sp = study.sp_label(l)
        for m in study.models[l]:
            if m.source not in cc.CONDITIONS:
                continue
            vals = [1.0 - raw[("W", l, m.label, k)] / raw[("W", l, sp, k)] for k in metric_set]
            rows.append({"layout": l, "source": m.source, "model": m.label, "fold": m.fold, "seed": m.seed,
                         "score": float(np.nanmean(vals))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# K1: task performance
# ----------------------------------------------------------------------------------------------------------------------

def fig_task_performance(plt, success, eps, layouts, out):
    fig, axes = plt.subplots(3, len(layouts), figsize=(7.0, 6.4), sharey="row")
    x = {s: i for i, s in enumerate(SOURCES)}
    rng = np.random.default_rng(0)
    counts = {}
    for j, l in enumerate(layouts):
        # Row 1: success rate. Per-model dots (shape = fold), condition mean with bootstrap CI.
        ax = axes[0, j]
        _grid(ax)
        for s in SOURCES:
            per = success[(success.layout == l) & (success.source == s) & (success.model != "(condition mean)")]
            if s in cc.CONDITIONS:
                for _, r in per.iterrows():
                    ax.plot(x[s] + rng.uniform(-0.18, 0.18), 100 * r.success_rate, marker=FOLD_MARKER[int(r.fold)],
                            ls="", mfc="white", mec=cc.COLORS[s], ms=4.5, mew=1, zorder=3)
                m = success[(success.layout == l) & (success.source == s) & (success.model == "(condition mean)")].iloc[0]
                ax.errorbar(x[s], 100 * m.success_rate, yerr=[[100 * (m.success_rate - m.ci_low)], [100 * (m.ci_high - m.success_rate)]],
                            fmt="_", ms=16, mew=2.5, color=cc.COLORS[s], elinewidth=1.2, capsize=0, zorder=4)
                k, n = per.successes.sum(), per.episodes.sum()
            else:
                r = per.iloc[0]
                ax.plot(x[s], 100 * r.success_rate, marker="D", ls="", color=cc.COLORS[s], ms=5.5, zorder=4)
                k, n = r.successes, r.episodes
            counts[(j, s)] = f"{int(k)}\nof {int(n)}"
        ax.set_ylim(-8, 108)
        ax.set_title(short(l))
        if j == 0:
            ax.set_ylabel("Success (%)")

        # Rows 2-3: time to goal and path efficiency of successful episodes.
        for row, metric, label in ((1, "time_to_goal_s", "Time to goal (s)"), (2, "path_efficiency", "Path efficiency")):
            ax = axes[row, j]
            _grid(ax)
            for s in SOURCES:
                v = eps[(eps.layout == l) & (eps.source == s) & (eps.success == 1)][metric].dropna().to_numpy()
                if len(v) == 0:
                    ax.text(x[s], 0.5, "none", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5, color=MUTED)
                    continue
                ax.scatter(x[s] + rng.uniform(-0.22, 0.22, len(v)), v, s=5, color=cc.COLORS[s], alpha=0.35, lw=0, zorder=2)
                if len(v) >= cc.MIN_SUCCESSES:
                    q1, med, q3 = np.percentile(v, [25, 50, 75])
                    ax.vlines(x[s] + 0.32, q1, q3, color=cc.COLORS[s], lw=2.5, zorder=3)
                    ax.plot(x[s] + 0.32, med, marker="o", color="white", mec=cc.COLORS[s], ms=4.5, mew=1.3, zorder=4)
                else:
                    ax.text(x[s], 1.0, f"n = {len(v)}", transform=ax.get_xaxis_transform(), ha="center", va="top",
                            fontsize=6.5, color=MUTED)
            if metric == "path_efficiency":
                ax.axhline(1.0, color=MUTED, lw=0.7, ls=":", zorder=1)
            if j == 0:
                ax.set_ylabel(label)
    for ax in axes.flat:
        ax.set_xticks(range(len(SOURCES)))
        ax.set_xticklabels([NAMES[s] if s != cc.SP else "Shortest\npath" for s in SOURCES])
        ax.set_xlim(-0.6, len(SOURCES) - 0.4)
    for j, ax in enumerate(axes[0]):
        ax.set_xticklabels([counts[(j, s)] for s in SOURCES], fontsize=6.5)
    axes[1, 0].set_ylim(bottom=0)
    _fold_legend(axes[0, 0], loc="lower left", bbox_to_anchor=(0, 1.12), ncol=3, columnspacing=1)
    axes[2, -1].text(1.0, -0.42, "Bars: interquartile range, circle: median (sources with ≥ 5 successes). "
                     "Bold dash: condition mean with 95 % CI.", transform=axes[2, -1].transAxes, ha="right", fontsize=6.5, color=MUTED)
    fig.subplots_adjust(hspace=0.35, wspace=0.08)
    return cc._save(fig, out, "fig_k1_task_performance")


# ----------------------------------------------------------------------------------------------------------------------
# K2: behavioural profile
# ----------------------------------------------------------------------------------------------------------------------

def fig_behaviour(plt, eps, layouts, out):
    cols = [("stop_share", "Stop share (%)", 100), ("yaw_rate_mean_deg_s", "Abs. yaw rate (°/s)", 1),
            ("wall_distance_mean_m", "Wall dist., walking (m)", 1), ("contacts_per_min", "Wall contacts (/min)", 1)]
    fig, axes = plt.subplots(len(layouts), len(cols), figsize=(7.0, 4.6), sharex="col", sharey=True)
    y = {s: len(SOURCES) - 1 - i for i, s in enumerate(SOURCES)}
    for i, l in enumerate(layouts):
        for j, (col, label, scale) in enumerate(cols):
            ax = axes[i, j]
            _grid(ax, "x")
            means = {s: scale * eps[(eps.layout == l) & (eps.source == s)][col].mean() for s in SOURCES}
            ax.axvline(means[cc.HUMAN], color=cc.COLORS[cc.HUMAN], lw=1, ls="--", zorder=1)
            for s in SOURCES:
                ax.hlines(y[s], means[cc.HUMAN], means[s], color=cc.COLORS[s], lw=1.6, alpha=0.6, zorder=2)
                ax.plot(means[s], y[s], "o", color=cc.COLORS[s], ms=6, mec="white", mew=1, zorder=3)
            if i == 0:
                ax.set_title(label, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(short(l), fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_yticks([y[s] for s in SOURCES])
        ax.set_yticklabels([NAMES[s] for s in SOURCES])
    for ax in axes.flat:
        ax.set_ylim(-0.6, len(SOURCES) - 0.4)
        ax.set_xlim(left=0)
    fig.text(0.99, -0.01, "Dot: mean over episodes; dashed line and connector: human mean.", ha="right", fontsize=6.5, color=MUTED)
    fig.subplots_adjust(hspace=0.15, wspace=0.12)
    return cc._save(fig, out, "fig_k2_behaviour")


# ----------------------------------------------------------------------------------------------------------------------
# K3: distance heatmap (multiples of the noise floor)
# ----------------------------------------------------------------------------------------------------------------------

def fig_distance_heatmap(plt, distances, layouts, out):
    from matplotlib.colors import LogNorm
    srcs = [cc.SP, "A", "B", "C"]
    fig, axes = plt.subplots(1, len(layouts), figsize=(7.0, 2.9), sharey=True)
    norm = LogNorm(vmin=0.8, vmax=30)
    cmap = plt.get_cmap("Purples")
    for ax, l in zip(axes, layouts):
        M = np.full((len(ALL_METRICS), len(srcs)), np.nan)
        for i, k in enumerate(ALL_METRICS):
            for j, s in enumerate(srcs):
                r = distances[(distances.layout == l) & (distances.metric == k) & (distances.source == s)]
                if len(r) and np.isfinite(r.W_to_humans.iloc[0]) and r.noise_floor.iloc[0] > 0:
                    M[i, j] = r.W_to_humans.iloc[0] / r.noise_floor.iloc[0]
        ax.imshow(np.where(np.isfinite(M), M, np.nan), cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.1f}×" if v < 10 else f"{v:.0f}×", ha="center", va="center", fontsize=6.8,
                            color="white" if v > 6 else INK, fontweight="bold" if v <= 1.0 else "normal")
                else:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec=GRID, lw=0))
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color=MUTED)
        ax.axhline(len(SCORE_METRICS) - 0.5, color="white", lw=3)
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels(["Shortest\npath", "A", "B", "C"])
        ax.set_yticks(range(len(ALL_METRICS)))
        ax.set_yticklabels([METRIC_LABELS[k] for k in ALL_METRICS])
        ax.set_title(short(l))
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cb.set_label("Distance to held-out humans\n(× human noise floor; 1 = as close as a second human group)", fontsize=7)
    cb.set_ticks([1, 2, 5, 10, 20])
    cb.set_ticklabels(["1", "2", "5", "10", "20"])
    fig.text(0.01, -0.04, "Upper four rows enter the score. Wasserstein distance divided by W(F1 humans, F2 humans); "
             "n/a = fewer than 5 successes.", fontsize=6.5, color=MUTED)
    return cc._save(fig, out, "fig_k3_distance_heatmap")


# ----------------------------------------------------------------------------------------------------------------------
# K4: scores and differences
# ----------------------------------------------------------------------------------------------------------------------

def fig_scores(plt, scores, diffs, model_scores, layouts, out):
    names = layouts + ["overall"]
    label = lambda n: "Overall" if n == "overall" else short(n)  # noqa: E731
    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.45, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    _grid(ax)
    off = {"A": -0.24, "B": 0.0, "C": 0.24}
    rng = np.random.default_rng(1)
    for i, n in enumerate(names):
        fl = scores[(scores.layout == n) & (scores.source == "human noise floor")].iloc[0]
        ax.add_patch(plt.Rectangle((i - 0.4, fl.ci_low), 0.8, fl.ci_high - fl.ci_low, color=cc.COLORS[cc.HUMAN],
                                   alpha=0.12, lw=0, zorder=1))
        ax.hlines(fl.score, i - 0.4, i + 0.4, color=cc.COLORS[cc.HUMAN], lw=1.6, zorder=2)
        for c in cc.CONDITIONS:
            r = scores[(scores.layout == n) & (scores.source == c)].iloc[0]
            if n != "overall":
                for _, m in model_scores[(model_scores.layout == n) & (model_scores.source == c)].iterrows():
                    ax.plot(i + off[c] + 0.07 + rng.uniform(-0.03, 0.03), m.score, marker=FOLD_MARKER[int(m.fold)], ls="",
                            mfc="white", mec=cc.COLORS[c], ms=4, mew=0.9, zorder=3)
            ax.errorbar(i + off[c] - 0.04, r.score, yerr=[[r.score - r.ci_low], [r.ci_high - r.score]], fmt="o",
                        color=cc.COLORS[c], ms=6, mec="white", mew=0.8, elinewidth=1.4, capsize=0, zorder=4)
    ax.axhline(0, color=cc.COLORS[cc.SP], lw=1, ls="--", zorder=2)
    ax.text(len(names) - 0.45, 0.08, "Shortest path (S = 0)", ha="right", va="bottom", fontsize=6.5, color=cc.COLORS[cc.SP])
    ax.text(len(names) - 0.45, scores[(scores.layout == "overall") & (scores.source == "human noise floor")].score.iloc[0] + 0.08,
            "Second human group", ha="right", va="bottom", fontsize=6.5, color=cc.COLORS[cc.HUMAN])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([label(n) for n in names])
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylabel("Human-likeness score S")
    ax.set_title("(a) Score per layout: all conditions stay below the shortest-path baseline", pad=26)
    leg1 = _legend_sources(ax, list(cc.CONDITIONS), loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, columnspacing=0.8)
    ax.add_artist(leg1)
    _fold_legend(ax, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, columnspacing=0.8)
    ax.set_ylim(-6.4, 1.0)

    for col, (meas, title, unit) in enumerate((("score", "(b) Score difference ΔS", "ΔS"),
                                               ("success", "(c) Success difference", "Δ success (pp)"))):
        ax = fig.add_subplot(gs[1, col])
        _grid(ax, "x")
        ax.axvline(0, color=MUTED, lw=0.8)
        ys = {n: len(names) - 1 - i for i, n in enumerate(names)}
        for comp, dy, color in (("B - A", 0.14, cc.COLORS["B"]), ("C - B", -0.14, cc.COLORS["C"])):
            for n in names:
                r = diffs[(diffs.layout == n) & (diffs.comparison == comp) & (diffs.measure == meas)].iloc[0]
                excl = r.ci_low > 0 or r.ci_high < 0
                ax.errorbar(r.estimate, ys[n] + dy, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="o",
                            ms=5.5, color=color, mfc=color if excl else "white", mec=color, elinewidth=1.3, capsize=0,
                            label=f"{comp.replace('-', '−')}" if n == names[0] else None)
        ax.axhline(0.5, color=GRID, lw=0.8)
        ax.set_yticks([ys[n] for n in names])
        ax.set_yticklabels([label(n) for n in names] if col == 0 else [])
        ax.set_xlabel(unit)
        ax.set_title(title)
        if col == 0:
            ax.legend(frameon=False, loc="lower left", handletextpad=0.2)
    fig.text(0.01, 0.0, "95 % bootstrap intervals. Filled marker: interval excludes zero. "
             "Open markers in (a): individual models; orange band: S_floor with interval.", fontsize=6.5, color=MUTED)
    return cc._save(fig, out, "fig_k4_scores")


# ----------------------------------------------------------------------------------------------------------------------
# K5: score heatmap per metric
# ----------------------------------------------------------------------------------------------------------------------

def fig_score_heatmap(plt, distances, layouts, out):
    from matplotlib.colors import TwoSlopeNorm
    cols = ["A", "B", "C", "S_floor"]
    fig, axes = plt.subplots(1, len(layouts), figsize=(7.0, 2.3), sharey=True)
    norm = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=1)
    cmap = plt.get_cmap("BrBG")
    for ax, l in zip(axes, layouts):
        M = np.full((len(SCORE_METRICS), len(cols)), np.nan)
        inscore = []
        for i, k in enumerate(SCORE_METRICS):
            d = distances[(distances.layout == l) & (distances.metric == k)]
            for j, c in enumerate(cols):
                src = "A" if c == "S_floor" else c
                r = d[d.source == src]
                if len(r):
                    M[i, j] = r.score_floor.iloc[0] if c == "S_floor" else r.score.iloc[0]
            inscore.append(bool((d.in_layout_score == True).any()))  # noqa: E712
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6.8,
                            color="white" if (v < -4.5 or v > 0.75) else INK)
                else:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec=GRID, lw=0))
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color=MUTED)
            if not inscore[i]:
                ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False, ec=MUTED, lw=0.8, ls=":"))
        ax.axvline(2.5, color="white", lw=3)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(["A", "B", "C", "S_floor"])
        ax.set_yticks(range(len(SCORE_METRICS)))
        ax.set_yticklabels([METRIC_LABELS[k] for k in SCORE_METRICS])
        ax.set_title(short(l))
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(), shrink=0.9, pad=0.02)
    cb.set_label("S (1 = humans, 0 = baseline)", fontsize=7)
    cb.set_ticks([-8, -4, -2, 0, 1])
    fig.text(0.01, -0.06, "Dotted outline: metric not in the layout score (undefined for at least one condition).",
             fontsize=6.5, color=MUTED)
    return cc._save(fig, out, "fig_k5_score_heatmap")


# ----------------------------------------------------------------------------------------------------------------------
# K6: route side
# ----------------------------------------------------------------------------------------------------------------------

def fig_route_side(plt, route, layouts, out):
    ls = [l for l in layouts if l in set(route.layout)]
    fig, axes = plt.subplots(1, len(ls), figsize=(7.0, 2.0), sharey=True)
    y = {s: len(SOURCES) - 1 - i for i, s in enumerate(SOURCES)}
    seg = (("north", "North", "#4a4a4a", None), ("south", "South", "#b5b5b5", None), ("none", "No gate crossed", "white", "////"))
    for ax, l in zip(np.atleast_1d(axes), ls):
        for s in SOURCES:
            r = route[(route.layout == l) & (route.source == s)].iloc[0]
            shares = {"north": r.north / r.episodes, "south": r.south / r.episodes,
                      "none": (r.episodes - r.crossed) / r.episodes}
            left = 0.0
            for key, lab, color, hatch in seg:
                w = shares[key]
                ax.barh(y[s], w, left=left, height=0.62, color=color, hatch=hatch, ec="white" if hatch is None else GRID,
                        lw=1.5 if hatch is None else 0.5, label=lab if s == SOURCES[0] else None)
                if key != "none" and w >= 0.12:
                    ax.text(left + w / 2, y[s], f"{100 * w:.0f}%", ha="center", va="center", fontsize=6.5,
                            color="white" if key == "north" else INK)
                left += w
            ax.text(1.02, y[s], f"n = {int(r.crossed)}/{int(r.episodes)}", va="center", fontsize=6.5, color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50", "100 %"])
        ax.set_title(short(l))
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    np.atleast_1d(axes)[0].set_yticks([y[s] for s in SOURCES])
    np.atleast_1d(axes)[0].set_yticklabels([NAMES[s] for s in SOURCES])
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.12))
    fig.subplots_adjust(wspace=0.35)
    fig.text(0.99, -0.06, "Share of episodes passing north or south of the central obstacle on the Start → POI leg; "
             "n = episodes that crossed a gate.", ha="right", fontsize=6.5, color=MUTED)
    return cc._save(fig, out, "fig_k6_route_side")


# ----------------------------------------------------------------------------------------------------------------------
# K7: where failed episodes ended
# ----------------------------------------------------------------------------------------------------------------------

def fig_failures(plt, episodes, layouts_obj, layouts, out):
    ls = [l for l in layouts if short(l) in FAIL_REGIONS]
    fig, axes = plt.subplots(len(ls), 1, figsize=(7.0, 3.6))
    greys = ["#d9d9d9", "#8c8c8c", "#3f3f3f"]
    y = {c: len(cc.CONDITIONS) - 1 - i for i, c in enumerate(cc.CONDITIONS)}
    for ax, l in zip(np.atleast_1d(axes), ls):
        regions = FAIL_REGIONS[short(l)]
        for c in cc.CONDITIONS:
            fails = [e for e in episodes if e.layout == l and e.source == c and not e.metrics["success"]]
            if not fails:
                continue
            end = np.array([[e.keep["x"].iloc[-1], e.keep["z"].iloc[-1]] for e in fails])
            wall = float(np.median(layouts_obj[l].wall_distance(end)))
            poi = sum(e.metrics["pois_reached"] >= 1 for e in fails)
            left = 0.0
            for (lab, lo, hi), color in zip(regions, greys):
                w = float(np.mean((end[:, 0] > lo) & (end[:, 0] <= hi)))
                ax.barh(y[c], w, left=left, height=0.62, color=color, ec="white", lw=1.5, label=lab if c == "A" else None)
                if w >= 0.1:
                    ax.text(left + w / 2, y[c], f"{100 * w:.0f}%", ha="center", va="center", fontsize=6.5,
                            color="white" if color == greys[2] else INK)
                left += w
            ax.text(1.02, y[c], f"{len(fails)} failed · {poi} reached POI · wall {wall:.2f} m", va="center",
                    fontsize=6.3, color=MUTED)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "50", "100 %"])
        ax.set_title(short(l), pad=16)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks([y[c] for c in cc.CONDITIONS])
        ax.set_yticklabels(cc.CONDITIONS)
        ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=6.5, columnspacing=0.8)
    fig.subplots_adjust(hspace=0.75, right=0.6)
    fig.text(0.01, -0.08, "Share of failed episodes by end region; right: count, episodes that had reached the POI, "
             "median distance to the nearest wall at the end (contact threshold 0.30 m).", fontsize=6.5, color=MUTED)
    return cc._save(fig, out, "fig_k7_failures")


# ----------------------------------------------------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((res / "run_info.json").read_text())
    success = pd.read_csv(res / "success.csv")
    distances = pd.read_csv(res / "distances.csv")
    scores = pd.read_csv(res / "scores.csv")
    diffs = pd.read_csv(res / "differences.csv")
    route = pd.read_csv(res / "route_side.csv")
    eps = pd.read_csv(res / "episodes.csv")
    print("loading episodes (per-model scores, failure positions) ...")
    models, episodes, layouts_obj, layouts = load_all(info)
    model_scores = per_model_scores(models, episodes, distances)

    plt = cc._plt()
    _style(plt)
    written = []
    written += fig_task_performance(plt, success, eps, layouts, out)
    written += fig_behaviour(plt, eps, layouts, out)
    written += fig_distance_heatmap(plt, distances, layouts, out)
    written += fig_scores(plt, scores, diffs, model_scores, layouts, out)
    written += fig_score_heatmap(plt, distances, layouts, out)
    written += fig_route_side(plt, route, layouts, out)
    written += fig_failures(plt, episodes, layouts_obj, layouts, out)
    for p in written:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
