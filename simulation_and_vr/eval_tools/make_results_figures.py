#!/usr/bin/env python3
"""Thesis figures and tables for the results chapter, from one compare_conditions.py results folder.

Reads the CSVs compare_conditions.py wrote (success, distances, scores, differences, route_side, episodes, run_info) and
re-loads the trajectories once (same loaders and metric code as compare_conditions.py) for the distribution and
trajectory figures. Nothing is recomputed differently: every number in the tables comes from the results folder.

Outputs (in OUT_DIR):
  fig_success.(png|pdf)              success rate per layout and source
  fig_distributions.(png|pdf)        CDFs of stop share, |yaw rate|, wall distance, path efficiency per layout and source
  fig_scores.(png|pdf)               (a) human-likeness score per layout, (b) B-A and C-B differences
  fig_trajectories_<layout>.(png|pdf) all episodes per source on one layout
  fig_heatmap_differences.(png|pdf)  time share condition minus humans
  tables.md                          all result tables (Markdown)

Usage:
  python make_results_figures.py RESULTS_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import compare_conditions as cc

SHORT = {"CentralPillar_Code_102620": "CentralPillar", "MultiRoomDoorway_Code_551670": "MultiRoomDoorway",
         "TJunction_Code_99835": "TJunction"}
SOURCE_NAMES = {cc.SP: "Shortest path", "A": "A", "B": "B", "C": "C", cc.HUMAN: "Humans"}
ORDER = [cc.SP, "A", "B", "C", cc.HUMAN]


def short(l: str) -> str:
    return SHORT.get(l, l.split("_")[0])


def f(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def ci_txt(lo, hi, nd=2):
    return "" if not (np.isfinite(lo) and np.isfinite(hi)) else f" [{lo:.{nd}f}, {hi:.{nd}f}]"


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------

def load_episodes(info: dict):
    """The same episodes compare_conditions.py analysed (paths from run_info.json), with metrics and histograms."""
    warnings: list[str] = []
    layouts: dict = {}

    def layout_for(scene):
        if scene not in layouts:
            layouts[scene] = cc.load_layout(Path(info["layouts_dir"]) / f"{scene}.npz")
        return layouts[scene]

    runs = [Path(r) for r in info["runs"]]
    models, agent_eps = cc.load_agents(runs, layout_for, warnings)
    human_eps = cc.load_humans(Path(info["recordings"]), {m.layout for m in models}, layout_for, warnings)
    gates = json.loads(Path(info["gates"]).read_text()) if info.get("gates") and Path(info["gates"]).exists() else {}
    episodes = agent_eps + human_eps
    for e in episodes:
        cc.compute_metrics(e, layouts[e.layout], gates.get(e.layout))
    return episodes, layouts, sorted({m.layout for m in models})


# ----------------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------------

def fig_success(success: pd.DataFrame, layouts: list[str], out: Path):
    plt = cc._plt()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    width = 0.16
    for j, src in enumerate(ORDER):
        for i, l in enumerate(layouts):
            if src == cc.HUMAN:
                r = success[(success.layout == l) & (success.source == cc.HUMAN)].iloc[0]
                y, lo, hi = r.success_rate, np.nan, np.nan
            else:
                r = success[(success.layout == l) & (success.source == src) & (success.model == "(condition mean)")].iloc[0]
                y, lo, hi = r.success_rate, r.ci_low, r.ci_high
            x = i + (j - 2) * width
            ax.bar(x, 100 * y, width=width * 0.92, color=cc.COLORS[src], label=SOURCE_NAMES[src] if i == 0 else None)
            if np.isfinite(lo):
                ax.errorbar(x, 100 * y, yerr=[[100 * (y - lo)], [100 * (hi - y)]], fmt="none", ecolor="#333", capsize=2, lw=0.8)
            ax.text(x, 100 * (hi if np.isfinite(hi) else y) + 2, f"{100 * y:.0f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(range(len(layouts)))
    ax.set_xticklabels([short(l) for l in layouts])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 112)
    ax.legend(ncol=5, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return cc._save(fig, out, "fig_success")


def _cdf_episode(vals: np.ndarray):
    v = np.sort(vals[np.isfinite(vals)])
    return v, np.arange(1, len(v) + 1) / max(len(v), 1)


def fig_distributions(episodes, layouts: list[str], out: Path):
    plt = cc._plt()
    cols = [("stop_share", "Stop share", (0, 1)), ("yaw_rate_deg_s", "|Yaw rate| (°/s, 0.5 s)", (0, 90)),
            ("wall_distance_m", "Wall distance, walking (m)", (0, 4)), ("path_efficiency", "Path efficiency", (0.9, 2.0))]
    fig, axes = plt.subplots(len(layouts), len(cols), figsize=(9.6, 2.3 * len(layouts)), squeeze=False, layout="constrained")
    for i, l in enumerate(layouts):
        for j, (m, title, xlim) in enumerate(cols):
            ax = axes[i, j]
            for src in ORDER:
                eps = [e for e in episodes if e.layout == l and e.source == src]
                lw, ls = (2.0, "-") if src == cc.HUMAN else (1.2, "--" if src == cc.SP else "-")
                if m in cc.SAMPLE_METRICS:
                    h = [e.hists[m] for e in eps if e.hists[m] is not None]
                    if not h:
                        continue
                    cdf = np.cumsum(np.mean(h, axis=0))
                    ax.plot(cc.HIST_EDGES[m][1:], cdf, color=cc.COLORS[src], lw=lw, ls=ls, label=SOURCE_NAMES[src])
                else:
                    v = np.array([e.metrics[m] for e in eps], dtype=float)
                    if m == "path_efficiency" and np.isfinite(v).sum() < cc.MIN_SUCCESSES:
                        continue
                    x, y = _cdf_episode(v)
                    ax.step(np.concatenate([[xlim[0]], x]), np.concatenate([[0], y]), where="post", color=cc.COLORS[src],
                            lw=lw, ls=ls, label=SOURCE_NAMES[src])
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 1.02)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{short(l)}\ncumulative share", fontsize=8)
            if m == "path_efficiency":
                ax.text(0.97, 0.05, "successful episodes;\nsources with ≥ 5 shown", transform=ax.transAxes, ha="right",
                        fontsize=6, color="#555")
    axes[0, 0].legend(fontsize=6, frameon=False, loc="lower right")
    return cc._save(fig, out, "fig_distributions")


def fig_scores(scores: pd.DataFrame, diffs: pd.DataFrame, layouts: list[str], out: Path):
    plt = cc._plt()
    names = layouts + ["overall"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.2), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    off = {"A": -0.22, "B": 0.0, "C": 0.22}
    for i, l in enumerate(names):
        fl = scores[(scores.layout == l) & (scores.source == "human noise floor")].iloc[0]
        ax1.hlines(fl.score, i - 0.36, i + 0.36, color=cc.COLORS[cc.HUMAN], lw=2, label="Second human group (S_floor)" if i == 0 else None)
        for c in cc.CONDITIONS:
            r = scores[(scores.layout == l) & (scores.source == c)].iloc[0]
            ax1.errorbar(i + off[c], r.score, yerr=[[r.score - r.ci_low], [r.ci_high - r.score]], fmt="o", color=cc.COLORS[c],
                         capsize=2.5, ms=4, label=f"Condition {c}" if i == 0 else None)
            if np.isfinite(r.seed_min):
                ax1.vlines(i + off[c] + 0.07, r.seed_min, r.seed_max, color=cc.COLORS[c], lw=1, alpha=0.55)
    ax1.axhline(0, color=cc.COLORS[cc.SP], lw=0.9, ls="--", label="Shortest path (S = 0)")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([short(n) if n != "overall" else "Overall" for n in names], fontsize=7)
    ax1.set_ylabel("Human-likeness score S")
    ax1.set_title("(a) Score per layout (95 % CI; thin line = seed range)", fontsize=8)
    ax1.legend(fontsize=6, frameon=False, loc="lower left")

    sel = diffs[diffs.measure == "score"]
    comps = [("B - A", "#2a78d6"), ("C - B", "#0d3f80")]
    for i, l in enumerate(names):
        for k, (comp, col) in enumerate(comps):
            r = sel[(sel.layout == l) & (sel.comparison == comp)].iloc[0]
            x = i + (k - 0.5) * 0.3
            ax2.errorbar(x, r.estimate, yerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="s", color=col,
                         capsize=2.5, ms=4, label=comp.replace("-", "−") if i == 0 else None)
    ax2.axhline(0, color="#777", lw=0.8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([short(n) if n != "overall" else "Overall" for n in names], fontsize=7)
    ax2.set_ylabel("Score difference ΔS")
    ax2.set_title("(b) Differences between conditions (95 % CI)", fontsize=8)
    ax2.legend(fontsize=6, frameon=False)
    return cc._save(fig, out, "fig_scores")


def fig_trajectories(episodes, layouts_obj: dict, layouts: list[str], out: Path):
    plt = cc._plt()
    paths = []
    for l in layouts:
        L = layouts_obj[l]
        fig, axes = plt.subplots(1, len(ORDER), figsize=(12.0, 2.9), layout="constrained")
        for ax, src in zip(axes, ORDER):
            cc._draw_layout(ax, L)
            eps = [e for e in episodes if e.layout == l and e.source == src]
            alpha = 0.75 if src in (cc.HUMAN, cc.SP) else 0.28
            for e in eps:
                k = e.keep
                ax.plot(k["x"], k["z"], color=cc.COLORS[src], lw=0.7, alpha=alpha, zorder=3,
                        ls="-" if e.metrics["success"] else (0, (2, 1.5)))
            cc._draw_markers(ax, L)
            n_ok = int(sum(e.metrics["success"] for e in eps))
            ax.set_title(f"{SOURCE_NAMES[src]}  ({n_ok}/{len(eps)})", fontsize=8)
        paths += cc._save(fig, out, f"fig_trajectories_{short(l)}")
        plt.close(fig)
    return paths


def fig_heatmap_differences(episodes, layouts_obj: dict, layouts: list[str], out: Path):
    plt = cc._plt()
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(len(layouts), 3, figsize=(7.6, 2.5 * len(layouts)), squeeze=False, layout="constrained")
    for i, l in enumerate(layouts):
        L = layouts_obj[l]
        hum = cc._time_share([e for e in episodes if e.layout == l and e.source == cc.HUMAN], L)
        diffs = {c: 100 * (cc._time_share([e for e in episodes if e.layout == l and e.source == c], L) - hum) for c in cc.CONDITIONS}
        vmax = max(float(np.percentile(np.abs(d[d != 0]), 99)) for d in diffs.values())
        for j, c in enumerate(cc.CONDITIONS):
            ax = axes[i, j]
            im = ax.imshow(np.ma.masked_equal(diffs[c], 0), origin="lower", extent=L.extent, cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest", zorder=2)
            cc._draw_layout(ax, L)
            cc._draw_markers(ax, L)
            if i == 0:
                ax.set_title(f"{c} − humans", fontsize=8)
            if j == 0:
                ax.set_ylabel(short(l), fontsize=8)
        fig.colorbar(im, ax=axes[i, :].tolist(), shrink=0.85, label="Δ time share (%)")
    return cc._save(fig, out, "fig_heatmap_differences")


# ----------------------------------------------------------------------------------------------------------------------
# Tables (Markdown)
# ----------------------------------------------------------------------------------------------------------------------

def tables(res: Path, episodes, layouts: list[str], info: dict) -> str:
    success = pd.read_csv(res / "success.csv")
    dist = pd.read_csv(res / "distances.csv")
    scores = pd.read_csv(res / "scores.csv")
    diffs = pd.read_csv(res / "differences.csv")
    route = pd.read_csv(res / "route_side.csv")
    eps = pd.read_csv(res / "episodes.csv")
    md = []

    # Data overview
    md.append("### Table: evaluated data\n")
    md.append("| Layout | Shortest path | A | B | C | Humans (test walks) |\n|---|---|---|---|---|---|")
    for l in layouts:
        row = [short(l)]
        for src in ORDER:
            n = len(eps[(eps.layout == l) & (eps.source == src)])
            models = eps[(eps.layout == l) & (eps.source == src)].model.nunique()
            row.append(f"{n}" if src in (cc.HUMAN,) else f"{n} ({models} {'run' if models == 1 else 'models'})")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Task performance
    md.append("### Table: task performance\n")
    md.append("| Layout | Source | Success (%) [95 % CI] | Successful episodes | Time to goal (s), median [IQR] | Path efficiency, median [IQR] |")
    md.append("|---|---|---|---|---|---|")
    for l in layouts:
        for src in ORDER:
            e = eps[(eps.layout == l) & (eps.source == src)]
            if src == cc.HUMAN:
                r = success[(success.layout == l) & (success.source == cc.HUMAN)].iloc[0]
                s = f"{100 * r.success_rate:.0f}"
            else:
                r = success[(success.layout == l) & (success.source == src) & (success.model == "(condition mean)")].iloc[0]
                s = f"{100 * r.success_rate:.1f}{ci_txt(100 * r.ci_low, 100 * r.ci_high, 1)}"
            ok = e[e.success == 1]
            t = ok.time_to_goal_s
            p = ok.path_efficiency
            tt = f"{t.median():.1f} [{t.quantile(.25):.1f}, {t.quantile(.75):.1f}]" if len(ok) else "–"
            pp = f"{p.median():.2f} [{p.quantile(.25):.2f}, {p.quantile(.75):.2f}]" if len(ok) else "–"
            md.append(f"| {short(l)} | {SOURCE_NAMES[src]} | {s} | {len(ok)}/{len(e)} | {tt} | {pp} |")
    for src in cc.AGENT_SOURCES:
        r = success[(success.layout == "overall") & (success.source == src)].iloc[0]
        md.append(f"| All layouts | {SOURCE_NAMES[src]} | {100 * r.success_rate:.1f}{ci_txt(100 * r.ci_low, 100 * r.ci_high, 1)} | | | |")
    md.append("")

    # Per-model success
    md.append("### Table: success per model (successful / 20 episodes)\n")
    models = sorted(eps[eps.source.isin(cc.CONDITIONS)].model.unique())
    md.append("| Model | " + " | ".join(short(l) for l in layouts) + " |\n|---|" + "---|" * len(layouts))
    for m in models:
        cells = []
        for l in layouts:
            e = eps[(eps.layout == l) & (eps.model == m)]
            cells.append(f"{int(e.success.sum())}/{len(e)}")
        md.append(f"| {m.replace('ImitationAgent_', '')} | " + " | ".join(cells) + " |")
    md.append("")

    # Behaviour descriptives
    md.append("### Table: behavioural metrics (mean over episodes; each episode weighted equally)\n")
    md.append("| Layout | Source | Stop share | Abs. yaw rate (°/s) | Wall distance, walking (m) | Wall contacts (/min) | POI reached (%) |")
    md.append("|---|---|---|---|---|---|---|")
    for l in layouts:
        for src in ORDER:
            e = eps[(eps.layout == l) & (eps.source == src)]
            md.append(f"| {short(l)} | {SOURCE_NAMES[src]} | {e.stop_share.mean():.3f} | {e.yaw_rate_mean_deg_s.mean():.1f} | "
                      f"{e.wall_distance_mean_m.mean():.2f} | {e.contacts_per_min.mean():.2f} | {100 * (e.pois_reached >= 1).mean():.0f} |")
    md.append("")

    # Wasserstein distances
    md.append("### Table: Wasserstein distance to the held-out humans (95 % CI)\n")
    md.append("| Layout | Metric | Shortest path | A | B | C | Noise floor W(H₁, H₂) |\n|---|---|---|---|---|---|---|")
    titles = {"stop_share": "Stop share", "yaw_rate_deg_s": "Abs. yaw rate (°/s)", "wall_distance_m": "Wall distance (m)",
              "path_efficiency": "Path efficiency", "time_to_goal_s": "Time to goal (s)", "contacts_per_min": "Wall contacts (/min)"}
    for l in layouts:
        for k in cc.DIST_METRICS:
            sel = dist[(dist.layout == l) & (dist.metric == k)]
            cells = []
            for src in cc.AGENT_SOURCES:
                r = sel[sel.source == src].iloc[0]
                cells.append(f"{f(r.W_to_humans, 3)}{ci_txt(r.ci_low, r.ci_high, 3)}" if np.isfinite(r.W_to_humans) else "undefined")
            r0 = sel.iloc[0]
            md.append(f"| {short(l)} | {titles[k]} | " + " | ".join(cells) + f" | {f(r0.noise_floor, 3)}{ci_txt(r0.noise_floor_ci_low, r0.noise_floor_ci_high, 3)} |")
    md.append("")

    # Per-metric scores
    md.append("### Table: human-likeness score S per metric (95 % CI); metrics in the layout score marked ✓\n")
    md.append("| Layout | Metric | A | B | C | S_floor | In score |\n|---|---|---|---|---|---|---|")
    for l in layouts:
        for k in cc.SCORE_METRICS:
            sel = dist[(dist.layout == l) & (dist.metric == k) & dist.source.isin(cc.CONDITIONS)]
            cells = [f"{f(r.score)}{ci_txt(r.score_ci_low, r.score_ci_high)}" for r in
                     (sel[sel.source == c].iloc[0] for c in cc.CONDITIONS)]
            r0 = sel.iloc[0]
            used = "✓" if r0.in_layout_score else f"– ({r0.excluded_reason})"
            md.append(f"| {short(l)} | {titles[k]} | " + " | ".join(cells) + f" | {f(r0.score_floor)} | {used} |")
    md.append("")

    # Layout scores
    md.append("### Table: layout and overall human-likeness scores (95 % CI; seed range in parentheses)\n")
    md.append("| Layout | A | B | C | Second human group (S_floor) |\n|---|---|---|---|---|")
    for l in layouts + ["overall"]:
        cells = []
        for c in cc.CONDITIONS:
            r = scores[(scores.layout == l) & (scores.source == c)].iloc[0]
            sr = f" ({r.seed_min:.2f} to {r.seed_max:.2f})" if np.isfinite(r.seed_min) else ""
            cells.append(f"{f(r.score)}{ci_txt(r.ci_low, r.ci_high)}{sr}")
        r = scores[(scores.layout == l) & (scores.source == "human noise floor")].iloc[0]
        md.append(f"| {short(l) if l != 'overall' else 'Overall'} | " + " | ".join(cells) + f" | {f(r.score)}{ci_txt(r.ci_low, r.ci_high)} |")
    md.append("")

    # Differences
    md.append("### Table: differences between conditions (95 % CI)\n")
    md.append("| Layout | ΔS B − A | ΔS C − B | ΔSuccess B − A (pp) | ΔSuccess C − B (pp) |\n|---|---|---|---|---|")
    for l in layouts + ["overall"]:
        cells = []
        for meas, nd in (("score", 2), ("success", 1)):
            for comp in ("B - A", "C - B"):
                r = diffs[(diffs.layout == l) & (diffs.comparison == comp) & (diffs.measure == meas)].iloc[0]
                cells.append(f"{r.estimate:+.{nd}f}{ci_txt(r.ci_low, r.ci_high, nd)}")
        md.append(f"| {short(l) if l != 'overall' else 'Overall'} | " + " | ".join([cells[0], cells[1], cells[2], cells[3]]) + " |")
    md.append("")

    # Route side
    md.append("### Table: route side around the central obstacle (start → POI leg)\n")
    md.append("| Layout | Source | Episodes | Crossed a gate | North | South | North share [95 % CI] | Difference to humans [95 % CI] |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in route.iterrows():
        diff = "" if r.source == cc.HUMAN else f"{r.diff_to_humans:+.2f}{ci_txt(r.diff_ci_low, r.diff_ci_high)}"
        md.append(f"| {short(r.layout)} | {SOURCE_NAMES[r.source]} | {r.episodes} | {r.crossed} | {r.north} | {r.south} | "
                  f"{f(r.share_north)}{ci_txt(r.ci_low, r.ci_high)} | {diff} |")
    md.append("")
    md.append(f"Excluded score metrics: {json.dumps(info.get('excluded_score_metrics', {}))}")
    md.append(f"Warnings: {json.dumps(info.get('warnings', []))}")
    return "\n".join(md)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    res, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((res / "run_info.json").read_text())
    print("loading episodes ...")
    episodes, layouts_obj, layouts = load_episodes(info)
    success = pd.read_csv(res / "success.csv")
    scores = pd.read_csv(res / "scores.csv")
    diffs = pd.read_csv(res / "differences.csv")
    fig_success(success, layouts, out)
    fig_distributions(episodes, layouts, out)
    fig_scores(scores, diffs, layouts, out)
    fig_trajectories(episodes, layouts_obj, layouts, out)
    fig_heatmap_differences(episodes, layouts_obj, layouts, out)
    (out / "tables.md").write_text(tables(res, episodes, layouts, info), encoding="utf-8")
    print(f"wrote figures and tables.md to {out}")


if __name__ == "__main__":
    main()
