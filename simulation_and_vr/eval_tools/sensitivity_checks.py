#!/usr/bin/env python3
"""Robustness checks for the results chapter, on the same episodes and the same bootstrap replicates as
compare_conditions.py (same RNG seed, same resampling order, so the main differences are reproduced exactly).

Checks:
  1. per-model layout and overall scores (training variance, fold effect)
  2. A and the shortest path scored against each human fold separately (is the fold effect in the reference?)
  3. B - A and C - B without TJunction (layout where 99 % of agent episodes fail)
  4. B - A and C - B with every metric kept in every replicate (no noise-floor drop-out inside the bootstrap)
  5. how often a metric drops out of the layout score inside the bootstrap

Outputs in RESULTS_DIR/sensitivity/: per_model_scores.csv, fold_reference.csv, sensitivity.csv, dropout.csv

Usage:
  python sensitivity_checks.py RESULTS_DIR [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_conditions as cc
from make_results_figures import short
from make_thesis_figures import load_all


def layout_scores_all_metrics(study: cc.Study, raw: dict, sets: dict) -> dict:
    """Layout and overall scores like summarise(), but a metric never drops out when W(shortest path) falls within the
    noise floor of a replicate: S = 1 - W(X) / W(SP) whenever W(SP) > 0."""
    out = {}
    for c in cc.CONDITIONS:
        ls = []
        for l in study.layouts:
            w_sp = {k: raw[("W", l, study.sp_label(l), k)] for k in sets[l]}
            vals = []
            for k in sets[l]:
                w = np.array([raw[("W", l, lab, k)] for lab in study.labels(l, c)])
                wc = float(np.nanmean(w)) if np.isfinite(w).any() else np.nan
                vals.append(1.0 - wc / w_sp[k] if np.isfinite(w_sp[k]) and w_sp[k] > 0 else np.nan)
            out[(l, c)] = float(np.nanmean(vals))
            ls.append(out[(l, c)])
        out[("overall", c)] = float(np.mean(ls))
    return out


def fold_reference(study: cc.Study, raw_point: dict, sets: dict) -> pd.DataFrame:
    """A (each model) and the shortest path against fold-1 and fold-2 humans separately."""
    ones = {l: np.ones(study.n_configs[l], dtype=np.int64) for l in study.layouts}
    hw1 = np.ones(study.n_participants, dtype=np.int64)
    rows = []
    for l in study.layouts:
        sp = study.sp_label(l)
        for f in study.folds:
            hum = study.groups[(l, cc.HUMAN, f)]
            hw = hum.weights(hw1)

            def W(label, k):
                g = study.groups[(l, label)]
                w = g.weights(ones[l])
                if not g.defined(k, w, threshold=True):
                    return np.nan
                return cc.distance(g, w, hum, hw, k)

            for c in ("A",):
                per_model = []
                for lab in study.labels(l, c):
                    vals = [1.0 - W(lab, k) / W(sp, k) for k in sets[l]]
                    per_model.append(float(np.nanmean(vals)))
                rows.append({"layout": short(l), "reference_fold": f, "source": c, "score_mean_over_models": np.mean(per_model),
                             "model_min": np.min(per_model), "model_max": np.max(per_model)})
            for k in sets[l]:
                rows.append({"layout": short(l), "reference_fold": f, "source": "W_shortest_path", "metric": k, "W": W(sp, k)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    info = json.loads((args.results / "run_info.json").read_text())
    n_boot = args.n_boot or info["settings"]["n_boot"]
    seed = info["settings"]["seed"]
    out = args.results / "sensitivity"
    out.mkdir(exist_ok=True)

    models, episodes, layouts_obj, layouts = load_all(info)
    study = cc.Study(models, episodes)
    ones = {l: np.ones(study.n_configs[l], dtype=np.int64) for l in study.layouts}
    raw_point = study.evaluate(ones, np.ones(study.n_participants, dtype=np.int64))
    point = cc.summarise(study, raw_point, None)
    sets = point["metric_sets"]
    assert sets == info["score_metrics_per_layout"], (sets, info["score_metrics_per_layout"])

    # 1. per-model scores
    rows = []
    for m in models:
        if m.source not in cc.CONDITIONS:
            continue
        l, sp = m.layout, study.sp_label(m.layout)
        vals = [1.0 - raw_point[("W", l, m.label, k)] / raw_point[("W", l, sp, k)] for k in sets[l]]
        rows.append({"layout": short(l), "source": m.source, "fold": m.fold, "seed": m.seed, "model": m.label,
                     "score": float(np.nanmean(vals))})
    pm = pd.DataFrame(rows)
    overall = pm.groupby(["source", "fold", "seed", "model"], as_index=False).score.mean().assign(layout="overall")
    pm = pd.concat([pm, overall[pm.columns]], ignore_index=True)
    pm.to_csv(out / "per_model_scores.csv", index=False)

    # 2. fold reference
    fr = fold_reference(study, raw_point, sets)
    fr.to_csv(out / "fold_reference.csv", index=False)

    # 3-5. bootstrap on the identical replicates
    rng = np.random.default_rng(seed)
    keys = []
    for r in range(n_boot):
        cfg_counts = {l: np.bincount(rng.integers(0, study.n_configs[l], study.n_configs[l]), minlength=study.n_configs[l])
                      for l in study.layouts}
        hum_counts = np.zeros(study.n_participants, dtype=np.int64)
        for f in study.folds:
            idx = study.fold_index[f]
            hum_counts += np.bincount(idx[rng.integers(0, len(idx), len(idx))], minlength=study.n_participants)
        raw = study.evaluate(cfg_counts, hum_counts)
        s = cc.summarise(study, raw, sets)
        a = layout_scores_all_metrics(study, raw, sets)
        rec = {}
        for name, x, y in (("B - A", "B", "A"), ("C - B", "C", "B")):
            rec[(name, "main")] = s[("dLS", name, "overall")]
            no_tj = [s[("LS", l, x)] - s[("LS", l, y)] for l in study.layouts if short(l) != "TJunction"]
            rec[(name, "without TJunction")] = float(np.mean(no_tj))
            rec[(name, "all metrics kept")] = a[("overall", x)] - a[("overall", y)]
            for l in study.layouts:
                rec[(name, f"all metrics kept: {short(l)}")] = a[(l, x)] - a[(l, y)]
        for l in study.layouts:
            for k in sets[l]:
                rec[("dropout", short(l), k)] = float(not np.isfinite(s[("S", l, "A", k)]))
        keys.append(rec)
        if (r + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {r + 1}/{n_boot}", flush=True)

    a_point = layout_scores_all_metrics(study, raw_point, sets)
    rows = []
    for name, x, y in (("B - A", "B", "A"), ("C - B", "C", "B")):
        est = {"main": point[("dLS", name, "overall")],
               "without TJunction": float(np.mean([point[("LS", l, x)] - point[("LS", l, y)] for l in study.layouts
                                                   if short(l) != "TJunction"])),
               "all metrics kept": a_point[("overall", x)] - a_point[("overall", y)]}
        for l in study.layouts:
            est[f"all metrics kept: {short(l)}"] = a_point[(l, x)] - a_point[(l, y)]
        for variant, e in est.items():
            v = np.array([k[(name, variant)] for k in keys])
            v = v[np.isfinite(v)]
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append({"comparison": name, "variant": variant, "estimate": e, "ci_low": lo, "ci_high": hi,
                         "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    sens = pd.DataFrame(rows)
    sens.to_csv(out / "sensitivity.csv", index=False)

    drop = [{"layout": short(l), "metric": k, "share_of_replicates_dropped": np.mean([r[("dropout", short(l), k)] for r in keys])}
            for l in study.layouts for k in sets[l]]
    pd.DataFrame(drop).to_csv(out / "dropout.csv", index=False)

    pd.set_option("display.width", 160)
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(drop).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
