#!/usr/bin/env python3
"""Quick sanity check: EvalHarness runs (SimulationLogs/EvalRun_*/episodes.csv) as one table, straight from Unity's own
episodes.csv columns. The thesis results (metrics against held-out humans, Wasserstein distances, figures) come from
compare_conditions.py, which recomputes every metric from the trajectories so agents and humans are measured alike.

Usage:
  python3 summarize_eval.py                       # every EvalRun_* under the default SimulationLogs folder
  python3 summarize_eval.py RUN_DIR [RUN_DIR ...] # specific runs
  python3 summarize_eval.py --ref 45 --csv out.csv

Standard library only. Needs harness_version 6 folders (agent_kind column; ShortestPath runs leave the
model, nudge and collision columns empty, shown as "-").
Success = outcome == "success" (episodes flagged invalid_speed count as failures and are reported separately).
"success@ref" applies the human-comparable cutoff (--ref, default 45 sim seconds) to success_time_s / episode length.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "unity_files", "SimulationLogs")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def blocked_seconds_from_model_debug(run_dir, config_id, agent_id):
    """Sum of tick durations where the agent was wall-blocked, from modeldebug_config<C>_agent<A>.csv.
    None if not logged (EvalHarness.logModelDebugTicks off, or ShortestPath: a NavMeshAgent cannot be blocked)."""
    path = os.path.join(run_dir, f"modeldebug_config{config_id}_agent{agent_id}.csv")
    if not os.path.exists(path):
        return None
    total, prev_t, measured = 0.0, None, False
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            t = fnum(row.get("t"))
            if t is None:
                continue
            if row.get("blocked") not in (None, ""):
                measured = True
            if prev_t is not None and row.get("blocked") == "1":
                total += t - prev_t
            prev_t = t
    return total if measured else None


def load_run(run_dir):
    cfg = {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    with open(os.path.join(run_dir, "episodes.csv")) as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return cfg, rows


def summarise(run_dir, ref):
    cfg, rows = load_run(run_dir)
    n = len(rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    invalid = [r for r in rows if r["outcome"] == "invalid_speed"]
    lo, hi = wilson(len(ok), n)

    def succ_within_ref(r):
        t = fnum(r.get("success_time_s")) or fnum(r.get("episode_length_seconds"))
        return r["outcome"] == "success" and t is not None and t <= ref

    ref_k = sum(1 for r in rows if succ_within_ref(r))
    rlo, rhi = wilson(ref_k, n)

    blocked = [blocked_seconds_from_model_debug(run_dir, r["config_id"], r["agent_id"]) for r in rows]
    kind = cfg.get("agent_kind") or (rows[0].get("agent_kind") if rows else "") or "?"

    return {
        "run": os.path.basename(run_dir.rstrip("/")).split("_")[-2][-4:] + "_" + os.path.basename(run_dir.rstrip("/")).split("_")[-1] if os.path.basename(run_dir.rstrip("/")).count("_") >= 2 else os.path.basename(run_dir.rstrip("/")),
        "dir": os.path.basename(run_dir.rstrip("/")),
        "kind": kind,
        "scene": (cfg.get("scene") or (rows[0].get("scene") if rows else "")) or "",
        "model": (cfg.get("model_sha256") or (rows[0].get("model_sha256") if rows else "")) or "",
        "nudge": "-" if kind == "ShortestPath" else ("on" if (cfg.get("nudge_enabled") if cfg else (rows and rows[0].get("nudge_enabled") == "1")) else "off"),
        "task": ("random" if cfg.get("randomize_markers") else "scene") if cfg else "?",
        "pois": cfg.get("task_number_of_needs", rows[0].get("num_pois_total") if rows else ""),
        "timeout": cfg.get("episode_timeout_sim_s", rows[0].get("timeout_s") if rows else ""),
        "jitter": f"{cfg.get('start_jitter_position_m', 0)}m/{cfg.get('start_jitter_heading_deg', 0)}deg",
        "n": n,
        "succ": len(ok),
        "succ_rate": len(ok) / n if n else float("nan"),
        "ci": (lo, hi),
        "succ_ref": ref_k,
        "ref_rate": ref_k / n if n else float("nan"),
        "ref_ci": (rlo, rhi),
        "invalid": len(invalid),
        "t_succ_med": med([fnum(r.get("success_time_s")) or fnum(r["episode_length_seconds"]) for r in ok]),
        "t_poi_med": med([fnum(r.get("time_to_first_poi_s")) for r in rows]),
        "spl_mean": mean([fnum(r.get("spl")) for r in rows]),
        "spl_succ": mean([min(1.0, fnum(r["optimal_path_length_m"]) / max(fnum(r["path_length_m"]), fnum(r["optimal_path_length_m"])))
                          for r in ok if fnum(r.get("path_length_m")) and fnum(r.get("optimal_path_length_m"))]),
        "final_dist_mean": mean([fnum(r.get("final_dist_to_goal_m")) for r in rows]),
        "stall_mean": mean([fnum(r.get("stall_seconds")) for r in rows]),
        "longest_stall_med": med([fnum(r.get("longest_stall_s")) for r in rows]),
        "coll_events_mean": mean([fnum(r.get("collision_events")) for r in rows]),
        "blocked_mean": mean(blocked),
        "nudges_mean": mean([fnum(r.get("nudge_count")) for r in rows]),
        "stall_positions": [r.get("longest_stall_pos") for r in rows if r.get("longest_stall_pos")],
        "n_speed_flag": sum(1 for r in rows if r.get("speed_violation") == "1"),
        "harness": rows[0].get("harness_version", "?") if rows else "?",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="EvalRun_* folders (default: all under SimulationLogs)")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--ref", type=float, default=45.0, help="human-comparable cutoff in sim seconds (default 45)")
    ap.add_argument("--episodes", action="store_true", help="also print one line per episode")
    ap.add_argument("--csv", help="also write the summary table to this CSV")
    args = ap.parse_args()

    runs = args.runs or sorted(r for r in glob.glob(os.path.join(args.root, "EvalRun_*")) +
                               glob.glob(os.path.join(args.root, "EvalQueue_*", "EvalRun_*")) if "_incomplete_" not in r)
    runs = [r for r in runs if os.path.exists(os.path.join(r, "episodes.csv"))]
    if not runs:
        raise SystemExit("No EvalRun_* folders with episodes.csv found.")

    sums = [summarise(r, args.ref) for r in runs]

    print(f"\nsuccess = outcome 'success' (invalid_speed counts as failure).  success@{args.ref:g}s = success within the human-comparable cutoff.")
    print("Wilson 95% CI in brackets.  Times are simulated seconds.  SPLs = path efficiency over successful episodes (optimal/actual path, nudge ignored).\n")
    hdr = f"{'run':12s} {'kind':12s} {'nudge':5s} {'jitter':10s} {'n':>3s}  {'success':>8s} {'CI':>11s}  {'succ@'+format(args.ref,'g')+'s':>9s} {'CI':>11s}  {'t_succ':>6s} {'t_poi':>6s} {'SPLs':>5s} {'d_goal':>6s} {'stall_s':>7s} {'long_st':>7s} {'coll_ev':>7s} {'blk_s':>6s} {'inval':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for s in sums:
        ci = f"[{s['ci'][0]*100:3.0f},{s['ci'][1]*100:3.0f}]"
        rci = f"[{s['ref_ci'][0]*100:3.0f},{s['ref_ci'][1]*100:3.0f}]"
        print(f"{s['run']:12s} {s['kind']:12s} {s['nudge']:5s} {s['jitter'].replace('deg','°'):10s} {s['n']:>3d}  "
              f"{s['succ']:>3d}/{s['n']:<3d}{'':1s} {ci:>11s}  {s['succ_ref']:>4d}/{s['n']:<4d} {rci:>11s}  "
              f"{fmt(s['t_succ_med']):>6s} {fmt(s['t_poi_med']):>6s} {fmt(s['spl_succ'],2):>5s} {fmt(s['final_dist_mean']):>6s} "
              f"{fmt(s['stall_mean']):>7s} {fmt(s['longest_stall_med']):>7s} {fmt(s['coll_events_mean']):>7s} {fmt(s['blocked_mean']):>6s} {s['invalid']:>5d}")

    print()
    for s in sums:
        if s["stall_positions"]:
            print(f"{s['run']:12s} longest-stall positions: {'; '.join(s['stall_positions'][:6])}{' ...' if len(s['stall_positions']) > 6 else ''}")

    print("\nComparability checks:")
    factors = {"kind": "agent kind", "scene": "scene", "model": "model", "task": "task mode", "pois": "#POIs", "timeout": "timeout", "nudge": "nudge", "jitter": "start jitter"}
    differing = [name for key, name in factors.items() if len({str(x[key]) for x in sums}) > 1]
    print(f"  factors that differ between the listed runs: {', '.join(differing) if differing else 'none'}")
    if "agent kind" in differing:
        differing = [d for d in differing if d not in ("model", "nudge")]  # these follow from the agent kind
    if len(differing) > 1:
        print("  NOTE: more than one factor differs - only pairs of runs that differ in exactly one factor are a clean comparison.")
    if any(s["n_speed_flag"] for s in sums):
        print("  WARNING: some episodes were flagged speed_violation - inspect them before trusting the numbers.")
    if any(s["n"] < 20 for s in sums):
        print("  NOTE: n < 20 in at least one run; the CIs are wide. With no start jitter the episodes are near-copies, so the effective sample size is even smaller.")

    if args.episodes:
        for r_dir, sm in zip(runs, sums):
            _, rows = load_run(r_dir)
            print(f"\n{sm['run']} ({sm['kind']}, nudge {sm['nudge']}, jitter {sm['jitter']}):")
            print(f"  {'ep':>2s} {'outcome':8s} {'t_end':>6s} {'t_poi':>6s} {'d_goal':>6s} {'stall_s':>7s} {'long_st':>7s} {'coll_ev':>7s} {'off_m':>5s} {'hd°':>5s}  longest-stall pos")
            for r in rows:
                print(f"  {r['episode_id']:>2s} {r['outcome']:8s} {fmt(fnum(r.get('episode_length_seconds'))):>6s} {fmt(fnum(r.get('time_to_first_poi_s'))):>6s} "
                      f"{fmt(fnum(r.get('final_dist_to_goal_m'))):>6s} {fmt(fnum(r.get('stall_seconds'))):>7s} {fmt(fnum(r.get('longest_stall_s'))):>7s} "
                      f"{r.get('collision_events') or '-':>7s} {fmt(fnum(r.get('start_offset_m')),2):>5s} {fmt(fnum(r.get('start_heading_jitter_deg'))):>5s}  {r.get('longest_stall_pos','')}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            cols = [k for k in sums[0].keys() if k not in ("stall_positions",)]
            w.writerow(cols)
            for s in sums:
                w.writerow([s[c] for c in cols])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
