"""
Aggregates one or more EvalHarness runs (Assets/Scripts/DataRecording/EvalHarness.cs's
episodes.csv output, under SimulationLogs/EvalRun_<scene>_<timestamp>/) into the summary
numbers EvalHarness's own docstring calls "a deliberately separate downstream (pandas) step":
goal-completion rate, collision counts, episode length, and path efficiency (SPL-style ratio).

Usage:
    python summarize_eval_run.py <run_dir_or_episodes.csv> [<run_dir_or_episodes.csv> ...]
    python summarize_eval_run.py --out summary.csv <run_dir> <run_dir> ...

Each argument is either a run folder (containing episodes.csv) or a direct path to an
episodes.csv. Passing several prints one row per run so runs can be compared side by side -
e.g. an in-distribution CentralPillar baseline against a zero-shot unseen-layout run.
"""

import argparse
import os
import sys

import pandas as pd

OUTCOME_SUCCESS = "success"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_OTHER = "other"  # configuration-level failure (e.g. navmesh placement failed) - no agent ever ran


def resolve_episodes_csv(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "episodes.csv")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"No episodes.csv found in '{path}'.")
        return candidate
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"'{path}' is neither a run folder nor a file.")


def summarize_one(path: str) -> dict:
    csv_path = resolve_episodes_csv(path)
    df = pd.read_csv(csv_path, sep=";")

    run_label = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)

    skipped = df[df["outcome"] == OUTCOME_OTHER]
    episodes = df[df["outcome"] != OUTCOME_OTHER]  # real agent runs only

    n = len(episodes)
    if n == 0:
        return {
            "run": run_label,
            "layout_mode": "n/a",
            "n_episodes": 0,
            "n_skipped_configs": len(skipped),
            "goal_completion_rate": float("nan"),
            "timeout_rate": float("nan"),
            "mean_collisions": float("nan"),
            "median_collisions": float("nan"),
            "mean_episode_length_s": float("nan"),
            "mean_spl": float("nan"),
        }

    successes = episodes[episodes["outcome"] == OUTCOME_SUCCESS]
    timeouts = episodes[episodes["outcome"] == OUTCOME_TIMEOUT]

    # SPL-style path efficiency (Success weighted by Path Length): for successful episodes only,
    # optimal_path_length_m / max(path_length_m, optimal_path_length_m), 0 for non-successes.
    # EvalHarness.cs notes optimal_path_length_m is straight spawn->final-goal, not threaded
    # through every POI actually visited - a rough efficiency signal, not an exact SPL.
    spl = 0.0
    if len(successes) > 0:
        denom = pd.concat([successes["path_length_m"], successes["optimal_path_length_m"]], axis=1).max(axis=1)
        spl_per_episode = successes["optimal_path_length_m"] / denom.replace(0, pd.NA)
        spl = spl_per_episode.fillna(0).sum() / n  # averaged over ALL episodes, not just successes

    layout_modes = episodes["layout_mode"].unique() if "layout_mode" in episodes.columns else ["n/a (older run, column missing)"]
    layout_mode = layout_modes[0] if len(layout_modes) == 1 else f"MIXED: {list(layout_modes)}"

    return {
        "run": run_label,
        "layout_mode": layout_mode,
        "n_episodes": n,
        "n_skipped_configs": len(skipped),
        "goal_completion_rate": len(successes) / n,
        "timeout_rate": len(timeouts) / n,
        "mean_collisions": episodes["collision_count"].mean(),
        "median_collisions": episodes["collision_count"].median(),
        "mean_episode_length_s": episodes["episode_length_seconds"].mean(),
        "mean_spl": spl,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run folder(s) (containing episodes.csv) or direct episodes.csv path(s).")
    parser.add_argument("--out", default=None, help="Optional path to also write the summary table as CSV.")
    args = parser.parse_args()

    rows = []
    for run_path in args.runs:
        try:
            rows.append(summarize_one(run_path))
        except FileNotFoundError as e:
            print(f"[summarize_eval_run] SKIPPING '{run_path}': {e}", file=sys.stderr)

    if not rows:
        print("[summarize_eval_run] No runs could be summarized.", file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(summary.to_string())

    if args.out:
        summary.to_csv(args.out)
        print(f"\n[summarize_eval_run] Wrote summary to '{args.out}'.")


if __name__ == "__main__":
    main()
