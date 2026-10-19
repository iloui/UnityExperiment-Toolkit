"""Leak and balance check for the fold x task experiment (methodology 3.8). Run it before and after training.

1. splits.json is consistent (make_splits.load_splits) and covers every human participant in the recordings.
2. For each fold and condition B, C: the exact selection and split train_2DCNN_discrete.py performs (same --seed,
   --val_ratio, --val_participants, --synthetic_sources) contains no participant of the other fold, B contains no
   test-task walk, C contains no c_excluded walk, and B and C have the same participants, the same validation
   participant and the same number of human episodes.
3. Every sidecar .json given: its train_episodes / val_episodes (the lists the model was actually trained and validated
   on) obey the same rules for the sidecar's condition and fold.

    python check_splits.py --seed 1 ../imitation_agent_model/ImitationAgent_*_s1.json
Exit code 0 = clean, 1 = a leak or an inconsistency (printed).
"""
import argparse
import json
import sys
from pathlib import Path

from data_contract import FOLDS, SOURCE_HUMAN, TASK_ROLE_TEST, read_manifest
from make_splits import DRY_RUN_PREFIX, HUMAN_CONDITIONS, condition_humans, heldout_humans, load_splits
from train_2DCNN_discrete import HUMAN, select_episodes, split_train_val


def rule_violations(names, manifests, condition, fold, splits):
    """Problems with a list of episode ids for a condition/fold: held-out participants, wrong tasks."""
    problems = []
    heldout = set(heldout_humans(splits, condition, fold)) if splits else set()
    for name in names:
        m = manifests.get(name)
        if m is None or m["source"] != SOURCE_HUMAN:
            continue
        pid, trial = m["participant_id"], m["trial"]
        if condition == "A" or pid in heldout:
            problems.append(f"{name}: participant {pid} must not be loaded by {condition} fold {fold}")
        elif condition == "B" and trial["task_role"] == TASK_ROLE_TEST:
            problems.append(f"{name}: B contains a test-task walk ({pid} {m['scene_name']})")
        elif condition == "C" and trial["c_excluded"]:
            problems.append(f"{name}: C contains the excluded {trial['task_id']} ({pid} {m['scene_name']})")
    return problems


def main():
    parser = argparse.ArgumentParser(description="Check folds, tasks and held-out participants of the A/B/C models.")
    parser.add_argument("sidecars", nargs="*", help="Model sidecar .json files written by train_2DCNN_discrete.py.")
    parser.add_argument("--recordings_dir", type=str, default="../zarr_recordings")
    parser.add_argument("--splits", type=str, default="../splits.json")
    parser.add_argument("--synthetic_sources", type=str, nargs="+", default=["synthetic_rrw_v3"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--val_participants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    recordings = Path(args.recordings_dir)
    splits = load_splits(args.splits)
    problems = []

    episodes = [(ep, read_manifest(ep)) for ep in sorted(p for p in recordings.iterdir()
                                                          if p.is_dir() and not p.name.startswith("."))]
    manifests = {ep.name: m for ep, m in episodes}
    in_folds = {p for f in FOLDS for p in splits["folds"][str(f)]}
    recorded = {m["participant_id"] for _, m in episodes
                if m["source"] == SOURCE_HUMAN and not m["participant_id"].startswith(DRY_RUN_PREFIX)}
    if recorded - in_folds:
        problems.append(f"human participants in the recordings but in no fold of {args.splits}: {sorted(recorded - in_folds)}")

    for fold in FOLDS:
        summary = {}
        for condition in HUMAN_CONDITIONS:
            humans = condition_humans(splits, condition, fold)
            heldout = heldout_humans(splits, condition, fold)
            selected, _ = select_episodes(episodes, args.synthetic_sources, humans, heldout, condition)
            train, val, val_pids = split_train_val(selected, args.val_ratio, args.val_participants, args.seed)
            names = [Path(e[0]).name for e in train + val]
            problems += [f"fold {fold} {condition}: {p}" for p in rule_violations(names, manifests, condition, fold, splits)]
            human = [e for e in train + val if e[1] == HUMAN]
            summary[condition] = (sorted({e[2] for e in human}), val_pids, len(human),
                                  sum(1 for e in train if e[1] == HUMAN), sum(1 for e in train if e[1] != HUMAN))
            print(f"[CHECK] fold {fold} {condition}: humans {summary[condition][0]} (val {val_pids or 'none'}) | "
                  f"human episodes {summary[condition][2]} ({summary[condition][3]} train) | "
                  f"synthetic train {summary[condition][4]}")
        if summary["B"][:3] != summary["C"][:3]:
            problems.append(f"fold {fold}: B and C differ in participants, validation participant or human episode count "
                            f"(B {summary['B'][:3]}, C {summary['C'][:3]})")

    for path in args.sidecars:
        sidecar = json.loads(Path(path).read_text())
        condition, fold = sidecar.get("condition"), sidecar.get("fold")
        if condition in HUMAN_CONDITIONS and fold not in FOLDS:
            problems.append(f"{path}: condition {condition} without a fold (trained with the old splits design?)")
            continue
        expected_heldout = heldout_humans(splits, condition, fold) if condition in HUMAN_CONDITIONS else None
        if expected_heldout is not None and sorted(sidecar.get("heldout_participants", [])) != expected_heldout:
            problems.append(f"{path}: held out {sidecar.get('heldout_participants')}, splits say {expected_heldout}")
        for key in ("train_episodes", "val_episodes"):
            if key not in sidecar:
                problems.append(f"{path}: no {key} list")
                continue
            missing = [n for n in sidecar[key] if n not in manifests]
            if missing:
                problems.append(f"{path}: {len(missing)} {key} not in {recordings} (e.g. {missing[0]}) - cannot verify")
            problems += [f"{path} {key}: {p}" for p in rule_violations(sidecar[key], manifests, condition, fold, splits)]
        print(f"[CHECK] {path}: condition {condition}, fold {fold or '-'}, seed {sidecar.get('seed')}, "
              f"{len(sidecar.get('train_episodes', []))} train / {len(sidecar.get('val_episodes', []))} val episodes checked")

    if problems:
        print("[CHECK] FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("[CHECK] OK - no model loads the other fold, B has no test-task walks, C has no excluded task, B and C are balanced.")


if __name__ == "__main__":
    main()
