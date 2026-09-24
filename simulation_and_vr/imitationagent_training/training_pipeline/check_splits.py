"""Leak check for the A/B/C experiment: no held-out participant may appear in any training or validation episode list.

1. splits.json is consistent (make_splits.load_splits) and covers every human participant in the recordings.
2. For each condition A, B, C: the exact selection and split train_2DCNN_discrete.py performs (same --seed,
   --val_ratio, --val_participants, --synthetic_sources) contains no held-out participant's episode.
3. Every sidecar .json given: its train_episodes / val_episodes (the lists the model was actually trained and
   validated on) contain no held-out participant's episode, and it was trained from the same heldout list.

    python check_splits.py --seed 1 ../imitation_agent_model/ImitationAgent_*_s1.json
Exit code 0 = clean, 1 = a leak or an inconsistency (printed).
"""
import argparse
import json
import sys
from pathlib import Path

from data_contract import SOURCE_HUMAN, read_manifest
from make_splits import CONDITIONS, condition_humans, human_participants, load_splits
from train_2DCNN_discrete import HUMAN, select_episodes, split_train_val


def main():
    parser = argparse.ArgumentParser(description="Check that no held-out participant is trained or validated on.")
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
    heldout = set(splits["heldout"])
    problems = []

    unlisted = sorted(set(human_participants(recordings)) - heldout - set(splits["train"]))
    if unlisted:
        problems.append(f"human participants in the recordings but not in {args.splits}: {unlisted}")

    episodes = [(ep, read_manifest(ep)) for ep in sorted(p for p in recordings.iterdir()
                                                          if p.is_dir() and not p.name.startswith("."))]
    for condition in CONDITIONS:
        selected, _ = select_episodes(episodes, args.synthetic_sources, condition_humans(splits, condition))
        train, val, val_pids = split_train_val(selected, args.val_ratio, args.val_participants, args.seed)
        leaked = sorted({pid for _, src, pid in train + val if src == HUMAN and pid in heldout})
        humans = lambda items: sorted({pid for _, src, pid in items if src == HUMAN})
        print(f"[CHECK] condition {condition}: train {len(train)} episodes, humans {humans(train) or 'none'} | "
              f"val {len(val)} episodes, humans {humans(val) or 'none'} | held-out leaked: {leaked or 'none'}")
        if leaked:
            problems.append(f"condition {condition} selects held-out participants {leaked}")

    manifest_pid = {ep.name: (m["source"], m["participant_id"]) for ep, m in episodes}
    for path in args.sidecars:
        sidecar = json.loads(Path(path).read_text())
        if set(sidecar.get("heldout_participants", [])) != heldout:
            problems.append(f"{path}: trained with heldout {sidecar.get('heldout_participants')}, splits say {sorted(heldout)}")
        for key in ("train_episodes", "val_episodes"):
            if key not in sidecar:
                problems.append(f"{path}: no {key} list")
                continue
            missing = [name for name in sidecar[key] if name not in manifest_pid]
            leaked = sorted({manifest_pid[n][1] for n in sidecar[key] if n in manifest_pid
                             and manifest_pid[n][0] == SOURCE_HUMAN and manifest_pid[n][1] in heldout})
            if missing:
                problems.append(f"{path}: {len(missing)} {key} not in {recordings} (e.g. {missing[0]}) - cannot verify")
            if leaked:
                problems.append(f"{path}: {key} contain held-out participants {leaked}")
        print(f"[CHECK] {path}: condition {sidecar.get('condition')}, {len(sidecar.get('train_episodes', []))} train / "
              f"{len(sidecar.get('val_episodes', []))} val episodes checked")

    if problems:
        print("[CHECK] FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"[CHECK] OK - held-out participants {sorted(heldout)} appear in no training or validation episode list.")


if __name__ == "__main__":
    main()
