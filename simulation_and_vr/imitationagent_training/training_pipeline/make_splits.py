"""Participant-level splits for the A/B/C experiment, written once after all human sessions are recorded.

Reads every human manifest in --recordings_dir and writes

    {"heldout": [...], "train": [...], "subset_B": [...], "seed": ..., "episodes_per_participant": {...}}

- heldout: never trained or validated on; the human baseline for EvalHarness.
- train: the humans of condition C (every training participant).
- subset_B: the humans of condition B, a subset of train.
Condition A uses no humans. Participants are sorted, shuffled with --seed and cut by the counts, so the same seed,
counts and participant set always give the same file. dryrun_* participants (ParticipantSession dry runs) are ignored.

    python make_splits.py --n_heldout 4 --n_subset_B 4 --seed 1
"""
import argparse
import json
import random
import sys
from pathlib import Path

from data_contract import SOURCE_HUMAN, read_manifest

DRY_RUN_PREFIX = "dryrun_"
CONDITIONS = ("A", "B", "C")


def human_participants(recordings_dir: Path) -> dict:
    """{participant_id: episode count} over every human manifest in recordings_dir (dry runs excluded).
    Raises on an invalid manifest, like load_episodes, so a broken episode can't silently drop a participant."""
    counts = {}
    for ep in sorted(p for p in recordings_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        manifest = read_manifest(ep)
        pid = manifest["participant_id"]
        if manifest["source"] == SOURCE_HUMAN and not pid.startswith(DRY_RUN_PREFIX):
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def make_splits(participants, n_heldout: int, n_subset_b: int, seed: int) -> dict:
    ids = sorted(participants)
    n_train = len(ids) - n_heldout
    if n_heldout < 1 or n_train < 2:
        raise ValueError(f"{len(ids)} participants: need n_heldout >= 1 and at least 2 left for training "
                         f"(n_heldout={n_heldout})")
    if not 1 <= n_subset_b < n_train:
        raise ValueError(f"n_subset_B must be in [1, {n_train - 1}] so that B has fewer humans than C (got {n_subset_b})")
    random.Random(seed).shuffle(ids)
    heldout, train = ids[:n_heldout], ids[n_heldout:]
    return {"heldout": sorted(heldout), "train": sorted(train), "subset_B": sorted(train[:n_subset_b]), "seed": seed}


def load_splits(path: str) -> dict:
    """Read and check splits.json: heldout and train disjoint, subset_B a strict subset of train."""
    with open(path) as f:
        splits = json.load(f)
    missing = [k for k in ("heldout", "train", "subset_B") if not isinstance(splits.get(k), list)]
    if missing:
        raise ValueError(f"{path}: missing list(s) {missing} - write it with make_splits.py")
    heldout, train, subset_b = set(splits["heldout"]), set(splits["train"]), set(splits["subset_B"])
    problems = []
    if heldout & train:
        problems.append(f"participants both held out and training: {sorted(heldout & train)}")
    if not subset_b < train:
        problems.append(f"subset_B is not a strict subset of train (extra: {sorted(subset_b - train)})")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return splits


def condition_humans(splits: dict, condition: str) -> list:
    """Human participants condition trains (and validates) on: A none, B subset_B, C train. Never a held-out one."""
    humans = {"A": [], "B": splits["subset_B"], "C": splits["train"]}[condition]
    assert not set(humans) & set(splits["heldout"])
    return sorted(humans)


def main():
    parser = argparse.ArgumentParser(description="Write participant-level splits.json for conditions A/B/C.")
    parser.add_argument("--recordings_dir", type=str, default="../zarr_recordings")
    parser.add_argument("--output", type=str, default="../splits.json")
    parser.add_argument("--n_heldout", type=int, required=True, help="Participants never trained on (human baseline).")
    parser.add_argument("--n_subset_B", type=int, required=True, help="Training participants used by condition B.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing splits file.")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"[ERROR] {out} exists. Models trained on it would no longer match; pass --force to overwrite.")
        sys.exit(1)
    participants = human_participants(Path(args.recordings_dir))
    splits = make_splits(participants, args.n_heldout, args.n_subset_B, args.seed)
    splits["episodes_per_participant"] = dict(sorted(participants.items()))
    out.write_text(json.dumps(splits, indent=2))
    print(f"[SPLITS] {len(participants)} participants -> heldout {splits['heldout']}, train {splits['train']}, "
          f"subset_B {splits['subset_B']}\n[SPLITS] written to {out}")


if __name__ == "__main__":
    main()
