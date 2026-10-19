"""Fold x task splits for the A/B/C experiment (methodology 3.8), written once after all human sessions are recorded.

Every human manifest already carries its design labels (manifest["trial"], written by ParticipantSession): the
participant fold (1 = odd participant numbers, 2 = even), the task walked (task_id "train_0".."train_2" or "test",
task_role "train"/"test") and whether condition C leaves this training task out (c_excluded). This script only
collects and checks them and writes

    {"design": "folds_x_tasks_v1",
     "folds": {"1": [participants], "2": [participants]},
     "c_excluded": {scene_name: task_id},
     "test_episodes": {"1": [episode ids], "2": [...]},      # each fold's test-task walks = the human reference
     "episodes_per_participant": {...}, "repeats": [...]}

A model trained on fold f never loads the other fold's participants; the other fold's test_episodes are its reference.

    A: synthetic only (no humans, no fold)
    B: synthetic + fold f humans, the three training tasks per layout
    C: synthetic + fold f humans, the test task + the two training tasks that are not c_excluded

B and C have the same participants and the same number of episodes; they differ only in whether the evaluated route was
seen. dryrun_* participants are ignored.

    python make_splits.py
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from data_contract import FOLDS, SOURCE_HUMAN, TASK_ROLE_TEST, TASK_ROLE_TRAIN, read_manifest

DESIGN = "folds_x_tasks_v1"
DRY_RUN_PREFIX = "dryrun_"
CONDITIONS = ("A", "B", "C")
HUMAN_CONDITIONS = ("B", "C")
TRAINING_TASKS_PER_LAYOUT = 3
LAYOUT_DIR = Path(__file__).resolve().parents[2] / "eval_tools" / "layouts"


def human_episodes(recordings_dir: Path) -> list:
    """[(episode_dir, manifest)] of every human, non-dry-run episode. Raises on an invalid manifest (read_manifest),
    so a broken episode can't silently drop out of a condition."""
    result = []
    for ep in sorted(p for p in recordings_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        manifest = read_manifest(ep)
        if manifest["source"] == SOURCE_HUMAN and not manifest["participant_id"].startswith(DRY_RUN_PREFIX):
            result.append((ep, manifest))
    return result


def participant_number(pid: str) -> int:
    m = re.search(r"(\d+)$", pid)
    if not m:
        raise ValueError(f"participant id {pid!r} does not end in a number")
    return int(m.group(1))


def in_condition(trial: dict, condition: str) -> bool:
    """Whether a human episode with these labels belongs to condition B or C (fold filtering is separate)."""
    if condition == "B":
        return trial["task_role"] == TASK_ROLE_TRAIN
    if condition == "C":
        return trial["task_role"] == TASK_ROLE_TEST or (trial["task_role"] == TASK_ROLE_TRAIN and not trial["c_excluded"])
    return False


def make_splits(episodes: list) -> tuple:
    """-> (splits dict, problems list). Problems are design violations that would make B and C incomparable."""
    problems = []
    folds = defaultdict(set)
    c_excluded = defaultdict(set)
    walks = defaultdict(list)  # (pid, scene, task_id) -> episode ids
    test_episodes = defaultdict(list)
    repeats = []
    per_participant = defaultdict(int)

    for ep, m in episodes:
        pid, scene, trial = m["participant_id"], m["scene_name"], m["trial"]
        fold = trial["fold"]
        expected_fold = 1 if participant_number(pid) % 2 else 2
        if fold != expected_fold:
            problems.append(f"{ep.name}: {pid} is labelled fold {fold}, but its number puts it in fold {expected_fold}")
        folds[fold].add(pid)
        per_participant[pid] += 1
        walks[(pid, scene, trial["task_id"])].append(ep.name)
        if trial["c_excluded"]:
            c_excluded[scene].add(trial["task_id"])
        if trial["task_role"] == TASK_ROLE_TEST:
            test_episodes[fold].append(ep.name)
        if trial.get("attempt_index", 0) > 0:
            repeats.append({"episode": ep.name, "participant": pid, "scene": scene, "task_id": trial["task_id"],
                            "attempt_index": trial["attempt_index"]})

    both = folds[1] & folds[2]
    if both:
        problems.append(f"participants in both folds: {sorted(both)}")
    for scene, tasks in sorted(c_excluded.items()):
        if len(tasks) != 1:
            problems.append(f"{scene}: episodes disagree on the C-excluded task {sorted(tasks)}")
    c_map = {scene: next(iter(tasks)) for scene, tasks in c_excluded.items() if len(tasks) == 1}

    # Cross-check with the generator's export (the tasks fixed before recording).
    for scene, task_id in sorted(c_map.items()):
        exported = LAYOUT_DIR / f"{scene}_human_tasks.json"
        if exported.exists():
            marked = [t["id"] for t in json.loads(exported.read_text())["tasks"] if t.get("c_excluded")]
            if marked != [task_id]:
                problems.append(f"{scene}: manifests say C leaves out {task_id}, {exported.name} says {marked}")

    # Every participant walks every task of every layout exactly once.
    scenes = sorted({m["scene_name"] for _, m in episodes})
    task_ids = [f"train_{i}" for i in range(TRAINING_TASKS_PER_LAYOUT)] + ["test"]
    for pid in sorted(per_participant):
        for scene in scenes:
            for task_id in task_ids:
                n = len(walks.get((pid, scene, task_id), []))
                if n != 1:
                    problems.append(f"{pid} {scene} {task_id}: {n} completed walks (expected 1)"
                                    + (f" {walks[(pid, scene, task_id)]}" if n > 1 else ""))
    for scene in scenes:
        if scene not in c_map:
            problems.append(f"{scene}: no episode marks a C-excluded task")

    splits = {
        "design": DESIGN,
        "folds": {str(f): sorted(folds[f]) for f in FOLDS},
        "c_excluded": dict(sorted(c_map.items())),
        "test_episodes": {str(f): sorted(test_episodes[f]) for f in FOLDS},
        "episodes_per_participant": dict(sorted(per_participant.items())),
        "repeats": repeats,
    }
    return splits, problems


def load_splits(path: str) -> dict:
    """Read and check splits.json (fold x task design): two disjoint, non-empty folds."""
    with open(path) as f:
        splits = json.load(f)
    if splits.get("design") != DESIGN:
        raise ValueError(f"{path}: design is {splits.get('design')!r}, expected {DESIGN!r} - rewrite it with make_splits.py")
    folds = {int(k): set(v) for k, v in splits["folds"].items()}
    if set(folds) != set(FOLDS):
        raise ValueError(f"{path}: folds {sorted(folds)}, expected {list(FOLDS)}")
    if folds[1] & folds[2]:
        raise ValueError(f"{path}: participants in both folds: {sorted(folds[1] & folds[2])}")
    return splits


def condition_humans(splits: dict, condition: str, fold: int) -> list:
    """Participants a condition trains (and validates) on: none for A, the fold's participants for B and C."""
    if condition == "A":
        return []
    return sorted(splits["folds"][str(fold)])


def heldout_humans(splits: dict, condition: str, fold: int) -> list:
    """Participants a model must never load: the other fold (A: every participant, it loads no humans)."""
    if condition == "A":
        return sorted(p for f in FOLDS for p in splits["folds"][str(f)])
    return sorted(splits["folds"][str(3 - fold)])


def main():
    parser = argparse.ArgumentParser(description="Write the fold x task splits.json for conditions A/B/C.")
    parser.add_argument("--recordings_dir", type=str, default="../zarr_recordings")
    parser.add_argument("--output", type=str, default="../splits.json")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing splits file.")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Write the file despite missing/duplicate walks (e.g. a participant who dropped out).")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"[ERROR] {out} exists. Models trained on it would no longer match; pass --force to overwrite.")
        sys.exit(1)
    episodes = human_episodes(Path(args.recordings_dir))
    if not episodes:
        print(f"[ERROR] No human episodes in {args.recordings_dir}.")
        sys.exit(1)
    splits, problems = make_splits(episodes)
    for f in FOLDS:
        print(f"[SPLITS] fold {f}: {splits['folds'][str(f)]} ({len(splits['test_episodes'][str(f)])} test-task walks)")
    print(f"[SPLITS] C leaves out: {splits['c_excluded']}")
    if splits["repeats"]:
        print(f"[SPLITS] {len(splits['repeats'])} walks were repeats after an abort: "
              + ", ".join(f"{r['participant']} {r['scene']} {r['task_id']}" for r in splits["repeats"]))
    if problems:
        print("[SPLITS] PROBLEMS:\n  " + "\n  ".join(problems))
        if not args.allow_incomplete:
            print("[ERROR] Not written. Fix the recordings, or pass --allow_incomplete if this is intended.")
            sys.exit(1)
    splits["problems_at_write"] = problems
    out.write_text(json.dumps(splits, indent=2))
    print(f"[SPLITS] written to {out}")


if __name__ == "__main__":
    main()
