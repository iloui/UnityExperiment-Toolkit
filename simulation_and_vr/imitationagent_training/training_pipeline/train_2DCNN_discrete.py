"""Behaviour cloning with a discrete action head (successor of train_onAllRecordings_2DCNNModel.py / M4).

Why: M4 regressed yaw with MSE. Wherever the demonstrations turn both ways from the same view - a wall ahead with
the goal behind it, the T-junction - the MSE optimum is the average, ~0 deg/s, and the agent pushed into the wall
until the timeout (training frames with a wall < 1.5 m ahead and the goal within +-20 deg: human 6 left / 7 right,
v2 14 left / 5 right). The flip augmentation makes those frames exactly symmetric. Here:

  - yaw is a 7-way classification (straight + weak/medium/strong per side, see data_contract.YAW_BIN_*), decoded in
    Unity by picking the most likely side and averaging only within it, so the two sides can't cancel;
  - forward is a walk/stop classification (labels are 0 or 1.3 m/s), so standing still to turn is its own decision;
  - pitch stays an MSE regression;
  - human and synthetic frames are drawn 50/50 per batch (--human_share);
  - stop frames are up-weighted to --target_stop_share (the v2 bot stood in 4.3% of samples, humans in 20.5%).

Experiment conditions (--condition, --fold; splits.json from make_splits.py, methodology 3.8):
  A = synthetic only (no splits, no fold);
  B = synthetic + the --fold participants' walks of the three training tasks per layout;
  C = synthetic + the same participants' walks of the test task and the two training tasks that are not c_excluded.
The other fold's participants are never loaded; their test-task walks are the model's human reference. Human validation
is leave-k-participants-out (--val_participants) from the fold's own participants (the same participant for B and C at
the same seed); synthetic validation is per episode.

B vs C: same participants, same number of episodes, and --human_share fixes the human share of every batch. They differ
only in whether the agent has seen humans walk the evaluated (test) route: A->B measures transfer, B->C memorisation.

Same trunk, inputs, augmentations and ONNX input/output names as M4; only the output size changes
(data_contract.DISCRETE_OUTPUT_SIZE), which ImitationAgent detects at load time.

Own motion (--proprio, default on; data_contract.PROPRIO_*): the 2026-09-28 fixed-task evals of the single-frame model
(ImitationAgent_A_s1) froze in all three layouts. Spawned 3.6 m in front of a wall with the goal straight behind it, the
agent stood still and swung +-6 deg around the goal bearing for 88 s: it turned one way, the goal left the view centre,
and the next frame said "turn back". The demonstrations don't do that. A standing bot keeps standing in 95.6% of the
next samples and a turning bot reverses side in only ~10%, but a single frame can't tell standing from walking or
turning left from turning right. The model therefore also gets its measured speed and recent yaw, the same values
Unity measures on the agent's body. Two standard counters to the "copycat" failure (the model just continues its own
motion and ignores the camera):
  - --proprio_dropout hides the motion input in a share of training frames (flag = 0), so vision alone stays predictive;
  - --walk_keyframe_weight / --yaw_keyframe_weight up-weight the frames where the demonstrated action differs from the
    current motion (start or stop walking; start, end or reverse a turn), the only frames where vision has to overrule
    motion.
Walk/stop then becomes a per-tick hazard (P(walk | standing) is small), so ImitationAgent samples the walk decision
instead of thresholding it at 0.5 (stochasticWalk), or the agent could never start walking again.

v2 (same evening), after the v1 models (3 motion values, dropout 0.3, one keyframe weight 3) got out of the start area
but then stood and turned in place through 180-360 deg without walking off. Held-out frames showed the walk head was
copying motion: 0.98 walk/stop accuracy with the motion input vs 0.85 without, but only 28-53% of "start walking" and
5-21% of "stop" frames right. Changes: previous speed and turn-since-last-walk added to the motion input
(data_contract.PROPRIO_*), dropout 0.5, start/stop frames weighted 6x (turn keyframes stay at 3x).
"""
import argparse
import copy
import fnmatch
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from data_contract import (PROPRIO_AVAILABLE, PROPRIO_SIZE, PROPRIO_SPEED, PROPRIO_SPEED_SCALE_MPS, PROPRIO_YAW,
                           PROPRIO_YAW_SCALE_DEG, SOURCE_HUMAN,
                           WALK_THRESHOLD_MPS, YAW_BIN_CENTERS_DEG, YAW_BIN_EDGES_DEG, YAW_CLASS_COUNT, YAW_STRAIGHT_CLASS)
from dataset import SingleEpisodeDataset, load_episodes
from make_splits import CONDITIONS, DRY_RUN_PREFIX, HUMAN_CONDITIONS, condition_humans, heldout_humans, in_condition, load_splits
from model import DiscreteImitation2DCNNModel, export_onnx

HUMAN = SOURCE_HUMAN


# --- 1. Label discretisation (mirrored by ImitationAgent.DecodeDiscreteYaw in Unity) ---
def yaw_to_class(yaw_deg: float) -> int:
    """Signed 0.5 s mean yaw (deg/sample) -> class 0..6 (strong-left .. straight .. strong-right)."""
    magnitude_bin = int(np.searchsorted(YAW_BIN_EDGES_DEG, abs(yaw_deg), side="right"))
    return YAW_STRAIGHT_CLASS + (magnitude_bin if yaw_deg > 0 else -magnitude_bin)


CLASS_CENTERS = torch.tensor(
    [np.sign(c - YAW_STRAIGHT_CLASS) * YAW_BIN_CENTERS_DEG[abs(c - YAW_STRAIGHT_CLASS)] for c in range(YAW_CLASS_COUNT)],
    dtype=torch.float32)


def decode_yaw(yaw_logits: torch.Tensor) -> torch.Tensor:
    """Most likely side (left / straight / right by summed probability), then the probability-weighted mean of that
    side's bin centers. Same rule as ImitationAgent.DecodeDiscreteYaw (without its hysteresis)."""
    p = yaw_logits.softmax(dim=1)
    s = YAW_STRAIGHT_CLASS
    left, right = p[:, :s], p[:, s + 1:]
    side_p = torch.stack([left.sum(1), p[:, s], right.sum(1)], dim=1)
    side = side_p.argmax(1)
    c = CLASS_CENTERS.to(p.device)
    left_val = (left * c[:s]).sum(1) / left.sum(1).clamp_min(1e-8)
    right_val = (right * c[s + 1:]).sum(1) / right.sum(1).clamp_min(1e-8)
    return torch.where(side == 0, left_val, torch.where(side == 2, right_val, torch.zeros_like(left_val)))


# --- 2. Dataset: continuous M4 labels -> (walk, yaw class, pitch) ---
class DiscreteLabelDataset(Dataset):
    """Wraps SingleEpisodeDataset (same inputs, flip/rotation/photometric augmentation, 0.5 s yaw mean) and bins its
    labels AFTER augmentation, so a flipped frame gets the mirrored class and a rotation-corrected label its new bin.
    Target row: [walk 0/1, yaw class, pitch deg/sample, continuous yaw deg/sample (diagnostics only)]."""

    def __init__(self, inner: Dataset):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        sensory, actions = self.inner[idx]
        forward, yaw, pitch = float(actions[0]), float(actions[1]), float(actions[2])
        target = torch.tensor([1.0 if forward > WALK_THRESHOLD_MPS else 0.0, yaw_to_class(yaw), pitch, yaw],
                              dtype=torch.float32)
        return sensory, target


# --- 3. Episode selection and participant-level split ---
def episode_source(manifest: dict) -> str:
    """'human' for human recordings (manifest source), else the synthetic participant id
    (synthetic_rrw_v3 = bot with stop behaviours, synthetic_dagger = DaggerHarness)."""
    return HUMAN if manifest["source"] == SOURCE_HUMAN else manifest["participant_id"]


def select_episodes(episodes, synthetic_sources, humans, heldout=(), condition=None):
    """-> (selected [(episode_dir, source, participant_id)], skipped {"who (why)": episode count}).
    Human episodes are kept only for participants in humans (the fold's list, never a held-out one) and only for the
    condition's tasks (make_splits.in_condition: B = training tasks, C = test task + non-excluded training tasks);
    synthetic_sources may contain 'all' = every synthetic* participant id in the manifests, and shell-style patterns
    (synthetic_dagger_run1_r*)."""
    available = {episode_source(m) for _, m in episodes if episode_source(m).startswith("synthetic")}
    wanted = set(available) if "all" in synthetic_sources else expand_sources(synthetic_sources, available)
    humans, heldout = set(humans), set(heldout)
    selected, skipped = [], {}
    for ep, manifest in episodes:
        src, pid = episode_source(manifest), manifest["participant_id"]
        task_ok = src != HUMAN or condition is None or in_condition(manifest["trial"], condition)
        if (src == HUMAN and pid in humans and task_ok) or (src != HUMAN and src in wanted):
            selected.append((ep, src, pid))
        else:
            if src == HUMAN:
                why = ("held out" if pid in heldout else "not in this fold" if pid not in humans
                       else f"task {manifest['trial']['task_id']} not in condition {condition}")
            else:
                why = "not in --synthetic_sources"
            key = f"{pid if src == HUMAN else src} ({why})"
            skipped[key] = skipped.get(key, 0) + 1
    return selected, skipped


def expand_sources(patterns, available) -> set:
    """Synthetic participant ids in `available` matched by any of `patterns` (exact ids or fnmatch patterns)."""
    return {src for src in available if any(fnmatch.fnmatchcase(src, p) for p in patterns)}


def parse_source_weights(items) -> list:
    """['synthetic_rrw_v5=0.7', 'synthetic_dagger_*=0.3'] -> [(pattern, weight)], in the given order."""
    out = []
    for item in items or []:
        pattern, sep, weight = item.rpartition("=")
        if not sep or not pattern:
            raise ValueError(f"--source_weights entry {item!r} is not PATTERN=WEIGHT")
        out.append((pattern, float(weight)))
    return out


def split_train_val(selected, val_ratio: float, val_participants: int, seed: int):
    """Human episodes: leave-k-participants-out (all episodes of val_participants participants go to validation, at
    least one participant stays in training). Synthetic: val_ratio of the episodes, per source.
    -> (train, val, validation participant ids). Deterministic given seed."""
    rng = random.Random(seed)
    train, val = [], []
    by_source = {}
    for item in selected:
        if item[1] != HUMAN:
            by_source.setdefault(item[1], []).append(item)
    for src, items in sorted(by_source.items()):
        items = sorted(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 1 else 0
        val += items[:n_val]
        train += items[n_val:]
    pids = sorted({pid for _, src, pid in selected if src == HUMAN})
    random.Random(seed).shuffle(pids)  # own stream: the choice must not depend on the synthetic episode count
    val_pids = set(pids[:min(val_participants, max(0, len(pids) - 1))])
    for item in sorted(i for i in selected if i[1] == HUMAN):
        (val if item[2] in val_pids else train).append(item)
    return train, val, sorted(val_pids)


def assert_no_heldout(episode_lists, heldout) -> None:
    """Raise if any (episode_dir, source, participant_id) in any of the lists belongs to a held-out participant."""
    heldout = set(heldout)
    leaked = sorted({(pid, Path(ep).name) for items in episode_lists for ep, src, pid in items
                     if src == HUMAN and pid in heldout})
    if leaked:
        raise RuntimeError(f"Held-out participant episodes selected for training/validation: {leaked}")


def build_dataset(episodes, augment: bool, args):
    parts, sources = [], []
    for ep, src, _ in episodes:
        kwargs = {"proprio": args.proprio}
        if augment:
            kwargs.update(rot_aug_deg=args.rot_aug_deg, rot_aug_prob=args.rot_aug_prob, rot_correction=args.rot_correction,
                          proprio_dropout=args.proprio_dropout)
        ds = SingleEpisodeDataset(ep, augment=augment, **kwargs)
        if len(ds) > 0:
            parts.append(ds)
            sources += [src] * len(ds)
    return DiscreteLabelDataset(ConcatDataset(parts)), sources


def source_balanced_sampler(sources, human_share: float) -> WeightedRandomSampler:
    """human_share of every batch (in expectation) from human frames, the rest spread over all synthetic frames.
    Only for conditions with humans (condition A shuffles plainly); both sources must have frames."""
    is_human = np.array(sources) == HUMAN
    n_h, n_s = int(is_human.sum()), int((~is_human).sum())
    if n_h == 0 or n_s == 0:
        raise ValueError(f"source-balanced sampling needs human and synthetic training frames (human {n_h}, synthetic {n_s})")
    weights = np.where(is_human, human_share / n_h, (1.0 - human_share) / n_s)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(sources), replacement=True)


def source_frame_weights(sources, human_share: float, source_weights) -> np.ndarray:
    """Per-frame sampling probability (sums to 1). Human frames share human_share. The synthetic share is spread over all
    synthetic frames, or - with source_weights [(pattern, weight)] - split between the patterns in proportion to their
    weights and spread evenly within each (every synthetic source must match a pattern; the first match counts). That
    keeps long DAgger episodes, which are mostly a stuck agent, from taking over the synthetic share by frame count."""
    src = np.array(sources)
    is_human = src == HUMAN
    n_h, n_s = int(is_human.sum()), int((~is_human).sum())
    if human_share > 0 and n_h == 0:
        raise ValueError(f"human_share {human_share} but no human training frames")
    if n_s == 0 and human_share < 1:
        raise ValueError("no synthetic training frames")
    weights = np.zeros(len(src), dtype=np.float64)
    if n_h:
        weights[is_human] = human_share / n_h
    if not source_weights:
        if n_s:
            weights[~is_human] = (1.0 - human_share) / n_s
        return weights
    group = np.full(len(src), -1)
    for i, s in enumerate(src):
        if s == HUMAN:
            continue
        for g, (pattern, _) in enumerate(source_weights):
            if fnmatch.fnmatchcase(s, pattern):
                group[i] = g
                break
        else:
            raise ValueError(f"synthetic source {s!r} matches no --source_weights pattern")
    present = [g for g in range(len(source_weights)) if (group == g).any()]
    total = sum(source_weights[g][1] for g in present)
    if total <= 0:
        raise ValueError("--source_weights of the sources present sum to 0")
    for g in present:
        mask = group == g
        weights[mask] = (1.0 - human_share) * source_weights[g][1] / total / mask.sum()
    return weights


def frame_sampler(weights: np.ndarray) -> WeightedRandomSampler:
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def git_revision() -> str:
    """HEAD of the repository this script lives in, with '-dirty' if tracked files are modified; 'unknown' without git."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def count_by(items, key) -> dict:
    counts = {}
    for item in items:
        counts[key(item)] = counts.get(key(item), 0) + 1
    return dict(sorted(counts.items()))


# --- 4. Loss ---
def label_weights_from(dataset: DiscreteLabelDataset, sources, human_share: float, power: float, target_stop_share: float):
    """Loss weights from the label frequencies under the sampling mix actually used.
    Yaw: (1/frequency)^power per class (power 0.5 = inverse sqrt), normalised to mean 1 - strong turns are 3-6% of frames
    and would otherwise be absorbed by the neighbouring bin.
    Walk: a weight on stop frames so they carry target_stop_share of the walk/stop loss (never below 1)."""
    yaw_counts = {HUMAN: np.zeros(YAW_CLASS_COUNT), "syn": np.zeros(YAW_CLASS_COUNT)}
    stop_counts = {HUMAN: 0, "syn": 0}
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    i = 0
    for _, target in loader:
        for cls, walk in zip(target[:, 1].long().tolist(), target[:, 0].tolist()):
            key = HUMAN if sources[i] == HUMAN else "syn"
            yaw_counts[key][cls] += 1
            stop_counts[key] += walk < 0.5
            i += 1
    freq = np.zeros(YAW_CLASS_COUNT)
    stop_freq = 0.0
    for key, share in ((HUMAN, human_share), ("syn", 1.0 - human_share)):
        n = yaw_counts[key].sum()
        if n > 0:
            freq += share * yaw_counts[key] / n
            stop_freq += share * stop_counts[key] / n
    total = freq.sum()
    freq, stop_freq = freq / total, stop_freq / total
    print("[LABELS] yaw class frequency under the sampling mix: " + " ".join(f"{f:.3f}" for f in freq))
    w = (1.0 / np.maximum(freq, 1e-4)) ** power
    stop_weight = 1.0
    if 0.0 < stop_freq < target_stop_share:
        stop_weight = (target_stop_share / (1.0 - target_stop_share)) * ((1.0 - stop_freq) / stop_freq)
    print(f"[LABELS] stop frequency under the sampling mix {stop_freq:.3f} -> stop weight {stop_weight:.2f} "
          f"(target share {target_stop_share:.2f})")
    return torch.tensor(w / w.mean(), dtype=torch.float32), stop_weight


def label_weights_from_frames(dataset: DiscreteLabelDataset, frame_weights: np.ndarray, power: float, target_stop_share: float):
    """label_weights_from for an arbitrary per-frame sampling mix (frame_weights = each training frame's sampling
    probability, e.g. from source_frame_weights with --source_weights). Same weighting rules."""
    freq = np.zeros(YAW_CLASS_COUNT)
    stop_freq = 0.0
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    i = 0
    for _, target in loader:
        for cls, walk in zip(target[:, 1].long().tolist(), target[:, 0].tolist()):
            freq[cls] += frame_weights[i]
            stop_freq += frame_weights[i] * (walk < 0.5)
            i += 1
    total = freq.sum()
    freq, stop_freq = freq / total, stop_freq / total
    print("[LABELS] yaw class frequency under the sampling mix: " + " ".join(f"{f:.3f}" for f in freq))
    w = (1.0 / np.maximum(freq, 1e-4)) ** power
    stop_weight = 1.0
    if 0.0 < stop_freq < target_stop_share:
        stop_weight = (target_stop_share / (1.0 - target_stop_share)) * ((1.0 - stop_freq) / stop_freq)
    print(f"[LABELS] stop frequency under the sampling mix {stop_freq:.3f} -> stop weight {stop_weight:.2f} "
          f"(target share {target_stop_share:.2f})")
    return torch.tensor(w / w.mean(), dtype=torch.float32), stop_weight


def motion_state(sensory: torch.Tensor):
    """From the proprio block at the end of sensory_inputs -> (available, walking now 0/1, turn side now -1/0/+1).
    Walking = speed above WALK_THRESHOLD_MPS, turning = |recent yaw| at least the straight bin edge."""
    motion = sensory[:, -PROPRIO_SIZE:]
    speed = motion[:, PROPRIO_SPEED] * PROPRIO_SPEED_SCALE_MPS
    yaw = motion[:, PROPRIO_YAW] * PROPRIO_YAW_SCALE_DEG
    side = torch.where(yaw.abs() >= YAW_BIN_EDGES_DEG[0], torch.sign(yaw), torch.zeros_like(yaw))
    return motion[:, PROPRIO_AVAILABLE] > 0.5, (speed > WALK_THRESHOLD_MPS).float(), side


def keyframes(sensory: torch.Tensor, target: torch.Tensor):
    """(walk keyframe, yaw keyframe) masks: the demonstrated action differs from the current motion. False where the
    motion input is hidden (dropout)."""
    available, walking, side = motion_state(sensory)
    label_side = torch.sign(target[:, 1] - YAW_STRAIGHT_CLASS)
    return available & (target[:, 0] != walking), available & (label_side != side)


def compute_losses(out, target, sensory, yaw_class_weights, pitch_std, args):
    walk_loss = F.binary_cross_entropy_with_logits(out[:, 0], target[:, 0], reduction="none")
    walk_loss = walk_loss * torch.where(target[:, 0] < 0.5, args.stop_weight, 1.0)
    yaw_loss = F.cross_entropy(out[:, 1:1 + YAW_CLASS_COUNT], target[:, 1].long(), weight=yaw_class_weights,
                               label_smoothing=args.label_smoothing, reduction="none")
    if args.proprio:
        walk_key, yaw_key = keyframes(sensory, target)
        walk_loss = walk_loss * torch.where(walk_key, args.walk_keyframe_weight, 1.0)
        yaw_loss = yaw_loss * torch.where(yaw_key, args.yaw_keyframe_weight, 1.0)
    pitch_loss = ((out[:, -1] - target[:, 2]) / pitch_std) ** 2
    return walk_loss, yaw_loss, pitch_loss


# --- 5. Held-out diagnostics ---
@torch.no_grad()
def evaluate(model, loader, device, yaw_class_weights, pitch_std, args, hide_proprio=False):
    """hide_proprio: feed every frame with the motion input marked unavailable (the dropout state), to measure how much
    the policy still gets from the camera and goal alone."""
    model.eval()
    sums = dict(walk=0.0, yaw=0.0, pitch=0.0, n=0)
    rows = []
    pixels = 64 * 64 * 4
    for sensory, target in loader:
        sensory, target = sensory.to(device), target.to(device)
        if hide_proprio:
            sensory = sensory.clone()
            sensory[:, 4 + pixels:] = 0.0
        out = model(sensory)
        wl, yl, pl = compute_losses(out, target, sensory, yaw_class_weights, pitch_std, args)
        sums["walk"] += wl.sum().item(); sums["yaw"] += yl.sum().item(); sums["pitch"] += pl.sum().item()
        sums["n"] += len(target)
        # Depth straight ahead (metres) and goal bearing, to isolate the "wall ahead, goal behind it" frames.
        vis = sensory[:, 4:4 + pixels].view(-1, 64, 64, 4)
        front_m = vis[:, 24:40, 28:36, 3].mean(dim=(1, 2)) * 50.0
        bearing = torch.rad2deg(torch.atan2(sensory[:, 0], sensory[:, 2]))
        if args.proprio:
            walk_key, yaw_key = keyframes(sensory, target)
        else:
            walk_key = yaw_key = torch.zeros_like(target[:, 0], dtype=torch.bool)
        rows.append(torch.stack([
            target[:, 0], target[:, 1], target[:, 3],
            (out[:, 0] > 0).float(), out[:, 1:1 + YAW_CLASS_COUNT].argmax(1).float(), decode_yaw(out[:, 1:1 + YAW_CLASS_COUNT]),
            front_m, bearing, walk_key.float(), yaw_key.float()], dim=1).cpu())
    n = max(1, sums["n"])
    losses = {k: sums[k] / n for k in ("walk", "yaw", "pitch")}
    losses["total"] = losses["walk"] * args.walk_loss_weight + losses["yaw"] + losses["pitch"] * args.pitch_loss_weight
    return losses, torch.cat(rows).numpy()


def report(rows: np.ndarray, tag: str) -> None:
    walk, cls, yaw, walk_hat, cls_hat, yaw_hat, front_m, bearing, walk_key, yaw_key = rows.T
    s = YAW_STRAIGHT_CLASS
    print(f"\n[DIAGNOSTIC {tag}] held-out frames: n={len(rows)}")
    print(f"  walk/stop accuracy {np.mean(walk == walk_hat):.3f} | demo stop share {1 - walk.mean():.3f} | predicted stop share {1 - walk_hat.mean():.3f}")
    print(f"  yaw class accuracy {np.mean(cls == cls_hat):.3f} | within one bin {np.mean(np.abs(cls - cls_hat) <= 1):.3f}")
    turning = np.abs(yaw) >= YAW_BIN_EDGES_DEG[1]
    if turning.sum() > 0:
        same_side = np.sign(yaw_hat[turning]) == np.sign(yaw[turning])
        print(f"  turning frames (|demo| >= {YAW_BIN_EDGES_DEG[1]} deg/sample): n={int(turning.sum())}, decoded same side {same_side.mean():.3f}, "
              f"decoded straight {np.mean(yaw_hat[turning] == 0):.3f}, |decoded|/|demo| median {np.median(np.abs(yaw_hat[turning]) / np.abs(yaw[turning])):.2f}")
    # The M4 failure case: wall close ahead, goal roughly behind it. A committed turn either way is fine; ~0 is not.
    wall = (front_m < 1.5) & (np.abs(bearing) < 20) & turning
    if wall.sum() > 0:
        print(f"  wall-ahead / goal-behind turning frames: n={int(wall.sum())}, decoded |yaw| >= {YAW_BIN_EDGES_DEG[0]}: "
              f"{np.mean(np.abs(yaw_hat[wall]) >= YAW_BIN_EDGES_DEG[0]):.3f} (M4 regression averaged these to ~0)")
    # Keyframes (--proprio): the demonstration changes what the walker is currently doing. Continuing the current motion
    # gets every other frame right, so these are where a copycat policy shows up.
    if walk_key.sum() > 0:
        starts, stops = (walk_key > 0) & (walk > 0.5), (walk_key > 0) & (walk < 0.5)
        print(f"  walk keyframes: start walking n={int(starts.sum())} predicted walk {np.mean(walk_hat[starts] > 0.5) if starts.any() else float('nan'):.3f} | "
              f"stop n={int(stops.sum())} predicted stop {np.mean(walk_hat[stops] < 0.5) if stops.any() else float('nan'):.3f}")
    if yaw_key.sum() > 0:
        k = yaw_key > 0
        demo_side = np.sign(cls[k] - s)
        print(f"  yaw keyframes (turn starts/ends/reversals): n={int(k.sum())}, predicted side = demo side "
              f"{np.mean(np.sign(yaw_hat[k]) == demo_side):.3f}")
    confusion = np.zeros((YAW_CLASS_COUNT, YAW_CLASS_COUNT), dtype=int)
    for a, b in zip(cls.astype(int), cls_hat.astype(int)):
        confusion[a, b] += 1
    print("  yaw confusion (rows demo, cols predicted; strong-left .. straight .. strong-right):")
    for c in range(YAW_CLASS_COUNT):
        print("   " + ("*" if c == s else " ") + " ".join(f"{v:6d}" for v in confusion[c]))


# --- 6. Main ---
def main():
    parser = argparse.ArgumentParser(description="2D CNN behaviour cloning with discrete walk/yaw head")
    parser.add_argument("--recordings_dir", type=str, default="../zarr_recordings")
    parser.add_argument("--condition", type=str, required=True, choices=CONDITIONS,
                        help="A = synthetic only; B = synthetic + the fold's training-task walks; C = synthetic + the fold's test-task and non-excluded training-task walks.")
    parser.add_argument("--fold", type=int, choices=(1, 2), default=None,
                        help="Training fold for B and C (1 = odd participant numbers, 2 = even). The other fold is held out.")
    parser.add_argument("--splits", type=str, default="../splits.json",
                        help="Fold x task splits written by make_splits.py (B and C only). Held-out participants are never loaded.")
    parser.add_argument("--name_tag", type=str, default="proprio2",
                        help="Model variant in the default output name ImitationAgent_<condition>_<tag>[_f<fold>]_s<seed>.onnx.")
    parser.add_argument("--synthetic_sources", type=str, nargs="+", default=["synthetic_rrw_v3"],
                        help="Synthetic participant ids to train on (synthetic_rrw_v3 = bot, synthetic_dagger* = "
                             "DaggerHarness), shell-style patterns (synthetic_dagger_run1_r*), or 'all' = every "
                             "synthetic* participant id in the manifests.")
    parser.add_argument("--source_weights", type=str, nargs="+", default=None, metavar="PATTERN=WEIGHT",
                        help="Split the synthetic share of every batch between source patterns by these weights instead "
                             "of by frame count, e.g. synthetic_rrw_v5=0.7 'synthetic_dagger_*=0.3'. Every synthetic "
                             "source must match one. Also applies to condition A.")
    parser.add_argument("--init_from", type=str, default=None,
                        help="Start from this checkpoint (.pt, written next to every exported .onnx) instead of random "
                             "weights: fine-tuning, or warm-starting a DAgger round.")
    parser.add_argument("--freeze_trunk", action="store_true",
                        help="Keep the convolutional encoder (weights and BatchNorm statistics) fixed; only FiLM and the "
                             "head train. For fine-tuning on few human episodes; needs --init_from.")
    parser.add_argument("--output_onnx", type=str, default=None,
                        help="Default ../imitation_agent_model/ImitationAgent_<condition>_<name_tag>[_f<fold>]_s<seed>.onnx.")
    parser.add_argument("--reason", type=str, default="manual", help="Why this run was started; written to the sidecar.")
    parser.add_argument("--human_share", type=float, default=0.5,
                        help="Expected share of human frames per batch (B, C; A has none). Fixed, so B and C weigh the "
                             "human data equally and differ only in how many participants it comes from.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation share of synthetic episodes, per source.")
    parser.add_argument("--val_participants", type=int, default=1,
                        help="Human validation: all episodes of this many of the condition's participants (at least "
                             "one participant always stays in training).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--class_weight_power", type=float, default=0.5,
                        help="Yaw class weight = (1/frequency)^power. 0 = unweighted, 1 = full inverse frequency.")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--walk_loss_weight", type=float, default=1.0)
    parser.add_argument("--target_stop_share", type=float, default=0.2,
                        help="Up-weight stop frames until they carry this share of the walk/stop loss. 0 disables. The "
                             "default 0.2 is the pilot's standing share; the thesis participants P01-P08 stand in 8.7%% "
                             "of samples (eval_tools/compare_bot_to_humans.py), so new runs should pass ~0.09.")
    parser.add_argument("--pitch_loss_weight", type=float, default=0.25)
    parser.add_argument("--rot_aug_deg", type=float, default=15.0)
    parser.add_argument("--rot_aug_prob", type=float, default=0.7)
    parser.add_argument("--rot_correction", type=float, default=0.15)
    parser.add_argument("--proprio", action=argparse.BooleanOptionalAction, default=True,
                        help="Append own motion to the input (data_contract.PROPRIO_*). --no-proprio = the "
                             "single-frame model evaluated on 2026-09-28.")
    parser.add_argument("--proprio_dropout", type=float, default=0.5,
                        help="Training frames whose motion input is hidden (marked unavailable). v1 used 0.3.")
    parser.add_argument("--walk_keyframe_weight", type=float, default=6.0,
                        help="Walk/stop loss weight of frames where the demonstration starts or stops walking. 1 disables. "
                             "v1 used 3.")
    parser.add_argument("--yaw_keyframe_weight", type=float, default=3.0,
                        help="Yaw loss weight of frames where the demonstrated turn side differs from the current turn. "
                             "1 disables.")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.condition in HUMAN_CONDITIONS and args.fold is None:
        parser.error(f"condition {args.condition} needs --fold 1 or --fold 2")
    if args.condition == "A" and args.fold is not None:
        parser.error("condition A uses no humans and has no fold; drop --fold")
    if args.freeze_trunk and not args.init_from:
        parser.error("--freeze_trunk needs --init_from (a frozen random trunk learns nothing)")
    try:
        source_weights = parse_source_weights(args.source_weights)
    except ValueError as e:
        parser.error(str(e))
    if args.output_onnx is None:
        fold_part = f"_f{args.fold}" if args.fold else ""
        tag_part = f"_{args.name_tag}" if args.name_tag else ""
        args.output_onnx = f"../imitation_agent_model/ImitationAgent_{args.condition}{tag_part}{fold_part}_s{args.seed}.onnx"
    print(f"[RUN] condition {args.condition}, fold {args.fold or '-'}, seed {args.seed}, reason: {args.reason}")

    episodes = load_episodes(Path(args.recordings_dir))  # manifests only; raises on any contract violation
    if args.condition in HUMAN_CONDITIONS:
        splits = load_splits(args.splits)
        humans = condition_humans(splits, args.condition, args.fold)
        heldout = heldout_humans(splits, args.condition, args.fold)
        listed = {p for pids in splits["folds"].values() for p in pids}
        print(f"[SPLITS] {args.splits}: fold {args.fold} trains on {humans}, held out {heldout} | "
              f"C leaves out {splits['c_excluded']}")
    else:  # A: no humans at all, every human is held out
        splits = None
        humans = []
        heldout = sorted({m["participant_id"] for _, m in episodes if m["source"] == SOURCE_HUMAN})
        listed = set(heldout)
    selected, skipped = select_episodes(episodes, args.synthetic_sources, humans, heldout, args.condition)
    train_eps, val_eps, val_pids = split_train_val(selected, args.val_ratio, args.val_participants, args.seed)
    assert_no_heldout([selected, train_eps, val_eps], heldout)
    train_pids = sorted({pid for _, src, pid in train_eps if src == HUMAN})

    print("[DATASET] episodes per source: " + ", ".join(f"{k}={v}" for k, v in count_by(selected, lambda e: e[1]).items()))
    if skipped:
        print("[DATASET] skipped: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
    unknown = sorted({m["participant_id"] for _, m in episodes if m["source"] == SOURCE_HUMAN
                      and m["participant_id"] not in listed and not m["participant_id"].startswith(DRY_RUN_PREFIX)})
    if unknown:
        print(f"[DATASET] WARNING: human participants in no fold of {args.splits} (never used): {unknown}")
    present = {e[1] for e in selected}
    missing = [s for s in args.synthetic_sources if s != "all" and not expand_sources([s], present)]
    if missing or not any(src != HUMAN for _, src, _ in selected):
        print(f"[ERROR] No synthetic episodes for {missing or args.synthetic_sources} in '{args.recordings_dir}'")
        sys.exit(1)
    if humans and not train_pids:
        print(f"[ERROR] Condition {args.condition}: no training episodes for humans {humans} in '{args.recordings_dir}'")
        sys.exit(1)
    missing_humans = sorted(set(humans) - {pid for _, src, pid in selected if src == HUMAN})
    if missing_humans:
        print(f"[DATASET] WARNING: condition {args.condition} participants without episodes: {missing_humans}")
    print(f"[DATASET] episodes train={len(train_eps)} val={len(val_eps)} | human train participants {train_pids or 'none'}, "
          f"human val participants {val_pids or 'none'}")
    manifest_by_dir = {str(ep): m for ep, m in episodes}
    human_tasks = count_by([e for e in selected if e[1] == HUMAN],
                           lambda e: f"{manifest_by_dir[str(e[0])]['scene_name']}:{manifest_by_dir[str(e[0])]['trial']['task_id']}")
    if human_tasks:
        print("[DATASET] human episodes per layout:task: " + ", ".join(f"{k}={v}" for k, v in sorted(human_tasks.items())))

    train_ds, train_sources = build_dataset(train_eps, True, args)
    val_ds, _ = build_dataset(val_eps, False, args)
    print(f"[DATASET] frames train={len(train_ds)} val={len(val_ds)} | human frames in train: "
          f"{sum(s == HUMAN for s in train_sources)}")

    if source_weights:  # explicit per-source shares (also for condition A)
        human_share = args.human_share if humans else 0.0
        frame_weights = source_frame_weights(train_sources, human_share, source_weights)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=frame_sampler(frame_weights))
    elif humans:
        human_share = args.human_share
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=source_balanced_sampler(train_sources, human_share))
    else:  # condition A: synthetic only, plain shuffling, no source balancing
        human_share = 0.0
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"[SAMPLER] human share per batch {human_share:.0%}")
    if source_weights:
        shares = {}
        for w, src in zip(frame_weights, train_sources):
            shares[src] = shares.get(src, 0.0) + w
        print("[SAMPLER] share per batch by source: " + ", ".join(f"{k}={v:.0%}" for k, v in sorted(shares.items())))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[HARDWARE] Training on device: {device}")

    if source_weights:
        yaw_class_weights, args.stop_weight = label_weights_from_frames(train_ds, frame_weights, args.class_weight_power,
                                                                        args.target_stop_share)
    else:
        yaw_class_weights, args.stop_weight = label_weights_from(train_ds, train_sources, human_share,
                                                                 args.class_weight_power, args.target_stop_share)
    yaw_class_weights = yaw_class_weights.to(device)
    print("[LABELS] yaw class weights: " + " ".join(f"{w:.2f}" for w in yaw_class_weights.tolist()))
    pitch_std = torch.tensor(max(1e-3, float(np.std([float(train_ds[i][1][2]) for i in
                                                     np.random.choice(len(train_ds), min(4000, len(train_ds)), replace=False)]))),
                             device=device)

    proprio_size = PROPRIO_SIZE if args.proprio else 0
    print(f"[MODEL] own-motion input: {'on' if proprio_size else 'off'}"
          + (f" ({proprio_size} values, dropout {args.proprio_dropout:.2f}, keyframe weight walk {args.walk_keyframe_weight:.1f} "
             f"/ yaw {args.yaw_keyframe_weight:.1f})" if proprio_size else ""))
    model = DiscreteImitation2DCNNModel(proprio_size=proprio_size)
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu")
        if checkpoint.get("proprio_size") != proprio_size:
            print(f"[ERROR] --init_from {args.init_from} has proprio_size {checkpoint.get('proprio_size')}, this run "
                  f"{proprio_size}")
            sys.exit(1)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"[MODEL] initialised from {args.init_from} (condition {checkpoint.get('condition')}, "
              f"sources {checkpoint.get('synthetic_sources')})")
    if args.freeze_trunk:
        for p in model.conv2d_encoder.parameters():
            p.requires_grad_(False)
        print("[MODEL] convolutional encoder frozen")
    model = model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                 weight_decay=args.weight_decay)
    min_best_epoch = max(1, args.epochs // 4)
    best = (float("inf"), -1, None)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_trunk:
            model.conv2d_encoder.eval()  # keep the pretrained BatchNorm statistics
        running, seen = 0.0, 0
        for sensory, target in train_loader:
            sensory, target = sensory.to(device), target.to(device)
            optimizer.zero_grad()
            wl, yl, pl = compute_losses(model(sensory), target, sensory, yaw_class_weights, pitch_std, args)
            loss = (args.walk_loss_weight * wl + yl + args.pitch_loss_weight * pl).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            running += loss.item() * len(target)
            seen += len(target)

        val, _ = evaluate(model, val_loader, device, yaw_class_weights, pitch_std, args)
        is_best = epoch >= min_best_epoch and val["total"] < best[0]
        if is_best:
            best = (val["total"], epoch, copy.deepcopy(model.state_dict()))
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] train {running / max(1, seen):.4f} | val total {val['total']:.4f} "
              f"(walk {val['walk']:.4f}, yaw {val['yaw']:.4f}, pitch {val['pitch']:.4f}){' (best)' if is_best else ''}")

    print(f"\n[CHECKPOINT] Using epoch {best[1]} (val total {best[0]:.4f})")
    model.load_state_dict(best[2])
    _, rows = evaluate(model, val_loader, device, yaw_class_weights, pitch_std, args)
    report(rows, "best checkpoint")
    if proprio_size:
        _, rows = evaluate(model, val_loader, device, yaw_class_weights, pitch_std, args, hide_proprio=True)
        report(rows, "best checkpoint, motion input hidden (camera + goal only)")
    model.cpu()
    export_onnx(model, args.output_onnx)
    checkpoint_path = os.path.splitext(args.output_onnx)[0] + ".pt"
    torch.save({"state_dict": model.state_dict(), "proprio_size": proprio_size, "condition": args.condition,
                "fold": args.fold, "seed": args.seed, "best_epoch": best[1],
                "synthetic_sources": sorted({e[1] for e in selected if e[1] != HUMAN}), "args": vars(args)},
               checkpoint_path)
    print(f"[EXPORT] PyTorch checkpoint written to {checkpoint_path}")

    sidecar = os.path.splitext(args.output_onnx)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump({"head": "discrete", "output_layout": "walk_logit, yaw_logit_0..6, pitch_deg_per_sample",
                   "input_layout": "goal_xyz, proximity, rgbd_64x64x4" + (", speed, prev_speed, recent_yaw, turn_since_walk, available" if proprio_size else ""),
                   "proprio_size": proprio_size,
                   "yaw_bin_edges_deg": YAW_BIN_EDGES_DEG, "yaw_bin_centers_deg": YAW_BIN_CENTERS_DEG,
                   "walk_threshold_mps": WALK_THRESHOLD_MPS,
                   "condition": args.condition, "seed": args.seed, "git_revision": git_revision(), "reason": args.reason,
                   "fold": args.fold, "splits": os.path.abspath(args.splits) if splits else None,
                   "splits_design": splits["design"] if splits else None,
                   "c_excluded": splits["c_excluded"] if splits else None,
                   "human_episodes_per_task": human_tasks, "heldout_participants": heldout,
                   "condition_participants": humans, "train_participants": train_pids, "val_participants": val_pids,
                   "human_share": human_share, "synthetic_sources": sorted({e[1] for e in selected if e[1] != HUMAN}),
                   "episode_counts": {"train": count_by(train_eps, lambda e: e[2] if e[1] == HUMAN else e[1]),
                                      "val": count_by(val_eps, lambda e: e[2] if e[1] == HUMAN else e[1])},
                   "frame_counts": {"train": len(train_ds), "val": len(val_ds),
                                    "train_human": sum(s == HUMAN for s in train_sources)},
                   "train_episodes": sorted(Path(e[0]).name for e in train_eps),
                   "val_episodes": sorted(Path(e[0]).name for e in val_eps),
                   "init_from": os.path.abspath(args.init_from) if args.init_from else None,
                   "freeze_trunk": args.freeze_trunk,
                   "source_weights": dict(source_weights) if source_weights else None,
                   "checkpoint": os.path.abspath(checkpoint_path),
                   "best_epoch": best[1], "args": vars(args)}, f, indent=2)
    print(f"[EXPORT] Contract sidecar written to {sidecar}")


if __name__ == "__main__":
    main()
