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

Experiment conditions (--condition, participant lists from splits.json, see make_splits.py):
  A = synthetic only, B = synthetic + subset_B humans, C = synthetic + every training human.
Held-out participants are never loaded. Human validation is leave-k-participants-out (--val_participants) from the
condition's own humans; synthetic validation is per episode.

B vs C: --human_share fixes the human share of every batch, whatever the number of human frames. B and C therefore give
the human data the same total weight; C spreads it over more participants (and more distinct frames, each drawn less
often). The B/C comparison measures human diversity, not how much the model is pushed towards human behaviour.

Same trunk, inputs, augmentations and ONNX input/output names as M4; only the output size changes
(data_contract.DISCRETE_OUTPUT_SIZE), which ImitationAgent detects at load time.
"""
import argparse
import copy
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

from data_contract import (SOURCE_HUMAN, WALK_THRESHOLD_MPS, YAW_BIN_CENTERS_DEG, YAW_BIN_EDGES_DEG, YAW_CLASS_COUNT,
                           YAW_STRAIGHT_CLASS)
from dataset import SingleEpisodeDataset, load_episodes
from make_splits import CONDITIONS, DRY_RUN_PREFIX, condition_humans, load_splits
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


def select_episodes(episodes, synthetic_sources, humans, heldout=()):
    """-> (selected [(episode_dir, source, participant_id)], skipped {"who (why)": episode count}).
    Human episodes are kept only for participants in humans (the condition's list, never a held-out one);
    synthetic_sources may contain 'all' = every synthetic* participant id in the manifests."""
    wanted = set(synthetic_sources)
    if "all" in wanted:
        wanted = {episode_source(m) for _, m in episodes if episode_source(m).startswith("synthetic")}
    humans, heldout = set(humans), set(heldout)
    selected, skipped = [], {}
    for ep, manifest in episodes:
        src, pid = episode_source(manifest), manifest["participant_id"]
        if (src == HUMAN and pid in humans) or (src != HUMAN and src in wanted):
            selected.append((ep, src, pid))
        else:
            why = "held out" if pid in heldout else "not in this condition" if src == HUMAN else "not in --synthetic_sources"
            key = f"{pid if src == HUMAN else src} ({why})"
            skipped[key] = skipped.get(key, 0) + 1
    return selected, skipped


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
        kwargs = {}
        if augment:
            kwargs.update(rot_aug_deg=args.rot_aug_deg, rot_aug_prob=args.rot_aug_prob, rot_correction=args.rot_correction)
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


def compute_losses(out, target, yaw_class_weights, pitch_std, args):
    walk_loss = F.binary_cross_entropy_with_logits(out[:, 0], target[:, 0], reduction="none")
    walk_loss = walk_loss * torch.where(target[:, 0] < 0.5, args.stop_weight, 1.0)
    yaw_loss = F.cross_entropy(out[:, 1:1 + YAW_CLASS_COUNT], target[:, 1].long(), weight=yaw_class_weights,
                               label_smoothing=args.label_smoothing, reduction="none")
    pitch_loss = ((out[:, -1] - target[:, 2]) / pitch_std) ** 2
    return walk_loss, yaw_loss, pitch_loss


# --- 5. Held-out diagnostics ---
@torch.no_grad()
def evaluate(model, loader, device, yaw_class_weights, pitch_std, args):
    model.eval()
    sums = dict(walk=0.0, yaw=0.0, pitch=0.0, n=0)
    rows = []
    for sensory, target in loader:
        sensory, target = sensory.to(device), target.to(device)
        out = model(sensory)
        wl, yl, pl = compute_losses(out, target, yaw_class_weights, pitch_std, args)
        sums["walk"] += wl.sum().item(); sums["yaw"] += yl.sum().item(); sums["pitch"] += pl.sum().item()
        sums["n"] += len(target)
        # Depth straight ahead (metres) and goal bearing, to isolate the "wall ahead, goal behind it" frames.
        vis = sensory[:, 4:].view(-1, 64, 64, 4)
        front_m = vis[:, 24:40, 28:36, 3].mean(dim=(1, 2)) * 50.0
        bearing = torch.rad2deg(torch.atan2(sensory[:, 0], sensory[:, 2]))
        rows.append(torch.stack([
            target[:, 0], target[:, 1], target[:, 3],
            (out[:, 0] > 0).float(), out[:, 1:1 + YAW_CLASS_COUNT].argmax(1).float(), decode_yaw(out[:, 1:1 + YAW_CLASS_COUNT]),
            front_m, bearing], dim=1).cpu())
    n = max(1, sums["n"])
    losses = {k: sums[k] / n for k in ("walk", "yaw", "pitch")}
    losses["total"] = losses["walk"] * args.walk_loss_weight + losses["yaw"] + losses["pitch"] * args.pitch_loss_weight
    return losses, torch.cat(rows).numpy()


def report(rows: np.ndarray, tag: str) -> None:
    walk, cls, yaw, walk_hat, cls_hat, yaw_hat, front_m, bearing = rows.T
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
                        help="A = synthetic only, B = synthetic + subset_B humans, C = synthetic + every training human.")
    parser.add_argument("--splits", type=str, default="../splits.json",
                        help="Participant splits written by make_splits.py. Held-out participants are never loaded.")
    parser.add_argument("--synthetic_sources", type=str, nargs="+", default=["synthetic_rrw_v3"],
                        help="Synthetic participant ids to train on (synthetic_rrw_v3 = bot, synthetic_dagger = "
                             "DaggerHarness), or 'all' = every synthetic* participant id in the manifests.")
    parser.add_argument("--output_onnx", type=str, default=None,
                        help="Default ../imitation_agent_model/ImitationAgent_<condition>_s<seed>.onnx.")
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
                        help="Up-weight stop frames until they carry this share of the walk/stop loss (humans stand in "
                             "~20%% of samples, v2 synthetic in 4%%). 0 disables.")
    parser.add_argument("--pitch_loss_weight", type=float, default=0.25)
    parser.add_argument("--rot_aug_deg", type=float, default=15.0)
    parser.add_argument("--rot_aug_prob", type=float, default=0.7)
    parser.add_argument("--rot_correction", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_onnx is None:
        args.output_onnx = f"../imitation_agent_model/ImitationAgent_{args.condition}_s{args.seed}.onnx"
    print(f"[RUN] condition {args.condition}, seed {args.seed}, reason: {args.reason}")
    splits = load_splits(args.splits)
    humans = condition_humans(splits, args.condition)
    print(f"[SPLITS] {args.splits}: held out {splits['heldout']} | condition {args.condition} humans {humans or 'none'}")

    episodes = load_episodes(Path(args.recordings_dir))  # manifests only; raises on any contract violation
    selected, skipped = select_episodes(episodes, args.synthetic_sources, humans, splits["heldout"])
    train_eps, val_eps, val_pids = split_train_val(selected, args.val_ratio, args.val_participants, args.seed)
    assert_no_heldout([selected, train_eps, val_eps], splits["heldout"])
    train_pids = sorted({pid for _, src, pid in train_eps if src == HUMAN})

    print("[DATASET] episodes per source: " + ", ".join(f"{k}={v}" for k, v in count_by(selected, lambda e: e[1]).items()))
    if skipped:
        print("[DATASET] skipped: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
    listed = set(splits["heldout"]) | set(splits["train"])
    unknown = sorted({m["participant_id"] for _, m in episodes if m["source"] == SOURCE_HUMAN
                      and m["participant_id"] not in listed and not m["participant_id"].startswith(DRY_RUN_PREFIX)})
    if unknown:
        print(f"[DATASET] WARNING: human participants not in {args.splits} (never used): {unknown}")
    missing = [s for s in args.synthetic_sources if s != "all" and s not in {e[1] for e in selected}]
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

    train_ds, train_sources = build_dataset(train_eps, True, args)
    val_ds, _ = build_dataset(val_eps, False, args)
    print(f"[DATASET] frames train={len(train_ds)} val={len(val_ds)} | human frames in train: "
          f"{sum(s == HUMAN for s in train_sources)}")

    if humans:
        human_share = args.human_share
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                  sampler=source_balanced_sampler(train_sources, human_share))
    else:  # condition A: synthetic only, plain shuffling, no source balancing
        human_share = 0.0
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f"[SAMPLER] human share per batch {human_share:.0%}")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[HARDWARE] Training on device: {device}")

    yaw_class_weights, args.stop_weight = label_weights_from(train_ds, train_sources, human_share,
                                                             args.class_weight_power, args.target_stop_share)
    yaw_class_weights = yaw_class_weights.to(device)
    print("[LABELS] yaw class weights: " + " ".join(f"{w:.2f}" for w in yaw_class_weights.tolist()))
    pitch_std = torch.tensor(max(1e-3, float(np.std([float(train_ds[i][1][2]) for i in
                                                     np.random.choice(len(train_ds), min(4000, len(train_ds)), replace=False)]))),
                             device=device)

    model = DiscreteImitation2DCNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    min_best_epoch = max(1, args.epochs // 4)
    best = (float("inf"), -1, None)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for sensory, target in train_loader:
            sensory, target = sensory.to(device), target.to(device)
            optimizer.zero_grad()
            wl, yl, pl = compute_losses(model(sensory), target, yaw_class_weights, pitch_std, args)
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
    model.cpu()
    export_onnx(model, args.output_onnx)

    sidecar = os.path.splitext(args.output_onnx)[0] + ".json"
    with open(sidecar, "w") as f:
        json.dump({"head": "discrete", "output_layout": "walk_logit, yaw_logit_0..6, pitch_deg_per_sample",
                   "yaw_bin_edges_deg": YAW_BIN_EDGES_DEG, "yaw_bin_centers_deg": YAW_BIN_CENTERS_DEG,
                   "walk_threshold_mps": WALK_THRESHOLD_MPS,
                   "condition": args.condition, "seed": args.seed, "git_revision": git_revision(), "reason": args.reason,
                   "splits": os.path.abspath(args.splits), "heldout_participants": splits["heldout"],
                   "condition_participants": humans, "train_participants": train_pids, "val_participants": val_pids,
                   "human_share": human_share, "synthetic_sources": sorted({e[1] for e in selected if e[1] != HUMAN}),
                   "episode_counts": {"train": count_by(train_eps, lambda e: e[2] if e[1] == HUMAN else e[1]),
                                      "val": count_by(val_eps, lambda e: e[2] if e[1] == HUMAN else e[1])},
                   "frame_counts": {"train": len(train_ds), "val": len(val_ds),
                                    "train_human": sum(s == HUMAN for s in train_sources)},
                   "train_episodes": sorted(Path(e[0]).name for e in train_eps),
                   "val_episodes": sorted(Path(e[0]).name for e in val_eps),
                   "best_epoch": best[1], "args": vars(args)}, f, indent=2)
    print(f"[EXPORT] Contract sidecar written to {sidecar}")


if __name__ == "__main__":
    main()
