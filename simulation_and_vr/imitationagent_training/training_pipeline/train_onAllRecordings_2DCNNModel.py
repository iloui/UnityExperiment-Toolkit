"""Legacy M4 trainer (continuous MSE head). Superseded by train_2DCNN_discrete.py; model and dataset live in
model.py / dataset.py."""
import copy
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from dataset import SingleEpisodeDataset, load_episodes
from model import SimpleImitation2DCNNModel, export_onnx

# --- 1. Per-channel label scale (forward is m/s, yaw/pitch are degrees - very different natural scales) ---
def compute_label_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    all_actions = torch.cat([actions for _, actions in loader], dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0).clamp_min(1e-6)
    return mean, std

# --- 2. Open-loop diagnostics on held-out episodes ---
@torch.no_grad()
def report_turn_response(model: nn.Module, dataset: Dataset, device: torch.device, tag: str) -> None:
    """Does the model actually turn towards an off-axis goal? Plain val MSE can't tell: with bursty human
    yaw the MSE-optimal output is a weak, near-constant turn, which is exactly what pins the agent against
    a wall in closed loop. Prints predicted vs demonstrated yaw per goal-bearing bin and a global
    undershoot ratio, on clean (non-augmented) held-out frames."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    bearings, preds, targets = [], [], []
    for sensory, actions in loader:
        out = model(sensory.to(device)).cpu()
        g = sensory[:, :3]
        bearings.append(torch.rad2deg(torch.atan2(g[:, 0], g[:, 2])))
        preds.append(out)
        targets.append(actions)
    bearing = torch.cat(bearings).numpy()
    pred = torch.cat(preds).numpy()
    tgt = torch.cat(targets).numpy()

    print(f"\n[DIAGNOSTIC {tag}] yaw (deg/sample) vs goal bearing, held-out frames")
    print("   bearing bin      n   demo yaw  pred yaw   pred/demo")
    edges = [-180, -90, -60, -30, -10, 10, 30, 60, 90, 180]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (bearing >= a) & (bearing < b)
        if m.sum() < 5:
            continue
        d, p_ = tgt[m, 1].mean(), pred[m, 1].mean()
        ratio = p_ / d if abs(d) > 0.05 else float("nan")
        print(f"  [{a:4d},{b:4d})  {int(m.sum()):5d}  {d:8.2f}  {p_:8.2f}   {ratio:8.2f}")
    turning = np.abs(tgt[:, 1]) > 3.0
    if turning.sum() > 5:
        slope = np.polyfit(tgt[turning, 1], pred[turning, 1], 1)[0]
        print(f"  turning frames (|demo yaw| > 3): n={int(turning.sum())}, slope of predicted vs demonstrated yaw = {slope:.2f} (1.0 = full response)")
    print(f"  forward: demo mean {tgt[:, 0].mean():.2f} | pred mean {pred[:, 0].mean():.2f} m/s")


# --- 3. Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Simplified 2D Behavior Cloning Pipeline")
    parser.add_argument("--recordings_dir", type=str, default="./recordings")
    parser.add_argument("--output_onnx", type=str, default="../Assets/ImitationModel/ImitationAgentModel.onnx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio (episode level)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turn_weight_alpha", type=float, default=2.0,
                         help="Per-sample train loss weight = 1 + alpha*|yaw_normalized|. Large "
                         "reorientation frames are only ~7%% of recorded frames, so plain per-frame "
                         "MSE averages them away and the model learns a weak, near-constant turn "
                         "response regardless of how much turning is actually needed (open-loop "
                         "eval showed a ~0.43 median undershoot ratio and flat yaw output from 10 "
                         "to 180 degrees of required turn). Upweighting by demonstrated |yaw| forces "
                         "the rare hard-turn frames to actually move the gradient. 0 disables it.")
    parser.add_argument("--rot_aug_deg", type=float, default=15.0,
                         help="Max simulated heading error (degrees) for the corrective rotation augmentation. 0 disables.")
    parser.add_argument("--rot_aug_prob", type=float, default=0.7, help="Probability of applying it to a training frame.")
    parser.add_argument("--rot_correction", type=float, default=0.15,
                         help="Yaw-label correction, deg/sample per degree of simulated heading error.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    recordings_path = Path(args.recordings_dir)
    episodes = [ep for ep, _ in load_episodes(recordings_path)]

    if not episodes:
        print(f"[ERROR] No valid Zarr episode folders found in '{args.recordings_dir}'")
        return

    # Episode-level Train/Val Split
    random.shuffle(episodes)
    val_count = max(1, int(len(episodes) * args.val_ratio))
    val_episodes = episodes[:val_count]
    train_episodes = episodes[val_count:]

    print(f"[DATASET] Split total {len(episodes)} episodes -> Train: {len(train_episodes)}, Val: {len(val_episodes)}")

    # Augment only the training split - val must stay a clean, deterministic measurement of
    # generalization to unseen (real, non-mirrored/non-jittered) episodes.
    train_dataset = ConcatDataset([
        SingleEpisodeDataset(ep, augment=True, rot_aug_deg=args.rot_aug_deg, rot_aug_prob=args.rot_aug_prob,
                             rot_correction=args.rot_correction) for ep in train_episodes])
    val_dataset = ConcatDataset([SingleEpisodeDataset(ep, augment=False) for ep in val_episodes])
    print(f"[TRAINING] rot_aug_deg={args.rot_aug_deg} rot_aug_prob={args.rot_aug_prob} "
          f"rot_correction={args.rot_correction} turn_weight_alpha={args.turn_weight_alpha}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[HARDWARE] Training on device: {device}")

    # forward (m/s) and yaw/pitch (degrees) live on very different natural scales; without
    # normalizing them the shared trunk's loss is dominated by whichever channel has the
    # largest variance (yaw), starving the others of gradient signal. Normalize only for the
    # loss - the model itself keeps predicting real-world units, so nothing downstream changes.
    label_mean, label_std = compute_label_stats(train_dataset)
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    print(f"[LABEL STATS] mean(forward,yaw,pitch)={label_mean.tolist()} std={label_std.tolist()}")

    model = SimpleImitation2DCNNModel().to(device)
    # weight_decay=1e-4 and grad_clip_norm=1.0 were previously tuned to curb erratic yaw/pitch
    # extrapolation in live Unity testing, but open-loop (teacher-forcing) evaluation against
    # the model's own training episodes showed that clipping the *global* gradient norm to 1.0
    # across all ~287k shared-trunk parameters throttles every update so hard that the easy
    # channel (forward) still converges in 20 epochs while yaw/pitch - which need a harder,
    # noisier mapping learned from the shared trunk - never get there: predicted yaw/pitch
    # variance collapsed to ~20-30% of the recorded variance (near mean prediction) even on
    # data the model was trained on. That's underfitting, not the runaway extrapolation this
    # was guarding against - min_best_epoch below already rules out shipping an early unstable
    # checkpoint, so the clip/decay no longer need to carry that job alone. Loosened to let the
    # harder channels actually fit; still non-zero as a safety net against a single bad batch.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # reduction="none" so the training loop can apply a per-sample turn_weight before averaging;
    # the validation loop below still collapses this to the same plain mean MSE as before, so
    # checkpoint selection (best_val_loss) stays an unweighted, apples-to-apples measurement.
    criterion = nn.MSELoss(reduction="none")
    YAW_CHANNEL = 1

    # BatchNorm's running stats (momentum 0.1) and each output channel's scale are still
    # settling in the first few epochs, so an early epoch can score deceptively well on the
    # small (4-episode) val split by luck without actually being calibrated yet - e.g. it
    # picked epoch 4/40 once and shipped a model whose forward output was 4-8x real walking
    # speed. Don't let the checkpoint search consider epochs before the model's had a real
    # chance to settle.
    min_best_epoch = max(1, args.epochs // 4)
    print(f"[CHECKPOINT] Ignoring epochs before {min_best_epoch} when selecting the best model")

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        weight_sum = 0.0
        for sensory, actions in train_loader:
            sensory, actions = sensory.to(device), actions.to(device)

            optimizer.zero_grad()
            predictions = model(sensory)
            norm_pred = (predictions - label_mean) / label_std
            norm_actions = (actions - label_mean) / label_std
            per_sample_loss = criterion(norm_pred, norm_actions).mean(dim=1)  # (B,)
            # Rare hard-turn frames (~7% of frames, per open-loop analysis) get averaged away
            # under plain per-frame MSE, so the model learns a weak, near-constant turn response
            # regardless of how much turning is actually needed. Weight each frame's loss by how
            # large the demonstrated yaw actually was, so those frames pull the gradient harder.
            turn_weight = 1.0 + args.turn_weight_alpha * norm_actions[:, YAW_CHANNEL].abs()
            loss = (per_sample_loss * turn_weight).mean()
            loss.backward()
            # Caps any single bad batch from yanking the shared trunk's weights into a regime
            # that extrapolates wildly on live (slightly out-of-distribution) camera frames.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            train_loss += loss.item() * sensory.size(0)
            weight_sum += turn_weight.sum().item()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_mean_turn_weight = weight_sum / len(train_dataset)

        # Validation Phase - plain unweighted mean MSE, so it stays a fair, consistent metric
        # for checkpoint selection regardless of how turn_weight_alpha is tuned.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sensory, actions in val_loader:
                sensory, actions = sensory.to(device), actions.to(device)
                predictions = model(sensory)
                loss = criterion((predictions - label_mean) / label_std, (actions - label_mean) / label_std).mean()
                val_loss += loss.item() * sensory.size(0)

        epoch_val_loss = val_loss / len(val_dataset)

        is_best = epoch >= min_best_epoch and epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        marker = " (best)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} "
              f"| Mean Turn Weight: {epoch_mean_turn_weight:.3f}{marker}")

    # Validation loss can bounce between epochs on a small dataset; blindly exporting whatever
    # the final epoch happened to land on (as before) risks shipping a worse-than-typical
    # snapshot. Export the checkpoint that actually scored best on held-out episodes instead.
    print(f"\n[CHECKPOINT] Using epoch {best_epoch} (best Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(best_state_dict)
    report_turn_response(model, val_dataset, device, "best checkpoint")
    model.cpu()
    export_onnx(model, args.output_onnx)

if __name__ == "__main__":
    main()