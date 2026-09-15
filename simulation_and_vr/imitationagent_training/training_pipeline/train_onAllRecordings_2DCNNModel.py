import copy
import os
import random
import argparse
from pathlib import Path
import zarr
import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

# --- 1. Streamlined 2D CNN Model (Sentis Contract Preserved) ---
class SimpleImitation2DCNNModel(nn.Module):
    def __init__(self, img_res: int = 64):
        super().__init__()
        self.H = self.W = img_res

        # Lightweight 2D CNN over a single 4-channel frame (RGB + Depth)
        self.conv2d_encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(), # TODO: Try LeakyReLU()
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(), # TODO: Try LeakyReLU()
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(), # TODO: Try LeakyReLU()
        )

        # FiLM conditioning (Perez et al. 2017): the goal vector produces a per-channel
        # scale/shift applied directly to the conv feature maps, instead of only being
        # concatenated after flattening where it's just 3 of 4099 inputs to the head and easy
        # for the network to learn to give near-zero weight - which is what previously produced
        # a purely reactive wall-following policy that ignored where it was supposed to go.
        # FiLM ties every visual channel's contribution to the goal instead of leaving that
        # coupling to a single late, easily-bypassed linear layer.
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
        )
        self.film_gamma = nn.Linear(32, 64)
        self.film_beta = nn.Linear(32, 64)

        self.flatten = nn.Flatten()

        # +3 for the goal vector, concatenated back in below on top of the FiLM modulation
        # above, so the head still gets a direct, unfiltered look at the goal too.
        self.network = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 3, 64),
            nn.ReLU(), # TODO: Try LeakyReLU()
            nn.Linear(64, 3)    # Motor Outputs: [Forward, Yaw, Pitch] (Unbounded)
        )

    def forward(self, sensory_inputs: torch.Tensor) -> torch.Tensor:
        # 1. Split off the goal vector (first 3 floats) from the visual payload
        goal_vec = sensory_inputs[:, :3]  # Shape: (Batch, 3)
        vis_flat = sensory_inputs[:, 3:]  # Shape: (Batch, 49152)

        # 2. Reshape visual payload to (Batch, T=3, H=64, W=64, C=4)
        x_vis = vis_flat.view(-1, 3, self.H, self.W, 4)

        # 3. Take only the most recent single frame (T=-1) -> Shape: (Batch, 64, 64, 4)
        x_latest = x_vis[:, -1, :, :, :]

        # 4. Permute to PyTorch CNN layout -> (Batch, C=4, H=64, W=64)
        x_latest = x_latest.permute(0, 3, 1, 2)

        # 5. Pass through the 2D CNN, then use the goal vector to modulate the resulting
        #    feature maps (FiLM) before flattening, then feed the goal vector into the MLP
        #    head again alongside the modulated visual features.
        vis_map = self.conv2d_encoder(x_latest)  # (Batch, 64, 8, 8)

        goal_hidden = self.goal_encoder(goal_vec)
        gamma = self.film_gamma(goal_hidden).unsqueeze(-1).unsqueeze(-1)  # (Batch, 64, 1, 1)
        beta = self.film_beta(goal_hidden).unsqueeze(-1).unsqueeze(-1)   # (Batch, 64, 1, 1)
        vis_map = vis_map * (1.0 + gamma) + beta

        vis_features = self.flatten(vis_map)
        combined = torch.cat([vis_features, goal_vec], dim=1)
        return self.network(combined)

# --- 2. Zarr Dataset Loader ---
class SingleEpisodeDataset(Dataset):
    def __init__(self, ep_path: Path, window_size: int = 3, augment: bool = False):
        zarr_root = zarr.open(str(ep_path / "dataset.zarr"), mode='r')
        self.rgb = zarr_root['rgb']
        self.depth = zarr_root['depth']
        self.goal = zarr_root['goal']
        self.action_vel = zarr_root['action_vel']
        self.action_rot = zarr_root['action_rot']

        self.window_size = window_size
        self.augment = augment
        # action_vel[i]/action_rot[i] (see ParticipantRecorder.LockedSamplingLoop) is the motion
        # that occurred BETWEEN sample i-1 and sample i - i.e. the action that produced
        # observation i, not the action to take from it. The label for the window ending at
        # idx+w-1 is therefore the action recorded one sample later, at idx+w, which requires
        # one fewer window than before (idx+w must stay in bounds).
        self.num_frames = self.rgb.shape[0] - window_size

    def __len__(self):
        return max(0, self.num_frames)

    def __getitem__(self, idx: int):
        w = self.window_size

        # Goal vector placeholder to meet Sentis memory layout
        goal_vec = np.array(self.goal[idx + w - 1], dtype=np.float32)

        # Interleaved (RGBD) frame sequence over window
        rgb_win = np.array(self.rgb[idx:idx+w], dtype=np.float32) / 255.0  # (T, 64, 64, 3)
        depth_win = np.array(self.depth[idx:idx+w], dtype=np.float32)[..., np.newaxis]  # (T, 64, 64, 1)

        # Target Motor Actions: the action to take FROM this observation, i.e. the one recorded
        # one sample after the window (see num_frames comment above).
        forward = np.array(self.action_vel[idx + w][2], dtype=np.float32)   # z = forward/back
        yaw = np.array(self.action_rot[idx + w][0], dtype=np.float32)       # index 0 = azimuth delta = yaw
        yaw = (yaw + 180.0) % 360.0 - 180.0                                 # undo azimuth wraparound (e.g. 357.6 -> -2.4)
        pitch = np.array(self.action_rot[idx + w][1], dtype=np.float32)     # index 1 = elevation delta = pitch

        if self.augment:
            # Random horizontal (left/right) flip. goal_vec is in the agent's local frame
            # (x=right, y=up, z=forward - see ParticipantRecorder.LockedSamplingLoop's
            # InverseTransformDirection), so mirroring the image left/right requires negating
            # goal_vec's x component and the yaw action; forward, pitch, and the goal's
            # up/forward components are unaffected by a left/right mirror. On a single fixed
            # layout this is close to free extra data: it stops the CNN from memorizing which
            # specific walls/textures mean "turn left" vs "turn right".
            if random.random() < 0.5:
                rgb_win = rgb_win[:, :, ::-1, :].copy()
                depth_win = depth_win[:, :, ::-1, :].copy()
                goal_vec[0] = -goal_vec[0]
                yaw = -yaw

            # Photometric jitter on RGB only, to fight overfitting to this layout's specific
            # lighting/texture rather than the underlying navigation cue.
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(0.8, 1.2)
            rgb_win = np.clip((rgb_win - 0.5) * contrast + 0.5, 0.0, 1.0)
            rgb_win = np.clip(rgb_win * brightness, 0.0, 1.0)

        rgbd = np.concatenate([rgb_win, depth_win], axis=-1).flatten()
        sensory_inputs = np.concatenate([goal_vec, rgbd])
        motor_actions = np.array([forward, yaw, pitch], dtype=np.float32)

        return torch.from_numpy(sensory_inputs), torch.from_numpy(motor_actions)

# --- 3. Per-channel label scale (forward is m/s, yaw/pitch are degrees - very different natural scales) ---
def compute_label_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    all_actions = torch.cat([actions for _, actions in loader], dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0).clamp_min(1e-6)
    return mean, std

# --- 5. ONNX Exporter ---
def export_onnx(model: nn.Module, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 49155, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['sensory_inputs'],
        output_names=['motor_actions'],
        dynamic_axes={
            'sensory_inputs': {0: 'batch_size'},
            'motor_actions': {0: 'batch_size'}
        }
    )

    # Recent torch versions transparently split the export into a small .onnx header plus a
    # separate .onnx.data file once the weights are large enough. Unity Sentis only understands
    # a single self-contained .onnx file, so re-save with everything embedded in one file.
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)

    print(f"\n[EXPORT SUCCESS] ONNX model exported to: {output_path}")

# --- 6. Training Loop ---
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
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    recordings_path = Path(args.recordings_dir)
    episodes = [p for p in recordings_path.iterdir() if p.is_dir() and (p / "dataset.zarr").exists()]
    
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
    train_dataset = ConcatDataset([SingleEpisodeDataset(ep, augment=True) for ep in train_episodes])
    val_dataset = ConcatDataset([SingleEpisodeDataset(ep, augment=False) for ep in val_episodes])

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
    criterion = nn.MSELoss()

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
        for sensory, actions in train_loader:
            sensory, actions = sensory.to(device), actions.to(device)

            optimizer.zero_grad()
            predictions = model(sensory)
            loss = criterion((predictions - label_mean) / label_std, (actions - label_mean) / label_std)
            loss.backward()
            # Caps any single bad batch from yanking the shared trunk's weights into a regime
            # that extrapolates wildly on live (slightly out-of-distribution) camera frames.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            train_loss += loss.item() * sensory.size(0)

        epoch_train_loss = train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sensory, actions in val_loader:
                sensory, actions = sensory.to(device), actions.to(device)
                predictions = model(sensory)
                loss = criterion((predictions - label_mean) / label_std, (actions - label_mean) / label_std)
                val_loss += loss.item() * sensory.size(0)

        epoch_val_loss = val_loss / len(val_dataset)

        is_best = epoch >= min_best_epoch and epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        marker = " (best)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}{marker}")

    # Validation loss can bounce between epochs on a small dataset; blindly exporting whatever
    # the final epoch happened to land on (as before) risks shipping a worse-than-typical
    # snapshot. Export the checkpoint that actually scored best on held-out episodes instead.
    print(f"\n[CHECKPOINT] Using epoch {best_epoch} (best Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(best_state_dict)
    model.cpu()
    export_onnx(model, args.output_onnx)

if __name__ == "__main__":
    main()