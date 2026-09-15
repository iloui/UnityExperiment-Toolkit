import os
import glob
import argparse
from pathlib import Path
import zarr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

# --- 1. PyTorch Architecture matching Unity Sentis Contract ---
class Imitation3DCNNModel(nn.Module):
    def __init__(self, temporal_window: int = 3, img_res: int = 64):
        super().__init__()
        self.T = temporal_window
        self.H = self.W = img_res

        # Conv3D encoder over (Batch, Channels=4, Time=3, Height=64, Width=64)
        self.conv3d_encoder = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU()
        )

        self.network = nn.Sequential(
            nn.Linear(8192 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Outputs: [Forward, Yaw, Pitch]
            nn.Tanh()           # Clamps predictions to [-1.0, 1.0]
        )

    def forward(self, sensory_inputs: torch.Tensor) -> torch.Tensor:
        # Split sensory_inputs (Shape: [Batch, 49155])
        goal_vec = sensory_inputs[:, :3]
        vis_flat = sensory_inputs[:, 3:]

        # Reshape interleaved pixels into tensor shape (Batch, C=4, T=3, H=64, W=64)
        x_vis = vis_flat.view(-1, self.T, self.H, self.W, 4)
        x_vis = x_vis.permute(0, 4, 1, 2, 3)

        vis_features = self.conv3d_encoder(x_vis)
        goal_features = self.goal_encoder(goal_vec)

        fused = torch.cat([vis_features, goal_features], dim=1)
        return self.network(fused)

# --- 2. Single Episode Zarr Dataset ---
class SingleEpisodeDataset(Dataset):
    def __init__(self, ep_path: Path, window_size: int = 3):
        self.rgb = zarr.open(str(ep_path / "rgb.zarr"), mode='r')
        self.depth = zarr.open(str(ep_path / "depth.zarr"), mode='r')
        self.goal = zarr.open(str(ep_path / "goal.zarr"), mode='r')
        self.action_vel = zarr.open(str(ep_path / "action_vel.zarr"), mode='r')
        self.action_rot = zarr.open(str(ep_path / "action_rot.zarr"), mode='r')

        self.window_size = window_size
        # action_vel[i]/action_rot[i] (see ParticipantRecorder.LockedSamplingLoop) is the motion
        # that occurred BETWEEN sample i-1 and sample i - i.e. the action that produced
        # observation i, not the action to take from it. The label for the window ending at
        # idx+w-1 is therefore the action recorded one sample later, at idx+w, which requires
        # one fewer window than before (idx+w must stay in bounds).
        self.num_frames = len(self.rgb) - window_size

    def __len__(self):
        return max(0, self.num_frames)

    def __getitem__(self, idx: int):
        w = self.window_size

        # Goal at window end frame (3 values)
        goal_vec = np.array(self.goal[idx + w - 1], dtype=np.float32)

        # Build interleaved (RGBD) frame sequence over temporal window
        rgb_win = np.array(self.rgb[idx:idx+w], dtype=np.float32) / 255.0  # (T, 64, 64, 3)
        depth_win = np.array(self.depth[idx:idx+w], dtype=np.float32)[..., np.newaxis]  # (T, 64, 64, 1)

        # Concatenate along channel dimension per pixel -> (T, 64, 64, 4)
        rgbd = np.concatenate([rgb_win, depth_win], axis=-1).flatten()

        # Pack full 49,155 array
        sensory_inputs = np.concatenate([goal_vec, rgbd])

        # Target Motor Actions: the action to take FROM this observation, i.e. the one recorded
        # one sample after the window (see num_frames comment above).
        forward = np.array(self.action_vel[idx + w][2], dtype=np.float32)  # z = forward/back
        yaw = np.array(self.action_rot[idx + w][0], dtype=np.float32)      # index 0 = azimuth delta = yaw
        yaw = (yaw + 180.0) % 360.0 - 180.0                                # undo azimuth wraparound (e.g. 357.6 -> -2.4)
        pitch = np.array(self.action_rot[idx + w][1], dtype=np.float32)    # index 1 = elevation delta = pitch
        motor_actions = np.array([forward, yaw, pitch], dtype=np.float32)

        return torch.from_numpy(sensory_inputs), torch.from_numpy(motor_actions)

# --- 3. Per-channel label scale (forward is m/s, yaw/pitch are degrees - very different natural scales) ---
def compute_label_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    all_actions = torch.cat([actions for _, actions in loader], dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0).clamp_min(1e-6)
    return mean, std

# --- 4. ONNX Export Function ---
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
    print(f"\n[EXPORT SUCCESS] ONNX model exported to: {output_path}")

# --- 5. Main Training Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="Full Dataset Offline Training")
    parser.add_argument("--recordings_dir", type=str, default="./recordings", help="Directory containing all episode folders")
    parser.add_argument("--output_onnx", type=str, default="../Assets/ImitationModel/ImitationAgentModel.onnx", help="Target output ONNX path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    # Discover all episode directories
    recordings_path = Path(args.recordings_dir)
    episodes = [p for p in recordings_path.iterdir() if p.is_dir() and (p / "rgb.zarr").exists()]
    
    if not episodes:
        print(f"[ERROR] No valid Zarr episode folders found in '{args.recordings_dir}'")
        return

    print(f"[DATASET] Aggregating {len(episodes)} recorded episodes for full training...")
    datasets = [SingleEpisodeDataset(ep) for ep in episodes]
    full_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Initialize Model, Optimizer, Loss
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
    label_mean, label_std = compute_label_stats(full_dataset)
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    print(f"[LABEL STATS] mean(forward,yaw,pitch)={label_mean.tolist()} std={label_std.tolist()}")

    model = Imitation3DCNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training Loop
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for sensory, actions in dataloader:
            sensory, actions = sensory.to(device), actions.to(device)

            optimizer.zero_grad()
            predictions = model(sensory)
            loss = criterion((predictions - label_mean) / label_std, (actions - label_mean) / label_std)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sensory.size(0)

        epoch_loss = total_loss / len(full_dataset)
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {epoch_loss:.6f}")

    # Export ONNX model to unity-readable destination
    model.cpu()
    export_onnx(model, args.output_onnx)

if __name__ == "__main__":
    main()