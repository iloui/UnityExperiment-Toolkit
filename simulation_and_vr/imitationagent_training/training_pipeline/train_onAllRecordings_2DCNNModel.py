import os
import random
import argparse
from pathlib import Path
import zarr
import numpy as np
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
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()                                          # Output: 4096 features
        )

        self.network = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 3),   # Motor Outputs: [Forward, Yaw, Pitch]
            nn.Tanh()            # Clamps predictions to [-1.0, 1.0]
        )

    def forward(self, sensory_inputs: torch.Tensor) -> torch.Tensor:
        # 1. Ignore goal vector (first 3 floats)
        vis_flat = sensory_inputs[:, 3:]  # Shape: (Batch, 49152)

        # 2. Reshape visual payload to (Batch, T=3, H=64, W=64, C=4)
        x_vis = vis_flat.view(-1, 3, self.H, self.W, 4)

        # 3. Take only the most recent single frame (T=-1) -> Shape: (Batch, 64, 64, 4)
        x_latest = x_vis[:, -1, :, :, :]

        # 4. Permute to PyTorch CNN layout -> (Batch, C=4, H=64, W=64)
        x_latest = x_latest.permute(0, 3, 1, 2)

        # 5. Pass through 2D CNN & MLP Head
        vis_features = self.conv2d_encoder(x_latest)
        return self.network(vis_features)

# --- 2. Zarr Dataset Loader ---
class SingleEpisodeDataset(Dataset):
    def __init__(self, ep_path: Path, window_size: int = 3):
        self.rgb = zarr.open(str(ep_path / "rgb.zarr"), mode='r')
        self.depth = zarr.open(str(ep_path / "depth.zarr"), mode='r')
        self.goal = zarr.open(str(ep_path / "goal.zarr"), mode='r')
        self.action_vel = zarr.open(str(ep_path / "action_vel.zarr"), mode='r')
        self.action_rot = zarr.open(str(ep_path / "action_rot.zarr"), mode='r')

        self.window_size = window_size
        self.num_frames = len(self.rgb) - window_size + 1

    def __len__(self):
        return max(0, self.num_frames)

    def __getitem__(self, idx: int):
        w = self.window_size

        # Goal vector placeholder to meet Sentis memory layout
        goal_vec = np.array(self.goal[idx + w - 1], dtype=np.float32)

        # Interleaved (RGBD) frame sequence over window
        rgb_win = np.array(self.rgb[idx:idx+w], dtype=np.float32) / 255.0  # (T, 64, 64, 3)
        depth_win = np.array(self.depth[idx:idx+w], dtype=np.float32)[..., np.newaxis]  # (T, 64, 64, 1)

        rgbd = np.concatenate([rgb_win, depth_win], axis=-1).flatten()
        sensory_inputs = np.concatenate([goal_vec, rgbd])

        # Target Motor Actions
        forward = np.array(self.action_vel[idx + w - 1][0], dtype=np.float32)
        yaw = np.array(self.action_rot[idx + w - 1][1], dtype=np.float32)
        pitch = np.array(self.action_rot[idx + w - 1][0], dtype=np.float32)
        motor_actions = np.array([forward, yaw, pitch], dtype=np.float32)

        return torch.from_numpy(sensory_inputs), torch.from_numpy(motor_actions)

# --- 3. ONNX Exporter ---
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

# --- 4. Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Simplified 2D Behavior Cloning Pipeline")
    parser.add_argument("--recordings_dir", type=str, default="./recordings")
    parser.add_argument("--output_onnx", type=str, default="../Assets/ImitationModel/ImitationAgentModel.onnx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio (episode level)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    recordings_path = Path(args.recordings_dir)
    episodes = [p for p in recordings_path.iterdir() if p.is_dir() and (p / "rgb.zarr").exists()]
    
    if not episodes:
        print(f"[ERROR] No valid Zarr episode folders found in '{args.recordings_dir}'")
        return

    # Episode-level Train/Val Split
    random.shuffle(episodes)
    val_count = max(1, int(len(episodes) * args.val_ratio))
    val_episodes = episodes[:val_count]
    train_episodes = episodes[val_count:]

    print(f"[DATASET] Split total {len(episodes)} episodes -> Train: {len(train_episodes)}, Val: {len(val_episodes)}")

    train_dataset = ConcatDataset([SingleEpisodeDataset(ep) for ep in train_episodes])
    val_dataset = ConcatDataset([SingleEpisodeDataset(ep) for ep in val_episodes])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARDWARE] Training on device: {device}")

    model = SimpleImitation2DCNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for sensory, actions in train_loader:
            sensory, actions = sensory.to(device), actions.to(device)

            optimizer.zero_grad()
            predictions = model(sensory)
            loss = criterion(predictions, actions)
            loss.backward()
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
                loss = criterion(predictions, actions)
                val_loss += loss.item() * sensory.size(0)

        epoch_val_loss = val_loss / len(val_dataset)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    model.cpu()
    export_onnx(model, args.output_onnx)

if __name__ == "__main__":
    main()