import json
import os
import sys
import tempfile

# Auto-inject python_embedded directory into sys.path so local packages are found
current_dir = os.path.dirname(os.path.abspath(__file__))
python_embedded_dir = os.path.abspath(os.path.join(current_dir, "..", "python_embedded"))
if python_embedded_dir not in sys.path:
    sys.path.insert(0, python_embedded_dir)

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import zarr


def parse_args():
    parser = argparse.ArgumentParser(description="DesignMind2 Lifecycle Engine - Dummy ML Pipeline")
    parser.add_argument("--csv_input", type=str, default=None, help="Legacy path to the recorded human CSV data")
    parser.add_argument("--episode_dir", type=str, default=None, help="Path to the recorded binary episode directory")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the Zarr dataset; if omitted it is inferred from the episode directory")
    parser.add_argument("--model_output", type=str, required=True, help="Target path for the final ONNX model")
    return parser.parse_args()


class DummyImitationModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(DummyImitationModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


def read_zarr_dataset(dataset_path: str):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Zarr dataset not found: {dataset_path}")

    root = zarr.open(dataset_path, mode="r")
    required = ["rgb", "depth", "goal", "pose", "action_vel", "action_rot", "timestamp"]
    missing = [name for name in required if name not in root]
    if missing:
        raise KeyError(f"Missing Zarr arrays: {missing}")

    rgb = root["rgb"]
    depth = root["depth"]
    goal = root["goal"]
    return rgb, depth, goal


def compute_input_dim_from_zarr(rgb, depth, goal):
    temporal_window = min(3, rgb.shape[0])
    rgb_depth_channels = 4 * rgb.shape[1] * rgb.shape[2]
    return (temporal_window * rgb_depth_channels) + goal.shape[1]


def load_legacy_csv(csv_input):
    df = pd.read_csv(csv_input)
    feature_columns = [col for col in df.columns if col.startswith('F')]
    return len(feature_columns) + 3


def main():
    args = parse_args()
    print("[Python AI Core] Initializing lifecycle loop update execution...")

    dataset_path = args.dataset_path
    if args.episode_dir is not None:
        if dataset_path is None:
            dataset_path = os.path.join(os.path.abspath(args.episode_dir), "dataset.zarr")
        print(f"[Python AI Core] Converting episode directory to Zarr at: {dataset_path}")
        from ingest_to_zarr import main as ingest_main
        import sys as _sys
        _sys.argv = ["ingest_to_zarr.py", "--episode_dir", args.episode_dir, "--output_zarr", dataset_path]
        ingest_main()
    elif args.csv_input is not None:
        dataset_path = args.csv_input
    else:
        print("[Python AI Core] ERROR: Provide either --episode_dir or --csv_input.")
        sys.exit(1)

    print(f"[Python AI Core] Reading dataset matrix from target path: {dataset_path}")

    if args.episode_dir is not None:
        try:
            rgb, depth, goal = read_zarr_dataset(dataset_path)
            input_dim = compute_input_dim_from_zarr(rgb, depth, goal)
            print(f"[Python AI Core] Loaded Zarr dataset with shapes: rgb={rgb.shape}, depth={depth.shape}, goal={goal.shape}")
            print(f"[Python AI Core] Computed runtime network input dimension: {input_dim}")
        except Exception as exc:
            print(f"[Python AI Core] Failed parsing Zarr dataset: {exc}")
            input_dim = 49155
    else:
        try:
            input_dim = load_legacy_csv(dataset_path)
            print(f"[Python AI Core] Computed runtime network input dimension: {input_dim}")
        except Exception as exc:
            print(f"[Python AI Core] Failed parsing tabular framework headers: {exc}")
            print("[Python AI Core] Falling back to default static tensor shapes for standard testing...")
            input_dim = 49155

    output_dim = 3
    model = DummyImitationModel(input_dim=input_dim, output_dim=output_dim)
    model.eval()

    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    output_dir = os.path.dirname(os.path.abspath(args.model_output))
    os.makedirs(output_dir, exist_ok=True)

    try:
        print("[Python AI Core] Compiling graph onto a secured isolated OS temp boundary...")
        temp_file_path = None
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False, suffix=".tmp") as tmp_file:
            temp_file_path = tmp_file.name

        torch.onnx.export(
            model,
            dummy_input,
            temp_file_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['sensory_inputs'],
            output_names=['motor_actions'],
        )

        if os.path.exists(args.model_output):
            os.remove(args.model_output)
        os.rename(temp_file_path, args.model_output)
        print(f"[Python AI Core] ATOMIC HOT-SWAP COMMIT SUCCESS: Saved file at -> {args.model_output}")

    except Exception as e:
        print(f"[Python AI Core] CRITICAL Error executing atomic asset compilation: {str(e)}")
        if temp_file_path is not None and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        sys.exit(2)


if __name__ == "__main__":
    main()
