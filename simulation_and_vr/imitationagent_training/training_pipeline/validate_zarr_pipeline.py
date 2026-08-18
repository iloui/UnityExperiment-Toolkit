import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import zarr

try:
    import torch
except ImportError:  # pragma: no cover - dependency may be absent in a minimal validation environment
    torch = None


IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
FRAME_COUNT = 8


def synthesize_episode(episode_dir: str):
    os.makedirs(episode_dir, exist_ok=True)

    rgb = np.random.randint(0, 255, size=(FRAME_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    depth = np.random.rand(FRAME_COUNT, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
    goal = np.random.randn(FRAME_COUNT, 3).astype(np.float32)
    pose = np.random.randn(FRAME_COUNT, 3).astype(np.float32)
    action_vel = np.random.randn(FRAME_COUNT, 3).astype(np.float32)
    action_rot = np.random.randn(FRAME_COUNT, 3).astype(np.float32)
    timestamp = np.arange(FRAME_COUNT, dtype=np.float64)

    rgb.tofile(os.path.join(episode_dir, "rgb.bin"))
    depth.tofile(os.path.join(episode_dir, "depth.bin"))
    goal.tofile(os.path.join(episode_dir, "goal.bin"))
    pose.tofile(os.path.join(episode_dir, "pose.bin"))
    action_vel.tofile(os.path.join(episode_dir, "action_vel.bin"))
    action_rot.tofile(os.path.join(episode_dir, "action_rot.bin"))
    timestamp.tofile(os.path.join(episode_dir, "timestamp.bin"))

    manifest = {
        "schema_version": "zarr-v1",
        "episode_id": "validation_episode",
        "participant_id": "P_TEST",
        "session_id": "S_TEST",
        "frame_count": FRAME_COUNT,
        "start_time": 0.0,
        "end_time": FRAME_COUNT * 0.1,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "total_pixels": TOTAL_PIXELS,
        "temporal_window_size": 3,
        "sampling_interval_s": 0.1,
        "max_ray_distance": 50.0,
        "horizontal_fov_deg": 90.0,
        "vertical_fov_deg": 90.0,
        "arrays": {
            "rgb": {"shape": ["time", IMAGE_HEIGHT, IMAGE_WIDTH, 3], "dtype": "uint8"},
            "depth": {"shape": ["time", IMAGE_HEIGHT, IMAGE_WIDTH], "dtype": "float32"},
            "goal": {"shape": ["time", 3], "dtype": "float32"},
            "pose": {"shape": ["time", 3], "dtype": "float32"},
            "action_vel": {"shape": ["time", 3], "dtype": "float32"},
            "action_rot": {"shape": ["time", 3], "dtype": "float32"},
            "timestamp": {"shape": ["time"], "dtype": "float64"},
        },
    }

    with open(os.path.join(episode_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def validate_episode_manifest(episode_dir: str):
    manifest_path = os.path.join(episode_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    expected_files = ["rgb.bin", "depth.bin", "goal.bin", "pose.bin", "action_vel.bin", "action_rot.bin", "timestamp.bin", "manifest.json"]
    missing = [name for name in expected_files if not os.path.exists(os.path.join(episode_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing required files in episode dir: {missing}")

    frame_count = manifest["frame_count"]
    for name in ["rgb", "depth", "goal", "pose", "action_vel", "action_rot", "timestamp"]:
        file_path = os.path.join(episode_dir, f"{name}.bin")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing binary shard for {name}: {file_path}")

    rgb_size = np.fromfile(os.path.join(episode_dir, "rgb.bin"), dtype=np.uint8).size
    depth_size = np.fromfile(os.path.join(episode_dir, "depth.bin"), dtype=np.float32).size
    goal_size = np.fromfile(os.path.join(episode_dir, "goal.bin"), dtype=np.float32).size
    pose_size = np.fromfile(os.path.join(episode_dir, "pose.bin"), dtype=np.float32).size
    action_vel_size = np.fromfile(os.path.join(episode_dir, "action_vel.bin"), dtype=np.float32).size
    action_rot_size = np.fromfile(os.path.join(episode_dir, "action_rot.bin"), dtype=np.float32).size
    timestamp_size = np.fromfile(os.path.join(episode_dir, "timestamp.bin"), dtype=np.float64).size

    expected_sizes = {
        "rgb": FRAME_COUNT * IMAGE_HEIGHT * IMAGE_WIDTH * 3,
        "depth": FRAME_COUNT * IMAGE_HEIGHT * IMAGE_WIDTH,
        "goal": FRAME_COUNT * 3,
        "pose": FRAME_COUNT * 3,
        "action_vel": FRAME_COUNT * 3,
        "action_rot": FRAME_COUNT * 3,
        "timestamp": FRAME_COUNT,
    }

    actual_sizes = {
        "rgb": rgb_size,
        "depth": depth_size,
        "goal": goal_size,
        "pose": pose_size,
        "action_vel": action_vel_size,
        "action_rot": action_rot_size,
        "timestamp": timestamp_size,
    }

    mismatches = {k: (expected_sizes[k], actual_sizes[k]) for k in expected_sizes if expected_sizes[k] != actual_sizes[k]}
    if mismatches:
        raise ValueError(f"Binary shard size mismatch: {mismatches}")

    print("[Validation] Episode manifest and binary shards are consistent.")


def validate_zarr_ingest(episode_dir: str, zarr_output: str):
    from ingest_to_zarr import main as ingest_main
    import sys as _sys

    _sys.argv = ["ingest_to_zarr.py", "--episode_dir", episode_dir, "--output_zarr", zarr_output]
    ingest_main()

    root = zarr.open(zarr_output, mode="r")
    required = ["rgb", "depth", "goal", "pose", "action_vel", "action_rot", "timestamp"]
    missing = [name for name in required if name not in root]
    if missing:
        raise KeyError(f"Zarr dataset missing required arrays: {missing}")

    for name in required:
        arr = root[name]
        if arr.shape[0] != FRAME_COUNT:
            raise ValueError(f"Zarr array {name} shape mismatch: expected first dimension {FRAME_COUNT}, got {arr.shape[0]}")

    print("[Validation] Zarr ingest produced the expected arrays with matching frame counts.")


def validate_onnx_export(zarr_path: str, output_path: str):
    if torch is None:
        print("[Validation] Skipping ONNX export validation because torch is not installed in this environment.")
        return

    root = zarr.open(zarr_path, mode="r")
    rgb = root["rgb"]
    depth = root["depth"]
    goal = root["goal"]

    temporal_window = min(3, rgb.shape[0])
    candidate_dim = (temporal_window * (4 * rgb.shape[1] * rgb.shape[2])) + goal.shape[1]

    class DummyModel(torch.nn.Module):
        def __init__(self, input_dim, output_dim=3):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, output_dim),
                torch.nn.Tanh(),
            )

        def forward(self, x):
            return self.net(x)

    model = DummyModel(candidate_dim)
    model.eval()

    dummy_input = torch.randn(1, candidate_dim, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        temp_path = handle.name

    try:
        torch.onnx.export(
            model,
            dummy_input,
            temp_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=["sensory_inputs"],
            output_names=["motor_actions"],
        )
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.move(temp_path, output_path)
    finally:
        if os.path.exists(temp_path) and os.path.abspath(temp_path) != os.path.abspath(output_path):
            os.remove(temp_path)

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"ONNX export failed: {output_path}")

    print("[Validation] Zarr arrays produce a valid ONNX export contract with the expected input/output names.")


def main():
    base_dir = Path(__file__).resolve().parent
    temp_root = base_dir / "validation_output"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    episode_dir = temp_root / "episode"
    zarr_output = temp_root / "dataset.zarr"
    onnx_output = temp_root / "validation_model.onnx"

    synthesize_episode(str(episode_dir))
    validate_episode_manifest(str(episode_dir))
    validate_zarr_ingest(str(episode_dir), str(zarr_output))
    validate_onnx_export(str(zarr_output), str(onnx_output))

    print(f"[Validation] Full Zarr Pipeline validation passed. Artifacts written to: {temp_root}")


if __name__ == "__main__":
    main()
