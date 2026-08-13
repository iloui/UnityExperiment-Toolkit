import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import zarr


DEFAULT_CHUNK_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Unity binary episode into a Zarr dataset.")
    parser.add_argument("--episode_dir", type=str, required=True, help="Directory containing the binary episode and manifest.json")
    parser.add_argument("--output_zarr", type=str, default=None, help="Target .zarr directory. Defaults to a sibling dataset.zarr in the same parent folder.")
    return parser.parse_args()


def read_manifest(episode_dir: str):
    manifest_path = os.path.join(episode_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest missing in episode directory: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _binary_file_path(episode_dir: str, name: str):
    return os.path.join(episode_dir, name)


def _frame_count_from_manifest(manifest):
    frame_count = manifest.get("frame_count")
    if frame_count is not None:
        return int(frame_count)
    raise ValueError("Manifest is missing frame_count; cannot determine dataset length.")


def _ensure_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected binary file not found: {path}")


def load_rgb(episode_dir: str, frame_count: int):
    path = _binary_file_path(episode_dir, "rgb.bin")
    _ensure_file(path)
    data = np.fromfile(path, dtype=np.uint8)
    expected = frame_count * 64 * 64 * 3
    if data.size != expected:
        raise ValueError(f"Unexpected rgb.bin size: expected {expected}, got {data.size}")
    return data.reshape(frame_count, 64, 64, 3)


def load_depth(episode_dir: str, frame_count: int):
    path = _binary_file_path(episode_dir, "depth.bin")
    _ensure_file(path)
    data = np.fromfile(path, dtype=np.float32)
    expected = frame_count * 64 * 64
    if data.size != expected:
        raise ValueError(f"Unexpected depth.bin size: expected {expected}, got {data.size}")
    return data.reshape(frame_count, 64, 64).astype(np.float32)


def load_vector_file(path: str, frame_count: int, vector_dim: int):
    _ensure_file(path)
    data = np.fromfile(path, dtype=np.float32)
    expected = frame_count * vector_dim
    if data.size != expected:
        raise ValueError(f"Unexpected data size for {path}: expected {expected}, got {data.size}")
    return data.reshape(frame_count, vector_dim).astype(np.float32)


def load_timestamp(episode_dir: str, frame_count: int):
    path = _binary_file_path(episode_dir, "timestamp.bin")
    _ensure_file(path)
    data = np.fromfile(path, dtype=np.float64)
    expected = frame_count
    if data.size != expected:
        raise ValueError(f"Unexpected timestamp.bin size: expected {expected}, got {data.size}")
    return data.astype(np.float64)


def create_dataset(output_zarr: str, manifest: dict):
    output_path = Path(output_zarr)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs["manifest"] = json.dumps(manifest)
    return group


def write_array(group, name: str, data: np.ndarray):
    chunks = tuple(max(1, min(DEFAULT_CHUNK_SIZE, s)) for s in data.shape)
    if data.ndim == 1:
        chunks = (max(1, min(DEFAULT_CHUNK_SIZE, data.shape[0])),)
    ds = group.create_dataset(name, data=data, chunks=chunks, overwrite=True)
    return ds


def main():
    args = parse_args()
    episode_dir = os.path.abspath(args.episode_dir)
    output_zarr = args.output_zarr
    if output_zarr is None:
        output_zarr = os.path.join(os.path.dirname(episode_dir), "dataset.zarr")

    manifest = read_manifest(episode_dir)
    frame_count = _frame_count_from_manifest(manifest)

    rgb = load_rgb(episode_dir, frame_count)
    depth = load_depth(episode_dir, frame_count)
    goal = load_vector_file(_binary_file_path(episode_dir, "goal.bin"), frame_count, 3)
    pose = load_vector_file(_binary_file_path(episode_dir, "pose.bin"), frame_count, 3)
    action_vel = load_vector_file(_binary_file_path(episode_dir, "action_vel.bin"), frame_count, 3)
    action_rot = load_vector_file(_binary_file_path(episode_dir, "action_rot.bin"), frame_count, 3)
    timestamp = load_timestamp(episode_dir, frame_count)

    group = create_dataset(output_zarr, manifest)
    write_array(group, "rgb", rgb)
    write_array(group, "depth", depth)
    write_array(group, "goal", goal)
    write_array(group, "pose", pose)
    write_array(group, "action_vel", action_vel)
    write_array(group, "action_rot", action_rot)
    write_array(group, "timestamp", timestamp)

    print(f"[Zarr Ingest] Saved converted dataset to: {output_zarr}")
    print(f"[Zarr Ingest] Frames ingested: {frame_count}")


if __name__ == "__main__":
    main()
