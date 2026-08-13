# Zarr data contract for the Unity imitation pipeline

This document fixes the target file schema before the Unity recorder is refactored away from CSV. The goal is to separate modalities cleanly and keep Python ingestion compatible with a chunked binary dataset format.

## 1. Target dataset layout

Each recorded run produces one episode directory:

```text
<dataset_root>/
  episode_<timestamp>/
    manifest.json
    rgb.zarr/
    depth.zarr/
    goal.zarr/
    pose.zarr/
    action_vel.zarr/
    action_rot.zarr/
    timestamp.zarr/
```

The canonical episode structure is deliberately not a single flat CSV file. Each array is stored separately so training can load only the required modality and avoid converting everything into strings.

## 2. Canonical array schema

- `rgb`: `uint8`, shape `(time, 64, 64, 3)`
- `depth`: `float32`, shape `(time, 64, 64)`
- `goal`: `float32`, shape `(time, 3)`
- `pose`: `float32`, shape `(time, 3)`
- `action_vel`: `float32`, shape `(time, 3)`
- `action_rot`: `float32`, shape `(time, 3)`
- `timestamp`: `float64`, shape `(time,)`

The current pipeline uses a fixed temporal window of 3 frames, therefore all modality arrays are appended in time order and are expected to be aligned by the same frame index.

## 3. Episode manifest

The `manifest.json` file is the canonical description for one recording. It contains at least:

```json
{
  "schema_version": "zarr-v1",
  "episode_id": "episode_20260813_123456",
  "participant_id": "P001",
  "session_id": "S001",
  "frame_count": 1200,
  "start_time": 0.0,
  "end_time": 120.0,
  "image_width": 64,
  "image_height": 64,
  "total_pixels": 4096,
  "temporal_window_size": 3,
  "sampling_interval_s": 0.1,
  "max_ray_distance": 50.0,
  "horizontal_fov_deg": 90.0,
  "vertical_fov_deg": 90.0,
  "arrays": {
    "rgb": { "shape": ["time", 64, 64, 3], "dtype": "uint8" },
    "depth": { "shape": ["time", 64, 64], "dtype": "float32" },
    "goal": { "shape": ["time", 3], "dtype": "float32" },
    "pose": { "shape": ["time", 3], "dtype": "float32" },
    "action_vel": { "shape": ["time", 3], "dtype": "float32" },
    "action_rot": { "shape": ["time", 3], "dtype": "float32" },
    "timestamp": { "shape": ["time"], "dtype": "float64" }
  }
}
```

## 4. Contract rules

1. `rgb` and `depth` must be aligned by frame index.
2. `goal`, `pose`, `action_vel`, and `action_rot` must be stored in the same time order as the camera frames.
3. `timestamp` is the authoritative per-frame time vector.
4. All arrays must use explicit dtypes; no implicit float/string conversions are permitted.
5. The Python ingestion script is responsible for validating lengths and throwing an error if the arrays are mismatched.

## 5. Why this contract

This format solves the current bottleneck in the CSV path:

- no string-building per row
- no pandas CSV parse cost for every training run
- natural chunked storage for large observation tensors
- direct compatibility with NumPy/Zarr/PyTorch batching

This contract is the fixed reference for the next implementation step: Unity binary recording and the Python ingestion layer.
