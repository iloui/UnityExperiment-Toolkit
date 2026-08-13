"""Canonical Zarr data contract for the Unity imitation dataset.

This module intentionally defines the target schema before any writing code is
implemented. The contract is shared by the Python training pipeline and the C#
Unity data recorder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_VERSION = "zarr-v1"
MANIFEST_FILENAME = "manifest.json"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

# The current imitation pipeline always reuses a temporal window of 3 frames.
TEMPORAL_WINDOW_SIZE = 3

# Unity captures RGB at 64x64x3 and depth as a single normalized scalar per pixel.
RGB_SHAPE = ("time", IMAGE_HEIGHT, IMAGE_WIDTH, 3)
DEPTH_SHAPE = ("time", IMAGE_HEIGHT, IMAGE_WIDTH)
GOAL_SHAPE = ("time", 3)
POSE_SHAPE = ("time", 3)
ACTION_VEL_SHAPE = ("time", 3)
ACTION_ROT_SHAPE = ("time", 3)
TIMESTAMP_SHAPE = ("time",)

# Storage types are explicit to avoid implicit conversion bugs during training.
RGB_DTYPE = "uint8"
DEPTH_DTYPE = "float32"
VECTOR_DTYPE = "float32"
TIMESTAMP_DTYPE = "float64"

ARRAY_NAMES = {
    "rgb": "rgb",
    "depth": "depth",
    "goal": "goal",
    "pose": "pose",
    "action_vel": "action_vel",
    "action_rot": "action_rot",
    "timestamp": "timestamp",
}

# Additional metadata recorded per episode.
REQUIRED_EPISODE_FIELDS = {
    "schema_version": DATASET_VERSION,
    "image_width": IMAGE_WIDTH,
    "image_height": IMAGE_HEIGHT,
    "total_pixels": TOTAL_PIXELS,
    "temporal_window_size": TEMPORAL_WINDOW_SIZE,
    "sampling_interval_s": 0.1,
    "max_ray_distance": 50.0,
    "horizontal_fov_deg": 90.0,
    "vertical_fov_deg": 90.0,
    "array_names": ARRAY_NAMES,
}


def make_episode_manifest(
    *,
    episode_id: str,
    participant_id: str,
    session_id: str,
    frame_count: int,
    start_time: float,
    end_time: float,
    extra_fields: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the canonical episode manifest for a recorded Unity session."""
    manifest: Dict[str, Any] = {
        "schema_version": DATASET_VERSION,
        "episode_id": episode_id,
        "participant_id": participant_id,
        "session_id": session_id,
        "frame_count": frame_count,
        "start_time": start_time,
        "end_time": end_time,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "total_pixels": TOTAL_PIXELS,
        "temporal_window_size": TEMPORAL_WINDOW_SIZE,
        "arrays": {
            "rgb": {"shape": list(RGB_SHAPE), "dtype": RGB_DTYPE},
            "depth": {"shape": list(DEPTH_SHAPE), "dtype": DEPTH_DTYPE},
            "goal": {"shape": list(GOAL_SHAPE), "dtype": VECTOR_DTYPE},
            "pose": {"shape": list(POSE_SHAPE), "dtype": VECTOR_DTYPE},
            "action_vel": {"shape": list(ACTION_VEL_SHAPE), "dtype": VECTOR_DTYPE},
            "action_rot": {"shape": list(ACTION_ROT_SHAPE), "dtype": VECTOR_DTYPE},
            "timestamp": {"shape": list(TIMESTAMP_SHAPE), "dtype": TIMESTAMP_DTYPE},
        },
    }
    if extra_fields:
        manifest.update(extra_fields)
    return manifest


def write_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    """Write a manifest to disk in JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    example = make_episode_manifest(
        episode_id="example_episode_01",
        participant_id="P001",
        session_id="S001",
        frame_count=100,
        start_time=0.0,
        end_time=10.0,
    )
    print(json.dumps(example, indent=2, sort_keys=True))
