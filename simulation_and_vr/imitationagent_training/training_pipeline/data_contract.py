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

# The current imitation pipeline reacts to a single most-recent frame - no temporal stacking.
TEMPORAL_WINDOW_SIZE = 1

# Yaw/pitch training targets are the MEAN per-sample rotation over this many consecutive samples
# starting at the sample after the observation (0.5 s at the 0.1 s sampling interval), not the
# single next sample. Recorded human yaw is bursty (61% of samples ~0 deg, rare 10-40 deg snaps),
# so a one-sample MSE target regresses to ~0 and the policy never turns toward an off-axis goal.
# Averaging keeps the units (deg per sample), so ImitationAgent's raw/RecordingSamplingInterval
# conversion and the ONNX output contract are unchanged. Training-side only: nothing is stored
# in the recordings, so existing episodes stay valid. 1 restores the old single-sample target.
ACTION_HORIZON_SAMPLES = 5

# --- Discrete action head (train_2DCNN_discrete.py) ---
# A single MSE yaw output averages "turn left" and "turn right" to ~0 wherever both are demonstrated
# (wall ahead with the goal behind it, T-junctions), which is what froze M4 against walls. The discrete
# model instead classifies the 0.5 s mean yaw (see ACTION_HORIZON_SAMPLES) into straight + 3 strengths per
# side. Edges/centers are |yaw| in deg per 0.1 s sample; centers are the per-bin label means of the human
# and v2 synthetic data. Must match ZarrDataContract.DiscreteHead in Unity.
YAW_BIN_EDGES_DEG = (0.5, 3.0, 8.0)          # straight | weak | medium | strong
YAW_BIN_CENTERS_DEG = (0.0, 1.5, 5.0, 12.0)  # magnitude per bin, straight first
YAW_CLASS_COUNT = 2 * (len(YAW_BIN_CENTERS_DEG) - 1) + 1  # 7: strong-left ... straight ... strong-right
YAW_STRAIGHT_CLASS = YAW_CLASS_COUNT // 2
# Forward labels are ~binary (0 or 1.3 m/s, 1.6% ramp samples in between): walk = forward above this.
WALK_THRESHOLD_MPS = 0.65
# ONNX 'motor_actions' layout of the discrete model: [walk_logit, yaw_logit_0..6, pitch_deg_per_sample].
DISCRETE_OUTPUT_SIZE = 1 + YAW_CLASS_COUNT + 1

# Unity captures RGB at 64x64x3 and depth as a single normalized scalar per pixel.
RGB_SHAPE = ("time", IMAGE_HEIGHT, IMAGE_WIDTH, 3)
DEPTH_SHAPE = ("time", IMAGE_HEIGHT, IMAGE_WIDTH)
GOAL_SHAPE = ("time", 3)
# Euclidean distance (meters) from the agent's head to its current POI/end target - the goal
# array above is a NORMALIZED direction only, so without this the network has no way to tell
# "goal 50m away" from "goal 0.5m away" except through pixels. Recorded raw (not pre-normalized,
# unlike depth) so the proximity transform applied at consumption time (see
# SingleEpisodeDataset/ImitationAgent.BuildSensoryTensor) can be changed later without
# re-recording. Optional on old episodes recorded before this field existed - loaders must
# handle its absence rather than assume every episode has it.
GOAL_DISTANCE_SHAPE = ("time",)
POSE_SHAPE = ("time", 3)
ACTION_VEL_SHAPE = ("time", 3)
ACTION_ROT_SHAPE = ("time", 3)
TIMESTAMP_SHAPE = ("time",)

# Storage types are explicit to avoid implicit conversion bugs during training.
RGB_DTYPE = "uint8"
DEPTH_DTYPE = "float32"
VECTOR_DTYPE = "float32"
GOAL_DISTANCE_DTYPE = "float32"
TIMESTAMP_DTYPE = "float64"

ARRAY_NAMES = {
    "rgb": "rgb",
    "depth": "depth",
    "goal": "goal",
    "goal_distance": "goal_distance",
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
            "goal_distance": {"shape": list(GOAL_DISTANCE_SHAPE), "dtype": GOAL_DISTANCE_DTYPE},
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
