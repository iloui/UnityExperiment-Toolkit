"""Canonical Zarr data contract for the Unity imitation dataset.

This module intentionally defines the target schema before any writing code is
implemented. The contract is shared by the Python training pipeline and the C#
Unity data recorder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_VERSION = "zarr-v2"
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

# Field of view (degrees) of the single capture camera; the image is square, so horizontal = vertical.
# RGB and depth come from the same pinhole camera: depth pixel (x, y) is the distance along the ray
# through the centre of RGB pixel (x, y), divided by MAX_RAY_DISTANCE (Unity ContractCamera). Row 0 is
# the BOTTOM image row (Texture2D.ReadPixels order) in both arrays. Must match
# ZarrDataContract.CameraFovDegrees.
CAMERA_FOV_DEG = 60.0
MAX_RAY_DISTANCE = 50.0

# Unity captures RGB at 64x64x3 and depth as a single normalized scalar per pixel.
RGB_SHAPE = ("time", IMAGE_HEIGHT, IMAGE_WIDTH, 3)
DEPTH_SHAPE = ("time", IMAGE_HEIGHT, IMAGE_WIDTH)
GOAL_SHAPE = ("time", 3)
# Euclidean distance (meters) from the agent's head to its current POI/end target - the goal
# array above is a NORMALIZED direction only, so without this the network has no way to tell
# "goal 50m away" from "goal 0.5m away" except through pixels. Recorded raw (not pre-normalized,
# unlike depth) so the proximity transform applied at consumption time (see
# SingleEpisodeDataset/ImitationAgent.BuildSensoryTensor) can be changed later without
# re-recording. Mandatory: ingest and training refuse an episode without it.
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

ARRAY_NAMES = (
    "rgb",
    "depth",
    "goal",
    "goal_distance",
    "pose",
    "action_vel",
    "action_rot",
    "timestamp",
)

# (shape, dtype) per array as written in the manifest's "arrays" list. Shape 0 on the first axis = time (frame_count).
ARRAY_SPECS = {
    "rgb": ([0, IMAGE_HEIGHT, IMAGE_WIDTH, 3], RGB_DTYPE),
    "depth": ([0, IMAGE_HEIGHT, IMAGE_WIDTH], DEPTH_DTYPE),
    "goal": ([0, 3], VECTOR_DTYPE),
    "goal_distance": ([0], GOAL_DISTANCE_DTYPE),
    "pose": ([0, 3], VECTOR_DTYPE),
    "action_vel": ([0, 3], VECTOR_DTYPE),
    "action_rot": ([0, 3], VECTOR_DTYPE),
    "timestamp": ([0], TIMESTAMP_DTYPE),
}

# --- Episode manifest (ZarrDataContract.EpisodeManifest) ---
# Everything Python needs about an episode without opening the Unity scene. check_manifest refuses an episode
# that lacks any of it; there is no fallback for older recordings.
SAMPLING_INTERVAL_S = 0.1
# Camera height above the NavMesh (ZarrDataContract.CameraHeightAboveNavMeshMeters / CameraHeightToleranceMeters).
CAMERA_HEIGHT_M = 1.5
CAMERA_HEIGHT_TOLERANCE_M = 0.03
SOURCE_HUMAN = "human"
SOURCE_SYNTHETIC = "synthetic"
EPISODE_SOURCES = (SOURCE_HUMAN, SOURCE_SYNTHETIC)
INPUT_MODE_DESKTOP = "desktop"

# Manifest key -> accepted Python type(s) after json.load.
_NUMBER = (int, float)
MANIFEST_FIELDS = {
    "schema_version": str,
    "recorder_version": str,
    "episode_id": str,
    "scene_name": str,
    "layout_typology": str,
    "layout_seed": int,
    "source": str,
    "participant_id": str,
    "session_id": str,
    "repetition_index": int,
    "input_mode": str,
    "task": dict,
    "frame_count": int,
    "start_time": _NUMBER,
    "end_time": _NUMBER,
    "image_width": int,
    "image_height": int,
    "total_pixels": int,
    "temporal_window_size": int,
    "sampling_interval_s": _NUMBER,
    "max_ray_distance": _NUMBER,
    "camera_height_m": _NUMBER,
    "camera_fov_deg": _NUMBER,
    "arrays": list,
}
# Must be non-empty strings (an empty participant_id would silently merge participants in the splits).
_NON_EMPTY_STRINGS = ("recorder_version", "episode_id", "scene_name", "layout_typology", "participant_id",
                      "session_id", "input_mode")


def _xyz(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(value.get(k), _NUMBER) for k in ("x", "y", "z"))


def make_episode_manifest(
    *,
    episode_id: str,
    scene_name: str,
    layout_typology: str,
    layout_seed: int,
    source: str,
    participant_id: str,
    session_id: str,
    repetition_index: int,
    task: Dict[str, Any],
    frame_count: int,
    start_time: float,
    end_time: float,
    camera_height_m: float = CAMERA_HEIGHT_M,
    input_mode: str = INPUT_MODE_DESKTOP,
    recorder_version: str = "python",
) -> Dict[str, Any]:
    """Build an episode manifest in exactly the layout ParticipantRecorder writes (for synthetic test episodes).
    task = {"start": {x,y,z}, "pois": [{x,y,z}, ...], "end": {x,y,z}} in Unity world coordinates."""
    manifest: Dict[str, Any] = {
        "schema_version": DATASET_VERSION,
        "recorder_version": recorder_version,
        "episode_id": episode_id,
        "scene_name": scene_name,
        "layout_typology": layout_typology,
        "layout_seed": layout_seed,
        "source": source,
        "participant_id": participant_id,
        "session_id": session_id,
        "repetition_index": repetition_index,
        "input_mode": input_mode,
        "task": task,
        "frame_count": frame_count,
        "start_time": start_time,
        "end_time": end_time,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "total_pixels": TOTAL_PIXELS,
        "temporal_window_size": TEMPORAL_WINDOW_SIZE,
        "sampling_interval_s": SAMPLING_INTERVAL_S,
        "max_ray_distance": MAX_RAY_DISTANCE,
        "camera_height_m": camera_height_m,
        "camera_fov_deg": CAMERA_FOV_DEG,
        "arrays": [{"name": name, "shape": shape, "dtype": dtype} for name, (shape, dtype) in ARRAY_SPECS.items()],
    }
    check_manifest(manifest, "make_episode_manifest")
    return manifest


def check_manifest(manifest: Dict[str, Any], source: str = "manifest") -> None:
    """Refuse an episode whose manifest is missing a field, was recorded under another schema or camera (e.g. zarr-v1:
    90 deg depth fan, unaligned with RGB), or has no goal_distance array. Raises ValueError listing every problem."""
    version = manifest.get("schema_version")
    if version != DATASET_VERSION:
        raise ValueError(f"{source}: schema_version is {version!r}, expected {DATASET_VERSION!r}. Re-record the episode.")

    problems = []
    for key, types in MANIFEST_FIELDS.items():
        if key not in manifest:
            problems.append(f"missing '{key}'")
        elif isinstance(manifest[key], bool) or not isinstance(manifest[key], types):
            problems.append(f"'{key}' has type {type(manifest[key]).__name__}")
    if problems:
        raise ValueError(f"{source}: incomplete manifest ({'; '.join(problems)}). Re-record the episode.")

    for key in _NON_EMPTY_STRINGS:
        if not manifest[key].strip():
            problems.append(f"'{key}' is empty")
    if manifest["participant_id"].strip() == "unknown":
        problems.append("'participant_id' is 'unknown'")
    if manifest["source"] not in EPISODE_SOURCES:
        problems.append(f"'source' is {manifest['source']!r}, expected one of {EPISODE_SOURCES}")
    if manifest["repetition_index"] < 0:
        problems.append("'repetition_index' is negative")
    if manifest["frame_count"] <= 0:
        problems.append("'frame_count' is not positive")

    task = manifest["task"]
    pois = task.get("pois")
    if not (_xyz(task.get("start")) and _xyz(task.get("end")) and isinstance(pois, list) and all(_xyz(p) for p in pois)):
        problems.append("'task' must be {start: {x,y,z}, pois: [{x,y,z}, ...], end: {x,y,z}}")

    fov = manifest["camera_fov_deg"]
    if abs(float(fov) - CAMERA_FOV_DEG) > 1e-3:
        problems.append(f"'camera_fov_deg' is {fov}, expected {CAMERA_FOV_DEG}")
    height = manifest["camera_height_m"]
    if abs(float(height) - CAMERA_HEIGHT_M) > CAMERA_HEIGHT_TOLERANCE_M:
        problems.append(f"'camera_height_m' is {height} (-1 = not measured), expected {CAMERA_HEIGHT_M} "
                        f"+- {CAMERA_HEIGHT_TOLERANCE_M}")
    if abs(float(manifest["sampling_interval_s"]) - SAMPLING_INTERVAL_S) > 1e-6:
        problems.append(f"'sampling_interval_s' is {manifest['sampling_interval_s']}, expected {SAMPLING_INTERVAL_S}")
    for key, expected in (("image_width", IMAGE_WIDTH), ("image_height", IMAGE_HEIGHT),
                          ("temporal_window_size", TEMPORAL_WINDOW_SIZE)):
        if manifest[key] != expected:
            problems.append(f"'{key}' is {manifest[key]}, expected {expected}")

    array_names = [a.get("name") for a in manifest["arrays"] if isinstance(a, dict)]
    missing_arrays = [name for name in ARRAY_NAMES if name not in array_names]
    if missing_arrays:
        problems.append(f"'arrays' lacks {missing_arrays}")

    if problems:
        raise ValueError(f"{source}: invalid manifest ({'; '.join(problems)}).")


def read_manifest(episode_dir: str | Path) -> Dict[str, Any]:
    """Load <episode_dir>/manifest.json and check it (raises on a missing file or an invalid manifest)."""
    path = Path(episode_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Manifest missing in episode directory: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    check_manifest(manifest, str(path))
    return manifest


def write_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    """Write a manifest to disk in JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    example = make_episode_manifest(
        episode_id="episode_20260924_120000_000",
        scene_name="Comb_Code_123456",
        layout_typology="Comb",
        layout_seed=123456,
        source=SOURCE_HUMAN,
        participant_id="P001",
        session_id="20260924_120000",
        repetition_index=0,
        task={"start": {"x": -12.0, "y": 1.05, "z": 0.0}, "pois": [{"x": 0.0, "y": 0.3, "z": -5.6}],
              "end": {"x": 12.0, "y": 1.0, "z": 0.0}},
        frame_count=100,
        start_time=0.0,
        end_time=10.0,
    )
    print(json.dumps(example, indent=2))
