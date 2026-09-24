"""Episode loading shared by the training scripts: ingest raw recordings, check every manifest against the data contract
(load_episodes) and turn one episode's dataset.zarr into (sensory_inputs, motor_actions) samples (SingleEpisodeDataset,
built identically to ImitationAgent.BuildSensoryTensor in Unity)."""
import random
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

import ingest_to_zarr
from data_contract import (ACTION_HORIZON_SAMPLES, CAMERA_FOV_DEG, IMAGE_HEIGHT, IMAGE_WIDTH, TEMPORAL_WINDOW_SIZE,
                           read_manifest)


def yaw_remap_indices(delta_deg: float, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH,
                      fov_deg: float = CAMERA_FOV_DEG):
    """Source (row, col) for every output pixel of the view after the camera yaws delta_deg to the right.

    Under the pinhole projection a yaw is not a column shift: the ray through output pixel (u, v) (normalised image
    coords in [-1, 1], tan(fov/2) at the border) is rotated about the vertical axis and re-projected, which moves
    columns non-linearly (more at the edges) and scales rows by 1/z'. This does that exactly for pixel centres and
    samples the nearest source pixel. What it cannot do is invent the strip that rotates into view: those pixels are
    clamped to the nearest edge pixel (replicating it), which is the only approximation. At 60 deg FOV and |delta| <=
    15 deg that strip is at most 17 of 64 columns. Depth is the distance along each pixel's ray, which a pure
    rotation leaves unchanged, so the same remap is exact for depth as well (it would not be for z-depth)."""
    t = np.tan(np.radians(fov_deg) / 2.0)
    u = ((np.arange(width) + 0.5) / width * 2.0 - 1.0) * t
    v = ((np.arange(height) + 0.5) / height * 2.0 - 1.0) * t
    x, y = np.meshgrid(u, v)  # (H, W) ray directions (x, y, 1) in the new camera frame
    c, sn = np.cos(np.radians(delta_deg)), np.sin(np.radians(delta_deg))
    xs, zs = x * c + sn, -x * sn + c  # same ray in the old camera frame (x = right, z = forward)
    zs = np.maximum(zs, 1e-6)
    col = (xs / zs / t + 1.0) / 2.0 * width - 0.5
    row = (y / zs / t + 1.0) / 2.0 * height - 0.5
    return (np.clip(np.rint(row), 0, height - 1).astype(np.intp),
            np.clip(np.rint(col), 0, width - 1).astype(np.intp))


def rotate_view(a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Apply yaw_remap_indices to a (T, H, W, C) array."""
    return a[:, rows, cols]


# Converts a raw goal distance (meters) into a bounded [0,1] proximity signal, with resolution
# concentrated near the goal (where distinguishing 0.5m from 2m matters far more than 40m from
# 45m) rather than an arbitrary linear /max_distance normalization. Must byte-for-byte match
# ImitationAgent.GoalDistanceToProximity in Unity. Distance is stored raw in zarr specifically so
# this transform can be changed later without re-recording.
def goal_distance_to_proximity(distance_m: float) -> float:
    return 1.0 / (1.0 + max(0.0, distance_m))


class SingleEpisodeDataset(Dataset):
    def __init__(self, ep_path: Path, window_size: int = TEMPORAL_WINDOW_SIZE, augment: bool = False,
                 action_horizon: int = ACTION_HORIZON_SAMPLES,
                 rot_aug_deg: float = 0.0, rot_aug_prob: float = 0.7, rot_correction: float = 0.15):
        zarr_root = zarr.open(str(ep_path / "dataset.zarr"), mode='r')
        self.rgb = zarr_root['rgb']
        self.depth = zarr_root['depth']
        self.goal = zarr_root['goal']
        if 'goal_distance' not in zarr_root:
            raise ValueError(f"{ep_path}: dataset.zarr has no goal_distance array. Re-record the episode.")
        self.goal_distance = zarr_root['goal_distance']
        self.rot_aug_deg = rot_aug_deg
        self.rot_aug_prob = rot_aug_prob
        self.rot_correction = rot_correction
        self.action_vel = zarr_root['action_vel']
        self.action_rot = zarr_root['action_rot']

        self.window_size = window_size
        self.augment = augment
        self.action_horizon = max(1, action_horizon)
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

        # Proximity signal (see goal_distance_to_proximity) - gives the network the "how close
        # am I" information the direction-only goal_vec above cannot provide on its own.
        goal_proximity = goal_distance_to_proximity(float(self.goal_distance[idx + w - 1]))

        # Interleaved (RGBD) frame sequence over window
        rgb_win = np.array(self.rgb[idx:idx+w], dtype=np.float32) / 255.0  # (T, 64, 64, 3)
        depth_win = np.array(self.depth[idx:idx+w], dtype=np.float32)[..., np.newaxis]  # (T, 64, 64, 1)

        # Target Motor Actions: the action to take FROM this observation, i.e. the one recorded
        # one sample after the window (see num_frames comment above).
        forward = np.array(self.action_vel[idx + w][2], dtype=np.float32)   # z = forward/back
        # Yaw/pitch are averaged over the next action_horizon samples (fewer at the very end of an
        # episode) - see ACTION_HORIZON_SAMPLES in data_contract.py for why. Wrap each yaw sample
        # BEFORE averaging so a 357.6 -> -2.4 wraparound can't corrupt the mean.
        rot = np.array(self.action_rot[idx + w: min(idx + w + self.action_horizon, self.rgb.shape[0])], dtype=np.float32)
        yaw = np.float32(((rot[:, 0] + 180.0) % 360.0 - 180.0).mean())      # index 0 = azimuth delta = yaw
        pitch = np.float32(rot[:, 1].mean())                                # index 1 = elevation delta = pitch

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

            # Simulated heading error with corrective relabelling (the PilotNet "shifted camera + steering
            # correction" idea). Humans never drive into a wall or arrive at a doorway with a bad
            # heading, so the recordings contain no examples of RECOVERING from one - and BC has no way
            # to learn recovery from demonstrations that never needed it. Here the view is rotated by a
            # random delta (positive = the agent turned right): both images are re-projected for that
            # yaw (yaw_remap_indices; RGB and depth share one camera, so one remap), the goal direction is rotated the opposite way in the local
            # frame, and the yaw label gets a proportional correction back towards the demonstrated
            # heading. rot_correction is deg/sample of yaw per degree of error (0.15 -> a 15 deg error
            # asks for 2.25 deg/sample = 22 deg/s, i.e. ~1 s to recover).
            if self.rot_aug_deg > 0.0 and random.random() < self.rot_aug_prob:
                delta = random.uniform(-self.rot_aug_deg, self.rot_aug_deg)
                rows, cols = yaw_remap_indices(delta)
                rgb_win = rotate_view(rgb_win, rows, cols)
                depth_win = rotate_view(depth_win, rows, cols)
                c, sn = np.cos(np.radians(delta)), np.sin(np.radians(delta))
                gx, gz = goal_vec[0], goal_vec[2]
                goal_vec[0] = gx * c - gz * sn
                goal_vec[2] = gx * sn + gz * c
                yaw = np.float32(yaw - self.rot_correction * delta)

            # Photometric jitter on RGB only, to fight overfitting to this layout's specific
            # lighting/texture rather than the underlying navigation cue.
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(0.8, 1.2)
            rgb_win = np.clip((rgb_win - 0.5) * contrast + 0.5, 0.0, 1.0)
            rgb_win = np.clip(rgb_win * brightness, 0.0, 1.0)

            # Small additive noise on depth only, to simulate raycast/sensor jitter instead of
            # letting the CNN treat exact depth readings on this one fixed layout as a
            # memorizable fingerprint. Depth is already normalized to [0,1] by MaxRayDistance
            # (ParticipantRecorder.LockedSamplingLoop), so a small fixed std stays proportionate
            # across the whole range; this never touches Unity's BuildSensoryTensor or the
            # parity test since it's train-time-only, applied after that shared feature layout.
            depth_win = np.clip(depth_win + np.random.normal(0.0, 0.02, depth_win.shape).astype(np.float32), 0.0, 1.0)

        rgbd = np.concatenate([rgb_win, depth_win], axis=-1).flatten()
        sensory_inputs = np.concatenate([goal_vec, np.array([goal_proximity], dtype=np.float32), rgbd])
        motor_actions = np.array([forward, yaw, pitch], dtype=np.float32)

        return torch.from_numpy(sensory_inputs), torch.from_numpy(motor_actions)


def ensure_all_episodes_ingested(recordings_dir: Path) -> None:
    """Scan recordings_dir for raw episode folders (manifest.json present) that haven't been
    converted to dataset.zarr yet, and convert them in place. Runs every time before training so
    freshly recorded episodes (e.g. new synthetic batches) are picked up automatically instead of
    silently being skipped by the episode discovery below."""
    if not recordings_dir.is_dir():
        return
    raw_episode_dirs = sorted(
        p for p in recordings_dir.iterdir()
        if p.is_dir() and (p / "manifest.json").exists() and not (p / "dataset.zarr").exists()
    )
    if not raw_episode_dirs:
        print("[INGEST] All episodes already converted to dataset.zarr - nothing to do.")
        return
    print(f"[INGEST] {len(raw_episode_dirs)} episode(s) missing dataset.zarr - converting now...")
    for ep in raw_episode_dirs:
        print(f"[INGEST] Converting {ep.name}...")
        ingest_to_zarr.convert_episode(str(ep), str(ep / "dataset.zarr"))
    print(f"[INGEST] Done - converted {len(raw_episode_dirs)} episode(s).")


def load_episodes(recordings_dir: Path) -> list:
    """Ingest what is new, then return [(episode_dir, manifest)] for every episode folder in recordings_dir.
    Every manifest is checked (data_contract.check_manifest): one episode without layout, participant, task or
    camera metadata, or without goal_distance, stops training instead of being silently mixed in."""
    ensure_all_episodes_ingested(recordings_dir)
    episodes = []
    bad = []
    for ep in sorted(p for p in recordings_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        try:
            manifest = read_manifest(ep)
            if not (ep / "dataset.zarr").exists():
                raise FileNotFoundError(f"{ep}: no dataset.zarr")
        except (OSError, ValueError) as ex:
            bad.append(str(ex))
            continue
        episodes.append((ep, manifest))
    if bad:
        raise ValueError(f"{len(bad)} episode(s) in {recordings_dir} violate the data contract - delete or re-record them:\n  "
                         + "\n  ".join(bad))
    return episodes
