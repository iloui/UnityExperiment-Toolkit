import copy
import os
import random
import argparse
from pathlib import Path
import zarr
import numpy as np
import onnx
import torch
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from data_contract import TEMPORAL_WINDOW_SIZE, ACTION_HORIZON_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT
from goal_reconstruction import reconstruct_goal_distance, DEFAULT_POI_XZ, DEFAULT_END_XZ

# Horizontal field of view of the two sensors, needed to turn a simulated view rotation (degrees)
# into a column shift. The RGB camera is copied from FirstPersonCharacter (60 deg vertical FOV, 64x64
# = square aspect -> 60 deg horizontal); the depth raycast fan is 90 deg (ZarrDataContract). They
# differ, so one heading change shifts the two images by different pixel counts.
RGB_HFOV_DEG = 60.0
DEPTH_HFOV_DEG = 90.0

# --- 1. Streamlined 2D CNN Model (Sentis Contract Preserved) ---
class SimpleImitation2DCNNModel(nn.Module):
    def __init__(self, img_res: int = 64, mask_proximity: bool = False):
        super().__init__()
        self.H = self.W = img_res
        # When the proximity input can't be trained on real values (see --proximity_mode), the model
        # multiplies it by 0 inside forward(), so whatever Unity feeds at run time is ignored. The
        # input layout (16388 floats) is unchanged, so no Unity change is needed. Exported as a constant.
        self.register_buffer("goal_input_scale", torch.tensor([1.0, 1.0, 1.0, 0.0 if mask_proximity else 1.0]))

        # Lightweight 2D CNN over a single 4-channel frame (RGB + Depth)
        self.conv2d_encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(), # TODO: Try LeakyReLU()
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(), # TODO: Try LeakyReLU()
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(), # TODO: Try LeakyReLU()
        )

        # FiLM conditioning (Perez et al. 2017): the goal vector produces a per-channel
        # scale/shift applied directly to the conv feature maps, instead of only being
        # concatenated after flattening where it's just 4 of 4100 inputs to the head and easy
        # for the network to learn to give near-zero weight - which is what previously produced
        # a purely reactive wall-following policy that ignored where it was supposed to go.
        # FiLM ties every visual channel's contribution to the goal instead of leaving that
        # coupling to a single late, easily-bypassed linear layer.
        #
        # 4th component is a bounded goal-proximity signal (1/(1+distance_m), see
        # SingleEpisodeDataset), not just direction - open-loop eval showed the model had zero
        # way to distinguish "goal 50m away" from "goal 0.5m away" and so never learned to slow
        # down/commit to a turn on final approach, instead orbiting the target.
        self.goal_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
        )
        self.film_gamma = nn.Linear(32, 64)
        self.film_beta = nn.Linear(32, 64)

        self.flatten = nn.Flatten()

        # +4 for the goal vector (direction + proximity), concatenated back in below on top of
        # the FiLM modulation above, so the head still gets a direct, unfiltered look at it too.
        self.network = nn.Sequential(
            nn.Linear(64 * 8 * 8 + 4, 64),
            nn.ReLU(), # TODO: Try LeakyReLU()
            nn.Linear(64, 3)    # Motor Outputs: [Forward, Yaw, Pitch] (Unbounded)
        )

    def forward(self, sensory_inputs: torch.Tensor) -> torch.Tensor:
        # 1. Split off the goal vector (direction xyz + proximity, first 4 floats) from the visual payload
        goal_vec = sensory_inputs[:, :4] * self.goal_input_scale  # Shape: (Batch, 4)
        vis_flat = sensory_inputs[:, 4:]  # Shape: (Batch, 16384)

        # 2. Reshape the single most-recent RGBD frame -> (Batch, 64, 64, 4). There is no
        #    temporal stacking anymore (TEMPORAL_WINDOW_SIZE == 1) - a 3-frame window used to be
        #    carried all the way through capture/storage/transport only for this forward() to
        #    immediately slice it back down to the latest frame, so the window was pure dead
        #    weight; the whole pipeline now only ever produces/consumes a single frame.
        x_latest = vis_flat.view(-1, self.H, self.W, 4)

        # 3. Permute to PyTorch CNN layout -> (Batch, C=4, H=64, W=64)
        x_latest = x_latest.permute(0, 3, 1, 2).contiguous()  # contiguous(): MPS conv rejects the strided permuted view

        # 4. Pass through the 2D CNN, then use the goal vector to modulate the resulting
        #    feature maps (FiLM) before flattening, then feed the goal vector into the MLP
        #    head again alongside the modulated visual features.
        vis_map = self.conv2d_encoder(x_latest)  # (Batch, 64, 8, 8)

        goal_hidden = self.goal_encoder(goal_vec)
        gamma = self.film_gamma(goal_hidden).unsqueeze(-1).unsqueeze(-1)  # (Batch, 64, 1, 1)
        beta = self.film_beta(goal_hidden).unsqueeze(-1).unsqueeze(-1)   # (Batch, 64, 1, 1)
        vis_map = vis_map * (1.0 + gamma) + beta

        vis_features = self.flatten(vis_map)
        combined = torch.cat([vis_features, goal_vec], dim=1)
        return self.network(combined)

def shift_columns(a: np.ndarray, n: int) -> np.ndarray:
    """Shift the image content of a (T, H, W, C) array n columns to the right (n < 0: left), replicating
    the edge column into the gap instead of wrapping around."""
    if n == 0:
        return a
    out = np.empty_like(a)
    if n > 0:
        out[:, :, n:] = a[:, :, :-n]
        out[:, :, :n] = a[:, :, :1]
    else:
        out[:, :, :n] = a[:, :, -n:]
        out[:, :, n:] = a[:, :, -1:]
    return out

# Converts a raw goal distance (meters) into a bounded [0,1] proximity signal, with resolution
# concentrated near the goal (where distinguishing 0.5m from 2m matters far more than 40m from
# 45m) rather than an arbitrary linear /max_distance normalization. Must byte-for-byte match
# ImitationAgent.GoalDistanceToProximity in Unity. Distance is stored raw in zarr specifically so
# this transform can be changed later without re-recording.
def goal_distance_to_proximity(distance_m: float) -> float:
    return 1.0 / (1.0 + max(0.0, distance_m))

# Sentinel proximity fed for episodes recorded before goal_distance.bin existed (see
# ingest_to_zarr.py) - equivalent to "infinitely far away", i.e. the same value a genuine
# reading would asymptote to, so it reads as "no proximity information" rather than "very close".
UNKNOWN_GOAL_DISTANCE_PROXIMITY = 0.0

# --- 2. Zarr Dataset Loader ---
class SingleEpisodeDataset(Dataset):
    def __init__(self, ep_path: Path, window_size: int = TEMPORAL_WINDOW_SIZE, augment: bool = False,
                 action_horizon: int = ACTION_HORIZON_SAMPLES, proximity_mode: str = "reconstruct",
                 rot_aug_deg: float = 0.0, rot_aug_prob: float = 0.7, rot_correction: float = 0.15,
                 poi_xz=DEFAULT_POI_XZ, end_xz=DEFAULT_END_XZ):
        zarr_root = zarr.open(str(ep_path / "dataset.zarr"), mode='r')
        self.rgb = zarr_root['rgb']
        self.depth = zarr_root['depth']
        self.goal = zarr_root['goal']
        try:
            self.goal_distance = zarr_root['goal_distance']
        except KeyError:
            self.goal_distance = None
            # Episodes recorded before goal_distance existed: rebuild it from the recorded pose (see
            # goal_reconstruction.py) so training sees the same proximity signal Unity feeds at run time.
            if proximity_mode == "reconstruct":
                self.goal_distance = reconstruct_goal_distance(
                    np.array(zarr_root['pose']), np.array(zarr_root['goal']), poi_xz, end_xz).distance_m
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
        # am I" information the direction-only goal_vec above cannot provide on its own. Falls
        # back to the unknown-distance sentinel on episodes recorded before this channel existed.
        if self.goal_distance is not None:
            goal_proximity = goal_distance_to_proximity(float(self.goal_distance[idx + w - 1]))
        else:
            goal_proximity = UNKNOWN_GOAL_DISTANCE_PROXIMITY

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
            # random delta (positive = the agent turned right): both images are shifted sideways by the
            # matching number of pixels, the goal direction is rotated the opposite way in the local
            # frame, and the yaw label gets a proportional correction back towards the demonstrated
            # heading. rot_correction is deg/sample of yaw per degree of error (0.15 -> a 15 deg error
            # asks for 2.25 deg/sample = 22 deg/s, i.e. ~1 s to recover).
            if self.rot_aug_deg > 0.0 and random.random() < self.rot_aug_prob:
                delta = random.uniform(-self.rot_aug_deg, self.rot_aug_deg)
                rgb_win = shift_columns(rgb_win, -int(round(delta * IMAGE_WIDTH / RGB_HFOV_DEG)))
                depth_win = shift_columns(depth_win, -int(round(delta * IMAGE_WIDTH / DEPTH_HFOV_DEG)))
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

# --- 3. Per-channel label scale (forward is m/s, yaw/pitch are degrees - very different natural scales) ---
def compute_label_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    all_actions = torch.cat([actions for _, actions in loader], dim=0)
    mean = all_actions.mean(dim=0)
    std = all_actions.std(dim=0).clamp_min(1e-6)
    return mean, std

# --- 4. Open-loop diagnostics on held-out episodes ---
@torch.no_grad()
def report_turn_response(model: nn.Module, dataset: Dataset, device: torch.device, tag: str) -> None:
    """Does the model actually turn towards an off-axis goal? Plain val MSE can't tell: with bursty human
    yaw the MSE-optimal output is a weak, near-constant turn, which is exactly what pins the agent against
    a wall in closed loop. Prints predicted vs demonstrated yaw per goal-bearing bin and a global
    undershoot ratio, on clean (non-augmented) held-out frames."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    bearings, preds, targets = [], [], []
    for sensory, actions in loader:
        out = model(sensory.to(device)).cpu()
        g = sensory[:, :3]
        bearings.append(torch.rad2deg(torch.atan2(g[:, 0], g[:, 2])))
        preds.append(out)
        targets.append(actions)
    bearing = torch.cat(bearings).numpy()
    pred = torch.cat(preds).numpy()
    tgt = torch.cat(targets).numpy()

    print(f"\n[DIAGNOSTIC {tag}] yaw (deg/sample) vs goal bearing, held-out frames")
    print("   bearing bin      n   demo yaw  pred yaw   pred/demo")
    edges = [-180, -90, -60, -30, -10, 10, 30, 60, 90, 180]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (bearing >= a) & (bearing < b)
        if m.sum() < 5:
            continue
        d, p_ = tgt[m, 1].mean(), pred[m, 1].mean()
        ratio = p_ / d if abs(d) > 0.05 else float("nan")
        print(f"  [{a:4d},{b:4d})  {int(m.sum()):5d}  {d:8.2f}  {p_:8.2f}   {ratio:8.2f}")
    turning = np.abs(tgt[:, 1]) > 3.0
    if turning.sum() > 5:
        slope = np.polyfit(tgt[turning, 1], pred[turning, 1], 1)[0]
        print(f"  turning frames (|demo yaw| > 3): n={int(turning.sum())}, slope of predicted vs demonstrated yaw = {slope:.2f} (1.0 = full response)")
    print(f"  forward: demo mean {tgt[:, 0].mean():.2f} | pred mean {pred[:, 0].mean():.2f} m/s")

# --- 5. ONNX Exporter ---
def export_onnx(model: nn.Module, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    sensory_input_dim = TEMPORAL_WINDOW_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * 4 + 4
    dummy_input = torch.randn(1, sensory_input_dim, dtype=torch.float32)

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

    # Recent torch versions transparently split the export into a small .onnx header plus a
    # separate .onnx.data file once the weights are large enough. Unity Sentis only understands
    # a single self-contained .onnx file, so re-save with everything embedded in one file.
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)

    print(f"\n[EXPORT SUCCESS] ONNX model exported to: {output_path}")

# --- 6. Training Loop ---
def main():
    parser = argparse.ArgumentParser(description="Simplified 2D Behavior Cloning Pipeline")
    parser.add_argument("--recordings_dir", type=str, default="./recordings")
    parser.add_argument("--output_onnx", type=str, default="../Assets/ImitationModel/ImitationAgentModel.onnx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio (episode level)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--turn_weight_alpha", type=float, default=2.0,
                         help="Per-sample train loss weight = 1 + alpha*|yaw_normalized|. Large "
                         "reorientation frames are only ~7%% of recorded frames, so plain per-frame "
                         "MSE averages them away and the model learns a weak, near-constant turn "
                         "response regardless of how much turning is actually needed (open-loop "
                         "eval showed a ~0.43 median undershoot ratio and flat yaw output from 10 "
                         "to 180 degrees of required turn). Upweighting by demonstrated |yaw| forces "
                         "the rare hard-turn frames to actually move the gradient. 0 disables it.")
    parser.add_argument("--proximity_mode", choices=["reconstruct", "mask", "raw"], default="reconstruct",
                         help="How to treat the goal-proximity input. All current recordings predate the "
                         "goal_distance array, so 'raw' (the old behaviour) trains with proximity=0 while Unity "
                         "feeds the real value at run time (train/serve skew on an input that was never "
                         "trained). 'reconstruct' rebuilds it from the recorded pose (validated; falls back to "
                         "'mask' if the validation fails). 'mask' makes the model ignore that input entirely.")
    parser.add_argument("--rot_aug_deg", type=float, default=15.0,
                         help="Max simulated heading error (degrees) for the corrective rotation augmentation. 0 disables.")
    parser.add_argument("--rot_aug_prob", type=float, default=0.7, help="Probability of applying it to a training frame.")
    parser.add_argument("--rot_correction", type=float, default=0.15,
                         help="Yaw-label correction, deg/sample per degree of simulated heading error.")
    parser.add_argument("--poi_xz", type=float, nargs=2, default=list(DEFAULT_POI_XZ), help="World xz of the POI marker (goal reconstruction).")
    parser.add_argument("--end_xz", type=float, nargs=2, default=list(DEFAULT_END_XZ), help="World xz of the end marker (goal reconstruction).")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    recordings_path = Path(args.recordings_dir)
    episodes = [p for p in recordings_path.iterdir() if p.is_dir() and (p / "dataset.zarr").exists()]
    
    if not episodes:
        print(f"[ERROR] No valid Zarr episode folders found in '{args.recordings_dir}'")
        return

    episodes_missing_goal_distance = [
        ep for ep in episodes if "goal_distance" not in zarr.open(str(ep / "dataset.zarr"), mode="r")
    ]
    proximity_mode = args.proximity_mode
    if episodes_missing_goal_distance:
        print(f"[DATASET] {len(episodes_missing_goal_distance)}/{len(episodes)} episodes have no goal_distance channel "
              f"(proximity_mode={proximity_mode}).")
        if proximity_mode == "raw":
            print(f"[DATASET] WARNING: 'raw' trains those frames with proximity={UNKNOWN_GOAL_DISTANCE_PROXIMITY} while Unity "
                  f"feeds the real value at run time - train/serve skew on an untrained input.")
        elif proximity_mode == "reconstruct":
            recs = []
            for ep in episodes_missing_goal_distance:
                zr = zarr.open(str(ep / "dataset.zarr"), mode="r")
                recs.append(reconstruct_goal_distance(np.array(zr["pose"]), np.array(zr["goal"]), tuple(args.poi_xz), tuple(args.end_xz)))
            ok_frac = sum(r.ok for r in recs) / len(recs)
            print(f"[DATASET] goal_distance reconstruction validated on {sum(r.ok for r in recs)}/{len(recs)} episodes "
                  f"(POI->end switch from pose vs recorded goal jump, max disagreement "
                  f"{max(r.switch_disagreement for r in recs)} frames)")
            if ok_frac < 0.8:
                print("[DATASET] WARNING: reconstruction validation failed (marker positions wrong for these recordings?) "
                      "- falling back to proximity_mode=mask. Check --poi_xz/--end_xz.")
                proximity_mode = "mask"

    # Episode-level Train/Val Split
    random.shuffle(episodes)
    val_count = max(1, int(len(episodes) * args.val_ratio))
    val_episodes = episodes[:val_count]
    train_episodes = episodes[val_count:]

    print(f"[DATASET] Split total {len(episodes)} episodes -> Train: {len(train_episodes)}, Val: {len(val_episodes)}")

    # Augment only the training split - val must stay a clean, deterministic measurement of
    # generalization to unseen (real, non-mirrored/non-jittered) episodes.
    ds_kwargs = dict(proximity_mode=proximity_mode, poi_xz=tuple(args.poi_xz), end_xz=tuple(args.end_xz))
    train_dataset = ConcatDataset([
        SingleEpisodeDataset(ep, augment=True, rot_aug_deg=args.rot_aug_deg, rot_aug_prob=args.rot_aug_prob,
                             rot_correction=args.rot_correction, **ds_kwargs) for ep in train_episodes])
    val_dataset = ConcatDataset([SingleEpisodeDataset(ep, augment=False, **ds_kwargs) for ep in val_episodes])
    print(f"[TRAINING] proximity_mode={proximity_mode} rot_aug_deg={args.rot_aug_deg} rot_aug_prob={args.rot_aug_prob} "
          f"rot_correction={args.rot_correction} turn_weight_alpha={args.turn_weight_alpha}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
    label_mean, label_std = compute_label_stats(train_dataset)
    label_mean, label_std = label_mean.to(device), label_std.to(device)
    print(f"[LABEL STATS] mean(forward,yaw,pitch)={label_mean.tolist()} std={label_std.tolist()}")

    model = SimpleImitation2DCNNModel(mask_proximity=(proximity_mode == "mask")).to(device)
    # weight_decay=1e-4 and grad_clip_norm=1.0 were previously tuned to curb erratic yaw/pitch
    # extrapolation in live Unity testing, but open-loop (teacher-forcing) evaluation against
    # the model's own training episodes showed that clipping the *global* gradient norm to 1.0
    # across all ~287k shared-trunk parameters throttles every update so hard that the easy
    # channel (forward) still converges in 20 epochs while yaw/pitch - which need a harder,
    # noisier mapping learned from the shared trunk - never get there: predicted yaw/pitch
    # variance collapsed to ~20-30% of the recorded variance (near mean prediction) even on
    # data the model was trained on. That's underfitting, not the runaway extrapolation this
    # was guarding against - min_best_epoch below already rules out shipping an early unstable
    # checkpoint, so the clip/decay no longer need to carry that job alone. Loosened to let the
    # harder channels actually fit; still non-zero as a safety net against a single bad batch.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # reduction="none" so the training loop can apply a per-sample turn_weight before averaging;
    # the validation loop below still collapses this to the same plain mean MSE as before, so
    # checkpoint selection (best_val_loss) stays an unweighted, apples-to-apples measurement.
    criterion = nn.MSELoss(reduction="none")
    YAW_CHANNEL = 1

    # BatchNorm's running stats (momentum 0.1) and each output channel's scale are still
    # settling in the first few epochs, so an early epoch can score deceptively well on the
    # small (4-episode) val split by luck without actually being calibrated yet - e.g. it
    # picked epoch 4/40 once and shipped a model whose forward output was 4-8x real walking
    # speed. Don't let the checkpoint search consider epochs before the model's had a real
    # chance to settle.
    min_best_epoch = max(1, args.epochs // 4)
    print(f"[CHECKPOINT] Ignoring epochs before {min_best_epoch} when selecting the best model")

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        weight_sum = 0.0
        for sensory, actions in train_loader:
            sensory, actions = sensory.to(device), actions.to(device)

            optimizer.zero_grad()
            predictions = model(sensory)
            norm_pred = (predictions - label_mean) / label_std
            norm_actions = (actions - label_mean) / label_std
            per_sample_loss = criterion(norm_pred, norm_actions).mean(dim=1)  # (B,)
            # Rare hard-turn frames (~7% of frames, per open-loop analysis) get averaged away
            # under plain per-frame MSE, so the model learns a weak, near-constant turn response
            # regardless of how much turning is actually needed. Weight each frame's loss by how
            # large the demonstrated yaw actually was, so those frames pull the gradient harder.
            turn_weight = 1.0 + args.turn_weight_alpha * norm_actions[:, YAW_CHANNEL].abs()
            loss = (per_sample_loss * turn_weight).mean()
            loss.backward()
            # Caps any single bad batch from yanking the shared trunk's weights into a regime
            # that extrapolates wildly on live (slightly out-of-distribution) camera frames.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            train_loss += loss.item() * sensory.size(0)
            weight_sum += turn_weight.sum().item()

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_mean_turn_weight = weight_sum / len(train_dataset)

        # Validation Phase - plain unweighted mean MSE, so it stays a fair, consistent metric
        # for checkpoint selection regardless of how turn_weight_alpha is tuned.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sensory, actions in val_loader:
                sensory, actions = sensory.to(device), actions.to(device)
                predictions = model(sensory)
                loss = criterion((predictions - label_mean) / label_std, (actions - label_mean) / label_std).mean()
                val_loss += loss.item() * sensory.size(0)

        epoch_val_loss = val_loss / len(val_dataset)

        is_best = epoch >= min_best_epoch and epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        marker = " (best)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} "
              f"| Mean Turn Weight: {epoch_mean_turn_weight:.3f}{marker}")

    # Validation loss can bounce between epochs on a small dataset; blindly exporting whatever
    # the final epoch happened to land on (as before) risks shipping a worse-than-typical
    # snapshot. Export the checkpoint that actually scored best on held-out episodes instead.
    print(f"\n[CHECKPOINT] Using epoch {best_epoch} (best Val Loss: {best_val_loss:.6f})")
    model.load_state_dict(best_state_dict)
    report_turn_response(model, val_dataset, device, "best checkpoint")
    model.cpu()
    export_onnx(model, args.output_onnx)

if __name__ == "__main__":
    main()