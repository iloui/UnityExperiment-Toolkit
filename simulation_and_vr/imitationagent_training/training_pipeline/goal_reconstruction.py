"""Reconstruct the missing goal_distance channel for episodes recorded before it existed.

Why this exists: every recording under zarr_recordings/ predates the goal_distance array, so training
fed the network proximity = 0.0 ("unknown") for all frames, while ImitationAgent feeds the REAL
proximity 1/(1+d) at run time. The 4th goal input was therefore never trained (only its random
initial weights act on it), and it takes values the network never saw exactly where it matters:
on the approach to the POI/goal.

The recorded goal array is a unit direction in the agent's (pitched) head frame, so it carries no
distance - but each recording also stores the agent's world pose, and the task is fixed (start ->
POI -> end, same markers every time). So:

    distance = sqrt( |target_xz - pose_xz|^2 + (pose_y - target_y)^2 )
    target   = POI until the agent is within the arrival radius of it, then the end marker

Validation (`Reconstruction.ok`): the POI->end switch frame predicted from the pose must agree with the frame
where the RECORDED goal direction jumps (two independent signals). If they don't, the assumed marker
positions are wrong and the caller should not trust the reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# CentralPillar_Code_927712 (the scene the recordings were made in): POI and end marker world xz.
# Taken from EvalHarness episodes.csv poi_positions/end_pos for that scene and cross-checked against
# the recordings (closest approach to this POI is <= 0.9 m in all 25 episodes; the POI->end switch
# predicted from pose matches the frame where the recorded goal vector jumps to within 1-3 frames).
DEFAULT_POI_XZ = (-0.01, -5.6)
DEFAULT_END_XZ = (11.25, 0.0)
ARRIVAL_RADIUS_M = 1.5  # ZarrDataContract.ArrivalRadiusMeters

# Marker heights (EvalHarness episodes.csv, same scene) - used only for the small vertical part of the distance.
POI_Y = 0.276
END_Y = 0.325

# Max allowed disagreement (frames) between the pose-predicted phase switch and the recorded goal jump.
MAX_SWITCH_DISAGREEMENT_FRAMES = 5


@dataclass
class Reconstruction:
    distance_m: np.ndarray            # (T,) float32, head-to-target distance
    switch_index: int                 # first frame that targets the end marker (-1 = never reached the POI)
    goal_jump_index: int              # frame where the recorded goal direction jumps the most
    switch_disagreement: int          # |switch_index - goal_jump_index| in frames (large = suspicious)
    ok: bool


def reconstruct_goal_distance(
    pose: np.ndarray,
    goal: np.ndarray,
    poi_xz: Tuple[float, float] = DEFAULT_POI_XZ,
    end_xz: Tuple[float, float] = DEFAULT_END_XZ,
    arrival_radius_m: float = ARRIVAL_RADIUS_M,
) -> Reconstruction:
    pose = np.asarray(pose, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    t = len(pose)
    poi = np.asarray(poi_xz, dtype=np.float64)
    end = np.asarray(end_xz, dtype=np.float64)

    d_poi = np.hypot(pose[:, 0] - poi[0], pose[:, 2] - poi[1])
    hit = np.where(d_poi <= arrival_radius_m)[0]
    switch = int(hit[0]) if len(hit) else -1

    target_xz = np.tile(end, (t, 1))
    target_y = np.full(t, END_Y)
    cut = t if switch < 0 else switch
    target_xz[:cut] = poi
    target_y[:cut] = POI_Y

    d_xz = np.hypot(pose[:, 0] - target_xz[:, 0], pose[:, 2] - target_xz[:, 1])
    dist = np.sqrt(d_xz ** 2 + (pose[:, 1] - target_y) ** 2)

    # Independent evidence for the phase switch: the biggest frame-to-frame change of the recorded goal direction.
    norms = np.maximum(np.linalg.norm(goal, axis=1), 1e-6)
    cos = (goal[1:] * goal[:-1]).sum(1) / (norms[1:] * norms[:-1])
    jump = int(np.argmin(cos)) + 1
    disagreement = abs(switch - jump) if switch >= 0 else 10 ** 6

    return Reconstruction(
        distance_m=dist.astype(np.float32),
        switch_index=switch,
        goal_jump_index=jump,
        switch_disagreement=int(disagreement),
        ok=bool(switch >= 0 and disagreement <= MAX_SWITCH_DISAGREEMENT_FRAMES),
    )


def _self_test(recordings_dir: str) -> None:
    import glob
    import os

    import zarr

    for ep in sorted(glob.glob(os.path.join(recordings_dir, "episode_*"))):
        z = zarr.open(os.path.join(ep, "dataset.zarr"), "r")
        r = reconstruct_goal_distance(np.array(z["pose"]), np.array(z["goal"]))
        print(f"{os.path.basename(ep)}: frames={len(r.distance_m)} switch@{r.switch_index} goal-jump@{r.goal_jump_index} "
              f"disagree={r.switch_disagreement} ok={r.ok} "
              f"dist range {r.distance_m.min():.1f}..{r.distance_m.max():.1f} m")


if __name__ == "__main__":
    import sys

    _self_test(sys.argv[1] if len(sys.argv) > 1 else "../zarr_recordings")
