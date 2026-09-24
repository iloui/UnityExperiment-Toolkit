# Episode data contract (`zarr-v3`)

The synthetic bot, human participants and the ImitationAgent all share one recording format. It is defined twice, and the two copies must match:

- Unity: `unity_files/Assets/Scripts/DataRecording/ZarrDataContract.cs` (written by `ParticipantRecorder`)
- Python: `training_pipeline/data_contract.py` (checked by `check_manifest`, read by ingestion and training)

Each episode is self-describing. Python never has to open the Unity scene to find out which layout, participant or task an episode belongs to. Nothing is kept for backward compatibility: an episode that does not match this document is refused, not patched.

## 1. On-disk layout

`ParticipantRecorder` writes one folder per episode into `imitationagent_training/zarr_recordings/`:

```text
zarr_recordings/
  episode_<yyyyMMdd_HHmmss_fff>/
    manifest.json         episode metadata (section 3)
    rgb.bin               raw little-endian arrays, one record per sample (section 2)
    depth.bin
    goal.bin
    goal_distance.bin
    pose.bin
    action_vel.bin
    action_rot.bin
    timestamp.bin
    dataset.zarr/         added by ingest_to_zarr.py: one Zarr group with the same eight arrays,
                          and the manifest copied into the group attribute "manifest" (JSON string)
```

`train_*` scripts call `ensure_all_episodes_ingested`, which converts every folder that has no `dataset.zarr` yet. `load_episodes` then checks every manifest. If any episode fails a check, training stops and lists the failing episodes.

## 2. Arrays

Sampling is locked to one sample every `sampling_interval_s` = 0.1 s. Every array has exactly `frame_count` rows, and row *i* of every array belongs to sample *i*.

| name | dtype | shape | meaning |
|---|---|---|---|
| `rgb` | uint8 | (T, 64, 64, 3) | colour image from the capture camera |
| `depth` | float32 | (T, 64, 64) | distance along each pixel's ray to the `Architecture_Obstacle` layer, divided by `max_ray_distance` (50 m); 1.0 = no hit |
| `goal` | float32 | (T, 3) | unit direction from the camera to the current target (POI, then end), in the camera frame (x right, y up, z forward) |
| `goal_distance` | float32 | (T,) | Euclidean distance in metres from the camera to the current target. **Mandatory.** Training turns it into proximity = 1 / (1 + d) |
| `pose` | float32 | (T, 5) | camera world position x, y, z (Unity metres, y up), heading (camera yaw, degrees in [0, 360), clockwise from above, 0 = +z) and pitch (degrees, + = looking down) |
| `action_vel` | float32 | (T, 3) | camera velocity between sample *i*-1 and *i*, in the camera frame, m/s (z = forward) |
| `action_rot` | float32 | (T, 3) | (yaw, pitch, 0) change between sample *i*-1 and *i*, degrees per sample. Row 0 is meaningless |
| `timestamp` | float64 | (T,) | seconds since the start of recording |

**Camera.** RGB and depth come from the same 60° square pinhole camera (`camera_fov_deg`; `ContractCamera` in Unity). Depth pixel (x, y) is the distance along the ray through the centre of RGB pixel (x, y), so the two are pixel-aligned. Row 0 is the **bottom** image row in both arrays (Unity `ReadPixels` order). The camera sits `camera_height_m` = 1.5 m above the NavMesh, for humans, the bot and the agent alike.

**Action alignment.** `action_*[i]` is the motion that produced observation *i*. The label for observation *i* is therefore read at *i* + 1 (`SingleEpisodeDataset`). Yaw and pitch labels are averaged over the next `ACTION_HORIZON_SAMPLES` samples.

**Model input** (built identically by `SingleEpisodeDataset` and `ImitationAgent.BuildSensoryTensor`): 16388 floats. That is `goal` (3), then proximity (1), then the 64 × 64 × 4 RGBD frame (RGB / 255, depth as stored). There is no temporal stacking (`temporal_window_size` = 1).

## 3. Manifest

`manifest.json` is written by `ParticipantRecorder.WriteEpisodeManifest` with Unity's `JsonUtility`. Example:

```json
{
  "schema_version": "zarr-v3",
  "recorder_version": "participant_recorder_v2",
  "episode_id": "episode_20260924_141503_120",

  "scene_name": "Comb_Code_482913",
  "layout_typology": "Comb",
  "layout_seed": 482913,

  "source": "human",
  "participant_id": "P07",
  "session_id": "20260924_141455",
  "repetition_index": 2,
  "input_mode": "desktop",

  "task": {
    "start": { "x": -12.5, "y": 1.05, "z": 0.0 },
    "pois":  [ { "x": 0.4, "y": 0.28, "z": -5.6 } ],
    "end":   { "x": 12.5, "y": 1.0, "z": 0.0 }
  },
  "start_jitter_heading_deg": 0.0,

  "frame_count": 412,
  "start_time": 3.21,
  "end_time": 44.43,
  "image_width": 64,
  "image_height": 64,
  "total_pixels": 4096,
  "temporal_window_size": 1,
  "sampling_interval_s": 0.1,
  "max_ray_distance": 50.0,
  "camera_height_m": 1.502,
  "camera_fov_deg": 60.0,

  "arrays": [
    { "name": "rgb",           "shape": [0, 64, 64, 3], "dtype": "uint8" },
    { "name": "depth",         "shape": [0, 64, 64],    "dtype": "float32" },
    { "name": "goal",          "shape": [0, 3],         "dtype": "float32" },
    { "name": "goal_distance", "shape": [0],            "dtype": "float32" },
    { "name": "pose",          "shape": [0, 5],         "dtype": "float32" },
    { "name": "action_vel",    "shape": [0, 3],         "dtype": "float32" },
    { "name": "action_rot",    "shape": [0, 3],         "dtype": "float32" },
    { "name": "timestamp",     "shape": [0],            "dtype": "float64" }
  ]
}
```

| field | filled from |
|---|---|
| `schema_version` | `ZarrDataContract.SchemaVersion`. Must equal `data_contract.DATASET_VERSION` |
| `recorder_version` | `ZarrDataContract.RecorderVersion`. Bump it whenever the recorder's behaviour changes |
| `scene_name` | the Unity scene the recorder is in |
| `layout_typology`, `layout_seed` | the scene's `LayoutInfo` component, which `Tools > Generate Layout` adds to the environment root. Recording refuses to start without exactly one `LayoutInfo` |
| `source` | `"human"` or `"synthetic"`. The recorder's `source` field; `RecordRandomWalksHarness` and `DaggerHarness` set `"synthetic"` |
| `participant_id` | the recorder's `participantId`. Recording refuses to start while it is empty or `"unknown"`. Synthetic episodes use the bot name (`synthetic_rrw_v3`, `synthetic_dagger`) |
| `session_id` | one sitting or one harness run. For humans: the recorder's `sessionId`, or the start time of the first episode in Play mode if left empty. For bots: `rrw_<time>` or `dagger_<time>` |
| `repetition_index` | the recorder's `repetitionIndex`, which increments after every episode. Harnesses set it to the index of the kept episode within the run |
| `input_mode` | `"desktop"` (keyboard + mouse), the only mode in use |
| `task` | world positions, fixed at the start of recording. `start` is the task start marker nearest the mover. `pois` are the POIs in visiting order and `end` is the end marker, both taken from the route `PoiProgressTracker` plans up front, so they are complete even when an episode stops early |
| `start_jitter_heading_deg` | random offset in degrees added to the start heading, which otherwise faces the first target. Set per episode by `RecordRandomWalksHarness` (`bot.startHeadingJitterDegrees`) and `DaggerHarness`; always 0 for humans |
| `frame_count` | number of samples written |
| `start_time`, `end_time` | Unity `Time.time` at start and stop |
| `camera_height_m` | measured camera height above the NavMesh, taken once at sample 10 (or at stop, for shorter episodes). -1 = could not be measured |
| `camera_fov_deg` | the capture camera's actual field of view |
| `arrays` | name, shape (0 = time axis) and dtype of every array in section 2 |

## 4. Checks (`data_contract.check_manifest`)

`ingest_to_zarr.convert_episode` and `load_episodes` (both training scripts) refuse an episode, with a `ValueError` naming every problem, when:

1. `schema_version` is not `zarr-v3`;
2. any field in section 3 is missing or has the wrong JSON type;
3. `recorder_version`, `episode_id`, `scene_name`, `layout_typology`, `participant_id`, `session_id` or `input_mode` is empty, or `participant_id` is `"unknown"`;
4. `source` is not `"human"` or `"synthetic"`, `repetition_index` < 0, or `frame_count` ≤ 0;
5. `task` is not `{start: {x,y,z}, pois: [{x,y,z}, ...], end: {x,y,z}}`;
6. `camera_fov_deg` ≠ 60, `camera_height_m` is outside 1.5 ± 0.03 m, `sampling_interval_s` ≠ 0.1, the image is not 64 × 64, or `temporal_window_size` ≠ 1;
7. `arrays` does not list all eight arrays, or lists one with another shape or dtype than section 2.

Ingestion also fails if any `.bin` file (including `goal_distance.bin`) is missing, or if its size does not equal `frame_count` × the per-sample record size. It then removes the partly written `dataset.zarr`.

## 5. Unified trajectory (evaluation)

Every evaluation source (ImitationAgent under conditions A/B/C, the ShortestPath baseline, held-out humans) is compared as one table, `data_contract.TRAJECTORY_COLUMNS` = `ZarrDataContract.Trajectory.Columns`:

`t, x, y, z, heading_deg, pitch_deg, speed_mps, yaw_rate_deg_s, phase, pois_reached`

One row per 0.1 s of sim time; x, y, z, heading and pitch are the eye/camera pose as in `pose`; `phase` is 0 while walking to a POI and 1 while walking to the end. `EvalHarness` writes it per episode as `traj_config<C>_agent<A>.csv` (`;`-separated), and the imitation model's per-tick outputs go to the optional `modeldebug_config<C>_agent<A>.csv`. `eval_tools/trajectories.py` loads eval runs (`load_eval_run`) and recorded episodes (`load_human_episode`, `load_recordings`) into the same DataFrame. It resamples every source onto the exact 0.1 s grid and recomputes speed and yaw rate from the resampled pose, so one definition applies to all of them.
