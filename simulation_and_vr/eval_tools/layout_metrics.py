#!/usr/bin/env python3
"""Wall distance and wall contacts from a layout's occupancy grid, the same way for every trajectory source.

The grid comes from Unity (OccupancyGridExporter, run by Tools > Thesis > Prepare Scene) as
eval_tools/layouts/<scene_name>.npz:
  occupancy        (H, W) bool   Architecture_Obstacle between floor + 0.1 m and eye height (1.5 m)
  origin_xz        (2,)          world x, z of the grid's corner; cell [r, c] covers
                                 x in [ox + c*cell, ox + (c+1)*cell), z in [oz + r*cell, oz + (r+1)*cell)
  cell_size        ()            metres (0.05)
  start, end       (3,)          task markers on the NavMesh (world x, y, z)
  pois             (P, 3)
  segment_lengths  (P+1,)        NavMesh shortest path per segment Start -> POI_0 -> ... -> End
  shortest_path    (K, 3)        corners of that path
  plus scene_name, layout_typology, layout_seed, floor_y, slice_y, navmesh_bake_radius, body_radius, arrival_radius.

Definitions (thesis metrics):
  wall distance  = distance from the body centre (trajectory x, z) to the nearest occupied cell's edge, from
                   scipy.ndimage.distance_transform_edt minus half a cell, looked up at the cell containing the point
                   (accurate to about one cell, 5 cm; walls are rasterised conservatively, so it errs short).
  contact        = wall distance < CONTACT_DISTANCE_M (body radius 0.25 + 0.05 m).
  contact event  = a run of contact samples; a new event starts only after >= EVENT_GAP_S without contact.

Trajectories: human/synthetic episodes (dataset.zarr or the raw pose.bin + timestamp.bin; pose is the camera, which sits
above the body centre) and EvalHarness ticks_config<C>_agent<A>.csv (t, x, z).

Usage:
  python layout_metrics.py plot  LAYOUT.npz [--episode DIR ...] [--ticks CSV ...] [-o out.png]
  python layout_metrics.py stats LAYOUT.npz [--episode DIR ...] [--ticks CSV ...]
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

LAYOUTS_DIR = Path(__file__).resolve().parent / "layouts"
FORMAT_VERSION = 1
BODY_RADIUS_M = 0.25  # ZarrDataContract.RecordingBody.Radius
CONTACT_MARGIN_M = 0.05
CONTACT_DISTANCE_M = BODY_RADIUS_M + CONTACT_MARGIN_M
EVENT_GAP_S = 1.0


@dataclass
class Layout:
    scene_name: str
    occupancy: np.ndarray        # (H, W) bool
    origin_xz: np.ndarray        # (2,)
    cell_size: float
    start: np.ndarray            # (3,)
    pois: np.ndarray             # (P, 3)
    end: np.ndarray              # (3,)
    segment_lengths: np.ndarray  # (P+1,)
    shortest_path: np.ndarray    # (K, 3)
    wall_distance_grid: np.ndarray  # (H, W) metres to the nearest occupied cell edge, 0 inside obstacles

    @property
    def extent(self):
        """imshow extent (x0, x1, z0, z1) with origin='lower'."""
        h, w = self.occupancy.shape
        ox, oz = self.origin_xz
        return (ox, ox + w * self.cell_size, oz, oz + h * self.cell_size)

    def wall_distance(self, xz: np.ndarray) -> np.ndarray:
        """Wall distance (m) at world points xz (N, 2). Points outside the grid get 0 (they are outside the building)."""
        xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
        cols = np.floor((xz[:, 0] - self.origin_xz[0]) / self.cell_size).astype(np.int64)
        rows = np.floor((xz[:, 1] - self.origin_xz[1]) / self.cell_size).astype(np.int64)
        h, w = self.occupancy.shape
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        out = np.zeros(len(xz))
        out[inside] = self.wall_distance_grid[rows[inside], cols[inside]]
        return out


def load_layout(path_or_scene: str | Path) -> Layout:
    """Load eval_tools/layouts/<scene>.npz (a path, or a scene name looked up in LAYOUTS_DIR)."""
    path = Path(path_or_scene)
    if path.suffix != ".npz":
        path = LAYOUTS_DIR / f"{path_or_scene}.npz"
    with np.load(path) as f:
        version = int(f["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {version}, expected {FORMAT_VERSION}. Re-run Prepare Scene.")
        occupancy = f["occupancy"].astype(bool)
        cell = float(f["cell_size"])
        # EDT gives the distance from each free cell centre to the nearest occupied cell centre; minus half a cell is the
        # distance to that cell's edge.
        dist = np.clip(distance_transform_edt(~occupancy) * cell - 0.5 * cell, 0.0, None)
        dist[occupancy] = 0.0
        return Layout(
            scene_name=str(f["scene_name"]),
            occupancy=occupancy,
            origin_xz=f["origin_xz"].astype(np.float64),
            cell_size=cell,
            start=f["start"],
            pois=f["pois"].reshape(-1, 3),
            end=f["end"],
            segment_lengths=f["segment_lengths"],
            shortest_path=f["shortest_path"].reshape(-1, 3),
            wall_distance_grid=dist,
        )


# ----------------------------------------------------------------------------------------------------------------------
# Trajectories -> (t seconds, xz (N, 2))
# ----------------------------------------------------------------------------------------------------------------------

def trajectory_from_episode(episode_dir: str | Path):
    """(t, xz, scene_name) of a recorded episode: dataset.zarr if it exists, else the raw pose.bin / timestamp.bin."""
    ep = Path(episode_dir)
    scene = json.loads((ep / "manifest.json").read_text(encoding="utf-8"))["scene_name"]
    if (ep / "dataset.zarr").exists():
        import zarr
        root = zarr.open_group(str(ep / "dataset.zarr"), mode="r")
        pose = np.asarray(root["pose"][:], dtype=np.float64)
        t = np.asarray(root["timestamp"][:], dtype=np.float64)
    else:
        pose = np.fromfile(ep / "pose.bin", dtype="<f4").reshape(-1, 3).astype(np.float64)
        t = np.fromfile(ep / "timestamp.bin", dtype="<f8")
    return t - t[0], pose[:, [0, 2]], scene


def trajectory_from_ticks(ticks_csv: str | Path):
    """(t, xz) from an EvalHarness ticks_config<C>_agent<A>.csv (';'-separated, columns t, x, z)."""
    t, xz = [], []
    with open(ticks_csv, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            try:
                t.append(float(row["t"]))
                xz.append((float(row["x"]), float(row["z"])))
            except (TypeError, ValueError, KeyError):
                continue
    t = np.asarray(t)
    return t - (t[0] if len(t) else 0.0), np.asarray(xz, dtype=np.float64).reshape(-1, 2)


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------

def contact_events(t: np.ndarray, in_contact: np.ndarray, gap_s: float = EVENT_GAP_S) -> list[tuple[float, float]]:
    """[(start_t, end_t)] of contact events: runs of contact samples, merged while the gap between them is < gap_s."""
    events: list[list[float]] = []
    for ti in np.asarray(t)[np.asarray(in_contact, dtype=bool)]:
        if events and ti - events[-1][1] < gap_s:
            events[-1][1] = ti
        else:
            events.append([ti, ti])
    return [(a, b) for a, b in events]


def wall_metrics(layout: Layout, t: np.ndarray, xz: np.ndarray) -> dict:
    """Per-trajectory wall-distance and contact summary."""
    d = layout.wall_distance(xz)
    contact = d < CONTACT_DISTANCE_M
    events = contact_events(t, contact)
    dt = np.diff(t, append=t[-1]) if len(t) else np.zeros(0)
    return {
        "samples": int(len(d)),
        "wall_dist_mean_m": float(d.mean()) if len(d) else float("nan"),
        "wall_dist_median_m": float(np.median(d)) if len(d) else float("nan"),
        "wall_dist_min_m": float(d.min()) if len(d) else float("nan"),
        "contact_fraction": float(contact.mean()) if len(d) else float("nan"),
        "contact_time_s": float(dt[contact].sum()),
        "contact_events": len(events),
    }


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------

def _gather(args, layout: Layout):
    trajs = []
    for ep in args.episode or []:
        t, xz, scene = trajectory_from_episode(ep)
        if scene != layout.scene_name:
            print(f"[warn] {ep} was recorded in '{scene}', not '{layout.scene_name}'")
        trajs.append((Path(ep).name, t, xz))
    for tick in args.ticks or []:
        t, xz = trajectory_from_ticks(tick)
        trajs.append((Path(tick).parent.name + "/" + Path(tick).stem, t, xz))
    return trajs


def cmd_stats(args):
    layout = load_layout(args.layout)
    print(f"{layout.scene_name}: grid {layout.occupancy.shape[1]}x{layout.occupancy.shape[0]} @ {layout.cell_size * 100:.0f} cm, "
          f"shortest path {layout.segment_lengths.sum():.2f} m (segments {np.round(layout.segment_lengths, 2).tolist()})")
    for name, t, xz in _gather(args, layout):
        m = wall_metrics(layout, t, xz)
        print(f"  {name}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layout = load_layout(args.layout)
    fig, ax = plt.subplots(figsize=(10, 10 * layout.occupancy.shape[0] / max(1, layout.occupancy.shape[1]) + 1))
    ax.imshow(np.where(layout.occupancy, np.nan, layout.wall_distance_grid), origin="lower", extent=layout.extent,
              cmap="Greys_r", alpha=0.35, interpolation="nearest")
    ax.imshow(np.ma.masked_where(~layout.occupancy, layout.occupancy), origin="lower", extent=layout.extent,
              cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    sp = layout.shortest_path
    ax.plot(sp[:, 0], sp[:, 2], "--", color="tab:green", lw=1.5, label=f"NavMesh shortest ({layout.segment_lengths.sum():.1f} m)")
    for name, t, xz in _gather(args, layout):
        d = layout.wall_distance(xz)
        line, = ax.plot(xz[:, 0], xz[:, 1], lw=1.2, label=name)
        contact = d < CONTACT_DISTANCE_M
        if contact.any():
            ax.scatter(xz[contact, 0], xz[contact, 1], s=10, color=line.get_color(), edgecolors="red", linewidths=0.8, zorder=4)
    ax.scatter(*layout.start[[0, 2]], marker="o", s=80, color="tab:blue", zorder=5, label="Start")
    ax.scatter(layout.pois[:, 0], layout.pois[:, 2], marker="*", s=160, color="tab:orange", zorder=5, label="POI")
    ax.scatter(*layout.end[[0, 2]], marker="s", s=80, color="tab:red", zorder=5, label="End")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.set_title(f"{layout.scene_name}  (red-edged points: wall distance < {CONTACT_DISTANCE_M:.2f} m)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    out = Path(args.output or LAYOUTS_DIR / f"{layout.scene_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plot", cmd_plot), ("stats", cmd_stats)):
        s = sub.add_parser(name)
        s.add_argument("layout", help="eval_tools/layouts/<scene>.npz, or just the scene name")
        s.add_argument("--episode", nargs="*", help="recorded episode folder(s)")
        s.add_argument("--ticks", nargs="*", help="EvalHarness ticks_config<C>_agent<A>.csv file(s)")
        if name == "plot":
            s.add_argument("-o", "--output", help="PNG path (default: eval_tools/layouts/<scene>.png)")
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
