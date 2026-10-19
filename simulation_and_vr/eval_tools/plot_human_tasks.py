"""Top-down figure of each layout's human tasks (test task + training tasks) over its occupancy grid.

Reads layouts/<scene>.npz (OccupancyGridExporter) and layouts/<scene>_human_tasks.json (HumanTaskGenerator).

    python plot_human_tasks.py                 # all layouts with a task file -> layouts/human_tasks.png
    python plot_human_tasks.py --out fig.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LAYOUT_DIR = Path(__file__).resolve().parent / "layouts"
TRAIN_COLORS = ["#2a78d6", "#1f9e5a", "#d4930d"]
TEST_COLOR = "#c8322d"


def plot_layout(ax, task_file: Path):
    tasks = json.loads(task_file.read_text())
    grid = np.load(LAYOUT_DIR / f"{tasks['scene_name']}.npz")
    occ = grid["occupancy"]
    ox, oz = np.ravel(grid["origin_xz"])[:2]
    cell = float(np.ravel(grid["cell_size"])[0])
    extent = [ox, ox + occ.shape[1] * cell, oz, oz + occ.shape[0] * cell]
    ax.imshow(occ, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest")

    train_index = 0
    for t in tasks["tasks"]:
        if t["role"] == "test":
            color, lw, label = TEST_COLOR, 2.4, "test"
        else:
            color, lw = TRAIN_COLORS[train_index % len(TRAIN_COLORS)], 1.4
            label = t["id"] + (" (C-excl.)" if t["c_excluded"] else "")
            train_index += 1
        route = np.array([[p["x"], p["z"]] for p in t["route"]])
        if len(route):
            ax.plot(route[:, 0], route[:, 1], color=color, lw=lw, ls="--" if t["c_excluded"] else "-",
                    label=f"{label}: {t['route_length_m']:.1f} m")
        s, p, e = (np.array([t[k]["x"], t[k]["z"]]) for k in ("start", "poi", "end"))
        ax.scatter(*s, marker="s", s=40, color=color, zorder=3)
        ax.scatter(*p, marker="o", s=40, color=color, zorder=3)
        ax.scatter(*e, marker="*", s=90, color=color, zorder=3)
        yaw = np.radians(t["start_yaw_deg"])
        ax.annotate("", xy=(s[0] + 1.5 * np.sin(yaw), s[1] + 1.5 * np.cos(yaw)), xytext=s,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    ax.set_title(f"{tasks['scene_name']}\ntest-route coverage {tasks['test_route_coverage']:.0%}", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="lower left", framealpha=0.85)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=LAYOUT_DIR / "human_tasks.png")
    args = parser.parse_args()
    files = sorted(LAYOUT_DIR.glob("*_human_tasks.json"))
    if not files:
        raise SystemExit(f"No *_human_tasks.json in {LAYOUT_DIR} - run Tools > Thesis > Human Tasks in Unity first.")
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 5.4), squeeze=False)
    for ax, f in zip(axes[0], files):
        plot_layout(ax, f)
    fig.text(0.5, 0.01, "square = start, circle = POI, star = end, arrow = start heading; dashed = left out in condition C",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
