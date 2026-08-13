import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


DEFAULT_CHUNK_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(description="Convert legacy CSV recordings to a Zarr dataset.")
    parser.add_argument("--csv_input", type=str, required=True, help="Legacy CSV file to convert")
    parser.add_argument("--output_zarr", type=str, default=None, help="Target .zarr directory")
    return parser.parse_args()


def infer_legacy_layout(df):
    feature_columns = [c for c in df.columns if c.startswith("F")]
    if not feature_columns:
        raise ValueError("No legacy feature columns starting with 'F' were found in the CSV.")
    frame_count = len(df)
    # Legacy data stores one row per timestep and a flattened 3-frame observation stack.
    # The old format uses 4 values per pixel: R,G,B,Depth; for 64x64 each frame is 16384 * 4 floats
    # and the flattened section includes 3 frames.
    # We convert the legacy row into a 2D representation for a simplified migration dataset.
    total_per_frame = 64 * 64 * 4
    legacy_dim = len(feature_columns)
    return frame_count, legacy_dim, total_per_frame


def csv_to_simple_zarr(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    frame_count, legacy_dim, total_per_frame = infer_legacy_layout(df)

    # Preserve the original flattened representation as a 2D Zarr array for migration safety.
    legacy_tensor = df[[c for c in df.columns if c.startswith("F")]].to_numpy(dtype=np.float32)

    if os.path.exists(output_path):
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)

    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(output_dir), mode="w")
    root.attrs["source_csv"] = csv_path
    root.attrs["schema_version"] = "legacy-to-zarr-migration-v1"
    root.attrs["frame_count"] = int(frame_count)
    root.attrs["legacy_feature_count"] = int(legacy_dim)

    root.create_dataset(
        "legacy_flat_features",
        data=legacy_tensor,
        chunks=(min(DEFAULT_CHUNK_SIZE, frame_count), legacy_dim),
        overwrite=True,
    )

    # Store the raw metadata as an additional array for downstream inspection.
    metadata = df[[c for c in df.columns if c.startswith(("Timestamp", "Pos_", "Goal_", "Action_"))]].to_numpy(dtype=np.float32)
    if metadata.size > 0:
        root.create_dataset(
            "metadata",
            data=metadata,
            chunks=(min(DEFAULT_CHUNK_SIZE, frame_count), metadata.shape[1]),
            overwrite=True,
        )

    manifest = {
        "schema_version": "legacy-to-zarr-migration-v1",
        "source_csv": csv_path,
        "frame_count": int(frame_count),
        "legacy_feature_count": int(legacy_dim),
        "legacy_total_per_frame": int(total_per_frame),
    }
    root.attrs["manifest"] = json.dumps(manifest)

    print(f"[CSV->Zarr] Converted legacy CSV to Zarr dataset: {output_path}")
    print(f"[CSV->Zarr] Legacy rows: {frame_count}, legacy feature columns: {legacy_dim}")


def main():
    args = parse_args()
    csv_input = os.path.abspath(args.csv_input)
    output_zarr = args.output_zarr
    if output_zarr is None:
        output_zarr = os.path.join(os.path.dirname(csv_input), os.path.splitext(os.path.basename(csv_input))[0] + ".zarr")
    csv_to_simple_zarr(csv_input, output_zarr)


if __name__ == "__main__":
    main()
