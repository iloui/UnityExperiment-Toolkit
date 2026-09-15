"""
Parity-test helper: prints the exact `sensory_inputs` tensor
train_onAllRecordings_2DCNNModel.SingleEpisodeDataset builds for the first temporal
window of one recorded episode, so it can be diffed against ImitationAgent.BuildSensoryTensor's
output on the same synthetic input from Unity (see Editor/SensoryTensorParityTest.cs).

Not part of the production training pipeline - this script only exists to give the parity
test a way to invoke the real SingleEpisodeDataset code path without spinning up training.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_onAllRecordings_2DCNNModel import SingleEpisodeDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Print the first sensory_inputs window of one recorded episode.")
    parser.add_argument("--episode_dir", type=str, required=True, help="Directory containing dataset.zarr")
    parser.add_argument("--out", type=str, required=True, help="Path to write the flat float32 tensor as newline-separated text")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = SingleEpisodeDataset(Path(args.episode_dir))
    if len(dataset) == 0:
        print(f"[print_first_window_tensor] ERROR: episode at {args.episode_dir} has fewer frames than the temporal window.")
        sys.exit(1)

    sensory_inputs, _motor_actions = dataset[0]
    values = sensory_inputs.numpy().tolist()

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(repr(v) for v in values))

    print(f"[print_first_window_tensor] Wrote {len(values)} values to {args.out}")


if __name__ == "__main__":
    main()
