import os
import sys

# Auto-inject python_embedded directory into sys.path so local packages are found
current_dir = os.path.dirname(os.path.abspath(__file__))
python_embedded_dir = os.path.abspath(os.path.join(current_dir, "..", "python_embedded"))
if python_embedded_dir not in sys.path:
    sys.path.insert(0, python_embedded_dir)

import argparse
import tempfile
import pandas as pd
import torch
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description="DesignMind2 Lifecycle Engine - Dummy ML Pipeline")
    parser.add_argument("--csv_input", type=str, default=None, help="Legacy path to the recorded human CSV data")
    parser.add_argument("--episode_dir", type=str, default=None, help="Path to the recorded binary episode directory")
    parser.add_argument("--model_output", type=str, required=True, help="Target path for the final ONNX model")
    return parser.parse_args()

class DummyImitationModel(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(DummyImitationModel, self).__init__()
        # A simple linear network to provide structural weight mapping for Sentis
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh() # Clamps continuous action outputs smoothly
        )
        
    def forward(self, x):
        return self.network(x)

def main():
    args = parse_args()
    print(f"[Python AI Core] Initializing lifecycle loop update execution...")

    dataset_path = args.episode_dir or args.csv_input
    if dataset_path is None:
        print("[Python AI Core] ERROR: Provide either --episode_dir or --csv_input.")
        sys.exit(1)

    print(f"[Python AI Core] Reading dataset matrix from target path: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"[Python AI Core] ERROR: Input file not found: {dataset_path}")
        sys.exit(1)

    # 1. Read CSV to dynamically verify structural alignment with C# data streams
    try:
        if args.episode_dir is not None:
            manifest_path = os.path.join(dataset_path, "manifest.json")
            if os.path.exists(manifest_path):
                print(f"[Python AI Core] Detected binary episode directory: {dataset_path}")
                input_dim = 49155
            else:
                raise FileNotFoundError(f"Manifest missing in episode dir: {manifest_path}")
        else:
            df = pd.read_csv(args.csv_input)
            print(f"[Python AI Core] Loaded data stream with shape: {df.shape}")

            # Determine the total features automatically based on headers
            # Columns 0 to 12 are metadata, goals, and actions (Timestamp, Pos, Goal_Dir, Action_Vel, Action_Rot)
            # All columns starting from index 13 represent the serialized historical vision/depth steps
            feature_columns = [col for col in df.columns if col.startswith('F')]

            # Account for Goal_Dir_X, Goal_Dir_Y, Goal_Dir_Z explicitly (3 elements)
            input_dim = len(feature_columns) + 3
            print(f"[Python AI Core] Computed runtime network input dimension: {input_dim}")

    except Exception as e:
        print(f"[Python AI Core] Failed parsing tabular framework headers: {str(e)}")
        print("[Python AI Core] Falling back to default static tensor shapes for standard testing...")
        # Fallback math matching 3 frames * 4096 pixels * 4 channels + 3 goal dimensions = 49155
        input_dim = 49155

    # 2. Build our structured runtime computational matrix
    output_dim = 3 # Fixed actions: Forward Locomotion, Yaw, Pitch
    model = DummyImitationModel(input_dim=input_dim, output_dim=output_dim)
    model.eval()

    # 3. Formulate structural dummy trace tracking tensor matching batch dimension format
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    # Ensure output target directories are fully generated on the file system
    output_dir = os.path.dirname(os.path.abspath(args.model_output))
    os.makedirs(output_dir, exist_ok=True)

    # 4. IMPLEMENTATION: Atomic Write-To-Temp-And-Rename Pattern
    # This prevents Unity Sentis from parsing bytes while the script is in mid-write
    try:
        print(f"[Python AI Core] Compiling graph onto a secured isolated OS temp boundary...")
        
        # Create a temp file inside the actual target folder to guarantee atomic cross-volume renaming
        with tempfile.NamedTemporaryFile(dir=output_dir, delete=False, suffix=".tmp") as tmp_file:
            temp_file_path = tmp_file.name
        
        # Export the ONNX model structure into the temporary file path
        torch.onnx.export(
            model,
            dummy_input,
            temp_file_path,
            export_params=True,
            opset_version=16, # <-- Shift from 15 to 16/17 for Sentis 2.1.3 compatibility
            do_constant_folding=True,
            input_names=['sensory_inputs'],
            output_names=['motor_actions']
        )
        
        # Atomic switch over the finalized target tracking path
        if os.path.exists(args.model_output):
            os.remove(args.model_output)
            
        os.rename(temp_file_path, args.model_output)
        print(f"[Python AI Core] ATOMIC HOT-SWAP COMMIT SUCCESS: Saved file at -> {args.model_output}")
        
    except Exception as e:
        print(f"[Python AI Core] CRITICAL Error executing atomic asset compilation: {str(e)}")
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        sys.exit(2)

if __name__ == "__main__":
    main()