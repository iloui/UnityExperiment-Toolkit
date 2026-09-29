"""Model and ONNX export shared by the training scripts: the M4 2D-CNN trunk with FiLM goal conditioning
(SimpleImitation2DCNNModel, continuous [forward, yaw, pitch] head) and its discrete-head variant
(DiscreteImitation2DCNNModel, train_2DCNN_discrete.py). The ONNX input/output names are the Sentis contract."""
import os

import onnx
import torch
import torch.nn as nn

from data_contract import DISCRETE_OUTPUT_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, TEMPORAL_WINDOW_SIZE


class SimpleImitation2DCNNModel(nn.Module):
    def __init__(self, img_res: int = 64):
        super().__init__()
        self.H = self.W = img_res

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
        goal_vec = sensory_inputs[:, :4]  # Shape: (Batch, 4)
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


class DiscreteImitation2DCNNModel(SimpleImitation2DCNNModel):
    """M4 trunk, discrete head: motor_actions = [walk_logit, yaw_logit_0..6, pitch_deg_per_sample].
    proprio_size > 0: sensory_inputs ends with that many own-motion values (data_contract.PROPRIO_*). They join the goal
    vector everywhere it is used (FiLM conditioning and the head), so "already turning left" can modulate the same
    visual features the goal does."""

    def __init__(self, img_res: int = 64, proprio_size: int = 0):
        super().__init__(img_res=img_res)
        self.proprio_size = proprio_size
        cond = 4 + proprio_size
        if proprio_size:
            self.goal_encoder[0] = nn.Linear(cond, self.goal_encoder[0].out_features)
            self.network[0] = nn.Linear(64 * 8 * 8 + cond, self.network[0].out_features)
        self.network[-1] = nn.Linear(self.network[-1].in_features, DISCRETE_OUTPUT_SIZE)

    def forward(self, sensory_inputs: torch.Tensor) -> torch.Tensor:
        if not self.proprio_size:
            return super().forward(sensory_inputs)
        pixels = self.H * self.W * 4
        cond = torch.cat([sensory_inputs[:, :4], sensory_inputs[:, 4 + pixels:]], dim=1)
        x = sensory_inputs[:, 4:4 + pixels].reshape(-1, self.H, self.W, 4).permute(0, 3, 1, 2).contiguous()
        vis_map = self.conv2d_encoder(x)
        hidden = self.goal_encoder(cond)
        gamma = self.film_gamma(hidden).unsqueeze(-1).unsqueeze(-1)
        beta = self.film_beta(hidden).unsqueeze(-1).unsqueeze(-1)
        vis_map = vis_map * (1.0 + gamma) + beta
        return self.network(torch.cat([self.flatten(vis_map), cond], dim=1))


def export_onnx(model: nn.Module, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    sensory_input_dim = TEMPORAL_WINDOW_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * 4 + 4 + getattr(model, "proprio_size", 0)
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

