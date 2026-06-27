"""Differentiable forward kinematics for skeletal motion."""

import numpy as np
import torch
import torch.nn as nn

from models.geometry import rotation_6d_to_matrix


class ForwardKinematics(nn.Module):
    def __init__(self, parents: list[int], bone_offsets: np.ndarray):
        super().__init__()
        self.parents = parents
        self.register_buffer(
            "offsets", torch.from_numpy(bone_offsets).float()
        )  # [num_joints, 3]
        self.num_joints = len(parents)

    def forward(self, local_rotations_6d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_rotations_6d: Tensor of shape [Batch, num_joints, 6]
        Returns:
            global_positions: Tensor of shape [Batch, num_joints * 3]
        """
        batch_size = local_rotations_6d.shape[0]
        device = local_rotations_6d.device

        local_rot_mats = rotation_6d_to_matrix(local_rotations_6d)

        global_rot_list: list[torch.Tensor] = []
        global_pos_list: list[torch.Tensor] = []

        for i in range(self.num_joints):
            parent = self.parents[i]
            if parent == -1:
                g_rot = local_rot_mats[:, i]
                g_pos = torch.zeros((batch_size, 3), device=device, dtype=local_rotations_6d.dtype)
            else:
                p_rot = global_rot_list[parent]
                g_rot = torch.matmul(p_rot, local_rot_mats[:, i])
                offset = self.offsets[i].view(1, 3, 1).expand(batch_size, -1, -1)
                rotated_offset = torch.matmul(p_rot, offset).squeeze(-1)
                g_pos = global_pos_list[parent] + rotated_offset

            global_rot_list.append(g_rot)
            global_pos_list.append(g_pos)

        global_positions = torch.stack(global_pos_list, dim=1)
        return global_positions.reshape(batch_size, -1)
