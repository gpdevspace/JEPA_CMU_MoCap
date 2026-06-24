"""VICReg and kinematic bone-length losses."""

import torch
import torch.nn.functional as F


def vicreg_loss(x, y, sim_w=25.0, var_w=25.0, cov_w=1.0, eps=1e-4):
  # 1. Invariance (Mean Squared Error)
    sim_loss = F.mse_loss(x, y)

    # 2. Variance Regularization
    x_centered = x - x.mean(dim=0)
    y_centered = y - y.mean(dim=0)
    std_x = torch.sqrt(x_centered.var(dim=0) + eps)
    std_y = torch.sqrt(y_centered.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))

    # 3. Covariance Regularization
    batch_size = x.size(0)
    cov_x = (x_centered.T @ x_centered) / (batch_size - 1)
    cov_y = (y_centered.T @ y_centered) / (batch_size - 1)

    diag_mask = torch.eye(cov_x.size(0), device=x.device).bool()
    cov_x = cov_x.clone()
    cov_y = cov_y.clone()
    cov_x[diag_mask] = 0.0
    cov_y[diag_mask] = 0.0
    cov_loss = (cov_x.pow(2).sum() + cov_y.pow(2).sum()) / x.size(1)

    return (sim_w * sim_loss) + (var_w * var_loss) + (cov_w * cov_loss)


def kinematic_bone_loss(pred_poses, ref_bone_lengths, joint_map):
    """
    Penalizes deviations from rigid body structural assumptions.
    pred_poses: Reshaped to (B, num_joints, 3)
    """
    loss = 0.0
    for (parent, child), ref_len in ref_bone_lengths.items():
        p_pos = pred_poses[:, parent, :]
        c_pos = pred_poses[:, child, :]
        predicted_len = torch.norm(p_pos - c_pos, p=2, dim=-1)
        ref_tensor = ref_len if isinstance(ref_len, torch.Tensor) else torch.tensor(
            ref_len, device=pred_poses.device, dtype=pred_poses.dtype
        )
        loss = loss + F.mse_loss(predicted_len, ref_tensor.expand_as(predicted_len))
    return loss
