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


def compute_jepa_loss(pred_delta, target_repr, online_repr, config):
    """
    pred_delta: [B, repr_dim] -> Predicted shift (Output of Online Predictor)
    target_repr: [B, repr_dim] -> Output of EMA Target Encoder
    online_repr: [B, repr_dim] -> Output of Online Encoder/Projector
    """
    # Calculate the true target shift
    target_delta = target_repr - online_repr
    
    # 1. Directional Invariance Loss (MSE against Target Delta)
    invariance_loss = F.mse_loss(pred_delta, target_delta)
    
    # 2. Variance regularization on the unit sphere of the reconstructed representation.
    # Reconstruct predicted representation: online_repr + pred_delta
    pred_repr = online_repr + pred_delta
    pred_norm = F.normalize(pred_repr, p=2, dim=-1)
    
    # On a d-dim unit sphere each dimension has std ~ 1/sqrt(d) when uniform.
    # We want to push std ABOVE that floor, not just equal to it.
    # Using 1/sqrt(d) as threshold means the penalty barely fires at all.
    # Instead use a fixed threshold of 0.1 — meaningful pressure on a [-1,1] sphere.
    std_pred = torch.sqrt(pred_norm.var(dim=0) + 1e-4)
    variance_loss = torch.mean(F.relu(0.1 - std_pred))
    
    # 3. Total Loss Composition (Keep weights disciplined)
    # Avoid cranking weights to 500. Let the sphere handle the scale.
    total_loss = (config["loss"]["invariance_weight"] * invariance_loss) + (config["loss"]["variance_weight"] * variance_loss)
    
    return total_loss, invariance_loss, variance_loss