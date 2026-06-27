"""Clean JEPA objective: latent prediction + VICReg collapse guard + decoder reconstruction."""

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """Hinge on per-dimension std; pushes the batch to spread out (anti-collapse)."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal covariance; decorrelates the representation dimensions."""
    n, d = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


def compute_jepa_loss(s_y_hat, s_y, s_x, recon, target_poses, config):
    """
    s_y_hat       : predicted future embedding  g(f(x))        [B, D]
    s_y           : EMA target future embedding  f_ema(y)       [B, D] (already detached)
    s_x           : online context embedding     f(x)           [B, D]
    recon         : decoder reconstruction       d(f(.))        [B, pose_dim]
    target_poses  : ground-truth poses for recon                [B, pose_dim]
    """
    loss_cfg = config["training"]["loss"]
    var_w = float(loss_cfg["var_weight"])
    cov_w = float(loss_cfg["cov_weight"])
    recon_w = float(loss_cfg["recon_weight"])

    # 1. JEPA prediction: match the EMA target future embedding.
    pred_loss = F.mse_loss(s_y_hat, s_y)

    # 2. VICReg regularization on the online embeddings (s_y is detached, so it
    #    cannot receive these gradients — apply to s_x and the prediction instead).
    vic_loss = var_w * (variance_loss(s_x) + variance_loss(s_y_hat)) + cov_w * (
        covariance_loss(s_x) + covariance_loss(s_y_hat)
    )

    # 3. Decoder reconstruction (latent -> pose) so the latent is visualizable.
    recon_loss = F.mse_loss(recon, target_poses)

    total = pred_loss + vic_loss + recon_w * recon_loss
    return total, pred_loss, vic_loss, recon_loss
