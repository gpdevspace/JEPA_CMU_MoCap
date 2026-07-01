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


def compute_jepa_loss(
    s_y_hat, s_y, s_x, recon, target_poses, config, pred_recon=None
):
    """
    s_y_hat       : predicted future embedding  g(f(x))        [B, D]
    s_y           : EMA target future embedding  f_ema(y)       [B, D] (already detached)
    s_x           : online context embedding     f(x)           [B, D]
    recon         : decoder reconstruction       d(f_ema(y))    [B, pose_dim] (decoder faithfulness)
    target_poses  : ground-truth poses (y) for the recon loss    [B, pose_dim]
    pred_recon    : decoder applied to the DETACHED predictor output  d(sg(s_y_hat))  [B, pose_dim]

    True JEPA: the predictor optimizes only latent MSE + VICReg. The decoder is a
    separate evaluation artifact. `recon` trains it to invert real EMA-target
    embeddings; `pred_recon` (fed a stop-gradient s_y_hat) additionally trains it to
    invert the predictor's own outputs so decoded PREDICTIONS land on the pose
    manifold — all without any gradient reaching the predictor.
    """
    loss_cfg = config["training"]["loss"]
    var_w = float(loss_cfg["var_weight"])
    cov_w = float(loss_cfg["cov_weight"])
    recon_w = float(loss_cfg["recon_weight"])

    # 1. JEPA prediction: match the EMA target future embedding (latent space).
    pred_loss = F.mse_loss(s_y_hat, s_y)

    # 2. VICReg regularization on the ONLINE CONTEXT embedding s_x only.
    #    Applying it to s_y_hat as well inflated the predictor's variance (std ~0.56)
    #    above the EMA target manifold (std ~0.36), pushing predictor outputs off the
    #    decoder's manifold so decoded predictions collapsed to ~mean pose. Anti-collapse
    #    still holds: VICReg spreads s_x, and the EMA stop-gradient target keeps the
    #    predictor honest (BYOL-style) — leaving the predictor free to land exactly on
    #    the EMA manifold the decoder was trained to invert.
    vic_loss = var_w * variance_loss(s_x) + cov_w * covariance_loss(s_x)

    # 3. Decoder reconstruction (latent -> pose) so the latent is visualizable.
    #    Trains the decoder only; gradients stop at the frozen EMA embedding.
    recon_loss = F.mse_loss(recon, target_poses)

    # 3b. Decode the DETACHED predictor output toward the true future pose. Teaches
    #     the decoder to invert predictor outputs (closing the off-manifold gap) with
    #     zero gradient to the predictor — keeps JEPA latent-space purity intact.
    if pred_recon is not None:
        recon_loss = recon_loss + F.mse_loss(pred_recon, target_poses)

    total = pred_loss + vic_loss + recon_w * recon_loss

    return total, pred_loss, vic_loss, recon_loss
