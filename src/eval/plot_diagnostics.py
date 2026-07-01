"""
viz/plot_diagnostics.py — JEPA model diagnostic plots

Generates six figures saved to outputs/viz/:

  1. pca_embedding.png        — 2D PCA of encoder representations coloured by action
  2. repr_variance.png        — Per-dimension std of s_x (VICReg health check)
  3. eigenspectrum.png        — Covariance eigenvalue spectrum + effective rank + cumulative variance
  4. prediction_quality.png   — JEPA pred MSE vs persistence baseline per action class
  5. cosine_similarity.png    — Cos-sim(s_y_hat, s_y) distribution per action
  6. recon_quality.png        — GT pose joints vs FK-decoded joints scatter

Usage:
  uv run python viz/plot_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn.functional as F

ROOT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT_SRC))

from data.dataset import SkeletonPairDataset          # noqa: E402
from models.jepa import JEPA                          # noqa: E402
from utils import (  # noqa: E402
    ACTION_CLASSES,
    jepa_conditioning_args,
    load_config,
    resolve_device,
    skeleton_fk_args,
    ROOT,
)

# ── palette ───────────────────────────────────────────────────────────────────
BG   = "#0d0d1a"
FG   = "#d0d0f0"
GRID = "#252540"
ACTION_COLORS = ["#00e676", "#ff6d00", "#4a9fff"]   # walking, jumping, boxing
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  FG,
    "xtick.color":      FG,
    "ytick.color":      FG,
    "text.color":       FG,
    "grid.color":       GRID,
    "grid.linewidth":   0.6,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "legend.facecolor": BG,
    "legend.edgecolor": GRID,
})

OUT_DIR = ROOT / "viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── model / data loading ──────────────────────────────────────────────────────

def load_model_and_meta(config: dict, device: torch.device) -> tuple[JEPA, dict]:
    processed_dir = ROOT / config["data"]["processed_dir"]
    with open(processed_dir / "skeleton.json") as f:
        meta = json.load(f)
    ckpt_path = ROOT / "checkpoints" / "jepa_latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = JEPA(
        pose_dim=meta["pose_dim"],
        repr_dim=config["model"]["repr_dim"],
        proj_dim=config["model"]["proj_dim"],
        pred_dim=config["model"]["pred_dim"],
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        use_latent=ckpt["config"]["training"]["use_latent"],
        **skeleton_fk_args(meta),
        **jepa_conditioning_args(ckpt["config"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, meta


@torch.no_grad()
def collect_embeddings(
    model: JEPA, config: dict, device: torch.device
) -> dict:
    """
    Run the full dataset through the model and collect:
      s_x, s_y, s_y_hat, recon, poses_x, poses_y, labels
    Returns a dict of numpy arrays.
    """
    processed_dir = ROOT / config["data"]["processed_dir"]
    dataset = SkeletonPairDataset(processed_dir=processed_dir, fixed_horizon=3)

    sx_list, sy_list, syhat_list = [], [], []
    recon_list, pose_x_list, pose_y_list = [], [], []
    label_list = []

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)

    for x, y, y_vel, labels, k in loader:
        x = x.to(device)
        y = y.to(device)
        y_vel = y_vel.to(device)
        k = k.to(device)
        y_aug = torch.cat([y, y_vel], dim=-1)        # velocity-augmented target [B, 2*pose_dim]
        s_y_hat, s_y, s_x = model(x, y_aug, k)
        recon_y = model.reconstruct(y_aug)           # EMA-decode the augmented target -> raw pose

        # x is velocity-augmented ([B, K, 2*pose_dim] or [B, 2*pose_dim]); recover the
        # raw current pose (last frame, first pose_dim dims) for persistence comparisons.
        pd = y.shape[-1]
        pose_x = x[:, -1, :pd] if x.dim() == 3 else x[:, :pd]

        sx_list.append(s_x.cpu())
        sy_list.append(s_y.cpu())
        syhat_list.append(s_y_hat.cpu())
        recon_list.append(recon_y.cpu())
        pose_x_list.append(pose_x.cpu())
        pose_y_list.append(y.cpu())
        label_list.append(labels)

    return {
        "s_x":    torch.cat(sx_list).numpy(),
        "s_y":    torch.cat(sy_list).numpy(),
        "s_y_hat": torch.cat(syhat_list).numpy(),
        "recon":  torch.cat(recon_list).numpy(),
        "pose_x": torch.cat(pose_x_list).numpy(),
        "pose_y": torch.cat(pose_y_list).numpy(),
        "labels": torch.cat(label_list).numpy(),
    }


# ── 1  PCA embedding ──────────────────────────────────────────────────────────

def plot_pca_embedding(data: dict) -> None:
    from sklearn.decomposition import PCA

    sx = data["s_x"]
    labels = data["labels"]
    num_classes = int(labels.max()) + 1

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(sx)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in range(num_classes):
        mask = labels == cls
        ax.scatter(
            z[mask, 0], z[mask, 1],
            c=ACTION_COLORS[cls % len(ACTION_COLORS)],
            label=ACTION_CLASSES[cls],
            alpha=0.55, s=18, linewidths=0,
        )
        # centroid cross
        cx, cy = z[mask, 0].mean(), z[mask, 1].mean()
        ax.scatter(cx, cy, c=ACTION_COLORS[cls % len(ACTION_COLORS)],
                   s=180, marker="*", zorder=5, edgecolors="white", linewidths=0.8)

    var = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC 1  ({var[0]:.1f}% var)")
    ax.set_ylabel(f"PC 2  ({var[1]:.1f}% var)")
    ax.set_title("Encoder Representations — PCA (action labels not used in training)")
    ax.legend(markerscale=1.8, framealpha=0.3)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    out = OUT_DIR / "pca_embedding.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {out.name}")


# ── 2  Per-dimension variance ─────────────────────────────────────────────────

def plot_repr_variance(data: dict) -> None:
    sx    = data["s_x"]
    syhat = data["s_y_hat"]
    eps   = 1e-4

    std_sx    = np.sqrt(sx.var(axis=0) + eps)
    std_syhat = np.sqrt(syhat.var(axis=0) + eps)

    order = np.argsort(std_sx)[::-1]
    d = len(order)
    idx = np.arange(d)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, std, label, color in zip(
        axes,
        [std_sx[order], std_syhat[order]],
        ["s_x  (encoder output)", "s_ŷ  (predictor output)"],
        ["#00e676", "#ff6d00"],
    ):
        ax.bar(idx, std, color=color, alpha=0.7, width=1.0)
        ax.axhline(1.0, color="white", linewidth=0.9, linestyle="--", alpha=0.6,
                   label="VICReg γ = 1.0")
        ax.set_ylabel("Std dev")
        ax.set_title(f"Per-dimension std — {label}")
        ax.legend(framealpha=0.25)
        ax.grid(True, axis="y", alpha=0.4)
        alive = int((std >= 0.5).sum())
        ax.text(0.98, 0.92, f"{alive}/{d} dims active (std ≥ 0.5)",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                color=FG, alpha=0.8)

    axes[1].set_xlabel("Dimension (sorted by s_x std, descending)")
    fig.tight_layout(h_pad=1.0)
    out = OUT_DIR / "repr_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {out.name}")


# ── 3  Eigenvalue spectrum + effective rank ───────────────────────────────────

def plot_eigenspectrum(data: dict) -> None:
    """
    Two-panel figure:
      Left  — covariance eigenvalue spectrum on a log y-axis for s_x and s_ŷ,
               annotated with effective rank (Roy & Vetterli 2007).
      Right — cumulative explained variance, showing how many dims are needed
               to capture 90% of the total variance.

    Effective rank = exp(H(p)) where p_i = λ_i / Σλ and H is Shannon entropy.
    A value near D (=256) means all dims equally used; near 1 means near-collapse.
    """
    def compute_eigenvalues(z: np.ndarray) -> np.ndarray:
        z_c = z - z.mean(axis=0, keepdims=True)
        # SVD of the data matrix is numerically stabler than eig of the covariance.
        _, s, _ = np.linalg.svd(z_c, full_matrices=False)
        return (s ** 2) / (len(z) - 1)   # eigenvalues of the sample covariance

    def effective_rank(eigenvalues: np.ndarray) -> float:
        p = eigenvalues / eigenvalues.sum()
        p = np.clip(p, 1e-10, None)
        return float(np.exp(-np.sum(p * np.log(p))))

    sx    = data["s_x"]
    syhat = data["s_y_hat"]

    eigs_sx    = compute_eigenvalues(sx)
    eigs_syhat = compute_eigenvalues(syhat)

    er_sx    = effective_rank(eigs_sx)
    er_syhat = effective_rank(eigs_syhat)

    d   = len(eigs_sx)
    idx = np.arange(d)

    cum_sx    = np.cumsum(eigs_sx    / eigs_sx.sum())
    cum_syhat = np.cumsum(eigs_syhat / eigs_syhat.sum())

    k90_sx    = int(np.searchsorted(cum_sx,    0.90)) + 1
    k90_syhat = int(np.searchsorted(cum_syhat, 0.90)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: eigenvalue spectrum (log scale) ──────────────────────────────────
    ax = axes[0]
    ax.semilogy(idx, eigs_sx,    color="#00e676", linewidth=1.5,
                label=f"s_x   eff. rank = {er_sx:.1f} / {d}")
    ax.semilogy(idx, eigs_syhat, color="#ff6d00", linewidth=1.5, linestyle="--",
                label=f"s_ŷ   eff. rank = {er_syhat:.1f} / {d}")
    # vertical markers at effective rank
    ax.axvline(er_sx,    color="#00e676", linewidth=0.9, linestyle=":", alpha=0.55)
    ax.axvline(er_syhat, color="#ff6d00", linewidth=0.9, linestyle=":", alpha=0.55)
    ax.set_xlabel("Eigenvalue index (sorted descending)")
    ax.set_ylabel("Eigenvalue  (log scale)")
    ax.set_title("Covariance eigenvalue spectrum")
    ax.legend(framealpha=0.3)
    ax.grid(True, which="both", alpha=0.3)

    # ── Right: cumulative explained variance ───────────────────────────────────
    ax = axes[1]
    ax.plot(idx, cum_sx    * 100, color="#00e676", linewidth=1.5, label="s_x")
    ax.plot(idx, cum_syhat * 100, color="#ff6d00", linewidth=1.5, linestyle="--",
            label="s_ŷ")
    ax.axhline(90, color="white", linewidth=0.9, linestyle="--", alpha=0.5,
               label="90 % threshold")
    ax.axvline(k90_sx,    color="#00e676", linewidth=0.9, linestyle=":", alpha=0.55)
    ax.axvline(k90_syhat, color="#ff6d00", linewidth=0.9, linestyle=":", alpha=0.55)
    # annotations: how many dims reach 90%
    y_anchor = 55
    for k, color, label in [(k90_sx, "#00e676", "s_x"), (k90_syhat, "#ff6d00", "s_ŷ")]:
        ax.text(k + 2, y_anchor, f"{label}: {k} dims",
                color=color, fontsize=9, va="top")
        y_anchor -= 10
    ax.set_xlabel("Number of dimensions")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Cumulative explained variance")
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    out = OUT_DIR / "eigenspectrum.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {out.name}")


# ── 4  Prediction quality vs persistence ─────────────────────────────────────

def plot_prediction_quality(data: dict) -> None:
    sx    = data["s_x"]
    sy    = data["s_y"]
    syhat = data["s_y_hat"]
    pose_x = data["pose_x"]
    pose_y = data["pose_y"]
    labels = data["labels"]
    num_classes = int(labels.max()) + 1

    # Prediction MSE in embedding space
    pred_mse = ((syhat - sy) ** 2).mean(axis=1)
    # Persistence MSE: predict y by repeating x
    pers_mse = ((sx - sy) ** 2).mean(axis=1)

    # Reconstruction MSE in pose space (decoder quality). recon decodes the
    # target frame y, so compare against pose_y.
    recon_mse_pose = ((data["recon"] - pose_y) ** 2).mean(axis=1)
    pers_mse_pose  = ((pose_x - pose_y) ** 2).mean(axis=1)

    action_names = [ACTION_CLASSES[i] for i in range(num_classes)]
    x_pos = np.arange(num_classes)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: embedding-space prediction quality
    ax = axes[0]
    jepa_means  = [pred_mse[labels == c].mean()  for c in range(num_classes)]
    pers_means  = [pers_mse[labels == c].mean()  for c in range(num_classes)]
    jepa_stds   = [pred_mse[labels == c].std()   for c in range(num_classes)]
    pers_stds   = [pers_mse[labels == c].std()   for c in range(num_classes)]

    b1 = ax.bar(x_pos - width/2, jepa_means, width, yerr=jepa_stds,
                label="JEPA prediction", color="#4a9fff", alpha=0.85,
                error_kw=dict(ecolor=FG, alpha=0.5, capsize=4))
    b2 = ax.bar(x_pos + width/2, pers_means, width, yerr=pers_stds,
                label="Persistence baseline", color="#888888", alpha=0.75,
                error_kw=dict(ecolor=FG, alpha=0.5, capsize=4))
    ax.set_xticks(x_pos); ax.set_xticklabels(action_names)
    ax.set_ylabel("MSE (embedding space)")
    ax.set_title("Prediction quality vs persistence\n(lower is better)")
    ax.legend(framealpha=0.3)
    ax.grid(True, axis="y", alpha=0.4)

    # Right: reconstruction quality (pose space)
    ax = axes[1]
    rec_means  = [recon_mse_pose[labels == c].mean() for c in range(num_classes)]
    rec_stds   = [recon_mse_pose[labels == c].std()  for c in range(num_classes)]
    per_means  = [pers_mse_pose[labels == c].mean()  for c in range(num_classes)]

    ax.bar(x_pos - width/2, rec_means, width, yerr=rec_stds,
           label="FK reconstruction", color="#00e676", alpha=0.85,
           error_kw=dict(ecolor=FG, alpha=0.5, capsize=4))
    ax.bar(x_pos + width/2, per_means, width,
           label="Persistence (pose space)", color="#888888", alpha=0.75)
    ax.set_xticks(x_pos); ax.set_xticklabels(action_names)
    ax.set_ylabel("MSE (pose space, Cartesian cm²)")
    ax.set_title("FK decoder reconstruction quality\n(lower is better)")
    ax.legend(framealpha=0.3)
    ax.grid(True, axis="y", alpha=0.4)

    fig.tight_layout()
    out = OUT_DIR / "prediction_quality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {out.name}")


# ── 4  Cosine similarity distribution ────────────────────────────────────────

def plot_cosine_similarity(data: dict) -> None:
    syhat  = torch.from_numpy(data["s_y_hat"])
    sy     = torch.from_numpy(data["s_y"])
    sx     = torch.from_numpy(data["s_x"])
    labels = data["labels"]
    num_classes = int(labels.max()) + 1

    cos_pred = F.cosine_similarity(syhat, sy, dim=1).numpy()   # predictor → target
    cos_pers = F.cosine_similarity(sx,    sy, dim=1).numpy()   # persistence → target

    fig, axes = plt.subplots(1, num_classes, figsize=(14, 4.5), sharey=True)
    bins = np.linspace(-1, 1, 50)

    for cls in range(num_classes):
        ax = axes[cls]
        mask = labels == cls
        ax.hist(cos_pers[mask], bins=bins, color="#888888", alpha=0.65,
                label="Persistence", density=True)
        ax.hist(cos_pred[mask], bins=bins, color=ACTION_COLORS[cls], alpha=0.75,
                label="JEPA pred", density=True)

        m_pred = cos_pred[mask].mean()
        m_pers = cos_pers[mask].mean()
        ax.axvline(m_pred, color=ACTION_COLORS[cls], linewidth=1.8, linestyle="--")
        ax.axvline(m_pers, color="#aaaaaa", linewidth=1.2, linestyle=":")
        ax.set_title(ACTION_CLASSES[cls].capitalize())
        ax.set_xlabel("Cosine similarity  s_ŷ · s_y")
        ax.text(0.05, 0.93, f"JEPA μ={m_pred:.3f}\nPers μ={m_pers:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                color=FG, alpha=0.9,
                bbox=dict(facecolor=BG, alpha=0.5, edgecolor=GRID, boxstyle="round,pad=0.3"))
        ax.grid(True, alpha=0.3)
        if cls == 0:
            ax.set_ylabel("Density")
            ax.legend(fontsize=9, framealpha=0.3)

    fig.suptitle("Cosine similarity: predicted embedding vs EMA target embedding",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "cosine_similarity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {out.name}")


# ── 5  Reconstruction scatter ─────────────────────────────────────────────────

def plot_recon_quality(data: dict) -> None:
    pose_x = data["pose_x"]    # [N, 93]
    recon  = data["recon"]     # [N, 93]
    labels = data["labels"]
    num_classes = int(labels.max()) + 1

    rng = np.random.default_rng(0)
    # Sample up to 2000 points to keep the scatter readable
    idx = rng.choice(len(pose_x), size=min(2000, len(pose_x)), replace=False)
    gt_pts = pose_x[idx].ravel()
    rc_pts = recon[idx].ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: overall scatter
    ax = axes[0]
    ax.scatter(gt_pts, rc_pts, alpha=0.04, s=3, color="#4a9fff", linewidths=0)
    lim = max(np.abs(gt_pts).max(), np.abs(rc_pts).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "w--", linewidth=1.0, alpha=0.7, label="y = x (perfect)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("GT joint position")
    ax.set_ylabel("FK-decoded position")
    ax.set_title("Reconstruction scatter (all joints, all frames)")
    ax.legend(fontsize=9, framealpha=0.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Compute per-dim R² for the annotation
    ss_res = ((pose_x - recon) ** 2).sum()
    ss_tot = ((pose_x - pose_x.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    ax.text(0.04, 0.96, f"R² = {r2:.4f}", transform=ax.transAxes,
            va="top", color=FG, fontsize=10,
            bbox=dict(facecolor=BG, alpha=0.5, edgecolor=GRID, boxstyle="round,pad=0.3"))

    # Right: per-action reconstruction MSE
    ax = axes[1]
    per_sample_mse = ((pose_x - recon) ** 2).mean(axis=1)
    positions = np.arange(num_classes)
    for cls in positions:
        mask = labels == cls
        vals = per_sample_mse[mask]
        ax.boxplot(
            vals, positions=[cls], widths=0.4,
            patch_artist=True,
            boxprops=dict(facecolor=ACTION_COLORS[cls], alpha=0.6),
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=FG, alpha=0.6),
            capprops=dict(color=FG, alpha=0.6),
            flierprops=dict(marker=".", color=FG, alpha=0.15, markersize=3),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([ACTION_CLASSES[c] for c in range(num_classes)])
    ax.set_ylabel("Reconstruction MSE per sample")
    ax.set_title("FK reconstruction error by action class")
    ax.grid(True, axis="y", alpha=0.35)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    fig.tight_layout()
    out = OUT_DIR / "recon_quality.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  saved {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    device = resolve_device(config)

    print("Loading model …")
    model, _meta = load_model_and_meta(config, device)

    print("Collecting embeddings over full dataset …")
    data = collect_embeddings(model, config, device)
    n = len(data["labels"])
    print(f"  {n} samples  |  s_x shape: {data['s_x'].shape}")

    print("Generating plots …")
    plot_pca_embedding(data)
    plot_repr_variance(data)
    plot_eigenspectrum(data)
    plot_prediction_quality(data)
    plot_cosine_similarity(data)
    plot_recon_quality(data)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
