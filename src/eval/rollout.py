"""Teacher-forced prediction (and optional autoregressive rollout) for Skeleton-JEPA."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.jepa import JEPA
from utils import load_config, resolve_device, skeleton_fk_args, ROOT


def load_model(config: dict, device: torch.device) -> tuple[JEPA, dict]:
    processed_dir = ROOT / config["data"]["processed_dir"]
    with open(processed_dir / "skeleton.json") as f:
        meta = json.load(f)

    pose_dim = meta["pose_dim"]
    ckpt_path = ROOT / "checkpoints" / "jepa_latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = JEPA(
        pose_dim=pose_dim,
        repr_dim=config["model"]["repr_dim"],
        proj_dim=config["model"]["proj_dim"],
        pred_dim=config["model"]["pred_dim"],
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        use_latent=ckpt["config"]["training"]["use_latent"],
        **skeleton_fk_args(meta),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, meta


def pick_clip(processed_dir: Path, max_frames: int) -> np.ndarray:
    """Pick the most *dynamic* clip (largest frame-to-frame motion) and trim it."""
    npz_files = sorted(processed_dir.glob("*.npz"))

    def motion(path: Path) -> float:
        poses = np.load(path)["poses"]
        if len(poses) < 2:
            return 0.0
        return float(np.mean(np.linalg.norm(np.diff(poses, axis=0), axis=-1)))

    # Prefer clips long enough to fill the video; among those, the most dynamic.
    min_frames = max(100, max_frames // 2)
    long_enough = [p for p in npz_files if len(np.load(p)["poses"]) >= min_frames]
    candidates = long_enough or npz_files
    best = max(candidates, key=motion)
    poses = np.load(best)["poses"]
    print(f"Showcase clip: {best.stem} ({len(poses)} frames)")
    return poses[:max_frames]


@torch.no_grad()
def predict_teacher_forced(
    model: JEPA, poses: np.ndarray, horizon: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each frame t, predict the pose `horizon` steps ahead from the *real* pose at t.
    Re-anchored on ground truth every step, so it cannot drift.
    Returns (ground_truth, predicted), aligned frame-for-frame.
    """
    x = torch.from_numpy(poses[:-horizon].astype(np.float32)).to(device)
    pred_emb = model.encode_for_rollout(x)          # g(f(x_t)) : predicted future embedding
    pred_poses = model.decode_pose(pred_emb).cpu().numpy()
    gt_poses = poses[horizon:]
    return gt_poses, pred_poses


def run_rollout(
    config_path: Path | None = None,
    steps: int = 150,
    horizon: int = 5,
) -> dict[str, np.ndarray]:
    config = load_config(config_path)
    device = resolve_device(config)
    processed_dir = ROOT / config["data"]["processed_dir"]

    model, _meta = load_model(config, device)
    poses = pick_clip(processed_dir, max_frames=steps + horizon)

    gt, pred = predict_teacher_forced(model, poses, horizon, device)
    results = {"ground_truth": gt, "predicted": pred}

    out_dir = ROOT / "outputs" / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, traj in results.items():
        np.savez_compressed(out_dir / f"rollout_{name}.npz", poses=traj)
        print(f"Saved '{name}' trajectory ({len(traj)} frames)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced JEPA prediction")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    run_rollout(args.config, steps=args.steps, horizon=args.horizon)


if __name__ == "__main__":
    main()
