"""Autoregressive latent rollout for Skeleton-JEPA."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.jepa import JEPA
from utils import ACTION_CLASSES, load_config, resolve_device, skeleton_fk_args, ROOT


def load_model(config: dict, device: torch.device) -> tuple[JEPA, dict]:
    processed_dir = ROOT / config["data"]["processed_dir"]
    with open(processed_dir / "skeleton.json") as f:
        meta = json.load(f)

    pose_dim = meta["pose_dim"]
    ckpt_path = ROOT / "checkpoints" / "jepa_latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    use_latent = ckpt["config"]["training"]["use_latent"]

    model = JEPA(
        pose_dim=pose_dim,
        repr_dim=config["model"]["repr_dim"],
        proj_dim=config["model"]["proj_dim"],
        pred_dim=config["model"]["pred_dim"],
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        use_latent=use_latent,
        **skeleton_fk_args(meta),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, meta


def get_start_pose(meta: dict, processed_dir: Path) -> np.ndarray:
    npz_files = sorted(processed_dir.glob("*.npz"))
    data = np.load(npz_files[0])
    poses = data["poses"]
    mid = len(poses) // 2
    return poses[mid]


@torch.no_grad()
def rollout(
    model: JEPA,
    pose_0: np.ndarray,
    action_label: int,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    current_pose = torch.from_numpy(pose_0.astype(np.float32)).to(device).unsqueeze(0)
    label = torch.tensor([action_label], device=device)

    poses = [current_pose.squeeze(0).cpu().numpy()]

    for _ in range(steps):
        repr_t = model.encode_for_rollout(
            current_pose, label if model.use_latent else None
        )
        repr_norm = F.normalize(repr_t, p=2, dim=-1)

        next_pose = model.decode_pose(repr_norm)
        poses.append(next_pose.squeeze(0).cpu().numpy())
        current_pose = next_pose

    return np.stack(poses, axis=0)


def run_rollout(
    config_path: Path | None = None,
    steps: int = 150,
    action_labels: list[int] | None = None,
) -> dict[str, np.ndarray]:
    config = load_config(config_path)
    device = resolve_device(config)
    processed_dir = ROOT / config["data"]["processed_dir"]

    model, meta = load_model(config, device)
    pose_0 = get_start_pose(meta, processed_dir)

    if action_labels is None:
        action_labels = [
            ACTION_CLASSES.index("walking"),
            ACTION_CLASSES.index("jumping"),
            ACTION_CLASSES.index("boxing"),
        ]

    results = {}
    for label_idx in action_labels:
        name = ACTION_CLASSES[label_idx]
        trajectory = rollout(model, pose_0, label_idx, steps, device)
        results[name] = trajectory

    out_dir = ROOT / "outputs" / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, traj in results.items():
        out_path = out_dir / f"rollout_{name}.npz"
        np.savez_compressed(out_path, poses=traj, action=name)
        print(f"Saved rollout for '{name}' to {out_path} ({len(traj)} frames)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoregressive JEPA rollout")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()
    run_rollout(args.config, steps=args.steps)


if __name__ == "__main__":
    main()
