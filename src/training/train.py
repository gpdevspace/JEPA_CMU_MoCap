"""Skeleton-JEPA training loop."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import SkeletonPairDataset
from models.jepa import JEPA
from training.losses import kinematic_bone_loss, vicreg_loss
from utils import load_config, resolve_device, ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_skeleton_meta(processed_dir: Path) -> dict:
    with open(processed_dir / "skeleton.json") as f:
        return json.load(f)


def build_ref_bone_lengths(meta: dict, device: torch.device, dtype: torch.dtype) -> dict:
    bone_pairs = meta["bone_pairs"]
    ref_lengths = meta["reference_bone_lengths"]
    ref_map = {}
    for parent, child in bone_pairs:
        key = f"{parent}_{child}"
        ref_map[(parent, child)] = torch.tensor(
            ref_lengths[key], device=device, dtype=dtype
        )
    return ref_map


def ema_momentum_for_step(
    step: int, total_steps: int, start: float, end: float
) -> float:
    if total_steps <= 1:
        return end
    progress = step / (total_steps - 1)
    return start + (end - start) * progress


def train(config_path: Path | None = None) -> None:
    config = load_config(config_path)
    set_seed(config["infrastructure"]["seed"])
    device = resolve_device(config)

    processed_dir = ROOT / config["data"]["processed_dir"]
    meta = load_skeleton_meta(processed_dir)
    pose_dim = meta["pose_dim"]
    num_joints = len(meta["joint_names"])

    dataset = SkeletonPairDataset(processed_dir=processed_dir)
    batch_size = min(config["data"]["batch_size"], len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    training_cfg = config["training"]
    use_latent = training_cfg["use_latent"]
    model = JEPA(
        pose_dim=pose_dim,
        repr_dim=config["model"]["repr_dim"],
        proj_dim=config["model"]["proj_dim"],
        pred_dim=config["model"]["pred_dim"],
        latent_dim=config["model"]["latent_dim"],
        num_classes=config["model"]["num_classes"],
        use_latent=use_latent,
    ).to(device)

    ref_bone_lengths = build_ref_bone_lengths(meta, device, torch.float32)
    vicreg_cfg = training_cfg["vicreg"]
    bone_weight = training_cfg["kinematic"]["bone_weight"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    epochs = training_cfg["epochs"]
    total_steps = max(1, epochs * len(loader))
    global_step = 0

    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y, labels, _k in loader:
            x = x.to(device)
            y = y.to(device)
            labels = labels.to(device)

            momentum = ema_momentum_for_step(
                global_step,
                total_steps,
                training_cfg["ema_momentum_start"],
                training_cfg["ema_momentum_end"],
            )
            model.set_ema_momentum(momentum)

            s_y_hat, s_y = model(x, y, labels if use_latent else None)

            loss = vicreg_loss(
                s_y_hat,
                s_y,
                sim_w=vicreg_cfg["sim_weight"],
                var_w=vicreg_cfg["var_weight"],
                cov_w=vicreg_cfg["cov_weight"],
            )

            if bone_weight > 0:
                pred_poses = model.decode_pose(s_y_hat).view(-1, num_joints, 3)
                loss = loss + bone_weight * kinematic_bone_loss(
                    pred_poses, ref_bone_lengths, meta["bone_pairs"]
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | ema_m={momentum:.6f}")

    ckpt_path = checkpoint_dir / "jepa_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "pose_dim": pose_dim,
            "skeleton_meta": meta,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Skeleton-JEPA")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
