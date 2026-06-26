"""Decoder training script for Skeleton-JEPA."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import SkeletonPairDataset
from models.jepa import JEPA
from utils import load_config, resolve_device, skeleton_fk_args, ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_skeleton_meta(processed_dir: Path) -> dict:
    with open(processed_dir / "skeleton.json") as f:
        return json.load(f)


def train_decoder(config_path: Path | None = None) -> None:
    config = load_config(config_path)
    set_seed(config["infrastructure"]["seed"])
    device = resolve_device(config)

    processed_dir = ROOT / config["data"]["processed_dir"]
    meta = load_skeleton_meta(processed_dir)
    pose_dim = meta["pose_dim"]

    dataset = SkeletonPairDataset(processed_dir=processed_dir, fixed_horizon=1)
    batch_size = min(config["data"]["batch_size"], len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    checkpoint_dir = ROOT / "checkpoints"
    ckpt_path = checkpoint_dir / "jepa_latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No JEPA checkpoint found at {ckpt_path}. Train the JEPA model first."
        )

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

    for p in model.parameters():
        p.requires_grad = False
    for p in model.vis_decoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.vis_decoder.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    epochs = 15
    model.eval()
    model.vis_decoder.train()

    print("Starting FK Spatial Decoder Training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        for x, y, labels, _k in loader:
            pose_t = x.to(device)
            pose_target = y.to(device)

            with torch.no_grad():
                repr_t = model.encode_for_rollout(
                    pose_t, labels.to(device) if model.use_latent else None
                )
                repr_t_norm = F.normalize(repr_t, p=2, dim=-1)

            predicted_pose = model.decode_pose(repr_t_norm)
            mse_loss = F.mse_loss(predicted_pose, pose_target)
            loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()

        avg_loss = epoch_loss / max(1, len(loader))
        avg_mse = epoch_mse / max(1, len(loader))
        print(
            f"Decoder Epoch {epoch + 1}/{epochs} | loss={avg_loss:.6f} | mse={avg_mse:.6f}"
        )

    ckpt["model_state_dict"] = model.state_dict()
    ckpt["skeleton_meta"] = meta
    torch.save(ckpt, ckpt_path)
    print(f"Saved updated checkpoint to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FK Visualization Decoder")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    train_decoder(args.config)


if __name__ == "__main__":
    main()
