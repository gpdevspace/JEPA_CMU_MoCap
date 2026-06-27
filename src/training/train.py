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
from training.losses import compute_jepa_loss
from utils import load_config, resolve_device, skeleton_fk_args, ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_skeleton_meta(processed_dir: Path) -> dict:
    with open(processed_dir / "skeleton.json") as f:
        return json.load(f)


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
        **skeleton_fk_args(meta),
    ).to(device)

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
        for x, y, _labels, _k in loader:
            x = x.to(device)
            y = y.to(device)

            momentum = ema_momentum_for_step(
                global_step,
                total_steps,
                training_cfg["ema_momentum_start"],
                training_cfg["ema_momentum_end"],
            )
            model.set_ema_momentum(momentum)

            # JEPA forward: predict the EMA target future embedding from the context.
            s_y_hat, s_y, s_x = model(x, y)

            # Joint FK-decoder reconstruction (detached latent: trains the decoder
            # to read the representation, not reshape it).
            recon = torch.cat([model.reconstruct(x), model.reconstruct(y)], dim=0)
            target_poses = torch.cat([x, y], dim=0)

            loss, pred_loss, vic_loss, rec_loss = compute_jepa_loss(
                s_y_hat, s_y, s_x, recon, target_poses, config
            )

            if global_step % 20 == 0:
                print(
                    f"  Step {global_step} | loss={loss.item():.4f} | "
                    f"pred={pred_loss.item():.4f} | vic={vic_loss.item():.4f} | "
                    f"rec={rec_loss.item():.4f}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / max(1, len(loader))
        print(
            f"Epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} | "
            f"pred={pred_loss.item():.4f} | vic={vic_loss.item():.4f} | "
            f"rec={rec_loss.item():.4f} | ema_m={momentum:.6f}"
        )

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