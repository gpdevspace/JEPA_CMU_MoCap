"""Linear probe evaluation on frozen encoder representations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F
from data.dataset import SkeletonPairDataset
from models.components import Encoder
from models.jepa import JEPA
from utils import (
    ACTION_CLASSES,
    augment_with_velocity,
    jepa_conditioning_args,
    load_config,
    resolve_device,
    skeleton_fk_args,
    ROOT,
)


def _context_window(poses: np.ndarray, t: int, context_len: int) -> np.ndarray:
    """Velocity-augmented K-frame window ending at t (mirrors the dataset's input).

    Returns [K, 2*pose_dim] (or [2*pose_dim] when context_len==1) so the probe sees
    the same augmented input the v3 encoder was trained on.
    """
    if context_len <= 1:
        return augment_with_velocity(poses[t].astype(np.float32))
    start = max(0, t - context_len + 1)
    w = poses[start : t + 1]
    if len(w) < context_len:
        pad = np.tile(w[[0]], (context_len - len(w), 1))
        w = np.concatenate([pad, w], axis=0)
    return augment_with_velocity(w.astype(np.float32))


class FrameClassificationDataset(Dataset):
    """Action-labelled samples. With context_len > 1 each sample is a [K, pose_dim]
    window so the probe sees the same fused input the v2 encoder was trained on."""

    def __init__(
        self,
        pair_dataset: SkeletonPairDataset,
        samples_per_clip: int = 32,
        context_len: int = 1,
    ):
        self.samples = []
        rng = np.random.default_rng(42)
        for clip in pair_dataset.clips:
            poses = clip["poses"]
            label = clip["label"]
            for _ in range(samples_per_clip):
                t = int(rng.integers(0, len(poses)))
                self.samples.append((_context_window(poses, t, context_len), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        window, label = self.samples[idx]
        return torch.from_numpy(window), torch.tensor(label, dtype=torch.long)


def split_clips(pair_dataset: SkeletonPairDataset, val_ratio: float = 0.2):
    clip_ids = [c["path"].stem for c in pair_dataset.clips]
    if len(clip_ids) == 1:
        return clip_ids, clip_ids
    rng = np.random.default_rng(42)
    rng.shuffle(clip_ids)
    n_val = max(1, int(len(clip_ids) * val_ratio))
    val_ids = set(clip_ids[:n_val])
    train_ids = [cid for cid in clip_ids if cid not in val_ids]
    return train_ids, list(val_ids)


def probe_accuracy(
    encode,
    classifier: nn.Linear,
    dataset: FrameClassificationDataset,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    classifier.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for poses, labels in loader:
            poses = poses.to(device)
            labels = labels.to(device)
            reps = encode(poses)
            preds = classifier(reps).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def train_classifier(
    encode,
    train_data: FrameClassificationDataset,
    repr_dim: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> nn.Linear:
    classifier = nn.Linear(repr_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(epochs):
        classifier.train()
        for poses, labels in loader:
            poses = poses.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                reps = encode(poses)
            loss = criterion(classifier(reps), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return classifier


def run_linear_probe(config_path: Path | None = None) -> None:
    config = load_config(config_path)
    device = resolve_device(config)
    processed_dir = ROOT / config["data"]["processed_dir"]

    with open(processed_dir / "skeleton.json") as f:
        meta = json.load(f)
    pose_dim = meta["pose_dim"]
    repr_dim = config["model"]["repr_dim"]
    num_classes = config["model"]["num_classes"]

    context_len = config["data"].get("context_len", 1)
    base_dataset = SkeletonPairDataset(processed_dir=processed_dir)
    train_ids, val_ids = split_clips(base_dataset)
    train_frames = FrameClassificationDataset(
        SkeletonPairDataset(processed_dir=processed_dir, clip_ids=train_ids),
        context_len=context_len,
    )
    val_frames = FrameClassificationDataset(
        SkeletonPairDataset(processed_dir=processed_dir, clip_ids=val_ids),
        context_len=context_len,
    )

    def build_jepa() -> JEPA:
        return JEPA(
            pose_dim=pose_dim,
            repr_dim=repr_dim,
            proj_dim=config["model"]["proj_dim"],
            pred_dim=config["model"]["pred_dim"],
            latent_dim=config["model"]["latent_dim"],
            num_classes=num_classes,
            use_latent=config["training"]["use_latent"],
            **skeleton_fk_args(meta),
            **jepa_conditioning_args(config),
        ).to(device)

    # Probe the full online representation path (temporal context + encoder), which
    # is the representation v2 actually learns — not the bare single-frame encoder.
    trained_jepa = build_jepa()
    checkpoint_path = ROOT / "checkpoints" / "jepa_latest.pt"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        trained_jepa.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        print("Warning: no checkpoint found; using untrained encoder weights")
    trained_jepa.eval()
    for p in trained_jepa.parameters():
        p.requires_grad = False

    random_jepa = build_jepa().eval()
    for p in random_jepa.parameters():
        p.requires_grad = False

    trained_encode = trained_jepa.encode_repr
    random_encode = random_jepa.encode_repr

    trained_clf = train_classifier(
        trained_encode, train_frames, repr_dim, num_classes, device
    )
    val_acc_trained = probe_accuracy(trained_encode, trained_clf, val_frames, device)
    train_acc_trained = probe_accuracy(trained_encode, trained_clf, train_frames, device)

    random_clf = train_classifier(
        random_encode, train_frames, repr_dim, num_classes, device
    )
    val_acc_random = probe_accuracy(random_encode, random_clf, val_frames, device)
    train_acc_random = probe_accuracy(random_encode, random_clf, train_frames, device)

    print("Linear Probe Results")
    print("=" * 40)
    print(f"Classes: {ACTION_CLASSES}")
    print(
        f"Trained encoder — train acc: {train_acc_trained:.3f}, val acc: {val_acc_trained:.3f}"
    )
    print(
        f"Random encoder  — train acc: {train_acc_random:.3f}, val acc: {val_acc_random:.3f}"
    )
    if val_acc_trained > val_acc_random:
        print("PASS: Trained representation outperforms random baseline on validation.")
    else:
        print("WARN: Trained representation did not beat random baseline on validation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe on frozen encoder")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run_linear_probe(args.config)


if __name__ == "__main__":
    main()
