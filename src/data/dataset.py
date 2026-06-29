"""Skeleton context-target pair dataset for JEPA training."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_config, ROOT


class SkeletonPairDataset(Dataset):
    def __init__(
        self,
        processed_dir: Path | None = None,
        horizons: list[int] | None = None,
        seed: int = 42,
        clip_ids: list[str] | None = None,
        fixed_horizon: int | None = None,
    ):
        config = load_config()
        self.processed_dir = Path(processed_dir or ROOT / config["data"]["processed_dir"])
        self.context_len = config["data"].get("context_len", 1)
        self.fixed_horizon = fixed_horizon
        if fixed_horizon is not None:
            self.horizons = [fixed_horizon]
        else:
            self.horizons = horizons or config["data"]["horizons"]
        self.max_horizon = max(self.horizons)
        self.rng = random.Random(seed)

        meta_path = self.processed_dir / "skeleton.json"
        with open(meta_path) as f:
            self.skeleton_meta = json.load(f)

        all_npz = sorted(self.processed_dir.glob("*.npz"))
        if clip_ids is not None:
            all_npz = [p for p in all_npz if p.stem in clip_ids]

        self.clips: list[dict] = []
        for npz_path in all_npz:
            data = np.load(npz_path)
            poses = data["poses"]
            if len(poses) <= self.max_horizon:
                continue
            self.clips.append(
                {
                    "path": npz_path,
                    "poses": poses,
                    "label": int(data["label"]),
                    "num_frames": len(poses),
                }
            )

        if not self.clips:
            raise RuntimeError(f"No valid clips found in {self.processed_dir}")

        # Deterministic flat index -> (clip_idx, t) map for full, even epoch coverage.
        # Every valid start frame t in [0, num_frames - max_horizon) is one sample.
        self.index_map: list[tuple[int, int]] = []
        for clip_idx, clip in enumerate(self.clips):
            for t in range(clip["num_frames"] - self.max_horizon):
                self.index_map.append((clip_idx, t))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_idx, t = self.index_map[idx]
        clip = self.clips[clip_idx]

        # Deterministic horizon per index (cycles through the configured horizons).
        k = self.horizons[idx % len(self.horizons)]

        poses = clip["poses"]

        # Context: the K frames ending at t (Phase 2). Left-pad with the first
        # available frame when there isn't enough history. K == 1 reproduces the
        # original single-frame [pose_dim] behaviour (squeezed below).
        K = self.context_len
        start = t - K + 1
        if start < 0:
            window = poses[0 : t + 1]
            pad = np.tile(window[[0]], (K - len(window), 1))
            window = np.concatenate([pad, window], axis=0)
        else:
            window = poses[start : t + 1]            # [K, pose_dim]

        x = window.astype(np.float32)                # [K, pose_dim]
        if K == 1:
            x = x[0]                                  # [pose_dim] (back-compat)
        y = poses[t + k].astype(np.float32)          # [pose_dim] target, unchanged
        label = clip["label"]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(k, dtype=torch.long),
        )