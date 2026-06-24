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
    ):
        config = load_config()
        self.processed_dir = Path(processed_dir or ROOT / config["data"]["processed_dir"])
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

        self._length = sum(c["num_frames"] - self.max_horizon for c in self.clips)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clip = self.rng.choice(self.clips)
        max_t = clip["num_frames"] - self.max_horizon - 1
        t = self.rng.randint(0, max_t)
        k = self.rng.choice(self.horizons)

        poses = clip["poses"]
        x = poses[t].astype(np.float32)
        y = poses[t + k].astype(np.float32)
        label = clip["label"]

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(k, dtype=torch.long),
        )
