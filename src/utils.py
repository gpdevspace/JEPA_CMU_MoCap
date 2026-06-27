"""Shared utilities for Skeleton-JEPA."""

from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_device(config: dict) -> torch.device:
    requested = config["infrastructure"]["device"]
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


ACTION_CLASSES = [
    "walking",
    "jumping",
    "boxing",
]

ACTION_KEYWORDS = {
    "walking": ["walk", "walkturn", "walkstop"],
    "jumping": ["jump", "hop", "leap"],
    "boxing": ["box", "punch", "kick"],
}


def infer_action_label(name: str) -> int:
    lower = name.lower()
    for class_idx, (class_name, keywords) in enumerate(ACTION_KEYWORDS.items()):
        if any(kw in lower for kw in keywords):
            return class_idx
    return len(ACTION_CLASSES) - 1


def skeleton_fk_args(meta: dict) -> dict:
    """Extract FK decoder arguments from skeleton metadata."""
    return {
        "num_joints": meta["num_joints"],
        "parents": meta["parents"],
        "bone_offsets": np.array(meta["bone_offsets"], dtype=np.float32),
    }
