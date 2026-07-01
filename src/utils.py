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


def augment_with_velocity(window: np.ndarray) -> np.ndarray:
    """Concatenate per-frame velocity (Δ = frameᵢ − frameᵢ₋₁) onto each frame.

    Gives the encoder an explicit motion signal ("standing still" vs "mid-jump")
    without an architecture redesign — the input dim simply doubles to 2·pose_dim.

    Shapes:
      [..., K, D] -> [..., K, 2D]   (velocity computed along the frame axis -2;
                                     the oldest frame's velocity is zero-padded)
      [D]         -> [2D]           (a lone frame has zero velocity)
    """
    w = np.asarray(window, dtype=np.float32)
    if w.ndim == 1:
        return np.concatenate([w, np.zeros_like(w)], axis=-1)
    vel = np.zeros_like(w)
    vel[..., 1:, :] = w[..., 1:, :] - w[..., :-1, :]
    return np.concatenate([w, vel], axis=-1)


def jepa_conditioning_args(config: dict) -> dict:
    """Horizon-conditioning + temporal-context kwargs for the JEPA constructor.

    Pulled from the (possibly checkpoint-saved) config so a loaded model is built
    with the same horizons/context_len it was trained with.
    """
    return {
        "horizons": config["data"]["horizons"],
        "horizon_emb_dim": config["model"].get("horizon_emb_dim", 16),
        "context_len": config["data"].get("context_len", 1),
    }
