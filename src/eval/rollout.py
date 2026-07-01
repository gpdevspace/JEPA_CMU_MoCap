"""Teacher-forced prediction (and optional autoregressive rollout) for Skeleton-JEPA."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.jepa import JEPA
from utils import (
    augment_with_velocity,
    jepa_conditioning_args,
    load_config,
    resolve_device,
    skeleton_fk_args,
    ROOT,
)


def build_context_windows(poses: np.ndarray, context_len: int, horizon: int) -> np.ndarray:
    """Build velocity-augmented sliding windows ending at each usable frame t.

    Mirrors the dataset's left-padding so eval-time inputs match training, then
    appends per-frame velocity: [T, K, 2*pose_dim] (or [T, 2*pose_dim] when K==1).
    """
    usable = poses[: len(poses) - horizon]
    if context_len <= 1:
        return augment_with_velocity(usable.astype(np.float32))

    windows = []
    for t in range(len(usable)):
        start = max(0, t - context_len + 1)
        w = poses[start : t + 1]
        if len(w) < context_len:
            pad = np.tile(w[[0]], (context_len - len(w), 1))
            w = np.concatenate([pad, w], axis=0)
        windows.append(w)
    return augment_with_velocity(np.stack(windows).astype(np.float32))


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
        **jepa_conditioning_args(ckpt["config"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, meta


def pick_clip(
    processed_dir: Path, max_frames: int, clip_name: str | None = None
) -> np.ndarray:
    """
    Return poses for the chosen clip (trimmed to max_frames).
    If clip_name is given, load that specific clip.
    Otherwise, auto-pick: prefer fully upright, long clips with high motion.
    """
    if clip_name is not None:
        path = processed_dir / f"{clip_name}.npz"
        poses = np.load(path)["poses"]
        print(f"Clip: {clip_name} ({len(poses)} frames)")
        return poses[:max_frames]

    npz_files = sorted(processed_dir.glob("*.npz"))

    def score(path: Path) -> float:
        d = np.load(path)
        poses = d["poses"]
        if len(poses) < 2:
            return -1.0
        # Disqualify clips where the skeleton goes upside-down.
        # Head is joint 16 in CMU 31-joint skeleton; Y < 0 means head below hips.
        head_y = poses.reshape(-1, 31, 3)[:, 16, 1]
        if (head_y < 0).any():
            return -1.0
        motion = float(np.mean(np.linalg.norm(np.diff(poses, axis=0), axis=-1)))
        return motion

    min_frames = max(100, max_frames // 2)
    long_enough = [p for p in npz_files if len(np.load(p)["poses"]) >= min_frames]
    candidates = long_enough or npz_files
    best = max(candidates, key=score)
    poses = np.load(best)["poses"]
    print(f"Auto-selected clip: {best.stem} ({len(poses)} frames, label={np.load(best)['label']})")
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
    context_len = model.context_len
    x = torch.from_numpy(
        build_context_windows(poses, context_len, horizon)
    ).to(device)

    # Condition the predictor on this horizon when it was trained on it; otherwise
    # fall back to the default (shortest) horizon.
    if horizon in model.predictor.horizons:
        k = torch.full((x.shape[0],), horizon, dtype=torch.long, device=device)
    else:
        k = None

    pred_emb = model.encode_for_rollout(x, k)       # g(f(x_t), k) : predicted future embedding
    pred_poses = model.decode_pose(pred_emb).cpu().numpy()
    gt_poses = poses[horizon:]
    return gt_poses, pred_poses


@torch.no_grad()
def predict_autoregressive(
    model: JEPA,
    poses: np.ndarray,
    n_steps: int,
    device: torch.device,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Feed the model's own decoded predictions back as context (honest drift test).

    Steps one frame at a time (horizon=1, matching the stride-1 training window):
    seed the rolling window with the first K real frames, then at each step encode
    -> predict -> decode -> append the decoded pose and slide. Real frames slide out
    as predicted frames slide in, so decoder error compounds — the honest test of
    whether the latent motion model is coherent.

    Returns (ground_truth, predicted), aligned frame-for-frame, each [n_steps, pose_dim].
    Requires len(poses) >= model.context_len + n_steps for the GT alignment.
    """
    K = model.context_len
    history = [poses[i].astype(np.float32) for i in range(K)]

    k = None
    if horizon in model.predictor.horizons:
        k = torch.full((1,), horizon, dtype=torch.long, device=device)

    preds = []
    for _ in range(n_steps):
        window = np.stack(history[-K:])                       # [K, pose_dim]
        x = augment_with_velocity(window)                    # [K, 2*pose_dim]
        if K == 1:
            x = x[0]
        x = torch.from_numpy(x).unsqueeze(0).to(device)      # [1, K, 2D] or [1, 2D]
        pred_emb = model.encode_for_rollout(x, k)
        pred_pose = model.decode_pose(pred_emb)[0].cpu().numpy()  # [pose_dim]
        preds.append(pred_pose)
        history.append(pred_pose)

    predicted = np.stack(preds)                              # [n_steps, pose_dim]
    gt_poses = poses[K : K + n_steps]                        # aligned ground truth
    return gt_poses, predicted


def run_rollout(
    config_path: Path | None = None,
    steps: int = 150,
    horizon: int = 5,
    clip_name: str | None = None,
    mode: str = "teacher",
) -> dict[str, np.ndarray]:
    config = load_config(config_path)
    device = resolve_device(config)
    processed_dir = ROOT / config["data"]["processed_dir"]

    model, _meta = load_model(config, device)

    if mode == "auto":
        poses = pick_clip(
            processed_dir, max_frames=model.context_len + steps, clip_name=clip_name
        )
        gt, pred = predict_autoregressive(model, poses, steps, device, horizon=1)
    else:
        poses = pick_clip(processed_dir, max_frames=steps + horizon, clip_name=clip_name)
        gt, pred = predict_teacher_forced(model, poses, horizon, device)
    results = {"ground_truth": gt, "predicted": pred}

    out_dir = ROOT / "outputs" / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, traj in results.items():
        np.savez_compressed(out_dir / f"rollout_{name}.npz", poses=traj)
        print(f"Saved '{name}' trajectory ({len(traj)} frames)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="JEPA prediction rollout")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--clip", type=str, default=None, help="Clip stem name, e.g. 35_02")
    parser.add_argument(
        "--rollout",
        choices=["teacher", "auto"],
        default="teacher",
        help="teacher = re-anchored on GT each step; auto = feed predictions back (drift test)",
    )
    args = parser.parse_args()
    run_rollout(
        args.config,
        steps=args.steps,
        horizon=args.horizon,
        clip_name=args.clip,
        mode=args.rollout,
    )


if __name__ == "__main__":
    main()
