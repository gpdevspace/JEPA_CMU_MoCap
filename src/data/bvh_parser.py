"""Parse raw BVH files into normalized NPZ caches for Skeleton-JEPA."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from npybvh import Bvh

from utils import ACTION_CLASSES, infer_action_label, load_config, ROOT


def _ordered_joints(root_joint) -> list:
    """Depth-first joint order excluding end sites."""
    joints = []

    def walk(joint):
        if joint.name.endswith("_end"):
            return
        joints.append(joint)
        for child in joint.children:
            walk(child)

    walk(root_joint)
    return joints


def _parse_bvh(path: Path) -> tuple[np.ndarray, list, list[tuple[int, int]]]:
    """Return global positions (T, J, 3), joint names, and parent-child index pairs."""
    bvh = Bvh()
    bvh.parse_file(str(path))
    all_positions, _ = bvh.all_frame_poses()
    joints = _ordered_joints(bvh.root)
    joint_order = list(bvh.joints.values())
    indices = [joint_order.index(j) for j in joints]
    positions = all_positions[:, indices, :]
    joint_names = [j.name for j in joints]
    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    bone_pairs = []
    for i, joint in enumerate(joints):
        if joint.parent is not None and not joint.parent.name.endswith("_end"):
            parent_idx = name_to_idx[joint.parent.name]
            bone_pairs.append((parent_idx, i))

    return positions, joint_names, bone_pairs


def _hips_center(positions: np.ndarray, hips_idx: int) -> np.ndarray:
    centered = positions - positions[:, hips_idx : hips_idx + 1, :]
    return centered


def _compute_bone_lengths(
    positions: np.ndarray, bone_pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    lengths = {}
    for parent, child in bone_pairs:
        diffs = positions[:, parent, :] - positions[:, child, :]
        lengths[(parent, child)] = float(np.mean(np.linalg.norm(diffs, axis=-1)))
    return lengths


def _reference_scale(positions: np.ndarray, hips_idx: int, left_leg_idx: int) -> float:
    diffs = positions[:, hips_idx, :] - positions[:, left_leg_idx, :]
    return float(np.mean(np.linalg.norm(diffs, axis=-1)))


def _downsample(positions: np.ndarray, rate: int) -> np.ndarray:
    return positions[::rate]


def discover_bvh_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.rglob("*.bvh"))
    if not files:
        raise FileNotFoundError(f"No BVH files found under {raw_dir}")
    return files


def process_dataset(
    raw_dir: Path,
    processed_dir: Path,
    downsample_rate: int,
    sanity_check: bool = False,
) -> dict:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    bvh_files = discover_bvh_files(raw_dir)
    global_scale: float | None = None
    skeleton_meta: dict | None = None
    all_bone_lengths: dict[tuple[int, int], list[float]] = {}

    for bvh_path in bvh_files:
        positions, joint_names, bone_pairs = _parse_bvh(bvh_path)
        hips_idx = joint_names.index("Hips") if "Hips" in joint_names else 0

        left_leg_candidates = ["LeftUpLeg", "LeftHip", "LHipJoint"]
        left_leg_idx = next(
            (joint_names.index(name) for name in left_leg_candidates if name in joint_names),
            1,
        )

        if global_scale is None:
            global_scale = _reference_scale(positions, hips_idx, left_leg_idx)
            skeleton_meta = {
                "joint_names": joint_names,
                "bone_pairs": bone_pairs,
                "hips_idx": hips_idx,
            }

        centered = _hips_center(positions, hips_idx)
        normalized = centered / global_scale
        downsampled = _downsample(normalized, downsample_rate)

        clip_bone_lengths = _compute_bone_lengths(downsampled, bone_pairs)
        for pair, length in clip_bone_lengths.items():
            all_bone_lengths.setdefault(pair, []).append(length)

        rel_parts = bvh_path.relative_to(raw_dir).parts
        label_name = rel_parts[0] if len(rel_parts) > 1 else bvh_path.stem
        label = infer_action_label(label_name)

        clip_id = bvh_path.stem
        poses = downsampled.reshape(downsampled.shape[0], -1).astype(np.float32)

        np.savez_compressed(
            processed_dir / f"{clip_id}.npz",
            poses=poses,
            label=np.int64(label),
            bone_lengths=np.array(
                [clip_bone_lengths[pair] for pair in bone_pairs], dtype=np.float32
            ),
        )

        if sanity_check:
            _sanity_visualize(downsampled, bone_pairs, clip_id)

    ref_bone_lengths = {
        pair: float(np.mean(lengths)) for pair, lengths in all_bone_lengths.items()
    }
    meta_path = processed_dir / "skeleton.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "joint_names": skeleton_meta["joint_names"],
                "bone_pairs": skeleton_meta["bone_pairs"],
                "hips_idx": skeleton_meta["hips_idx"],
                "global_scale": global_scale,
                "reference_bone_lengths": {
                    f"{p}_{c}": ref_bone_lengths[(p, c)]
                    for p, c in skeleton_meta["bone_pairs"]
                },
                "action_classes": ACTION_CLASSES,
                "pose_dim": len(skeleton_meta["joint_names"]) * 3,
            },
            f,
            indent=2,
        )

    return {
        "clips": len(bvh_files),
        "pose_dim": len(skeleton_meta["joint_names"]) * 3,
        "num_joints": len(skeleton_meta["joint_names"]),
        "global_scale": global_scale,
        "meta_path": str(meta_path),
    }


def _sanity_visualize(
    positions: np.ndarray,
    bone_pairs: list[tuple[int, int]],
    clip_id: str,
) -> None:
    """Plot a single frame to verify human posture."""
    frame_idx = min(10, len(positions) - 1)
    pos = positions[frame_idx]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c="crimson", s=20)

    for parent, child in bone_pairs:
        p = pos[parent]
        c = pos[child]
        ax.plot(
            [p[0], c[0]],
            [p[2], c[2]],
            [p[1], c[1]],
            color="navy",
            linewidth=1.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(f"Sanity check: {clip_id} frame {frame_idx}")
    out_dir = ROOT / "outputs" / "sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}_sanity.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sanity visualization to {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse BVH files to NPZ caches")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    raw_dir = ROOT / config["data"]["raw_dir"]
    processed_dir = ROOT / config["data"]["processed_dir"]
    downsample_rate = config["data"]["downsample_rate"]

    if not raw_dir.exists():
        example_src = ROOT / "third_party" / "npybvh" / "example.bvh"
        raw_dir.mkdir(parents=True, exist_ok=True)
        if example_src.exists():
            import shutil

            dest = raw_dir / "walking" / "example.bvh"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(example_src, dest)
            print(f"No raw data found; copied example BVH to {dest}")

    summary = process_dataset(
        raw_dir,
        processed_dir,
        downsample_rate,
        sanity_check=args.sanity_check,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
