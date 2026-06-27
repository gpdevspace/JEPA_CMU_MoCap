"""3-panel comparative skeleton animation renderer."""

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from eval.rollout import run_rollout
from utils import load_config, ROOT


def render_trajectory(
    ax,
    positions: np.ndarray,
    bone_pairs: list[tuple[int, int]],
    frame_idx: int,
    color: str = "crimson",
    axis_range: float = 8.0,
):
    num_joints = positions.shape[1]
    pos = positions[frame_idx]

    ax.cla()
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=color, s=20, depthshade=True)
    for parent, child in bone_pairs:
        if parent >= num_joints or child >= num_joints:
            continue
        p = pos[parent]
        c = pos[child]
        ax.plot(
            [p[0], c[0]],
            [p[2], c[2]],
            [p[1], c[1]],
            color="navy",
            linewidth=1.5,
        )

    # Fixed scale across all frames: eliminates the "jumping skeleton" visual artifact
    ax.set_xlim(-axis_range, axis_range)
    ax.set_ylim(-axis_range, axis_range)
    ax.set_zlim(-axis_range, axis_range)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=15, azim=-70)


def trajectories_to_positions(trajectories: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for name, traj in trajectories.items():
        # Using -1 allows numpy to automatically calculate the correct number of joints
        # based on the remaining dimensions (assuming 3 coordinates per joint)
        out[name] = traj.reshape(traj.shape[0], -1, 3)
    return out


def render_comparison(
    trajectories: dict[str, np.ndarray],
    bone_pairs: list[tuple[int, int]],
    output_path: Path,
    fps: int = 30,
    horizon: int = 5,
) -> None:
    positions = trajectories_to_positions(trajectories)
    names = list(positions.keys())
    n = len(names)
    num_frames = min(len(positions[name]) for name in names)

    # Compute a single fixed axis range from the ground-truth trajectory
    all_pos = positions[names[0]]  # GT is always first
    axis_range = float(np.max(np.abs(all_pos))) * 1.1
    axis_range = max(axis_range, 5.0)  # minimum sensible range

    titles = {
        "ground_truth": "Ground Truth",
        "predicted": f"JEPA Predicted  (+{horizon} frames / {horizon/30:.2f}s)",
    }

    fig = plt.figure(figsize=(5 * n, 5))
    fig.patch.set_facecolor("white")
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]
    colors = ["#2ecc71", "#e74c3c"]  # green = GT, red = predicted

    def update(frame_idx):
        artists = []
        for i, name in enumerate(names):
            color = colors[i % len(colors)]
            render_trajectory(
                axes[i], positions[name], bone_pairs, frame_idx, color, axis_range
            )
            axes[i].set_title(titles.get(name, name.replace("_", " ").title()), fontsize=11, pad=4)
            artists.extend(axes[i].collections)
            artists.extend(axes[i].lines)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        anim.save(str(output_path), writer=writer)
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        anim.save(str(gif_path), writer=animation.PillowWriter(fps=fps))
        print(f"FFmpeg unavailable; saved GIF to {gif_path}")
        plt.close(fig)
        return

    plt.close(fig)
    print(f"Saved comparison video to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ground-truth vs JEPA-predicted motion")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--clip", type=str, default=None, help="Clip stem name, e.g. 35_02")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = ROOT / config["data"]["processed_dir"]
    with open(processed_dir / "skeleton.json") as f:
        meta = json.load(f)

    trajectories = run_rollout(
        args.config, steps=args.steps, horizon=args.horizon, clip_name=args.clip
    )

    output_path = ROOT / "outputs" / "videos" / "prediction_vs_groundtruth.mp4"
    render_comparison(trajectories, meta["bone_pairs"], output_path, horizon=args.horizon)


if __name__ == "__main__":
    main()
