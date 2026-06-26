"""3-panel comparative skeleton animation renderer."""

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from eval.rollout import run_rollout
from utils import ACTION_CLASSES, load_config, ROOT


def render_trajectory(
    ax,
    positions: np.ndarray,
    bone_pairs: list[tuple[int, int]],
    frame_idx: int,
    color: str = "crimson",
):
    num_joints = positions.shape[1]
    pos = positions[frame_idx]

    ax.cla()
    ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1], c=color, s=15)
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
            linewidth=1.2,
        )

    # DYNAMIC SCALING:
    # Find the max extent of the skeleton to keep it in frame
    max_val = np.max(np.abs(pos)) + 0.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=20, azim=-60)


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
) -> None:
    positions = trajectories_to_positions(trajectories)
    names = list(positions.keys())
    num_frames = min(len(positions[n]) for n in names)

    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
    colors = ["forestgreen", "crimson", "darkorange"]

    def update(frame_idx):
        artists = []
        for i, name in enumerate(names[:3]):
            render_trajectory(axes[i], positions[name], bone_pairs, frame_idx, colors[i])
            axes[i].set_title(f"Z = {name}")
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
        anim.save(str(output_path), writer=animation.FFMpegWriter(fps=fps))
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        anim.save(str(gif_path), writer=animation.PillowWriter(fps=fps))
        print(f"FFmpeg unavailable; saved GIF to {gif_path}")
        plt.close(fig)
        return

    plt.close(fig)
    print(f"Saved comparison video to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render latent traversal comparison")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = ROOT / config["data"]["processed_dir"]
    with open(processed_dir / "skeleton.json") as f:
        meta = json.load(f)

    action_labels = [
        ACTION_CLASSES.index("walking"),
        ACTION_CLASSES.index("jumping"),
        ACTION_CLASSES.index("boxing"),
    ]
    trajectories = run_rollout(args.config, steps=args.steps, action_labels=action_labels)

    output_path = ROOT / "outputs" / "videos" / "latent_traversal_comparison.mp4"
    render_comparison(trajectories, meta["bone_pairs"], output_path)


if __name__ == "__main__":
    main()
