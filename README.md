# Skeleton-JEPA

Self-supervised motion model learning predictive representations of human kinetics from CMU MoCap BVH data.

## Setup

```bash
uv sync
```

Place raw `.bvh` files under `data/raw/` (optionally grouped in subfolders by action class).

## Pipeline

```bash
uv run python -m data.bvh_parser          # parse BVH → npz caches
uv run python -m training.train           # Phase 1 (use_latent: false)
uv run python -m eval.linear_probe        # representation sanity check
# Set training.use_latent: true in configs/config.yaml for Phase 2
uv run python -m eval.rollout
uv run python -m eval.render
```
