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
uv run python -m training.train           # train JEPA (encoder + predictor + FK decoder jointly)
uv run python -m eval.linear_probe        # representation sanity check (action linear probe)
uv run python -m eval.render              # render ground-truth vs JEPA-predicted motion (mp4)
```
