# JEPA Motion Prediction Research History

## Overview

This document summarizes the major discussions, architectural decisions,
pivots, experiments, and conclusions from this chat.

### 1. Initial Vision

-   Build a portfolio-grade JEPA project on the CMU MoCap dataset.
-   Demonstrate self-supervised predictive world models rather than
    simple sequence prediction.
-   Target Apple Silicon (M5/MPS), lean codebase, reproducible training.

### 2. Major Architectural Decisions

-   Started from a JEPA predicting future latent representations.
-   Switched prediction loss from MSE to VICReg.
-   Added EMA target encoder to prevent trivial solutions.
-   Retained projector after discussing whether it was necessary.
-   Added linear probe for representation evaluation.
-   Added visualization decoder solely for qualitative rollouts.

### 3. Physics Representation Discussion

Initial ideas included: - Predicting joint rotations (forward
kinematics) - Predicting Cartesian joint positions with geometric
constraints

Final direction: - Predict Cartesian joint coordinates. - Add
bone-length consistency loss. - Learn predictive latent representations
rather than exact physical simulation.

### 4. Data Pipeline

Discussed: - BVH parsing - Hip centering - Global normalization -
Downsampling - NPZ caching - Skeleton metadata generation

### 5. Dataset Design Pivot

Original implementation randomly sampled clips inside **getitem**.

Issue: - Sampling with replacement. - Poor epoch coverage.

Improved design: - Deterministic global index mapping. - Random temporal
horizon only.

### 6. Model Design

Final components: - Encoder - Projector - Predictor - EMA Target
Encoder - Action Embedding - Visualization Decoder

Discussion covered why projectors improve SSL by separating
representation learning from optimization constraints.

### 7. Network Capacity

Concluded: - 256-dimensional representation is appropriate. - Predictor
bottleneck (128) encourages meaningful latent conditioning. -
Architecture balanced for Apple Silicon.

### 8. Training

-   VICReg replaces plain MSE.
-   EMA encoder detached.
-   Phase 1:
    -   Vanilla JEPA
-   Phase 2:
    -   Latent-conditioned JEPA.

### 9. Evaluation

Added: - Linear probe. - PCA/manifold visualization. - Latent rollout
visualization. - Future comparison against random encoder.

### 10. Repository Engineering

Discussed: - Lean modular layout. - Unified train.py. - Config-driven
architecture. - uv environment.

### 11. CMU Dataset

Created ingestion strategy. Encountered: - Incorrect repository
structure assumptions. - Network download failures. - Recommended
retries, streaming downloads, and manual download fallback.

### 12. Rollout Rendering

Initially GIF. Migrated toward FFmpeg MP4 rendering.

### 13. Scientific Insights

Important realization: - The model can learn underlying physics while
encoding them nonlinearly. - Learning the correct dynamics and achieving
linear separability are distinct objectives.

### 14. Current Problem Statement

The project ultimately became:

Learn predictive latent representations of human motion from past
skeletal observations such that future motion is predictable in latent
space while preserving physical structure. Evaluate representation
quality using linear probes and qualitative latent-conditioned motion
rollouts.

### 15. Simplification Pivot (v1 "make it work")

The codebase had accumulated competing anti-collapse mechanisms that fought each
other, plus a syntax error that prevented training from running at all. Stripped
back to a clean, canonical JEPA:

-   Fixed hard blocker: stray text at top of `losses.py` (un-importable module).
-   Single objective: `MSE(predicted future embedding, EMA target)` + VICReg
    variance/covariance (collapse guard) + FK-decoder reconstruction.
-   Removed: prediction dropout, counterfactual/shuffled-label pass, hardcoded
    cosine "push" loss, kinematic bone loss (redundant — FK enforces rigid bones),
    and the `normalize(s_x + delta)` reconstruction trick.
-   Action conditioning dropped for v1 (unconditioned predictor); labels kept only
    for the linear probe / eval.
-   FK decoder trained **jointly** with the encoder/predictor (detached latent),
    replacing the fragile two-phase `train_decoder.py` + `balanced_repr` heuristic.
-   Dataset `__getitem__` made deterministic (flat index → `(clip, t)` map) for
    full, reproducible epoch coverage.
-   Fixed stale 3-class labels in the NPZ cache (regenerated via `bvh_parser`).
-   Visualization is now drift-free, teacher-forced predicted-vs-ground-truth
    (re-anchored each step) instead of a divergent closed-loop rollout.

Result: trains cleanly with no collapse (VICReg variance healthy), linear probe
beats random/chance, FK decoder produces valid rigid skeletons, and the
predicted-vs-GT comparison video renders.

### 16. Remaining Future Work

-   Sharpen the predictor (it currently only marginally beats a persistence
    baseline at short horizons) — try multi-step / sequence context instead of
    single-frame, or a temporal encoder.
-   Improve decoder fidelity (reconstruction is plausible but rough).
-   Re-introduce action conditioning cleanly (FiLM / wider token) for
    "one pose, three actions" conditioned rollouts.
-   Multi-action training, latent traversal, effective-rank analysis.
-   Stretch: stabilized autoregressive rollout with periodic re-grounding.
-   Portfolio-quality videos.
