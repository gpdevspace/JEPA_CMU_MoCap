# Skeleton-JEPA: Project Overview

## What We Are Building

Skeleton-JEPA is a self-supervised world model for human motion. It learns to understand how the human body moves — without any labels, annotations, or supervision — by training on raw motion capture recordings from the CMU MoCap dataset.

The core idea is simple: given a snapshot of a person's pose right now, the model learns to predict what that person's pose will look like a fraction of a second in the future. Rather than predicting exact joint coordinates (which are noisy and full of irrelevant detail), the model predicts an *abstract latent representation* of the future pose. This forces the model to learn what is genuinely informative about motion dynamics and discard everything else.

The result is an encoder that compresses any 3D skeletal pose into a 256-dimensional vector capturing its semantic content — what action is being performed, where in the motion cycle the body is — learned entirely from the temporal structure of movement.

---

## The Scientific Problem

### Why Representation Learning for Motion?

Human motion is high-dimensional and structured. A single pose from the CMU MoCap skeleton has 31 joints, each with 3D coordinates — 93 numbers in total. But most of that space is not equally informative: the specific millimeter-level position of a finger matters far less for understanding human movement than whether the person is mid-stride, mid-jump, or mid-swing. Good representations compress the 93-dimensional pose into a smaller space that preserves the semantically meaningful structure.

Traditional supervised learning would require labeled data: "this clip is walking, this one is jumping." Labels are expensive and limiting. Self-supervised approaches learn representations from the data's own structure, without labels, and often generalize better because they are not constrained to the label vocabulary.

### The Temporal Self-Supervision Signal

The key insight driving this project: **movement is inherently temporal**, and temporal consistency is a free supervision signal. If you know a person is mid-stride in a walking motion, you can predict with high confidence what they will look like 0.2 seconds from now. If the model can learn to make such predictions accurately, it must have learned something real about the structure of human motion.

This places Skeleton-JEPA in the family of *predictive world models*: systems that understand their domain by learning to predict the future state of a system from its current state.

---

## The JEPA Framework

### Joint Embedding Predictive Architecture

JEPA (Lecun 2022, Assran et al. 2023) is a self-supervised learning framework that learns by predicting in *latent space* rather than in *input space*. This distinction is important:

- **Autoencoders / Masked Autoencoders (MAE)**: predict the raw input — exact pixel values, exact joint coordinates. The model must reconstruct every detail, including noise and irrelevant variation.
- **JEPA**: predict the *embedding* of the future observation. The model only needs to predict what is semantically meaningful about the future — the structure of the future pose, not its noise.

The result is representations that are more abstract and semantically richer. A JEPA encoder does not need to encode the exact measurement noise of a given frame; it only needs to encode what determines the future.

### The Collapse Problem

The fundamental challenge in self-supervised learning is *representation collapse*: a model trained to make similar inputs produce similar outputs can cheat by mapping everything to the same constant vector. Trivially satisfies any similarity objective; produces useless representations.

JEPA solves this with two mechanisms:

1. **EMA target encoder**: The prediction target comes from a *different* copy of the encoder — an exponential moving average (EMA) of the online encoder that is never directly updated by backpropagation. The online encoder must chase a slowly-moving, improving target; it cannot collapse alongside the target.

2. **VICReg regularization**: Variance and covariance penalties force every dimension of the 256-d representation to carry non-redundant information. No dimension is allowed to be constant across poses or to duplicate another dimension.

---

## Architecture

### System Overview

```
Input pose at time t (93 floats)
         │
         ▼
  ┌─────────────┐
  │   Encoder   │  ──────────────────────────────────────────────┐
  └─────────────┘                                                 │
         │ 256-d representation                                   │ (detached)
         ▼                                                        ▼
  ┌─────────────┐                                       ┌────────────────┐
  │  Projector  │                                       │  FK Decoder    │
  └─────────────┘                                       └────────────────┘
         │ s_x                                                    │
         ▼                                                        ▼
  ┌─────────────┐                                       Reconstructed pose
  │  Predictor  │ ──► predicted future embedding ŝ_y   (for visualization)
  └─────────────┘
         │
         │ compare (MSE loss)
         │
         ▼
  EMA target embedding s_y  ◄── EMA Encoder + EMA Projector ◄── pose at time t+k
  (no gradient, slowly updated)
```

### Components

**Encoder** (`src/models/components.py`): The core learned function. Maps a 93-dimensional joint coordinate vector to a 256-dimensional semantic representation. Uses LayerNorm at the input to handle the different scales of joints (hip is far from origin, fingers are close). Three-layer MLP: `93 → 256 → 256 → 256`.

**Projector** (`src/models/components.py`): Sits on top of the encoder in the online branch only. Acts as a gradient buffer — VICReg's anti-collapse pressure is absorbed by the projector, leaving the encoder free to learn the representation best suited for downstream use. Discarded after training; the linear probe uses the raw encoder output.

**Predictor** (`src/models/components.py`): A small bottleneck network (`256 → 128 → 256`) that maps the current context embedding to a predicted future embedding. The bottleneck is intentional: it forces compression, preventing the predictor from memorizing mappings. It must learn to predict *meaningful motion state changes*, not noise.

**EMA Target Encoder**: A frozen copy of the encoder+projector, updated each training step as an exponential moving average of the online encoder. Provides stable prediction targets that slowly improve as training progresses. Momentum is scheduled from 0.996 → 0.999 over training (faster early, slower late).

**FK Decoder** (`src/models/components.py`, `src/models/kinematics.py`, `src/models/geometry.py`): Maps a latent embedding back to a physically valid 3D skeleton pose. Predicts 6D rotation vectors (one per joint), converts them to valid rotation matrices via Gram-Schmidt orthogonalization, then runs the forward kinematics chain through the skeleton hierarchy. Bone lengths are baked into the skeleton rest-pose offsets and cannot change — rigid bodies are mathematically guaranteed, not enforced by a loss.

---

## Loss Function

Three components, each serving a distinct purpose:

### 1. JEPA Prediction Loss

```
L_pred = MSE( predictor(s_x), s_y )
```

The online encoder+predictor learns to map the current pose embedding to the target's future embedding. Gradient flows through: predictor → projector → encoder. The target `s_y` is detached (no gradient).

### 2. VICReg Anti-Collapse Regularization

**Variance loss**: For each of the 256 embedding dimensions, the standard deviation across the batch must exceed a threshold. If any dimension collapses to a constant value, this loss fires. Prevents *dimensional collapse*.

**Covariance loss**: The off-diagonal entries of the 256×256 embedding covariance matrix are penalized. If two dimensions are correlated (always move together), this fires. Forces each dimension to specialize in a *different aspect* of the pose.

Applied to both `s_x` (context embeddings) and `ŝ_y` (predictions), with weights `var_weight=1.0`, `cov_weight=0.05`.

### 3. FK Reconstruction Loss

```
L_recon = MSE( FK_decoder( encoder(pose).detach() ), pose )
```

The `.detach()` is critical: the encoder does **not** learn to make itself easy to decode. The decoder learns to *read* whatever representation the JEPA objective produces. This keeps the encoder focused on temporal dynamics, not appearance reconstruction. The reconstruction loss only trains the FK decoder — giving it a signal while not biasing the encoder toward autoencoders.

### Total Loss

```
L = L_pred + L_vicreg + recon_weight × L_recon
```

---

## Data Pipeline

### Source Data

CMU MoCap BVH files, organized by action class into subfolders under `data/raw/`:

```
data/raw/
  walking/     ← e.g., 35_02.bvh, 35_03.bvh
  jumping/
  boxing/
```

### Processing (`src/data/bvh_parser.py`)

Each BVH file goes through:

1. **Parse BVH hierarchy**: Extract joint names, parent relationships, and bone offsets from the file header via depth-first traversal.
2. **Extract global positions**: Read per-frame joint rotations from the motion data section, compute global 3D positions for all 31 joints across all frames.
3. **Hip-center**: Subtract the hip joint's position from all joints at every frame. The model works in body-relative coordinates, not world coordinates.
4. **Normalize**: Divide by the average thigh bone length (~2.56 units) computed across the dataset. All clips share a common scale.
5. **Downsample**: 120 Hz → 30 Hz (keep every 4th frame). Reduces redundancy while preserving motion dynamics.
6. **Flatten**: `[T, 31, 3]` → `[T, 93]` for model consumption.
7. **Save**: Compressed NPZ files per clip with poses, action label (inferred from filename), and bone lengths.

A single `skeleton.json` file captures the skeleton topology (joint names, parent indices, bone offsets, global scale) shared across all clips.

### Dataset (`src/data/dataset.py`)

`SkeletonPairDataset` produces `(context, target)` pairs for JEPA training. For each valid `(clip, frame_t)` combination:

- **Context** `x`: pose at frame `t`
- **Target** `y`: pose at frame `t + k`, where `k ∈ {3, 5, 8}` frames (0.1–0.27 seconds)
- **Label**: action class (used only during evaluation, not training)

Horizons cycle deterministically across the flat index map (`k = horizons[idx % 3]`), ensuring each (clip, frame) pair is seen at all three horizons with equal frequency over full epochs. This is preferable to random per-call sampling, which produced non-uniform coverage.

---

## Forward Kinematics: Guaranteed Valid Poses

### The Problem with Naive Decoders

A decoder that directly outputs 93 floats (joint XYZ coordinates) has no constraint on bone lengths. It can produce physically impossible poses: disconnected joints, stretched limbs, asymmetric bodies. These are useless for visualization and poor as a learning target.

### The FK Solution

Instead of predicting positions, the FK decoder predicts *local joint rotations* (in 6D representation), then computes global positions by traversing the skeleton tree.

**6D rotation representation** (Zhou et al. 2019): The network outputs 6 free floats per joint. Gram-Schmidt orthogonalization converts them to a valid 3×3 rotation matrix (SO(3)) deterministically. This representation is:
- **Continuous**: small changes in the 6 floats produce small changes in rotation (unlike Euler angles or quaternions)
- **Singularity-free**: no gimbal lock
- **Unconstrained**: the network never needs to learn to satisfy a constraint (unlike quaternions, which must have unit norm)

**FK chain**: The skeleton is a tree rooted at the hips. Each joint's global position is computed from its parent's global rotation and the rest-pose bone offset:

```
global_pos[i] = global_pos[parent] + global_rot[parent] @ bone_offset[i]
global_rot[i] = global_rot[parent] @ local_rot[i]
```

Because `bone_offset[i]` is fixed (baked into `skeleton.json`), its magnitude (the bone length) is never modified — only rotated. Rigid bodies are guaranteed by the math.

---

## Evaluation

### Linear Probe

Freezes the trained encoder and trains a single linear layer to classify action from the frozen 256-d embeddings. This directly measures whether action structure is *linearly encoded* in the representation — without any labels during training.

**Results**: 68% accuracy vs. 62% random-encoder baseline (33% chance for 3 classes). The encoder learns action-discriminative representations purely from temporal prediction.

### Teacher-Forced Rollout (`src/eval/rollout.py`)

For each ground-truth frame `x_t`, the model predicts the embedding of frame `x_{t+k}`, decodes it via the FK decoder, and compares to the actual future pose. Each prediction is anchored to the real ground-truth frame (not to a prior prediction), isolating prediction quality from accumulation error.

### Visualization (`src/eval/render.py`)

Side-by-side 3D skeleton animations: ground truth on the left, FK-decoded JEPA prediction on the right. Rendered as MP4 using matplotlib 3D.

### Diagnostic Plots (`src/eval/plot_diagnostics.py`)

Five diagnostic views:
1. **PCA embedding**: 2D projection of encoder outputs colored by action — shows cluster structure learned without labels
2. **Per-dimension variance**: Confirms VICReg is working; all 256 dimensions should carry variance
3. **Prediction quality**: JEPA MSE vs. persistence baseline per action class
4. **Cosine similarity**: Distribution of alignment between predicted and target embeddings
5. **Reconstruction scatter**: FK-decoded joints vs. ground truth, with R² score

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Predict in latent space (not input space) | Encoder focuses on semantics, not noise or irrelevant detail |
| EMA target encoder | Prevents collapse without requiring negative pairs; provides stable, improving targets |
| VICReg regularization | Anti-collapse that also enforces dimensional diversity and specialization |
| Projector head | Absorbs VICReg gradient pressure; keeps encoder representation clean for downstream use |
| 6D rotation representation | Continuous, singularity-free rotation parameterization; no constraint the network must learn |
| FK decoder with fixed bone offsets | Rigid bodies guaranteed by math; bone lengths cannot be violated |
| Detached latent for reconstruction | Decoder learns to read JEPA representations without biasing encoder toward reconstruction |
| Single-frame context (no sequence) | Forces learning of intrinsic pose-to-future dynamics; no temporal memory shortcuts |
| Deterministic dataset indexing | Full epoch coverage; reproducible; avoids sampling bias in (clip, frame, horizon) triples |
| 30 Hz @ horizons {3, 5, 8} | 0.1–0.27s — short enough to be predictable, long enough to require understanding dynamics |

---

## Project Structure

```
JEPA_CMU_MoCap/
├── configs/
│   └── config.yaml          # All hyperparameters: model dims, training, loss weights
├── data/
│   ├── raw/                 # Input BVH files organized by action class
│   ├── processed/           # Parsed NPZ caches + skeleton.json
│   └── skeleton.json        # Joint hierarchy, bone offsets, global scale
├── src/
│   ├── data/
│   │   ├── bvh_parser.py    # BVH → hip-centered, normalized, downsampled NPZ
│   │   └── dataset.py       # SkeletonPairDataset: deterministic (context, target) pairs
│   ├── models/
│   │   ├── jepa.py          # JEPA orchestrator: wires encoder, EMA, predictor, decoder
│   │   ├── components.py    # Encoder, Projector, Predictor, VisualizationDecoder, MLP
│   │   ├── geometry.py      # 6D → SO(3) rotation matrix via Gram-Schmidt
│   │   └── kinematics.py    # Differentiable FK chain: local rotations → global positions
│   ├── training/
│   │   ├── train.py         # Training loop: forward, loss, EMA update, checkpoint
│   │   └── losses.py        # Three-part loss: JEPA pred + VICReg + FK reconstruction
│   └── eval/
│       ├── linear_probe.py  # Freeze encoder, train linear classifier on actions
│       ├── rollout.py       # Teacher-forced prediction rollout
│       ├── render.py        # Side-by-side 3D animation renderer (MP4)
│       └── plot_diagnostics.py  # Five diagnostic plots
├── checkpoints/
│   └── jepa_latest.pt       # Saved model weights + config + skeleton metadata
└── outputs/
    ├── videos/              # Rendered comparison animations
    ├── rollouts/            # NPZ prediction trajectories
    └── viz/                 # Diagnostic plots
```

---

## Execution Pipeline

```bash
# 1. Parse raw BVH files to normalized NPZ caches
uv run python -m data.bvh_parser

# 2. Train the JEPA model
uv run python -m training.train

# 3. Evaluate representation quality
uv run python -m eval.linear_probe

# 4. Render ground truth vs. predicted side-by-side animation
uv run python -m eval.render
```

---

## Key Numbers

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `pose_dim` | 93 | 31 joints × 3 coordinates |
| `repr_dim` | 256 | Encoder output dimension (used downstream) |
| Predictor bottleneck | 128 | Forces compression; prevents memorization |
| Prediction horizons | 3, 5, 8 frames | 0.10, 0.17, 0.27 seconds at 30 Hz |
| EMA momentum | 0.996 → 0.999 | Slow at end of training for stability |
| `var_weight` | 1.0 | VICReg variance hinge weight |
| `cov_weight` | 0.05 | VICReg covariance decorrelation weight |
| `recon_weight` | 1.0 | FK decoder reconstruction weight |
| Epochs | 15 | Full training run |
| Batch size | 256 | — |
| Linear probe accuracy | 68% | vs. 62% random encoder, 33% chance |

---

## Further Reading

- [JEPA Specifics](JEPA_Specifics.md) — deep-dive into every component, gradient flow, and implementation decisions anchored to file and line numbers
- [Design History](history.md) — what was tried, what was removed, and why
- [LeCun (2022)](https://openreview.net/forum?id=BZ55e1jC8cL) — "A Path Towards Autonomous Machine Intelligence" (original JEPA proposal)
- [Assran et al. (2023)](https://arxiv.org/abs/2301.08243) — I-JEPA: first JEPA implementation on images
- [Bardes et al. (2022)](https://arxiv.org/abs/2105.04906) — VICReg: variance-invariance-covariance regularization
- [Zhou et al. (2019)](https://arxiv.org/abs/1812.07035) — On the Continuity of Rotation Representations in Neural Networks (6D rotation representation)
