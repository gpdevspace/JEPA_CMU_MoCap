"""Neural network building blocks for Skeleton-JEPA."""

import numpy as np
import torch
import torch.nn as nn

from models.kinematics import ForwardKinematics


class MLP(nn.Module):
    def __init__(self, layers: list[int], activation: str = "gelu"):
        super().__init__()
        acts = []
        for i in range(len(layers) - 1):
            acts.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                acts.append(nn.GELU() if activation == "gelu" else nn.ReLU())
        self.net = nn.Sequential(*acts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, repr_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = MLP([input_dim, 256, 256, repr_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm(x))


class Projector(nn.Module):
    def __init__(self, repr_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.mlp = MLP([repr_dim, proj_dim, proj_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ActionEmbedding(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, latent_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class Predictor(nn.Module):
    """Predicts the future embedding from the context embedding.

    Phase 1: horizon-conditioned. The predictor is told which horizon k it is
    targeting via a learned embedding, so a single network can serve multiple
    horizons without confusing "3 steps ahead" and "8 steps ahead".

    Residual: the MLP predicts the *change* (delta) from the context embedding,
    and the output is `s_x + delta`. This anchors the prediction to the current
    real frame — even if the delta regresses to its mean, the output still varies
    frame-to-frame with the input, so the decoded skeleton always moves coherently
    instead of collapsing to a single static mean pose.
    """

    def __init__(
        self,
        proj_dim: int = 256,
        pred_dim: int = 256,
        horizons: list[int] | None = None,
        horizon_emb_dim: int = 16,
    ):
        super().__init__()
        self.horizons = horizons or [3]
        self._k_to_idx = {k: i for i, k in enumerate(self.horizons)}
        self.horizon_emb = nn.Embedding(len(self.horizons), horizon_emb_dim)
        self.mlp = MLP([proj_dim + horizon_emb_dim, 256, pred_dim])

    def _horizon_idx(self, s_x: torch.Tensor, k: torch.Tensor | None) -> torch.Tensor:
        if k is None:
            # Default to the shortest configured horizon (index 0).
            return torch.zeros(s_x.shape[0], device=s_x.device, dtype=torch.long)
        return torch.tensor(
            [self._k_to_idx[int(ki)] for ki in k],
            device=s_x.device,
            dtype=torch.long,
        )

    def forward(
        self, representation: torch.Tensor, k: torch.Tensor | None = None
    ) -> torch.Tensor:
        idx = self._horizon_idx(representation, k)
        h = self.horizon_emb(idx)
        delta = self.mlp(torch.cat([representation, h], dim=-1))
        return representation + delta


class TemporalContextModule(nn.Module):
    """Fuse K context frames into one effective pose representation (Phase 2).

    Each frame is passed through a shared per-frame transform, then combined with
    learned, softmax-normalized recency weights — recent frames can be weighted
    more heavily. Output is a single [B, pose_dim] vector consumed by the existing
    single-frame encoder, so all EMA machinery stays untouched.
    """

    def __init__(self, pose_dim: int, context_len: int):
        super().__init__()
        self.K = context_len
        # Linear ramp initialisation: most recent frame starts with highest weight.
        # Softmax of [1,2,...,K] gives a gentle recency prior; the network can still
        # learn any distribution, but starts from something useful rather than uniform.
        init_weights = torch.arange(1, context_len + 1, dtype=torch.float)
        self.weights = nn.Parameter(init_weights)
        self.mlp = MLP([pose_dim, 256, pose_dim])  # per-frame transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, pose_dim]
        B, K, D = x.shape
        w = torch.softmax(self.weights, dim=0)              # [K] learned recency
        encoded = self.mlp(x.reshape(B * K, D)).reshape(B, K, D)
        fused = (encoded * w.view(1, K, 1)).sum(dim=1)      # [B, pose_dim]
        return fused


class FKPoseDecoder(nn.Module):
    """Predicts local 6D joint rotations and resolves Cartesian positions via FK."""

    def __init__(
        self,
        repr_dim: int,
        num_joints: int,
        parents: list[int],
        bone_offsets: np.ndarray,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.rotation_decoder = nn.Sequential(
            nn.Linear(repr_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 6),
        )
        self.fk_layer = ForwardKinematics(parents, bone_offsets)

    def forward(self, latent_repr: torch.Tensor) -> torch.Tensor:
        batch_size = latent_repr.shape[0]
        pred_rotations_6d = self.rotation_decoder(latent_repr)
        pred_rotations_6d = pred_rotations_6d.view(batch_size, self.num_joints, 6)
        return self.fk_layer(pred_rotations_6d)


class VisualizationDecoder(FKPoseDecoder):
    """FK-backed spatial decoder; outputs Cartesian coordinates [B, pose_dim]."""
