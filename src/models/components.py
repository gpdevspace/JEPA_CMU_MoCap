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
    """Predicts the future embedding from the context embedding (unconditioned)."""

    def __init__(self, proj_dim: int = 256, pred_dim: int = 256):
        super().__init__()
        self.mlp = MLP([proj_dim, 128, pred_dim])

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.mlp(representation)


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
