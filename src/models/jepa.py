"""JEPA orchestrator with online/EMA encoders and predictor."""

import copy

import numpy as np
import torch
import torch.nn as nn

from models.components import (
    Encoder,
    Predictor,
    Projector,
    VisualizationDecoder,
)


class JEPA(nn.Module):
    def __init__(
        self,
        pose_dim: int,
        repr_dim: int = 256,
        proj_dim: int = 256,
        pred_dim: int = 256,
        latent_dim: int = 32,
        num_classes: int = 6,
        use_latent: bool = False,
        num_joints: int | None = None,
        parents: list[int] | None = None,
        bone_offsets: np.ndarray | None = None,
    ):
        super().__init__()
        self.use_latent = use_latent
        self.proj_dim = proj_dim
        self.pose_dim = pose_dim
        self.num_joints = num_joints or pose_dim // 3

        if parents is None or bone_offsets is None:
            raise ValueError("parents and bone_offsets are required for FK decoder")

        self.encoder = Encoder(pose_dim, repr_dim)
        self.projector = Projector(repr_dim, proj_dim)

        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_projector = copy.deepcopy(self.projector)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        for p in self.ema_projector.parameters():
            p.requires_grad = False

        self.predictor = Predictor(proj_dim, pred_dim)
        self.vis_decoder = VisualizationDecoder(
            pred_dim, self.num_joints, parents, bone_offsets
        )

        self._ema_momentum = 0.9999

    def set_ema_momentum(self, momentum: float) -> None:
        self._ema_momentum = momentum

    @torch.no_grad()
    def update_ema(self) -> None:
        m = self._ema_momentum
        for ema_p, online_p in zip(
            self.ema_encoder.parameters(), self.encoder.parameters()
        ):
            ema_p.data.mul_(m).add_(online_p.data, alpha=1 - m)
        for ema_p, online_p in zip(
            self.ema_projector.parameters(), self.projector.parameters()
        ):
            ema_p.data.mul_(m).add_(online_p.data, alpha=1 - m)

    def encode_online(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.projector(h)

    def encode_repr(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def encode_ema(self, y: torch.Tensor) -> torch.Tensor:
        h = self.ema_encoder(y)
        return self.ema_projector(h)

    def encode_for_rollout(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the future embedding from a pose (used for autoregressive rollout)."""
        return self.predictor(self.encode_online(x))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_x = self.encode_online(x)
        s_y = self.encode_ema(y)  # already detached (frozen EMA, no_grad)
        s_y_hat = self.predictor(s_x)  # predict the absolute future embedding
        return s_y_hat, s_y, s_x

    def reconstruct(self, poses: torch.Tensor) -> torch.Tensor:
        """Encode poses then FK-decode back to Cartesian coordinates (detached latent)."""
        s = self.encode_online(poses).detach()
        return self.vis_decoder(s)

    def decode_pose(self, representation: torch.Tensor) -> torch.Tensor:
        """Decode a latent embedding to rigid-bone Cartesian coordinates."""
        return self.vis_decoder(representation)
