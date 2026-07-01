"""JEPA orchestrator with online/EMA encoders and predictor."""

import copy

import numpy as np
import torch
import torch.nn as nn

from models.components import (
    Encoder,
    Predictor,
    Projector,
    TemporalContextModule,
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
        horizons: list[int] | None = None,
        horizon_emb_dim: int = 16,
        context_len: int = 1,
    ):
        super().__init__()
        self.use_latent = use_latent
        self.proj_dim = proj_dim
        self.pose_dim = pose_dim
        self.context_len = context_len
        self.num_joints = num_joints or pose_dim // 3

        if parents is None or bone_offsets is None:
            raise ValueError("parents and bone_offsets are required for FK decoder")

        # v3: the encoder consumes a velocity-augmented pose [pose ‖ Δpose], so its
        # input dim is 2·pose_dim. Both the online and EMA paths use this dim, which
        # keeps the EMA copy valid. Decoder output stays raw pose_dim (num_joints).
        input_dim = pose_dim * 2

        # Phase 2: fuse K context frames into one effective (augmented) pose before
        # the encoder. Online path only — the EMA target is a single augmented frame.
        self.temporal_ctx = (
            TemporalContextModule(input_dim, context_len) if context_len > 1 else None
        )

        self.encoder = Encoder(input_dim, repr_dim)
        self.projector = Projector(repr_dim, proj_dim)

        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_projector = copy.deepcopy(self.projector)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        for p in self.ema_projector.parameters():
            p.requires_grad = False

        self.predictor = Predictor(
            proj_dim, pred_dim, horizons=horizons, horizon_emb_dim=horizon_emb_dim
        )
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
        # x is [B, K, pose_dim] when temporal context is active, else [B, pose_dim].
        if self.temporal_ctx is not None:
            x = self.temporal_ctx(x)  # [B, K, pose_dim] -> [B, pose_dim]
        h = self.encoder(x)
        return self.projector(h)

    def encode_repr(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_ctx is not None:
            x = self.temporal_ctx(x)
        return self.encoder(x)

    @torch.no_grad()
    def encode_ema(self, y_aug: torch.Tensor) -> torch.Tensor:
        # y_aug is the velocity-augmented target [B, 2*pose_dim].
        h = self.ema_encoder(y_aug)
        return self.ema_projector(h)

    def encode_for_rollout(
        self, x: torch.Tensor, k: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict the future embedding from a pose (used for autoregressive rollout)."""
        return self.predictor(self.encode_online(x), k)

    def forward(
        self,
        x: torch.Tensor,
        y_aug: torch.Tensor,
        k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: velocity-augmented context [B, K, 2*pose_dim] (or [B, 2*pose_dim]).
        # y_aug: velocity-augmented target [B, 2*pose_dim].
        s_x = self.encode_online(x)
        s_y = self.encode_ema(y_aug)  # already detached (frozen EMA, no_grad)
        s_y_hat = self.predictor(s_x, k)  # predict the absolute future embedding
        return s_y_hat, s_y, s_x

    def reconstruct(self, y_aug: torch.Tensor) -> torch.Tensor:
        """FK-decode the augmented target via the EMA path to raw pose [B, pose_dim]."""
        s = self.encode_ema(y_aug)  # EMA encoder/projector, already no_grad
        return self.vis_decoder(s)

    def decode_pose(self, representation: torch.Tensor) -> torch.Tensor:
        """Decode a latent embedding to rigid-bone Cartesian coordinates."""
        return self.vis_decoder(representation)
