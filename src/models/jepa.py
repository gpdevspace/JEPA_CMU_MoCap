"""JEPA orchestrator with online/EMA encoders and predictor."""

import copy

import torch
import torch.nn as nn

from models.components import (
    ActionEmbedding,
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
    ):
        super().__init__()
        self.use_latent = use_latent
        self.proj_dim = proj_dim

        self.encoder = Encoder(pose_dim, repr_dim)
        self.projector = Projector(repr_dim, proj_dim)

        self.ema_encoder = copy.deepcopy(self.encoder)
        self.ema_projector = copy.deepcopy(self.projector)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        for p in self.ema_projector.parameters():
            p.requires_grad = False

        self.action_embedding = ActionEmbedding(num_classes, latent_dim)
        self.predictor = Predictor(proj_dim, pred_dim, latent_dim, use_latent)
        self.vis_decoder = VisualizationDecoder(pred_dim, pose_dim)

        self._ema_momentum = 0.996

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

    @torch.no_grad()
    def encode_ema(self, y: torch.Tensor) -> torch.Tensor:
        h = self.ema_encoder(y)
        return self.ema_projector(h)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_x = self.encode_online(x)
        s_y = self.encode_ema(y)

        z_emb = None
        if self.use_latent:
            if labels is None:
                raise ValueError("labels required when use_latent=True")
            z_emb = self.action_embedding(labels)

        s_y_hat = self.predictor(s_x, z_emb)
        return s_y_hat, s_y

    def decode_pose(self, representation: torch.Tensor) -> torch.Tensor:
        return self.vis_decoder(representation)
