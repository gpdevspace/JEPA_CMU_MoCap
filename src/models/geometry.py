"""Rotation geometry utilities."""

import torch


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation vectors to 3x3 rotation matrices.
    Args:
        d6: Tensor of shape (..., 6)
    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]

    x = torch.nn.functional.normalize(x_raw, p=2, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = torch.nn.functional.normalize(z, p=2, dim=-1)
    y = torch.cross(z, x, dim=-1)

    return torch.stack((x, y, z), dim=-1)
