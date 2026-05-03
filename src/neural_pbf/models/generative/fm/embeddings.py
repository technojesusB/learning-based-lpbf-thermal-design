"""Sinusoidal time embedding and AdaGroupNorm3d for FiLM-style conditioning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_time_embedding(tau: Tensor, dim: int) -> Tensor:
    """Map scalar τ ∈ [0, 1] to a (B, dim) sinusoidal embedding.

    Args:
        tau: Shape (B,), values in [0, 1].
        dim: Embedding dimensionality (must be even).

    Returns:
        Tensor of shape (B, dim).
    """
    assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
    device = tau.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)  # clamp avoids 0/0 when dim==2
    )  # (half,)
    # tau: (B,) → (B, 1); freqs: (half,) → (1, half)
    args = tau.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class AdaGroupNorm3d(nn.Module):
    """Adaptive Group Norm (FiLM-style) for 3D feature maps.

    Projects a conditioning vector to per-channel (scale, shift) and applies:
        y = GroupNorm(x) * (1 + scale) + shift

    Args:
        num_groups: Number of groups for GroupNorm.
        num_channels: Number of feature channels in the input.
        cond_dim: Dimensionality of the conditioning vector.
    """

    def __init__(self, num_groups: int, num_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=num_channels, affine=False
        )
        self.proj = nn.Linear(cond_dim, num_channels * 2)
        # Zero-init so at init the module is a pure GroupNorm
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x:    (B, C, Nz, Ny, Nx)
            cond: (B, cond_dim)

        Returns:
            (B, C, Nz, Ny, Nx)
        """
        x_norm = self.norm(x)
        scale_shift = self.proj(cond)  # (B, 2*C)
        scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, C)
        # Broadcast over spatial dims
        scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x_norm * (1.0 + scale) + shift
