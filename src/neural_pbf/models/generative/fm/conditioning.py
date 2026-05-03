"""ConditioningEncoder: maps scalar process parameters to an embedding vector."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ConditioningEncoder(nn.Module):
    """Three-layer MLP that maps (B, D_cond) scalars to (B, cond_embed_dim).

    Args:
        cond_in_dim:   Number of scalar conditioning inputs (e.g. 12).
        cond_embed_dim: Output embedding dimensionality.
    """

    def __init__(self, cond_in_dim: int, cond_embed_dim: int) -> None:
        super().__init__()
        hidden = max(cond_embed_dim, cond_in_dim * 4)
        self.net = nn.Sequential(
            nn.Linear(cond_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

    def forward(self, scalars: Tensor) -> Tensor:
        """
        Args:
            scalars: (B, cond_in_dim) z-score normalised conditioning scalars.

        Returns:
            (B, cond_embed_dim)
        """
        return self.net(scalars)
