from __future__ import annotations

import torch


def make_xy_grid(
    H: int, W: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns X, Y with shape [1,1,H,W] in [0,1].
    """
    ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X[None, None], Y[None, None]
