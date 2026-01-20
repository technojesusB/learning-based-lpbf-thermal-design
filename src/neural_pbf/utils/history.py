# utils/history.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """Returns [1,1,K,K] Gaussian kernel normalized to sum=1."""
    assert kernel_size % 2 == 1, "kernel_size should be odd"
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k = k / (k.sum() + 1e-12)
    return k[None, None]


@torch.no_grad()
def make_smooth_preheat_field(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    ambient: float = 0.0,
    amplitude: float = 0.05,
    kernel_size: int = 51,
    sigma: float = 12.0,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> torch.Tensor:
    """
    Create a smooth random temperature field T0(x,y) ~ ambient + low-freq noise.
    Output shape: [1,1,H,W]

    amplitude: typical deviation from ambient
    kernel_size/sigma: controls spatial correlation length (bigger = smoother)
    """
    noise = torch.randn((1, 1, H, W), device=device, dtype=dtype)
    k = _gaussian_kernel2d(kernel_size, sigma, device, dtype)

    # reflect padding -> avoids boundary artifacts
    pad = kernel_size // 2
    noise_pad = F.pad(noise, (pad, pad, pad, pad), mode="reflect")
    smooth = F.conv2d(noise_pad, k)

    # normalize to ~[-1,1]
    smooth = smooth - smooth.mean()
    smooth = smooth / (smooth.std() + 1e-12)

    T0 = ambient + amplitude * smooth

    if clamp_min is not None or clamp_max is not None:
        T0 = torch.clamp(
            T0,
            min=clamp_min if clamp_min is not None else -float("inf"),
            max=clamp_max if clamp_max is not None else float("inf"),
        )
    return T0
