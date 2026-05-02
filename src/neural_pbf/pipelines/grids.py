"""Device-safe coordinate grid construction for LPBF simulation domains."""

from __future__ import annotations

import torch

from neural_pbf.core.config import SimulationConfig


def make_coordinate_grids(
    sim_cfg: SimulationConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 3D meshgrid tensors for the simulation domain, placed on ``device``.

    Uses ``torch.meshgrid(z, y, x, indexing='ij')`` so each returned tensor has
    shape ``(Nz, Ny, Nx)``, matching the PyTorch convention for 3-D volumetric
    fields ``(batch, channel, D, H, W)``.  Each tensor resides on ``device`` and
    has the requested ``dtype``.

    Args:
        sim_cfg: Simulation configuration.  The physical lengths ``Lx_m``,
                 ``Ly_m``, ``Lz_m`` and grid counts ``Nx``, ``Ny``, ``Nz``
                 are read from this object.
        device:  Target device for all returned tensors.
        dtype:   Floating-point dtype (default ``torch.float32``).

    Returns:
        Tuple ``(X3, Y3, Z3)`` where each tensor has shape ``(Nz, Ny, Nx)``.
        ``X3[iz, iy, ix]`` is the x-coordinate of voxel ``(iz, iy, ix)``, and
        likewise for Y3 and Z3.

    Raises:
        ValueError: When ``sim_cfg`` is not 3D (``Lz`` is None or ``Nz <= 1``).
    """
    x = torch.linspace(0, sim_cfg.Lx_m, sim_cfg.Nx, device=device, dtype=dtype)
    y = torch.linspace(0, sim_cfg.Ly_m, sim_cfg.Ny, device=device, dtype=dtype)
    if not sim_cfg.is_3d:
        z = torch.zeros(1, device=device, dtype=dtype)
    else:
        z = torch.linspace(0, sim_cfg.Lz_m, sim_cfg.Nz, device=device, dtype=dtype)

    # Use (z, y, x) order so each grid has shape (Nz, Ny, Nx), matching
    # the PyTorch convention for 3-D fields: (batch, channel, D, H, W).
    Z3, Y3, X3 = torch.meshgrid(z, y, x, indexing="ij")
    return X3, Y3, Z3
