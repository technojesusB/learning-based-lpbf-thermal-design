# physics/operators.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (2.0 * a * b) / (a + b + eps)


def div_k_grad_2d(T: torch.Tensor, k_cell: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    div(k grad T) on a 2D regular grid using face-centered harmonic means.
    Boundary: reflect padding ~= Neumann (zero flux).
    """
    assert T.shape == k_cell.shape and T.ndim == 4 and T.shape[1] == 1

    Tpad = F.pad(T, (1, 1, 1, 1), mode="reflect")
    kpad = F.pad(k_cell, (1, 1, 1, 1), mode="reflect")

    Tc = Tpad[:, :, 1:-1, 1:-1]
    Tl = Tpad[:, :, 1:-1, 0:-2]
    Tr = Tpad[:, :, 1:-1, 2:]
    Tu = Tpad[:, :, 0:-2, 1:-1]
    Td = Tpad[:, :, 2:, 1:-1]

    kc = kpad[:, :, 1:-1, 1:-1]
    kl = kpad[:, :, 1:-1, 0:-2]
    kr = kpad[:, :, 1:-1, 2:]
    ku = kpad[:, :, 0:-2, 1:-1]
    kd = kpad[:, :, 2:, 1:-1]

    kx_r = harmonic_mean(kc, kr)
    kx_l = harmonic_mean(kc, kl)
    ky_d = harmonic_mean(kc, kd)
    ky_u = harmonic_mean(kc, ku)

    dTdx_r = (Tr - Tc) / dx
    dTdx_l = (Tc - Tl) / dx
    dTdy_d = (Td - Tc) / dy
    dTdy_u = (Tc - Tu) / dy

    fx_r = kx_r * dTdx_r
    fx_l = kx_l * dTdx_l
    fy_d = ky_d * dTdy_d
    fy_u = ky_u * dTdy_u

    div = (fx_r - fx_l) / dx + (fy_d - fy_u) / dy
    return div
