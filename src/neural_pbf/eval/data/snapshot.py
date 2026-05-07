"""Snapshot — a single point-in-time capture of the simulation state."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Snapshot:
    """Immutable record of one macro-step in a trajectory.

    All tensors retain the canonical LPBF shape: ``(1, 1, [Nz,] Ny, Nx)``.

    Attributes:
        T:             Temperature field [K].
        t:             Absolute simulation time at this snapshot [s].
        dt:            Time-step used to advance *to* this snapshot [s].
        Q_ext:         External volumetric heat source [W/m³] (optional).
        material_mask: Phase mask (0=Powder, 1=Solid/Liquid) (optional).
    """

    T: torch.Tensor
    t: float
    dt: float
    Q_ext: torch.Tensor | None = None
    material_mask: torch.Tensor | None = None
