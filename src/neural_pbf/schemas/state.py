# schemas/state.py
from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

Tensor = torch.Tensor


class FinalState(BaseModel):
    """
    Final fields after a full scan / event sequence.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    T: Tensor  # [1,1,H,W]
    E_acc: Tensor  # [1,1,H,W]
    t_since: Tensor  # [1,1,H,W]
    T_peak_global: Tensor  # [1,1,H,W]

    cooling_rate: Tensor | None = None  # [1,1,H,W]  (Toy units: per toy-second)


class SnapshotState(BaseModel):
    """
    Time-resolved snapshots for visualization / animation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    t: Tensor  # [N]
    event_idx: Tensor  # [N]
    T: Tensor  # [N,1,H,W]
    E_acc: Tensor  # [N,1,H,W]
    t_since: Tensor  # [N,1,H,W]

    cooling_rate: Tensor | None = None  # [N,1,H,W]


class StateMeta(BaseModel):
    """
    Lightweight run metadata (no heavy tensors).
    """

    model_config = ConfigDict(frozen=True)

    H: int
    W: int
    dt: float
    loss_h: float
    T_ambient: float
    description: str | None = None

    cooling_delta_t: float | None = None  # requested Δt (toy)
    cooling_delta_t_eff: float | None = None  # actual used Δt (delta_steps*dt)
    cooling_delta_steps: int | None = None


class ThermalStates(BaseModel):
    """
    Full container returned by a simulation run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    final: FinalState
    snapshots: SnapshotState
    meta: StateMeta
