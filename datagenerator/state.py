# data/state.py
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ThermalState:
    """
    Stateful maps carried across multiple dots/pulses.
    All tensors shape: [1,1,H,W]
    """
    T: torch.Tensor
    t_since: torch.Tensor
    E_acc: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.T.device

    @property
    def dtype(self) -> torch.dtype:
        return self.T.dtype

    def clone(self) -> "ThermalState":
        return ThermalState(self.T.clone(), self.t_since.clone(), self.E_acc.clone())


@torch.no_grad()
def init_state(
    H: int,
    W: int,
    device: torch.device,
    dtype: torch.dtype,
    T0: torch.Tensor,
    t_since_init: float = 1e6,
) -> ThermalState:
    assert T0.shape == (1, 1, H, W)
    t_since = torch.full((1, 1, H, W), t_since_init, device=device, dtype=dtype)
    E_acc = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    return ThermalState(T=T0.to(device=device, dtype=dtype).clone(), t_since=t_since, E_acc=E_acc)


@torch.no_grad()
def update_history_maps(
    state: ThermalState,
    spot: torch.Tensor,
    dt_event: float,
    energy: float,
    reset_threshold: float = 0.2,
) -> ThermalState:
    """
    Update t_since and E_acc after one dot event.

    spot: [1,1,H,W] spatial footprint (e.g., Gaussian spot), typically normalized [0..1]
    dt_event: elapsed time since last event (in your toy units)
    energy: scalar energy deposited for this dot (toy units)

    reset_threshold: pixels with spot>threshold are considered "hit" and have t_since reset.
    """
    assert spot.shape == state.T.shape

    # advance time since last hit everywhere
    state.t_since += dt_event

    # deposit energy map (very simple accumulation proxy)
    state.E_acc += float(energy) * spot

    # reset t_since where we "hit" strongly
    hit = spot > reset_threshold
    state.t_since = torch.where(hit, torch.zeros_like(state.t_since), state.t_since)

    return state
