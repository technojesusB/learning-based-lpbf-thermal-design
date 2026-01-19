# src/lpbf/state.py
from __future__ import annotations
from dataclasses import dataclass, field
import torch

@dataclass
class SimulationState:
    """
    Mutable state container for the simulation.
    Tensors are expected to be on the same device.
    """
    # Primary fields
    T: torch.Tensor          # Temperature [K], shape (B, 1, [D], H, W)
    t: float = 0.0          # Current simulation time [s]
    step: int = 0           # Step count

    # Auxiliary / History fields for analysis
    max_T: torch.Tensor | None = None          # Max T reached at each pixel
    cooling_rate: torch.Tensor | None = None   # Captured cooling rate [K/s] (e.g. at solidification)
    
    # Internal state for integrators (e.g. previous step T for dT/dt)
    T_prev: torch.Tensor | None = None
    
    def __post_init__(self):
        if self.max_T is None:
            self.max_T = self.T.clone()
        if self.cooling_rate is None:
            self.cooling_rate = torch.zeros_like(self.T)

    @property
    def device(self) -> torch.device:
        return self.T.device

    @property
    def dtype(self) -> torch.dtype:
        return self.T.dtype

    def clone(self) -> "SimulationState":
        return SimulationState(
            T=self.T.clone(),
            t=self.t,
            step=self.step,
            max_T=self.max_T.clone() if self.max_T is not None else None,
            cooling_rate=self.cooling_rate.clone() if self.cooling_rate is not None else None,
            T_prev=self.T_prev.clone() if self.T_prev is not None else None
        )
