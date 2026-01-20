# src/lpbf/state.py
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SimulationState:
    """
    Mutable container for the complete state of the thermal simulation.

    This class holds the current temperature field, simulation time, and auxiliary
    history variables (max temperature, cooling rates). It supports cloning for
    checkpointing or look-ahead.

    Attributes:
        T (torch.Tensor): Current Temperature field [K].
                          Shape: (Batch, Channel, [Depth], Height, Width).
                          Typically (1, 1, [Nz], Ny, Nx).
        t (float): Current simulation time [s].
        step (int): Current integer time step count.
        max_T (torch.Tensor | None): Field tracking the maximum temperature reached
                                     at each voxel throughout history [K].
                                     Initialized to T on creation if None.
        cooling_rate (torch.Tensor | None): Field capturing the instantaneous cooling rate [K/s]
                                            at the moment of solidification (crossing T_solidus).
        T_prev (torch.Tensor | None): Temperature field from the previous time step.
                                      Used for finite difference time derivatives (cooling rate).
    """

    # Primary fields
    T: torch.Tensor
    t: float = 0.0
    step: int = 0

    # Auxiliary / History fields for analysis
    max_T: torch.Tensor | None = None
    cooling_rate: torch.Tensor | None = None

    # Internal state for integrators (e.g. previous step T for dT/dt)
    T_prev: torch.Tensor | None = None

    def __post_init__(self):
        """Initialize auxiliary fields if not provided."""
        if self.max_T is None:
            self.max_T = self.T.clone()
        if self.cooling_rate is None:
            self.cooling_rate = torch.zeros_like(self.T)

    @property
    def device(self) -> torch.device:
        """Get the device of the tensor state."""
        return self.T.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the floating point dtype of the state."""
        return self.T.dtype

    def clone(self) -> SimulationState:
        """
        Create a deep copy of the state. tensors are cloned.

        Returns:
            SimulationState: Identify copy.
        """
        return SimulationState(
            T=self.T.clone(),
            t=self.t,
            step=self.step,
            max_T=self.max_T.clone() if self.max_T is not None else None,
            cooling_rate=self.cooling_rate.clone()
            if self.cooling_rate is not None
            else None,
            T_prev=self.T_prev.clone() if self.T_prev is not None else None,
        )
