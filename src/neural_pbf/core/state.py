# src/lpbf/state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from neural_pbf.core.config import SimulationConfig


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
        cooling_rate (torch.Tensor | None): Field capturing the instantaneous
            cooling rate [K/s] at the moment of solidification (crossing
            T_solidus).
        T_prev (torch.Tensor | None): Temperature field from the previous time
            step. Used for finite difference time derivatives (cooling rate).
    """

    # Primary fields
    T: torch.Tensor
    t: float = 0.0
    step: int = 0

    # Auxiliary / History fields for analysis
    max_T: torch.Tensor | None = None
    cooling_rate: torch.Tensor | None = None

    # Material Phase Mask: 0 = Powder, 1 = Solid / Liquid
    # Used for irreversible transitions (once melted, powder becomes solid).
    material_mask: torch.Tensor | None = None

    # Internal state for integrators (e.g. previous step T for dT/dt)
    T_prev: torch.Tensor | None = None

    # Diagnostic: CFL sub-steps taken by the last step_adaptive call.
    # Set by step_adaptive; None until first call. Not physically meaningful
    # across clone/checkpoint boundaries — treat as a last-call diagnostic.
    last_n_sub: int | None = None

    def __post_init__(self):
        """Initialize auxiliary fields if not provided."""
        if self.max_T is None:
            self.max_T = self.T.clone()
        if self.cooling_rate is None:
            self.cooling_rate = torch.zeros_like(self.T)
        if self.material_mask is None:
            # Default to Powder (0) everywhere
            self.material_mask = torch.zeros_like(self.T, dtype=torch.uint8)

    @property
    def device(self) -> torch.device:
        """Get the device of the tensor state."""
        return self.T.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the floating point dtype of the state."""
        return self.T.dtype

    @classmethod
    def zeros(
        cls,
        sim_cfg: SimulationConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        T_initial: float | None = None,
    ) -> SimulationState:
        """Create a zero-initialized SimulationState on the given device.

        Args:
            sim_cfg:   Simulation configuration (determines grid shape).
            device:    Target device for all tensors.
            dtype:     Floating-point dtype for T and related tensors (default float32).
            T_initial: Fill temperature [K]. Defaults to ``sim_cfg.T_ambient``.

        Returns:
            A freshly initialized :class:`SimulationState` where all tensors
            reside on ``device``.
        """
        fill_value: float = T_initial if T_initial is not None else sim_cfg.T_ambient

        if sim_cfg.is_3d:
            shape = (1, 1, sim_cfg.Nz, sim_cfg.Ny, sim_cfg.Nx)
        else:
            shape = (1, 1, sim_cfg.Ny, sim_cfg.Nx)

        T = torch.full(shape, fill_value, dtype=dtype, device=device)
        max_T = T.clone()
        cooling_rate = torch.zeros_like(T)
        material_mask = torch.zeros_like(T, dtype=torch.uint8)

        return cls(
            T=T,
            t=0.0,
            step=0,
            max_T=max_T,
            cooling_rate=cooling_rate,
            material_mask=material_mask,
        )

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
            material_mask=self.material_mask.clone()
            if self.material_mask is not None
            else None,
            T_prev=self.T_prev.clone() if self.T_prev is not None else None,
            last_n_sub=self.last_n_sub,
        )
