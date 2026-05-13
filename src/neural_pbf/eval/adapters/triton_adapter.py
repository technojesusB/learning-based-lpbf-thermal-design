"""TritonAdapter — wraps TimeStepper.step_adaptive as a BaseStepper."""

from __future__ import annotations

from typing import Any

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig


class TritonAdapter:
    """Adapts :class:`TimeStepper` to the :class:`BaseStepper` protocol.

    Args:
        sim_cfg:     Simulation domain configuration.
        mat_cfg:     Material configuration.
        use_triton:  Whether to activate the Triton GPU kernel path.
        name:        Display name used in reports (default ``"triton"``).
    """

    def __init__(
        self,
        sim_cfg: SimulationConfig,
        mat_cfg: MaterialConfig,
        use_triton: bool = True,
        name: str = "triton",
    ) -> None:
        self.name = name
        self._stepper = TimeStepper(sim_cfg, mat_cfg)
        self._use_triton = use_triton

    def step(
        self,
        state: SimulationState,
        Q_ext: torch.Tensor | None,
        dt: float,
        conditioning: dict[str, Any] | None = None,
    ) -> SimulationState:
        # step_adaptive mutates the state in-place — always pass a clone.
        return self._stepper.step_adaptive(
            state.clone(),
            dt_target=dt,
            Q_ext=Q_ext,
            use_triton=self._use_triton,
        )
