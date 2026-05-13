"""IdentityAdapter — returns the input state unchanged (regression baseline)."""

from __future__ import annotations

from typing import Any

import torch

from neural_pbf.core.state import SimulationState


class IdentityAdapter:
    """No-op stepper: returns a clone of the input state with updated time.

    Useful as a zero-error baseline to verify that the rollout engine and
    metric pipeline are producing correct results.
    """

    name: str = "identity"

    def step(
        self,
        state: SimulationState,
        Q_ext: torch.Tensor | None,
        dt: float,
        conditioning: dict[str, Any] | None = None,
    ) -> SimulationState:
        new_state = state.clone()
        new_state.t = state.t + dt
        new_state.step = state.step + 1
        return new_state
