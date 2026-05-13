"""BaseStepper protocol — the single interface all steppers must satisfy."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from neural_pbf.core.state import SimulationState


@runtime_checkable
class BaseStepper(Protocol):
    """Any object that can advance a SimulationState by one macro-step.

    The protocol is deliberately thin: adapters absorb all backend-specific
    complexity (CFL sub-stepping for TimeStepper, ODE integration for FM).

    Attributes:
        name: Human-readable identifier used in reports and MLflow tags.

    ``conditioning`` is an open dict so FM adapters can receive material
    fingerprints (``"vector"``) and future universal-surrogate channels
    (``"k"``, ``"cp"``, ``"phi"``) without polluting the protocol surface.
    """

    name: str

    def step(
        self,
        state: SimulationState,
        Q_ext: torch.Tensor | None,
        dt: float,
        conditioning: dict[str, Any] | None = None,
    ) -> SimulationState:
        ...
