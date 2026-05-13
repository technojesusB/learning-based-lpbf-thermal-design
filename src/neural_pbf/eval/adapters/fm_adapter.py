"""FMAdapter — wraps FMStepper as a BaseStepper.

conditioning dict contract
--------------------------
Required key:
    "vector": (cond_dim,) Tensor — z-score normalised scalar conditioning.

Future universal-surrogate keys (7–8 channel expansion):
    "k":   (1,1,[Nz,]Ny,Nx) Tensor — instantaneous thermal conductivity.
    "cp":  (1,1,[Nz,]Ny,Nx) Tensor — instantaneous heat capacity.
    "phi": (1,1,[Nz,]Ny,Nx) Tensor — melt fraction field.
"""

from __future__ import annotations

from typing import Any

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.fm_stepper import FMStepper
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.velocity_net import VelocityNet


class FMAdapter:
    """Adapts :class:`FMStepper` to the :class:`BaseStepper` protocol.

    Args:
        model:        Trained :class:`VelocityNet`.
        cond_encoder: Trained :class:`ConditioningEncoder`.
        sim_cfg:      Simulation domain configuration.
        fm_cfg:       Flow Matching configuration used during training.
        device:       Target device for inference.
        name:         Display name (default ``"fm_surrogate"``).
    """

    def __init__(
        self,
        model: VelocityNet,
        cond_encoder: ConditioningEncoder,
        sim_cfg: SimulationConfig,
        fm_cfg: FMConfig,
        device: torch.device,
        name: str = "fm_surrogate",
    ) -> None:
        self.name = name
        self._stepper = FMStepper(
            model=model,
            cond_encoder=cond_encoder,
            sim_cfg=sim_cfg,
            fm_cfg=fm_cfg,
            device=device,
        )

    def step(
        self,
        state: SimulationState,
        Q_ext: torch.Tensor | None,
        dt: float,
        conditioning: dict[str, Any] | None = None,
    ) -> SimulationState:
        if conditioning is None or "vector" not in conditioning:
            raise ValueError(
                "FMAdapter requires a conditioning dict with key 'vector' "
                "containing the (cond_dim,) conditioning tensor."
            )
        cond_vec: torch.Tensor = conditioning["vector"]
        return self._stepper.step(state=state, conditioning=cond_vec, dt_target=dt)
