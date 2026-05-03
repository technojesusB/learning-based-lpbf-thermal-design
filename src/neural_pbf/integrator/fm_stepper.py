"""FMStepper — inference engine for the Flow Matching surrogate.

Mirrors the TimeStepper.step_adaptive interface: takes a SimulationState
and advances it by one macro-step using the trained VelocityNet.
Immutability is enforced: input state is never mutated.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.velocity_net import VelocityNet


class FMStepper:
    """Inference engine that advances a SimulationState using the FM surrogate.

    Uses Euler ODE integration on τ ∈ [0, 1]:
        x_{τ + dτ} = x_τ + v_θ(x_τ, τ, cond) · dτ

    The mask and Q channels are kept constant throughout the ODE integration
    (they are conditioning inputs, not predicted quantities).

    Args:
        model:        Trained VelocityNet.
        cond_encoder: Trained ConditioningEncoder.
        sim_cfg:      SimulationConfig for the target domain.
        fm_cfg:       FMConfig used during training.
        device:       Target device for inference.
    """

    def __init__(
        self,
        model: VelocityNet,
        cond_encoder: ConditioningEncoder,
        sim_cfg: SimulationConfig,
        fm_cfg: FMConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.cond_encoder = cond_encoder.to(device)
        self.sim_cfg = sim_cfg
        self.fm_cfg = fm_cfg
        self.device = device

    def step(
        self,
        state: SimulationState,
        conditioning: Tensor,
        n_steps: int | None = None,
        dt_target: float | None = None,
        # SDE extension hook — None = pure ODE; callable adds stochasticity
        # TODO: activate for Euler-Maruyama SDE sampling
        noise_schedule: Callable[[Tensor, float], Tensor] | None = None,
    ) -> SimulationState:
        """Advance state by one macro-step using the FM surrogate.

        Args:
            state:        Input SimulationState (not mutated).
            conditioning: (cond_dim,) z-score normalised scalar conditioning vector.
            n_steps:      ODE integration steps. Defaults to fm_cfg.n_inference_steps.
            dt_target:    Macro time-step [s]. Defaults to sim_cfg.dt_base.

        Returns:
            A new SimulationState with updated T, t, step, and max_T.
        """
        if conditioning.ndim != 1:
            raise ValueError(
                f"conditioning must be a 1-D tensor of shape (cond_dim,), "
                f"got shape {tuple(conditioning.shape)}"
            )
        if state.T.ndim != 5:
            raise ValueError(
                f"FMStepper requires a 3-D SimulationState (T.ndim=5), "
                f"got T.ndim={state.T.ndim}"
            )

        n_steps = n_steps if n_steps is not None else self.fm_cfg.n_inference_steps
        dt_target = dt_target if dt_target is not None else self.sim_cfg.dt_base

        device = self.device
        cfg = self.fm_cfg

        self.model.eval()
        self.cond_encoder.eval()

        with torch.no_grad():
            # --- Prepare inputs ---
            T = state.T.to(device, dtype=torch.float32)
            mask = state.material_mask
            if mask is None:
                mask = torch.zeros_like(T, dtype=torch.float32)
            else:
                mask = mask.to(device, dtype=torch.float32)

            # Build Q channel: use Q from state if available, else zeros
            # VelocityNet expects shape (1, 3, Nz, Ny, Nx): [T_τ, mask, Q]
            # During inference we pass Q=0 unless the caller provides it via
            # a state field (not a standard SimulationState field, so zeros).
            Q_channel = torch.zeros_like(T)  # (1, 1, Nz, Ny, Nx)

            # Normalise T: (T - T_ambient) / T_ref
            T_norm = (T - cfg.T_ambient) / cfg.T_ref

            # Encode conditioning once
            cond_vec = conditioning.to(device, dtype=torch.float32).unsqueeze(0)
            cond_emb = self.cond_encoder(cond_vec)  # (1, cond_embed_dim)

            # Start from noise
            x_T_channel = torch.randn_like(T_norm)  # (1, 1, Nz, Ny, Nx)

            # Build 3-channel input tensor (T_τ, mask, Q)
            def _make_x_tau(t_channel: Tensor) -> Tensor:
                return torch.cat([t_channel, mask, Q_channel], dim=1)  # (1, 3, ...)

            # Euler ODE on τ ∈ [0, 1]
            dτ = 1.0 / n_steps
            for i in range(n_steps):
                tau_i = torch.full((1,), i * dτ, device=device, dtype=torch.float32)
                x_tau_full = _make_x_tau(x_T_channel)
                v = self.model(x_tau_full, tau_i, cond_emb)  # (1, 1, Nz, Ny, Nx)
                x_T_channel = x_T_channel + v * dτ

                if noise_schedule is not None:
                    # TODO: Euler-Maruyama SDE term (σ(τ) * sqrt(dτ) * ε)
                    pass

            # Denormalise: T_new = x_T_channel * T_ref + T_ambient
            T_new = x_T_channel * cfg.T_ref + cfg.T_ambient

        # --- Build new immutable state ---
        new_state = state.clone()
        new_state.T = T_new.to(state.T.device)
        new_state.t = state.t + dt_target
        new_state.step = state.step + 1

        if new_state.max_T is not None:
            new_state.max_T = torch.maximum(new_state.max_T, new_state.T)
        else:
            new_state.max_T = new_state.T.clone()

        return new_state

    def rollout(
        self,
        state: SimulationState,
        conditioning_seq: list[Tensor],
        n_steps: int | None = None,
        dt_target: float | None = None,
    ) -> list[SimulationState]:
        """Autoregressive rollout over a sequence of conditioning vectors.

        Args:
            state:            Initial SimulationState (not mutated).
            conditioning_seq: List of (cond_dim,) tensors, one per macro-step.
            n_steps:          ODE integration steps per macro-step.
            dt_target:        Time step per macro-step [s].

        Returns:
            List of SimulationStates, one per conditioning step.
        """
        results: list[SimulationState] = []
        current = state
        for cond in conditioning_seq:
            current = self.step(current, cond, n_steps=n_steps, dt_target=dt_target)
            results.append(current)
        return results
