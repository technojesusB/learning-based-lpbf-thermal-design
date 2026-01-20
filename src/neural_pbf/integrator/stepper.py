# src/lpbf/integrator/stepper.py
from __future__ import annotations

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.physics.material import MaterialConfig, cp_eff, k_eff
from neural_pbf.physics.ops import div_k_grad
from neural_pbf.scan.sources import HeatSource


class TimeStepper:
    """
    Solves the transient heat equation using Finite Differences.

    Governing Equation:
        rho * cp(T) * dT/dt = Div(k(T) * Grad(T)) + Q_source

    Implements:
        - Explicit Euler integration scheme.
        - Adaptive sub-stepping for numerical stability (CFL condition).
        - Cooling rate capture logic at solidification interface.
    """

    def __init__(self, sim_config: SimulationConfig, mat_config: MaterialConfig):
        """
        Initialize the time stepper.

        Args:
            sim_config (SimulationConfig): Grid and Domain settings.
            mat_config (MaterialConfig): Material properties.
        """
        self.sim = sim_config
        self.mat = mat_config

    def step(
        self,
        state: SimulationState,
        dt: float,
        heat_sources: list[HeatSource] | None = None,
        XY_grid: tuple[torch.Tensor, torch.Tensor] | None = None,
        Z_grid: torch.Tensor | None = None,
    ) -> SimulationState:
        """
        Perform a single time step of size dt.

        Note: This is a legacy interface. Use `step_adaptive` for robust integration.

        Args:
            state (SimulationState): Current state.
            dt (float): Time step [s].
            heat_sources (List[HeatSource]): List of heat source objects.
            XY_grid (tuple): Pre-computed meshgrid (X, Y).
            Z_grid (torch.Tensor): Pre-computed Z grid.

        Returns:
            SimulationState: Updated state.
        """
        # Placeholder for high-level step that handles source evaluation internally.
        # Currently not fully implemented with source evaluation.
        raise NotImplementedError("Use step_adaptive instead.")

    def step_explicit_euler(
        self, state: SimulationState, dt: float, Q_ext: torch.Tensor | None = None
    ) -> SimulationState:
        """
        Perform a single Explicit Euler integration step.

        Update rule:
            T_new = T + dt * (Div(k Grad T) + Q_ext) / (rho * cp)

        Side Effects:
            - Updates state.T, state.t, state.step.
            - Updates state.max_T.
            - Updates state.cooling_rate if solidification occurs.
            - Stores state.T_prev.

        Args:
            state (SimulationState): Current simulation state (mutated in place,
                but returned).
            dt (float): Timestep size [s]. MUST satisfy stability criterion.
            Q_ext (torch.Tensor | None): External volumetric heat source field
                [W/m^3].
                                         Assumed constant over the timestep.

        Returns:
            SimulationState: Ref to the updated state.
        """
        T = state.T

        # Material props
        k = k_eff(T, self.mat)
        cp = cp_eff(T, self.mat)
        rho = self.mat.rho  # constant density for now

        # Diffusion term: Div(k Grad T)
        # dx, dy, dz from sim config
        dz = self.sim.dz if self.sim.is_3d else None
        div_term = div_k_grad(T, k, self.sim.dx, self.sim.dy, dz)

        # Heat Source
        Q = Q_ext if Q_ext is not None else torch.zeros_like(T)

        # CRITICAL FIX: Unit Consistency for 2D
        # In 2D, Q_ext is often a surface flux [W/m^2] from GaussianBeam.
        # The PDE expects volumetric source [W/m^3].
        # We must divide by the effective layer thickness dz.
        # In 3D, Q_ext is already [W/m^3] (handled by source.intensity).
        if not self.sim.is_3d and Q_ext is not None:
             # dz is now the physical thickness from config
            Q = Q / self.sim.dz

        # Loss term (linear cooling)
        loss_term = -self.sim.loss_h * (T - self.sim.T_ambient)

        # RHS = Div + Q + Loss
        # Neumann BC is handled inside div_term (padding).
        rhs = div_term + Q + loss_term

        # Update Temperature
        dT = (dt / rho) * (rhs / (cp + 1e-9))
        T_new = T + dT

        # Update State
        state.T_prev = T  # store for cooling rate calculation
        state.T = T_new
        state.t += dt
        state.step += 1

        # Analysis: Max Temperature
        if state.max_T is not None:
            state.max_T = torch.maximum(state.max_T, T_new)

        # Analysis: Cooling Rate at Solidification
        # Condition: T_prev > T_solidus AND T_new <= T_solidus (crossing downwards)
        T_sol = self.mat.T_solidus

        crossing_mask = (state.T_prev > T_sol) & (T_new <= T_sol)

        if crossing_mask.any() and state.cooling_rate is not None:
            # CR = (T_prev - T_new) / dt   [K/s]
            # We record the POSITIVE instantaneous cooling rate.
            cr_inst = (state.T_prev - T_new) / dt

            # Update cooling rate map only at crossing pixels.
            # If a pixel remelts and solidifies again, this overrides the
            # previous value. This captures the *last* solidification event.
            state.cooling_rate = torch.where(crossing_mask, cr_inst, state.cooling_rate)

        return state

    def estimate_stability_dt(self, state: SimulationState) -> float:
        """
        Estimate the maximum stable timestep for the Explicit Euler scheme.

        Criterion:
            dt <= dx^2 / (2 * D * alpha_max)

        Where:
            D: Dimensionality (2 or 3).
            alpha_max: Maximum thermal diffusivity = max(k) / (rho * min(cp)).

        Args:
            state (SimulationState): Current state (unused here, but possibly
                needed for local alpha).

        Returns:
            float: Recommended maximum timestep [s] (including safety factor 0.9).
        """
        # Conservative estimates (max k, min cp)
        k_max = max(self.mat.k_powder, self.mat.k_solid, self.mat.k_liquid)
        cp_min = (
            self.mat.cp_base
        )  # ignore latent heat effective cp for stability bound (safety)
        rho = self.mat.rho

        alpha_max = k_max / (rho * cp_min)

        dx = self.sim.dx
        dy = self.sim.dy
        d_min = min(dx, dy)
        if self.sim.is_3d:
            d_min = min(d_min, self.sim.dz)

        # Stability limit: dt <= d^2 / (2*Dims * alpha)
        dims = 3 if self.sim.is_3d else 2
        dt_crit = (d_min**2) / (2.0 * dims * alpha_max)

        # Safety factor
        return 0.9 * dt_crit

    def step_adaptive(
        self,
        state: SimulationState,
        dt_target: float,
        Q_ext: torch.Tensor | None = None,
    ) -> SimulationState:
        """
        Advance the simulation by `dt_target` using adaptive sub-stepping.

        If `dt_target` exceeds the stability limit, it is broken down into `n`
            smaller steps.

        Args:
            state (SimulationState): Current state.
            dt_target (float): Desired macro timestep [s].
            Q_ext (torch.Tensor | None): External heat source field [W/m^3].
                                         Assumed constant across all sub-steps.

        Returns:
            SimulationState: Updated state after dt_target.
        """
        dt_crit = self.estimate_stability_dt(state)

        import math

        n_sub = math.ceil(dt_target / dt_crit)

        dt_sub = dt_target / n_sub

        for _ in range(n_sub):
            # Apply identical Q_ext at each sub-step
            state = self.step_explicit_euler(state, dt_sub, Q_ext)

        return state
