# src/lpbf/integrator/stepper.py
from __future__ import annotations

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.physics.material import MaterialConfig, cp_eff, k_eff
from neural_pbf.physics.ops import div_k_grad
from neural_pbf.scan.sources import HeatSource


def _assert_devices_match(
    state: SimulationState,
    Q_ext: torch.Tensor | None,
) -> None:
    """Raise RuntimeError if any state tensor or Q_ext is on a different device.

    Checks:
    - state.T.device == state.material_mask.device (when mask is not None)
    - state.T.device == Q_ext.device (when Q_ext is not None)

    Args:
        state:  Current simulation state.
        Q_ext:  Optional external heat source tensor.

    Raises:
        RuntimeError: When any device mismatch is detected.
    """
    state_device = state.T.device

    if state.material_mask is not None:
        mask_device = state.material_mask.device
        if mask_device.type != state_device.type:
            raise RuntimeError(
                f"Device mismatch: state.T on {state_device}, "
                f"state.material_mask on {mask_device}. "
                "All tensors must be on the same device."
            )

    if Q_ext is not None:
        q_device = Q_ext.device
        if q_device.type != state_device.type:
            raise RuntimeError(
                f"Device mismatch: state.T on {state_device}, "
                f"Q_ext on {q_device}. "
                "Move Q_ext to the same device as state.T before calling the stepper."
            )


try:
    from neural_pbf.physics.triton_ops import run_thermal_step_3d_triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


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
        # HIGH-1: LUT tensor cache keyed by device string to avoid repeated
        # CPU→GPU copies across sub-steps.  Populated lazily on first Triton call.
        # Key: str(device), Value: dict with "T_lut", "k_lut", "cp_lut" tensors.
        self._lut_tensors: dict[str, dict[str, torch.Tensor]] = {}

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
        self,
        state: SimulationState,
        dt: float,
        Q_ext: torch.Tensor | None = None,
        use_triton: bool = False,
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

        Note:
            state.material_mask is NOT updated here.  Mask updates happen at
            macro-step granularity inside step_adaptive (bitwise OR after all
            sub-steps complete) to avoid repeated large tensor allocations.

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
        _assert_devices_match(state, Q_ext)

        T = state.T

        if use_triton and HAS_TRITON and self.sim.is_3d:
            # FUSED TRITON PATH
            # Note: Triton kernel handles unit consistency internally for 3D.
            # Q_ext is [W/m^3] for 3D.
            Q = Q_ext if Q_ext is not None else torch.zeros_like(T)

            # The kernel currently expects [Nx, Ny, Nz] or [1,1,Nx,Ny,Nz]
            # We pass contiguous views
            mask = state.material_mask
            if mask is None:
                # Cache the default zero mask on the state to avoid re-allocating
                # 8+ million bytes every single substep (5000+ times per dt).
                state.material_mask = torch.zeros_like(T, dtype=torch.uint8)
                mask = state.material_mask

            # CRITICAL-2 fix: use squeeze(0).squeeze(0) instead of bare
            # squeeze() to remove ONLY the batch (dim 0) and channel (dim 1)
            # leading dimensions.  Bare squeeze() removes ALL size-1 dims,
            # which would silently drop a spatial dimension if Nz (or another
            # spatial dim) happened to be 1 — passing a 2D tensor to the 3D
            # kernel, causing illegal memory access.
            #
            # HIGH-1 fix: build LUT GPU tensors once per device and cache them
            # on self._lut_tensors to avoid 3 × n_sub CPU→GPU copies.
            dev_key = str(T.device)
            lut_cache: dict[str, torch.Tensor] | None = None
            if self.mat.use_lut and self.mat.T_lut is not None:
                if dev_key not in self._lut_tensors:
                    self._lut_tensors[dev_key] = {
                        "T_lut": torch.tensor(
                            self.mat.T_lut, device=T.device, dtype=T.dtype
                        ),
                        "k_lut": torch.tensor(
                            self.mat.k_lut, device=T.device, dtype=T.dtype
                        ),
                        "cp_lut": torch.tensor(
                            self.mat.cp_lut, device=T.device, dtype=T.dtype
                        ),
                    }
                lut_cache = self._lut_tensors[dev_key]

            T_new = run_thermal_step_3d_triton(
                T.squeeze(0).squeeze(0).contiguous(),
                mask.squeeze(0).squeeze(0).contiguous(),
                Q.squeeze(0).squeeze(0).contiguous(),
                self.sim,
                self.mat,
                dt,
                lut_tensors=lut_cache,
            )

            # Reshape back to (B, C, Nx, Ny, Nz)
            T_new = T_new.view_as(T)

            # NOTE: We intentionally DO NOT update the mask per sub-step here.
            # Doing `T_new > T_sol` and `|` allocates 16MB of tensors 5000+ times
            # per dt, stalling the GPU allocator. Mask updates should be done at
            # the macro-step level (in step_adaptive or externally).

            state.T_prev = T
            state.T = T_new
            state.t += dt
            state.step += 1

            # (Analysis and Cooling rate logic simplified for bench,
            # should ideally be in kernel but for now we do it here if needed)
            if state.max_T is not None:
                state.max_T = torch.maximum(state.max_T, T_new)

            return state

        # STANDARD PYTORCH PATH
        mask = state.material_mask
        if mask is None:
            state.material_mask = torch.zeros_like(T, dtype=torch.uint8)
            mask = state.material_mask

        k = k_eff(T, self.mat, mask)
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

        # NOTE: We intentionally DO NOT update the mask per sub-step here.
        # Mask updates should be done at the macro-step level (in step_adaptive).

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
        safety_factor: float = 0.9,
        use_triton: bool = False,
    ) -> SimulationState:
        """
        Advance the simulation by `dt_target` using adaptive sub-stepping.

        If `dt_target` exceeds the stability limit, it is broken down into `n`
            smaller steps.

        Args:
            state (SimulationState): Current state.
            dt_target (float): Desired macro timestep [s].
            Q_ext (torch.Tensor | None): External heat source field [W/m^3]
                                         (or Flux in 2D).
                                         Assumed constant across all sub-steps.
            safety_factor (float): Multiplier for CFL limit (default 0.9).

        Returns:
            SimulationState: Updated state after dt_target.
        """
        import math

        # 1. Estimate effective max diffusivity for current state?
        # For strict safety, use theoretical max of the material.
        # `estimate_stability_dt` uses conservative max(k)/min(cp).

        # Note: estimate_stability_dt generally assumes static properties.
        # The spike in cp REDUCES diffusivity (alpha = k/rho*cp), so it is SAFER.
        # The dangerous regime is high k, low cp.

        dt_crit = self.estimate_stability_dt(state) * safety_factor

        # 2. Determine number of sub-steps
        if dt_crit < 1e-12:
            # Fallback to avoid infinite loop if properties are broken
            dt_crit = 1e-6

        n_sub = math.ceil(dt_target / dt_crit)
        dt_sub = dt_target / n_sub

        # 3. Sub-step loop
        for _ in range(n_sub):
            # Apply identical Q_ext at each sub-step
            # Note: explicit Euler handles the Q unit normalization internally now.
            state = self.step_explicit_euler(
                state, dt_sub, Q_ext, use_triton=use_triton
            )

        # 4. Single macro-step mask update — promotes voxels that reached or
        #    crossed T_solidus during this macro-step.  Doing this once per
        #    macro-step rather than per sub-step avoids allocating large uint8
        #    tensors 5000+ times per dt_target.
        if state.T is not None and state.material_mask is not None:
            # CRITICAL-3 fix: remove .squeeze() — both tensors are already
            # (1, 1, Nx, Ny, Nz) so no squeeze is needed.  Bare squeeze()
            # would remove ALL size-1 dims (including spatial ones), making
            # view_as silently map wrong indices if a spatial dim is 1.
            newly_solid = (self.mat.T_solidus <= state.T).to(torch.uint8)
            state.material_mask = state.material_mask | newly_solid

        # Expose substep count for diagnostics.
        state.last_n_sub = n_sub

        return state
