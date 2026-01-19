# src/lpbf/integrator/stepper.py
from __future__ import annotations
from typing import List, Callable
import torch
from lpbf.config import SimulationConfig
from lpbf.physics.material import MaterialConfig, k_eff, cp_eff, melt_fraction
from lpbf.physics.ops import div_k_grad
from lpbf.state import SimulationState
from lpbf.scan.sources import HeatSource

class TimeStepper:
    def __init__(self, sim_config: SimulationConfig, mat_config: MaterialConfig):
        self.sim = sim_config
        self.mat = mat_config

    def step(self, 
             state: SimulationState, 
             dt: float, 
             heat_sources: List[HeatSource] = [], 
             XY_grid: tuple[torch.Tensor, torch.Tensor] | None = None,
             Z_grid: torch.Tensor | None = None) -> SimulationState:
        """
        Perform a single time step of size dt.
        Updates T, t, max_T, cooling_rate.
        """
        T = state.T
        t = state.t
        
        # 1. Update Material Properties
        k = k_eff(T, self.mat)
        cp = cp_eff(T, self.mat)
        
        # 2. Compute Heat Source Term
        # Sum all sources
        Q_source = torch.zeros_like(T)
        
        # We need grid coordinates to evaluate sources
        # Assuming caller manages grids to avoid re-generating every step
        # If grids not provided, we can't eval source accurately spatially
        if XY_grid is not None:
             X, Y = XY_grid
             # Z grid handling for 3D
             # If 2D, Z_grid is None or ignored by 2D source logic (handled in sources.py)
             
             for source in heat_sources:
                 # Evaluate source intensity
                 # Source usually has its own position/path logic? 
                 # Wait, HeatSource.intensity takes (X, Y, Z, x0, y0, z0).
                 # The 'ScanEvent' determines x0, y0.
                 # This 'step' function assumes the Source *is* the object at current location?
                 # No, typically 'HeatSource' is the Beam Profile. Position comes from Scan Engine.
                 # BUT, for the stepper, we need Q(x, t).
                 
                 # Refactoring: The 'sources' passed here should probably be 'Evaluated Field Q' 
                 # OR we pass a callback/function Q(t).
                 # Given the architecture, let's assume 'heat_sources' are objects that know where they are?
                 # OR we pass 'current_Q_field'.
                 
                 # Let's change the API to take 'Q_external' tensor directly for flexibility & speed.
                 pass
        
        # REVISION: Let's accept Q_force: torch.Tensor as argument. 
        # The integration loop (driver) will handle calling ScanEngine -> Event -> Source -> Q field.
        pass

    def step_explicit_euler(self, 
                            state: SimulationState, 
                            dt: float, 
                            Q_ext: torch.Tensor | None = None) -> SimulationState:
        """
        Explicit Euler step:
        rho * cp * (T_new - T) / dt = div(k grad T) + Q_ext - Loss
        """
        T = state.T
        
        # Material props
        k = k_eff(T, self.mat)
        cp = cp_eff(T, self.mat)
        rho = self.mat.rho # constant density for now
        
        # Diffusion term
        # dx, dy, dz from sim config
        dz = self.sim.dz if self.sim.is_3d else None
        div_term = div_k_grad(T, k, self.sim.dx, self.sim.dy, dz)
        
        # Heat Source
        Q = Q_ext if Q_ext is not None else torch.zeros_like(T)
        
        # RHS = Div + Q
        # (plus boundary losses if any, e.g. convection/radiation handled as volumetric sink or boundary flux?)
        # For simple prototyping, we assume adiabatic or simple Newton cooling "body term" if desired.
        # Let's rely on Neumann BC (zero flux) by default for bounds.
        # Maybe add crude surface cooling if 2D? (simulating top surface loss)
        # But 'div_term' handles internal flux. 
        
        rhs = div_term + Q
        
        # Update
        dT = (dt / rho) * (rhs / (cp + 1e-9))
        T_new = T + dT
        
        # Update State
        state.T_prev = T # store for cooling rate
        state.T = T_new
        state.t += dt
        state.step += 1
        
        # Analysis
        if state.max_T is not None:
             state.max_T = torch.maximum(state.max_T, T_new)
             
        # Cooling Rate Logic
        # Condition: T_prev > T_solidus AND T_new <= T_solidus
        T_sol = self.mat.T_solidus
        
        # Mask of crossing
        # We only care about cooling (T_new < T_prev), though < T_sol implies it.
        # But specifically crossing the threshold.
        crossing_mask = (state.T_prev > T_sol) & (T_new <= T_sol)
        
        if crossing_mask.any() and state.cooling_rate is not None:
            # CR = (T_prev - T_new) / dt
            # Absolute value of cooling rate (positive K/s)
            cr_inst = (state.T_prev - T_new) / dt
            state.cooling_rate = torch.where(crossing_mask, cr_inst, state.cooling_rate)
        
        return state

    def estimate_stability_dt(self, state: SimulationState) -> float:
        """
        Estimate max stable timestep for explicit Euler: dt < dx^2 / (4 * alpha).
        alpha = k / (rho * cp).
        We use max alpha across the domain (conservative).
        """
        # Conservative estimates (max k, min cp)
        # Or just use current max alpha
        k_max = max(self.mat.k_powder, self.mat.k_solid, self.mat.k_liquid)
        cp_min = self.mat.cp_base # ignore latent heat effective cp for stability bound (safety)
        rho = self.mat.rho
        
        alpha_max = k_max / (rho * cp_min)
        
        dx = self.sim.dx
        dy = self.sim.dy
        d_min = min(dx, dy)
        if self.sim.is_3d:
            d_min = min(d_min, self.sim.dz)
            
        # Stability limit: dt <= d^2 / (2*Dims * alpha)
        dims = 3 if self.sim.is_3d else 2
        dt_crit = (d_min ** 2) / (2.0 * dims * alpha_max)
        
        # Safety factor
        return 0.9 * dt_crit

    def step_adaptive(self, 
                      state: SimulationState, 
                      dt_target: float, 
                      Q_ext: torch.Tensor | None = None) -> SimulationState:
        """
        Take a macro-step of size dt_target, automatically sub-stepping for stability.
        """
        dt_crit = self.estimate_stability_dt(state)
        
        import math
        n_sub = math.ceil(dt_target / dt_crit)
        
        dt_sub = dt_target / n_sub
        
        # We can optimize Q_ext scaling here if it's constant over the step
        # If Q_ext is provided, it is likely "Average Power durnig dt".
        # So we apply Q_ext at every sub-step?
        # Yes, standard operator splitting approximation.
        
        for _ in range(n_sub):
            # perform one explicit step
            # We call an internal raw step function to avoid overhead/recursion specifics if needed
            # But step_explicit_euler is fine.
            # CAUTION: step_explicit_euler advances state.t and state.step.
            # We want state.step to increment by 1 macro step? Or N micro steps?
            # Usually users care about macro steps.
            # BUT TimeStepper.step_explicit_euler does logic.
            
            # Let's modify step_explicit_euler to NOT increment step count automatically?
            # Or just let it increment.
            # For now, let's just loop.
            state = self.step_explicit_euler(state, dt_sub, Q_ext)
            
        return state
