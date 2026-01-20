# src/lpbf/diagnostics/energy.py
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.physics.material import MaterialConfig

logger = logging.getLogger(__name__)


@dataclass
class EnergyStats:
    total_input_J: float = 0.0
    total_loss_J: float = 0.0
    initial_enthalpy_J: float = 0.0
    current_enthalpy_J: float = 0.0
    
    @property
    def error_J(self) -> float:
        """
        Energy Balance Error:
        Ideally: dH = Input - Loss
        So: CurrentH - InitialH = Input - Loss
        Error = (CurrentH - InitialH) - (Input - Loss)
        """
        lhs = self.current_enthalpy_J - self.initial_enthalpy_J
        rhs = self.total_input_J - self.total_loss_J
        return lhs - rhs


class EnergyMonitor:
    """
    Tracks global energy conservation in the simulation.
    
    Verifies that:
        Change in Enthalpy == Integrated Input Power - Integrated Boundary Loss
    """

    def __init__(self, sim_config: SimulationConfig, mat_config: MaterialConfig):
        self.sim = sim_config
        self.mat = mat_config
        
        self.stats = EnergyStats()
        self.is_initialized = False

        # Precompute cell volume
        # Note: sim.dz is now physical even in 2D
        self.cell_vol = self.sim.dx * self.sim.dy * self.sim.dz

    def _compute_enthalpy(self, T: torch.Tensor) -> float:
        """
        Compute total system enthalpy [J].
        H = Integral(rho * cp(T) * T) dV
        
        Note: strictly, H = Integral(rho * Integral(cp) dT). 
        Approximating H ~ rho * cp(T) * T is valid if cp is effectively 
        enthalpy-derivative. Our cp_eff includes latent heat bump, 
        so Integral(cp_eff)dT corresponds to Enthalpy change.
        
        However, simply multiplying rho * cp_eff(T) * T is an approximation 
        that might be slightly off for the latent heat part depending on definition.
        A more robust way is H = rho * (cp_base * T + L * phi(T)).
        Let's use the explicit phase fraction form for better accuracy if possible,
        or stick to the cp_eff integration if that's what the solver uses.
        
        Since solver uses cp_eff * dT, checking Energy with cp_eff * T is consistent 
        with the linearization error of Explicit Euler. 
        But let's try to be precise: H(T) = rho * [ cp_base * T + L * phi(T) ].
        This ignores the T_solidus non-zero offset but differences will cancel out.
        """
        # Option A: H = rho * (cp_base * T + L * melt_fraction(T))
        # This is the "True" Enthalpy relative to 0K solid.
        from neural_pbf.physics.material import melt_fraction
        
        phi = melt_fraction(T, self.mat)
        
        # Enthalpy density [J/m^3]
        # h_vol = rho * (cp_base * T + L * phi)
        h_vol = self.mat.rho * (self.mat.cp_base * T + self.mat.latent_heat_L * phi)
        
        total_J = h_vol.sum().item() * self.cell_vol
        return total_J

    def initialize(self, state: SimulationState):
        """Set initial energy baseline."""
        self.stats.initial_enthalpy_J = self._compute_enthalpy(state.T)
        self.stats.current_enthalpy_J = self.stats.initial_enthalpy_J
        self.stats.total_input_J = 0.0
        self.stats.total_loss_J = 0.0
        self.is_initialized = True

    def update(self, state: SimulationState, dt: float, power_in: float):
        """
        Update tracking.
        
        Args:
            state: Current simulation state (after step).
            dt: Timestep size [s].
            power_in: Input power [W] during the step. 
                      Note: If Q_ext was Field [W/m^3] or Flux [W/m^2], 
                      this needs to be the integrated value. 
                      If the scanner provides 'Power', use that.
        """
        if not self.is_initialized:
            self.initialize(state)
            return

        # 1. Update Enthalpy
        self.stats.current_enthalpy_J = self._compute_enthalpy(state.T)
        
        # 2. Accumulate checks
        self.stats.total_input_J += power_in * dt
        
        # 3. Calculate Loss (Simple volumetric loss model)
        # Loss Power = Integral( loss_h * (T - T_amb) ) dV
        # Note: The solver computes dE_loss = -loss_h * (T-T_amb) * dt.
        # We need the integral over volume.
        T_diff = state.T - self.sim.T_ambient
        # Only count positive losses (cooling) or negative (heating from ambient)
        loss_density_rate = (
            self.sim.loss_h * T_diff * self.mat.rho * self.mat.cp_base
        )
        # Solver implementation: loss_term = -self.sim.loss_h * (T - self.sim.T_ambient)
        # And dT = ... (rhs / cp).
        # This implies loss_term has units of [W/m^3] / [rho*cp]? No.
        # Let's check stepper again.
        # rhs = div + Q + loss.  Dimensions of rhs must be [W/m^3].
        # So loss_term is [W/m^3].
        # loss_h * (T - T_amb). So loss_h is [W/(m^3 K)].
        # BUT config says loss_h is "Linear cooling loss coefficient [1/s]".
        # If unit is [1/s], then it acts like a time constant decay?
        # Re-reading stepper: loss_term = -self.sim.loss_h * (T - self.sim.T_ambient).
        # if loss_h is 1/s, then T is K. Result is K/s.
        # Then dT = ... (rhs / cp).
        # If rhs is K/s, then dT should be rhs * dt.
        # BUT code: dT = (dt / rho) * (rhs / cp). 
        # If rhs is K/s, then rhs/cp is (K kg K) / (s J). Unlikely.
        # 
        # Standard Heat Eq: rho*cp*dT/dt = ... [W/m^3]
        # So RHS terms are [W/m^3].
        # If loss_h is [1/s], then loss term likely intended as
        # rho*cp*loss_h*(T-T_amb) ?
        #
        # Let's look exactly at stepper line 106:
        # loss_term = -self.sim.loss_h * (T - self.sim.T_ambient)
        # line 113: dT = (dt / rho) * (rhs / (cp + 1e-9))
        #
        # Dimensional analysis:
        # dT [K], dt [s], rho [kg/m^3], cp [J/(kg K)].
        # (dt/rho)/cp = s / (kg/m^3 * J/kg*K) = s / (J/m^3*K) = s * m^3 * K / J.
        # rhs must be [W/m^3] = [J/(s m^3)].
        # Then: (s m^3 K / J) * (J / s m^3) = K.  Correct.
        #
        # So rhs MUST be [W/m^3].
        # If loss_term is part of rhs, it must be [W/m^3].
        # If code calculates loss_term = loss_h * dT, and loss_h is [1/s],
        # then [1/s] * [K] = [K/s].  This is NOT [W/m^3].
        #
        # BUG SUSPICION: The current loss term implementation might be dimensionally wrong
        # unless loss_h is interpreted as [W/(m^3 K)].
        # But config says [1/s]. 
        # If it is simple Newtonian cooling dT/dt = -k(T-Tamb), then k is 1/s.
        # Then term in eq should be rho*cp * (-k(T-amb)).
        
        # For the monitor, we must integrate whatever the solver did.
        # Solver did: loss_flux_vol = -loss_h * (T - T_amb). 
        # Wait, if stepper adds it to rhs, it treats it as W/m^3.
        # So we treat it as W/m^3 here too for consistency, regardless of whether 
        # the physical value of 'loss_h' makes sense for the user.
        
        loss_vol_rate = self.sim.loss_h * T_diff # [W/m^3] assuming solver logic
        loss_rate_J = loss_vol_rate.sum().item() * self.cell_vol
        
        # Loss is negative current, so we subtract
        self.stats.total_loss_J -= loss_rate_J * dt
