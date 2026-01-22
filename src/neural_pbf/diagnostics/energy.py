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

        # ENERGY BALANCE CHECK:
        # self.sim.loss_h is [1/s].
        # Heat Equation term: -loss_h * (T - T_inf)   (Temperature rate of change [K/s])
        # To get Power [W], we need to integrate:
        # Integral_V { rho * cp * (loss_h * (T - T_inf)) } dV
        # Units: [kg/m^3] * [J/kg K] * [1/s] * [K] * [m^3] = [J/s] = [W]. Correct.

        # However, earlier code treated loss_h as [W/(m^3 K)]?
        # If config says [1/s], then above is correct.
        # If we just sum loss term from stepper: -loss_h*(T-T_amb) is added to dT/dt.
        # So it is explicitly [K/s].
        # So Power Loss Density = rho * cp * loss_h * (T - T_amb) [W/m^3].

        # Calculate loss power [W]
        # Approximation: Use current T (implicit explicit Euler lag)
        # We need volume integral.
        # sum() * dx * dy * dz

        # NOTE: This assumes cp is constant or using cp_base.
        # Ideally should use cp_eff(T) but that includes latent heat.
        # Sensible heat loss uses specific heat.

        loss_power_W = (
            self.sim.loss_h * T_diff * self.mat.rho * self.mat.cp_base
        ).sum() * (self.sim.dx * self.sim.dy * self.sim.dz)

        self.stats.total_loss_J += float(loss_power_W * dt)
