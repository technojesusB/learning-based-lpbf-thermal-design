# src/lpbf/physics/material.py
from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict, Field


class MaterialConfig(BaseModel):
    """
    Physical material properties for the simulation.
    All fields expect SI units:
    - Temperature: Kelvin [K]
    - Length: Meters [m]
    - Mass: Kilograms [kg]
    - Energy: Joules [J]
    - Power: Watts [W]
    - Time: Seconds [s]

    The configuration defines the base properties for powder, solid, and liquid states,
    as well as phase change parameters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Conductivity [W/(m K)]
    k_powder: float = Field(
        ..., gt=0.0, description="Thermal conductivity of the powder bed [W/(m K)]"
    )
    k_solid: float = Field(
        ..., gt=0.0, description="Thermal conductivity of the solid material [W/(m K)]"
    )
    k_liquid: float = Field(
        ..., gt=0.0, description="Thermal conductivity of the liquid material [W/(m K)]"
    )

    # Heat Capacity [J/(kg K)]
    cp_base: float = Field(
        ...,
        gt=0.0,
        description="Base specific heat capacity (sensible heat) [J/(kg K)]",
    )

    # Density [kg/m^3]
    rho: float = Field(
        ..., gt=0.0, description="Material density [kg/m^3] (assumed constant)"
    )

    # Phase Change Temperatures [K]
    T_solidus: float = Field(
        ..., gt=0.0, description="Solidus temperature (start of melting) [K]"
    )
    T_liquidus: float = Field(
        ..., gt=0.0, description="Liquidus temperature (end of melting) [K]"
    )

    # Latent Heat [J/kg]
    latent_heat_L: float = Field(
        ..., ge=0.0, description="Latent heat of fusion [J/kg]"
    )

    # Numerical parameters
    transition_sharpness: float = Field(
        default=5.0,
        gt=0.0,
        description="Sharpness parameter for the sigmoid phase transition smoothing [dimensionless].",
    )

    @property
    def latent_width(self) -> float:
        """
        Effective width of the melting temperature range [K].

        Returns:
            float: T_liquidus - T_solidus. Returns 1.0 if gap is negligible to
                avoid division by zero.
        """
        gap = self.T_liquidus - self.T_solidus
        if gap < 1e-6:
            return 1.0  # fallback to avoid div0, shouldn't happen with valid cfg
        return gap


def sigmoid_step(x: torch.Tensor, sharpness: float) -> torch.Tensor:
    """
    Compute a smooth step function traversing 0 -> 1.

    Formula:
        y = sigmoid(sharpness * x) = 1 / (1 + exp(-sharpness * x))

    Args:
        x (torch.Tensor): Input tensor. 0 corresponds to the midpoint of the transition.
        sharpness (float): Parameter controlling the steepness of the step.

    Returns:
        torch.Tensor: Values in range (0, 1).
    """
    return torch.sigmoid(sharpness * x)


def melt_fraction(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Compute the liquid phase fraction (phi) for a temperature field.
    The transition is modeled as a smooth sigmoid function centered between
    T_solidus and T_liquidus.

    phi = 0 implies completely solid (or powder).
    phi = 1 implies completely liquid.

    Args:
        T (torch.Tensor): Temperature field [K].
        cfg (MaterialConfig): Material configuration containing phase change parameters.

    Returns:
        torch.Tensor: Melt fraction field (0 to 1).
    """
    mid = 0.5 * (cfg.T_solidus + cfg.T_liquidus)
    half_width = 0.5 * (cfg.T_liquidus - cfg.T_solidus) + 1e-9

    # Normalized temperature coordinate: -1 at solidus, +1 at liquidus roughly
    s = (T - mid) / half_width

    # Using sharpness from config to tune how 'hard' the step is
    return sigmoid_step(s, cfg.transition_sharpness)


def k_eff(
    T: torch.Tensor,
    cfg: MaterialConfig,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the effective thermal conductivity field k(T) [W/(m K)].

    Physical Model:
       The conductivity is a phase-weighted average of the solid and liquid
       conductivities:
       k_phase = (1 - phi) * k_solid + phi * k_liquid

       If a `mask` is provided (0=Powder, 1=Solid), we further blend:
       k_eff = (1 - mask) * k_powder + mask * k_phase

       This allows modeling the irreversible transition from Powder -> Solid.

    Args:
        T (torch.Tensor): Temperature field [K].
        cfg (MaterialConfig): Material configuration.
        mask (torch.Tensor | None): Phase mask (0=Powder, 1=Solid).
                                    If None, assumes Solid state everywhere.

    Returns:
        torch.Tensor: Effective thermal conductivity field.
    """
    phi = melt_fraction(T, cfg)

    # Pure Phase Change k (Solid <-> Liquid)
    k_phase = (1.0 - phi) * cfg.k_solid + phi * cfg.k_liquid
    
    if mask is not None:
        # mask is 0 or 1.
        # k = (1 - mask)*k_powder + mask*k_phase
        # Use simple interpolation
        # logical_or mask with phi>0? No, mask is historical state.
        
        # Ensure mask is float for math
        m = mask.to(T.dtype)
        return (1.0 - m) * cfg.k_powder + m * k_phase

    return k_phase


def cp_eff(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Compute the Apparent Heat Capacity cp_eff(T) [J/(kg K)].

    Physical Model:
       The Apparent Heat Capacity method incorporates the Latent Heat of fusion (L)
       as a temperature-dependent bump in the specific heat capacity.

       cp_eff(T) = cp_base + L * (d(phi) / dT)

       We approximate d(phi)/dT using a Gaussian function centered in the
       melting range. This ensures that the integral of cp_eff over the melting
       range approximately evaluates to:
       Integral(cp_eff) dT ~ (cp_base * DeltaT) + L

    Args:
        T (torch.Tensor): Temperature field [K].
        cfg (MaterialConfig): Material configuration.

    Returns:
        torch.Tensor: Effective specific heat capacity field.
    """
    cp = torch.full_like(T, cfg.cp_base)

    if cfg.latent_heat_L <= 1e-9:
        return cp

    mid = 0.5 * (cfg.T_solidus + cfg.T_liquidus)
    width = cfg.T_liquidus - cfg.T_solidus
    if width < 1e-6:
        width = 1.0

    # Consistent Derivative of the Sigmoid used in melt_fraction
    # phi = sigmoid( s * sharpness ) where s = (T - mid)/half_width
    # d_phi/dT = d_phi/ds * ds/dT
    # ds/dT = 1 / half_width = 2 / width
    # d_phi/ds = sharpness * phi * (1 - phi)
    
    half_width = 0.5 * width
    s = (T - mid) / half_width
    phi = sigmoid_step(s, cfg.transition_sharpness)
    
    ds_dT = 1.0 / half_width
    d_phi_ds = cfg.transition_sharpness * phi * (1.0 - phi)
    
    d_phi_dT = d_phi_ds * ds_dT

    return cp + cfg.latent_heat_L * d_phi_dT
