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

    # --- Phase 3: Temperature Dependency (Optional) ---
    use_T_dep: bool = Field(
        default=False,
        description="Toggle temperature-dependent material properties (k, cp).",
    )
    
    # Simple linear model coefficients (Legacy support)
    T_ref: float = Field(
        default=293.15,
        gt=0.0,
        description="Reference temperature for temperature-dependent properties [K].",
    )
    k_solid_T_coeff: float = Field(
        default=0.0,
        description="Linear temperature coefficient for solid conductivity [1/K].",
    )
    k_liquid_T_coeff: float = Field(
        default=0.0,
        description="Linear temperature coefficient for liquid conductivity [1/K].",
    )
    cp_T_coeff: float = Field(
        default=0.0,
        description="Linear temperature coefficient for heat capacity [1/K].",
    )

    # --- Lookup Table (LUT) support ---
    use_lut: bool = Field(
        default=False,
        description="Use Lookup Tables (LUT) for material properties instead of linear coefficients.",
    )
    T_lut: list[float] | None = Field(
        default=None,
        description="Temperature points for the lookup table [K]. Must be monotonically increasing.",
    )
    k_lut: list[float] | None = Field(
        default=None,
        description="Thermal conductivity values at T_lut [W/(m K)].",
    )
    cp_lut: list[float] | None = Field(
        default=None,
        description="Specific heat capacity values at T_lut [J/(kg K)].",
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

    @classmethod
    def ss316l_preset(cls) -> MaterialConfig:
        """Returns physical properties for Stainless Steel 316L with LUT presets."""
        return cls(
            k_powder=0.2, k_solid=15.0, k_liquid=30.0,
            cp_base=450.0, rho=7900.0,
            T_solidus=1650.0, T_liquidus=1700.0,
            latent_heat_L=2.7e5,
            use_T_dep=True, use_lut=True,
            T_lut=[293, 500, 800, 1100, 1400, 1700, 2500, 4000],
            k_lut=[15.0, 17.0, 20.0, 24.0, 28.0, 30.0, 35.0, 40.0],
            cp_lut=[450, 480, 520, 560, 600, 650, 700, 750]
        )

    @classmethod
    def ti64_preset(cls) -> MaterialConfig:
        """Returns physical properties for Ti-6Al-4V with LUT presets."""
        return cls(
            k_powder=0.15, k_solid=7.0, k_liquid=20.0,
            cp_base=530.0, rho=4430.0,
            T_solidus=1878.0, T_liquidus=1923.0,
            latent_heat_L=3.7e5,
            use_T_dep=True, use_lut=True,
            T_lut=[293, 600, 900, 1200, 1500, 2000, 3000],
            k_lut=[7.0, 10.0, 13.0, 16.0, 19.0, 25.0, 30.0],
            cp_lut=[530, 580, 630, 680, 730, 800, 850]
        )


def interpolate_1d(T: torch.Tensor, T_lut: list[float], V_lut: list[float]) -> torch.Tensor:
    """
    Perform 1D linear interpolation on a temperature field.
    
    Args:
        T (torch.Tensor): Temperature field.
        T_lut (list[float]): Reference temperature points.
        V_lut (list[float]): Values at T_lut.
        
    Returns:
        torch.Tensor: Interpolated values.
    """
    # Convert LUT to tensors on same device/dtype as T
    tlut = torch.tensor(T_lut, device=T.device, dtype=T.dtype)
    vlut = torch.tensor(V_lut, device=T.device, dtype=T.dtype)
    
    # We use searchsorted to find indices
    # T: (B, C, ...)
    # tlut: (N,)
    # Output indices: where T would be inserted into tlut.
    idx = torch.searchsorted(tlut, T)
    
    # Clamp to valid range [1, N-1]
    idx = torch.clamp(idx, 1, len(tlut) - 1)
    
    # Neighbors
    t0, t1 = tlut[idx - 1], tlut[idx]
    v0, v1 = vlut[idx - 1], vlut[idx]
    
    # Interpolation factor
    # Handle cases where t1 == t0 (should not happen in valid LUT)
    denom = t1 - t0
    denom = torch.where(denom < 1e-9, torch.tensor(1.0, device=T.device, dtype=T.dtype), denom)
    
    alpha = (T - t0) / denom
    
    # Linear interpolation
    res = v0 + alpha * (v1 - v0)
    
    # Handle extrapolation (constant outside range)
    res = torch.where(T <= tlut[0], vlut[0], res)
    res = torch.where(T >= tlut[-1], vlut[-1], res)
    
    return res


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

    # Conductivity [W/(m K)]
    k_s = cfg.k_solid
    k_l = cfg.k_liquid
    
    if cfg.use_T_dep:
        if cfg.use_lut and cfg.T_lut is not None and cfg.k_lut is not None:
             # Use LUT for bulk conductivity
             k_bulk = interpolate_1d(T, cfg.T_lut, cfg.k_lut)
             # In LUT mode, we typically use the interpolated bulk value directly,
             # but to be consistent with phase weighting:
             # We assume k_lut describes the 'Solid' phase property, 
             # and we might still want to scale liquid differently or just use the LUT value.
             # BETTER: If LUT is present, it usually describes the 'material' property vs T.
             # We'll treat the LUT as the master source for Solid/Liquid k.
             k_phase = k_bulk
        else:
            # Simple linear scaling
            dT = T - cfg.T_ref
            k_s = k_s * (1.0 + cfg.k_solid_T_coeff * dT)
            k_l = k_l * (1.0 + cfg.k_liquid_T_coeff * dT)
            # Phase-weighted average
            k_phase = (1.0 - phi) * k_s + phi * k_l
    else:
        # Phase-weighted average (constant)
        k_phase = (1.0 - phi) * k_s + phi * k_l
    
    if mask is not None:
        # mask is 0 or 1.
        # k = (1 - mask)*k_powder + mask*k_phase
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
    cp_base = cfg.cp_base
    if cfg.use_T_dep:
        if cfg.use_lut and cfg.T_lut is not None and cfg.cp_lut is not None:
            cp_base = interpolate_1d(T, cfg.T_lut, cfg.cp_lut)
        else:
            dT = T - cfg.T_ref
            cp_base = cp_base * (1.0 + cfg.cp_T_coeff * dT)
        
    cp = torch.zeros_like(T) 
    cp = cp + cp_base

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
