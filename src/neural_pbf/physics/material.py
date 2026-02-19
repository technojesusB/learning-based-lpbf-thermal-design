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
        description=(
            "Sharpness parameter for the sigmoid phase transition smoothing "
            "[dimensionless]."
        ),
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
        description=(
            "Use Lookup Tables (LUT) for material properties instead of "
            "linear coefficients."
        ),
    )
    T_lut: list[float] | None = Field(
        default=None,
        description=(
            "Temperature points for the lookup table [K]. "
            "Must be monotonically increasing."
        ),
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
        """Stainless Steel 316L (Mills 2002)"""
        return cls(
            k_powder=0.2,
            k_solid=13.8,
            k_liquid=31.3,
            cp_base=483.0,
            rho=7950.0,
            T_solidus=1653.0,
            T_liquidus=1673.0,
            latent_heat_L=2.9e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[300, 600, 900, 1200, 1500, 1673],
            k_lut=[13.8, 20.0, 24.8, 28.4, 30.7, 31.3],
            cp_lut=[483, 537, 592, 646, 701, 732],
        )

    @classmethod
    def ss304_preset(cls) -> MaterialConfig:
        """Stainless Steel 304 (NIST)"""
        return cls(
            k_powder=0.2,
            k_solid=14.5,
            k_liquid=32.0,
            cp_base=490.0,
            rho=8000.0,
            T_solidus=1673.0,
            T_liquidus=1723.0,
            latent_heat_L=2.68e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[300, 600, 900, 1300, 1673, 1723],
            k_lut=[14.5, 19.4, 23.8, 28.5, 31.8, 32.0],
            cp_lut=[490, 545, 600, 680, 745, 750],
        )

    @classmethod
    def steel_17_4ph_preset(cls) -> MaterialConfig:
        """17-4PH Stainless Steel (AK Steel)"""
        return cls(
            k_powder=0.2,
            k_solid=18.0,
            k_liquid=28.5,
            cp_base=460.0,
            rho=7800.0,
            T_solidus=1677.0,
            T_liquidus=1713.0,
            latent_heat_L=2.7e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[300, 533, 755, 1200, 1677],
            k_lut=[18.0, 19.5, 22.6, 25.5, 28.5],
            cp_lut=[460, 500, 530, 580, 630],
        )

    @classmethod
    def maraging_preset(cls) -> MaterialConfig:
        """Maraging Steel 1.2709 (Renishaw)"""
        return cls(
            k_powder=0.2,
            k_solid=14.1,
            k_liquid=31.0,
            cp_base=450.0,
            rho=8100.0,
            T_solidus=1703.0,
            T_liquidus=1753.0,
            latent_heat_L=2.7e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[293, 873, 1573, 1703],
            k_lut=[14.1, 21.0, 29.0, 31.0],
            cp_lut=[450, 450, 450, 450],
        )

    @classmethod
    def ti64_preset(cls) -> MaterialConfig:
        """Ti-6Al-4V Grade 5 (Akwaboa/Boivineau)"""
        return cls(
            k_powder=0.15,
            k_solid=6.7,
            k_liquid=25.0,
            cp_base=520.0,
            rho=4430.0,
            T_solidus=1878.0,
            T_liquidus=1923.0,
            latent_heat_L=2.86e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[300, 373, 573, 873, 1300, 1878, 1923],
            k_lut=[6.7, 6.8, 9.4, 12.5, 18.0, 24.5, 25.0],
            cp_lut=[520, 414, 443, 448, 610, 740, 750],
        )

    @classmethod
    def cp_ti_preset(cls) -> MaterialConfig:
        """Commercially Pure Titanium Grade 2 (NIST)"""
        return cls(
            k_powder=0.15,
            k_solid=21.6,
            k_liquid=32.0,
            cp_base=522.0,
            rho=4510.0,
            T_solidus=1941.0,
            T_liquidus=1942.0,
            latent_heat_L=4.4e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[298, 700, 1158, 1700, 1941],
            k_lut=[21.6, 28.0, 30.0, 32.0, 32.0],
            cp_lut=[522, 700, 900, 1200, 1200],
        )

    @classmethod
    def alsi10mg_preset(cls) -> MaterialConfig:
        """AlSi10Mg (Akwaboa 2023)"""
        return cls(
            k_powder=0.5,
            k_solid=138.4,
            k_liquid=155.2,
            cp_base=800.0,
            rho=2680.0,
            T_solidus=843.0,
            T_liquidus=883.0,
            latent_heat_L=3.8e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[373, 473, 573, 673, 843],
            k_lut=[138.4, 153.9, 169.8, 155.2, 155.2],
            cp_lut=[800, 835, 862, 758, 758],
        )

    @classmethod
    def al6061_preset(cls) -> MaterialConfig:
        """Aluminum 6061 (ASM)"""
        return cls(
            k_powder=0.5,
            k_solid=160.0,
            k_liquid=200.0,
            cp_base=897.0,
            rho=2700.0,
            T_solidus=855.0,
            T_liquidus=925.0,
            latent_heat_L=4.0e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[300, 600, 855, 925],
            k_lut=[160, 190, 200, 200],
            cp_lut=[897, 1050, 1100, 1100],
        )

    @classmethod
    def in718_preset(cls) -> MaterialConfig:
        """Inconel 718 (Special Metals)"""
        return cls(
            k_powder=0.1,
            k_solid=10.0,
            k_liquid=25.3,
            cp_base=425.0,
            rho=8190.0,
            T_solidus=1533.0,
            T_liquidus=1609.0,
            latent_heat_L=2.27e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[300, 600, 900, 1200, 1533, 1609],
            k_lut=[10.0, 14.9, 20.1, 23.6, 25.0, 25.3],
            cp_lut=[425, 489, 550, 635, 635, 635],
        )

    @classmethod
    def in625_preset(cls) -> MaterialConfig:
        """Inconel 625 (Special Metals)"""
        return cls(
            k_powder=0.1,
            k_solid=9.8,
            k_liquid=30.0,
            cp_base=429.0,
            rho=8440.0,
            T_solidus=1563.0,
            T_liquidus=1623.0,
            latent_heat_L=2.3e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[296, 573, 873, 1255, 1563],
            k_lut=[9.8, 15.5, 21.3, 30.0, 30.0],
            cp_lut=[429, 480, 560, 650, 650],
        )

    @classmethod
    def h13_preset(cls) -> MaterialConfig:
        """H13 Tool Steel (SteelPro)"""
        return cls(
            k_powder=0.2,
            k_solid=24.3,
            k_liquid=30.0,
            cp_base=460.0,
            rho=7750.0,
            T_solidus=1608.0,
            T_liquidus=1748.0,
            latent_heat_L=2.8e5,
            use_T_dep=True,
            use_lut=True,
            T_lut=[300, 748, 1000, 1500, 1748],
            k_lut=[24.3, 26.0, 28.0, 30.0, 30.0],
            cp_lut=[460, 550, 600, 650, 650],
        )


def interpolate_1d(
    T: torch.Tensor, T_lut: list[float], V_lut: list[float]
) -> torch.Tensor:
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
    denom = torch.where(
        denom < 1e-9, torch.tensor(1.0, device=T.device, dtype=T.dtype), denom
    )

    alpha = (T - t0) / denom

    # Linear interpolation
    res = v0 + alpha * (v1 - v0)

    # Handle extrapolation (constant outside range)
    res = torch.where(tlut[0] >= T, vlut[0], res)
    res = torch.where(tlut[-1] <= T, vlut[-1], res)

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
    k_s: float | torch.Tensor = cfg.k_solid
    k_l: float | torch.Tensor = cfg.k_liquid

    if cfg.use_T_dep:
        if cfg.use_lut and cfg.T_lut is not None and cfg.k_lut is not None:
            # Use LUT for bulk conductivity
            k_bulk = interpolate_1d(T, cfg.T_lut, cfg.k_lut)
            # In LUT mode, we typically use the interpolated bulk value directly,
            # but to be consistent with phase weighting:
            # and we might still want to scale liquid differently or just
            # use the LUT value.
            # BETTER: If LUT is present, it usually describes the
            # 'material' property vs T.
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
    cp_base: float | torch.Tensor = cfg.cp_base
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
