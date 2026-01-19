# src/lpbf/physics/material.py
from __future__ import annotations
import torch
import math
from pydantic import BaseModel, Field, ConfigDict

class MaterialConfig(BaseModel):
    """
    Physical material properties.
    All units SI (W, m, K, J, kg).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Conductivity [W/(m K)]
    k_powder: float = Field(..., gt=0.0)
    k_solid: float = Field(..., gt=0.0)
    k_liquid: float = Field(..., gt=0.0)

    # Heat Capacity [J/(kg K)]
    cp_base: float = Field(..., gt=0.0)

    # Density [kg/m^3]
    rho: float = Field(..., gt=0.0)

    # Phase Change Temperatures [K]
    T_solidus: float = Field(..., gt=0.0)
    T_liquidus: float = Field(..., gt=0.0)

    # Latent Heat [J/kg]
    latent_heat_L: float = Field(..., ge=0.0)
    
    # Numerical parameters
    transition_sharpness: float = Field(default=5.0, gt=0.0, description="Sharpness of sigmoid transition")

    @property
    def latent_width(self) -> float:
        """Effective width of melting range for Gaussian spread."""
        gap = self.T_liquidus - self.T_solidus
        if gap < 1e-6:
            return 1.0 # fallback to avoid div0, shouldn't happen with valid cfg
        return gap


def sigmoid_step(x: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Smooth step function 0->1."""
    return torch.sigmoid(sharpness * x)

def melt_fraction(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Compute melt fraction phi (0=solid, 1=liquid).
    smoothed via sigmoid based on Solidus/Liquidus.
    """
    mid = 0.5 * (cfg.T_solidus + cfg.T_liquidus)
    half_width = 0.5 * (cfg.T_liquidus - cfg.T_solidus) + 1e-9
    
    # Normalized temperature coordinate: -1 at solidus, +1 at liquidus roughly
    # We want sigmoid to traverse 0.1 to 0.9 roughly within the range
    s = (T - mid) / half_width
    
    # Using sharpness from config to tune how 'hard' the step is
    return sigmoid_step(s, cfg.transition_sharpness)

def k_eff(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Effective thermal conductivity.
    Model:
      - Below solidus: Mix of k_powder and k_solid (consolidation proxy). 
        *Simplification*: For now, we assume if T < T_solidus, it could be powder OR solid. 
        However, the legacy code used melt_fraction as a proxy for 'consolidated status'.
        We will adopt a 'state-based' approach later, but for stateless k(T), 
        we usually assume 'powder' properties are valid only until first melt, 
        and 'solid' properties thereafter.
        
        Since this function is PURELY T-dependent (k(T)), it cannot know history.
        We will assume the caller handles history-dependent 'state' (powder vs solid).
        
        BUT, to match the legacy 'toy' behavior first:
        k = (1-f)*k_powder + f*k_liquid  <-- mixed with solid?
        
        Let's implement the standard k(T) = k_solid*(1-phi) + k_liquid*phi
        AND allow an external 'consolidation' mask if needed.
        
        For this stateless function, we'll return the Phase-change dependent k.
        
        IF we strictly follow the user request "Distinct powder / solid / liquid regimes":
        We need a 'state' tensor (is_powder).
    """
    phi = melt_fraction(T, cfg)
    
    # Pure Phase Change k (Solid <-> Liquid)
    k_phase = (1.0 - phi) * cfg.k_solid + phi * cfg.k_liquid
    
    # Note: Powder handling usually requires a history variable (has_melted).
    # We will compute the 'material k' here. The simulation loop will mix in k_powder
    # based on the history state. 
    return k_phase

def cp_eff(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Apparent Heat Capacity: cp_eff = cp_base + L * d(phi)/dT
    We approximate d(phi)/dT with a Gaussian bump.
    """
    cp = torch.full_like(T, cfg.cp_base)
    
    if cfg.latent_heat_L <= 1e-9:
        return cp

    mid = 0.5 * (cfg.T_solidus + cfg.T_liquidus)
    # The width of the physical melting range
    width = (cfg.T_liquidus - cfg.T_solidus)
    if width < 1e-6: width = 1.0 # safety

    # Gaussian approximation for delta function
    # Integral of Gaussian = sqrt(2*pi)*sigma
    # We want Integral = L
    # So height * sqrt(2*pi)*sigma = L  => height = L / (sqrt(2*pi)*sigma)
    
    # somewhat arbitrary sigma relative to width. 
    # Let's say +/- 2 sigma covers the width? sigma = width / 4
    sigma = width / 4.0 
    
    arg = (T - mid) / (sigma)
    gauss = torch.exp(-0.5 * arg**2)
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    
    d_phi_dT = norm * gauss
    
    return cp + cfg.latent_heat_L * d_phi_dT
