from __future__ import annotations

import torch
from pydantic import BaseModel, Field, ConfigDict
import math

class MaterialConfig(BaseModel):
    """
    Toy (but plausible) thermal material model with
    - strong conductivity contrast (powder vs. sold/liquid)
    - latent heat via apparent heat capacity bump

    All temperatures are in "normalized units" unless you device otherwise.
    """

    model_config = ConfigDict(frozen = True, extra = "forbid")

    # Conductivity
    k_powder: float = Field(default = 0.15, gt = 0.0)
    k_solid: float = Field(default = 1.50, gt = 0.0)
    k_liquid: float = Field(default = 1.10, gt = 0.0)

    # Base heat capacity
    cp_base: float = Field(default = 1.0, gt = 0.0)

    # Melting range
    T_solidus: float = Field(default = 0.60)
    T_liquidus:float = Field(default = 0.70)
    transition_sharpness: float = Field(40.0, gt=0.0)  # larger => sharper sigmoid

    # Latent heat parameters (apparent heat capacity)
    latent_heat_L: float = Field(default = 0.3, gt = 0.0) # "extra energy over melting range"
    latent_width: float = Field(default = 0.03, gt = 0.0) # width of bump around Tm

    # Reference density (kept constant toy model)
    rho: float = Field(default = 1.0, gt = 0.0)

def sigmoid_step(x: torch.Tensor, sharpness: float) -> torch.Tensor:
    return torch.sigmoid(sharpness * x)

def melt_fraction(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Smooth 0..1 fragtion based on solidus/liquidus.

    :param T: Temperature
    :type T: torch.Tensor
    :param cfg: Material Configuration
    :type cfg: MaterialConfig
    :return: segmoid_step from normalized coordinate with transition sharpness
    :rtype: torch.Tensor
    """
    mid = 0.5*(cfg.T_solidus + cfg.T_liquidus)
    half = 0.5*(cfg.T_liquidus-cfg.T_solidus)

    # normalized coordinate
    s = (T - mid) / (half + 1e-12)

    # smooth step
    return sigmoid_step(x = s, sharpness=cfg.transition_sharpness)

def k_eff(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Effective conductivity:
      - below melt range: blend powder -> solid (captures "powder is insulating")
      - within/above melt range: blend solid -> liquid via melt_fraction

    This is a toy design choice:
      - powder→solid transition is tied to temperature for simplicity.
      - To-Do for later: spatially fixed powder/solid, make k(x,y) instead
    
    :param T: Temperature
    :type T: torch.Tensor
    :param cfg: Metrial Config
    :type cfg: MaterialConfig
    :return: Effective conductivity
    :rtype: torch.Tensor
    """
    f = melt_fraction(T = T, cfg = cfg)

    # "powder vs consolidated" gating:
    # below solidus, we want k close to k_powder; above, move toward k_solid.
    # Reuse the same melt_fraction as a proxy for consolidation.
    k_consolidated = (1.0 - f) * cfg.k_solid + f * cfg.k_liquid
    k = (1.0 - f) * cfg.k_powder + f * k_consolidated
    return k

def cp_eff(T: torch.Tensor, cfg: MaterialConfig) -> torch.Tensor:
    """
    Apparent heat capacity to model latent heat in a smooth, differentiable way.
    Adds a Gaussian bump around Tm.

    :param T: Temperature
    :type T: torch.Tensor
    :param cfg: Metrial Config
    :type cfg: MaterialConfig
    :return: Effective heat capacity
    :rtype: torch.Tensor
    """

    cp = torch.full_like(input = T, fill_value = cfg.cp_base)

    if cfg.latent_heat_L <= 0.0:
        return cp
    
    Tm = 0.5 * (cfg.T_solidus + cfg.T_liquidus)
    w = cfg.latent_width

    bump = torch.exp(-0.5*((T-Tm) / w) ** 2)
    # scale bump so that integral-ish adds latent heat (roughly)
    # In toy form: L distributed over ~sqrt(2π)*w
    scale = cfg.latent_heat_L / (w * math.sqrt(2.0 * math.pi)) # 1/sqrt(2π) inverse

    return cp + scale * bump

