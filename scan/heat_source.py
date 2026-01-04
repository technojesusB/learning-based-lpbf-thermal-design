# scan/heat_source.py
from __future__ import annotations

import torch
from pydantic import BaseModel, Field, ConfigDict


class PulseConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # position in normalized [0,1] domain
    x0: float = Field(0.5, ge=0.0, le=1.0)
    y0: float = Field(0.5, ge=0.0, le=1.0)

    # pulse power and duration
    power: float = Field(1.0, ge=0.0)
    t_on: float = Field(0.00, ge=0.0)
    t_off: float = Field(0.10, ge=0.0)

    # gaussian width (normalized units)
    sigma: float = Field(0.02, gt=0.0)

    # absorption/scaling factor
    eta: float = Field(1.0, ge=0.0)

    # smoothness for on/off gating (higher = sharper)
    gate_sharpness: float = Field(80.0, gt=0.0)


def smooth_gate(t: torch.Tensor, t_on: float, t_off: float, sharpness: float) -> torch.Tensor:
    """
    Smooth approximation of a rectangular pulse: sigmoid(t-t_on) - sigmoid(t-t_off)
    """
    return torch.sigmoid(sharpness * (t - t_on)) - torch.sigmoid(sharpness * (t - t_off))


def gaussian_spot(X: torch.Tensor, Y: torch.Tensor, x0: float, y0: float, sigma: float) -> torch.Tensor:
    dx = X - x0
    dy = Y - y0
    return torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def pulse_Q(X: torch.Tensor, Y: torch.Tensor, t: torch.Tensor, cfg: PulseConfig) -> torch.Tensor:
    """
    Heat source Q(x,y,t) for a single stationary pulse dot.
    Shapes:
      X,Y: [1,1,H,W]
      t: scalar tensor []
    Returns:
      Q: [1,1,H,W]
    """
    gate = smooth_gate(t, cfg.t_on, cfg.t_off, cfg.gate_sharpness)
    spot = gaussian_spot(X, Y, cfg.x0, cfg.y0, cfg.sigma)
    return (cfg.eta * cfg.power) * gate * spot

# scan/heat_source.py (add this)
@torch.no_grad()
def spot_footprint(X: torch.Tensor, Y: torch.Tensor, cfg: PulseConfig) -> torch.Tensor:
    """
    Returns the spatial spot footprint (Gaussian) independent of pulse gating.
    Shape [1,1,H,W]
    """
    return gaussian_spot(X, Y, cfg.x0, cfg.y0, cfg.sigma)


def pulse_energy(cfg: PulseConfig) -> float:
    """
    Toy energy for the event: integral of power over time.
    """
    return float(cfg.power) * max(0.0, float(cfg.t_off - cfg.t_on))

