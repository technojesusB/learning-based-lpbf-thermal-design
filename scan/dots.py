# scan/dots.py
from __future__ import annotations

from dataclasses import dataclass
import torch
from scan.heat_source import gaussian_spot


@dataclass(frozen=True)
class DotEvent:
    x: float
    y: float
    power: float
    dwell: float      # laser-on duration
    travel: float     # time until next dot (laser off)
    sigma: float
    eta: float = 1.0


def make_dot_Q_fn(X: torch.Tensor, Y: torch.Tensor, event: DotEvent):
    """
    Returns Q_fn(t) for constant-on during dwell; the stepper controls duration.
    """
    spot = gaussian_spot(X, Y, event.x, event.y, event.sigma)

    def Q_fn(t: torch.Tensor) -> torch.Tensor:
        return (event.eta * event.power) * spot

    return Q_fn, spot


def make_zero_Q_fn(shape_like: torch.Tensor):
    def Q0(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(shape_like)
    return Q0
