# data/stepper.py
from __future__ import annotations

from typing import Callable
import torch

from physics.material import MaterialConfig, k_eff, cp_eff
from physics.operators import div_k_grad_2d


QFn = Callable[[torch.Tensor], torch.Tensor]  # takes scalar t -> Q [1,1,H,W]


@torch.no_grad()
def advance_temperature(
    T: torch.Tensor,                 # [1,1,H,W]
    mat: MaterialConfig,
    dx: float,
    dy: float,
    T_ambient: float,
    loss_h: float,
    dt: float,
    duration: float,
    Q_fn: QFn,
    t0: float = 0.0,
    step0: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Advance temperature from time t0 to t0+duration with explicit Euler + variable coefficients.

    Returns:
      T_new: [1,1,H,W]
      T_peak_during_interval: [1,1,H,W]
    """
    assert T.ndim == 4 and T.shape[1] == 1
    steps = max(1, int(duration / dt))
    # use exact duration by adjusting dt_eff
    dt_eff = duration / steps

    T_peak = T.clone()

    for n in range(steps):
        t = torch.tensor(t0 + (n + 0.5) * dt_eff, device=T.device, dtype=T.dtype)

        k = k_eff(T, mat)
        cp = cp_eff(T, mat)
        cp = torch.clamp(cp, min=1e-3)  # safety

        div_term = div_k_grad_2d(T, k, dx=dx, dy=dy)
        Q = Q_fn(t)

        rhs = div_term + Q - loss_h * (T - T_ambient)
        T = T + dt_eff * (rhs / (mat.rho * cp))

        T_peak = torch.maximum(T_peak, T)

    return T, T_peak, step0 + steps
