"""ReLoBRaLo adaptive loss weighting for physics-informed training."""

from __future__ import annotations

import math
import random


class ReLoBRaLoWeighter:
    """Relative Loss Balancing with Random Lookback (ReLoBRaLo).

    Tracks EMA of loss magnitudes for the FM (data) and physics (PDE) losses.
    On each step, updates lambda_phys so that:
        lambda_phys * loss_phys ≈ loss_fm
    using the relative change from a reference (EMA or initial value selected
    randomly with probability beta).

    Designed to handle astronomical physics magnitudes (e+26) without gradient-norm
    computation (which would require retain_graph=True and a second backward pass).

    Reference: ReLoBRaLo — Relative Loss Balancing Residual Regularisation
    (Bischof & Kraus, 2021, https://arxiv.org/abs/2110.09813).
    """

    def __init__(
        self,
        alpha: float = 0.999,
        beta: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self._ema_fm: float | None = None
        self._ema_phys: float | None = None
        self._init_fm: float | None = None
        self._init_phys: float | None = None
        self.lambda_phys: float = 1.0

    def step(self, loss_fm: float, loss_phys: float) -> float:
        """Update lambda and return the new physics weight.

        Args:
            loss_fm:    Current FM flow loss (scalar Python float).
            loss_phys:  Current physics PDE residual (scalar Python float).

        Returns:
            Updated lambda_phys.
        """
        if not (math.isfinite(loss_fm) and math.isfinite(loss_phys)):
            return self.lambda_phys

        if self._ema_fm is None:
            self._ema_fm = loss_fm
            self._ema_phys = loss_phys
            self._init_fm = loss_fm
            self._init_phys = loss_phys
            return self.lambda_phys

        # Single Bernoulli draw per step so both losses share the
        # same reference baseline
        use_init = random.random() > self.beta
        ref_fm = self._init_fm if use_init else self._ema_fm
        ref_phys = self._init_phys if use_init else self._ema_phys

        rho_fm = loss_fm / (ref_fm + self.eps)
        rho_phys = loss_phys / (ref_phys + self.eps)

        magnitude_ratio = ref_fm / (ref_phys + self.eps)
        self.lambda_phys = magnitude_ratio * (rho_fm / (rho_phys + self.eps))

        # Guard against NaN/Inf from extreme ratios
        if not math.isfinite(self.lambda_phys) or self.lambda_phys <= 0:
            self.lambda_phys = 1.0

        self._ema_fm = self.alpha * self._ema_fm + (1 - self.alpha) * loss_fm
        self._ema_phys = self.alpha * self._ema_phys + (1 - self.alpha) * loss_phys

        return self.lambda_phys
