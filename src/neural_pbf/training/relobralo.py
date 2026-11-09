"""ReLoBRaLo loss balancing for the data (flow-matching) and physics terms."""

from __future__ import annotations

import math
import random


class ReLoBRaLoWeighter:
    """Relative Loss Balancing with Random Lookback (Bischof & Kraus 2021, https://arxiv.org/abs/2110.09813).

    For the two terms i ∈ {data, physics}, with m = 2:

        λ_i^bal(t, t') = m · softmax_i( L_i(t) / (T·L_i(t') + eps) )         (Eq. 11)
        λ_i^hist(t)    = ρ·λ_i(t−1) + (1−ρ)·λ_i^bal(t, 0),   ρ ~ Bernoulli(beta)
        λ_i(t)         = α·λ_i^hist(t) + (1−α)·λ_i^bal(t, t−1)

    A term that progresses slowly relative to its reference gets the larger weight; the two weights sum to m.
    The data term keeps the fixed weight 1 in training; only ``lambda_phys`` is used.

    ``max_lambda`` caps the weight that is used: a residual that cannot reach zero stagnates and would otherwise
    be up-weighted towards m. The update keeps building on the uncapped value, ``lambda_raw``; ``cap_hit`` says
    whether the cap acted on this step. A step with a non-finite loss is skipped: the weights are held and
    ``skipped`` is set.

    The version before 2026-11-09 weighted the term that improved faster more (``ref/loss`` instead of
    ``loss/ref``), the opposite of the paper.
    """

    def __init__(
        self,
        alpha: float = 0.999,
        beta: float = 0.9,
        temperature: float = 0.1,
        eps: float = 1e-8,
        max_lambda: float | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta (E[rho]) must lie in [0, 1], got {beta}")
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(f"temperature must be a finite value > 0, got {temperature}")
        if max_lambda is not None and not (math.isfinite(max_lambda) and max_lambda > 0):
            raise ValueError(f"max_lambda must be a finite value > 0 or None, got {max_lambda}")
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.eps = eps
        self.max_lambda = max_lambda
        self._init: tuple[float, float] | None = None
        self._prev: tuple[float, float] | None = None
        self.lambda_fm: float = 1.0
        self.lambda_raw: float = 1.0  # physics weight before the cap
        self.lambda_phys: float = 1.0  # physics weight after the cap: the one that is used
        self.cap_hit: bool = False
        self.skipped: bool = False

    def _balance(self, losses: tuple[float, float], ref: tuple[float, float]) -> tuple[float, float]:
        z = [loss / (self.temperature * r + self.eps) for loss, r in zip(losses, ref, strict=True)]
        top = max(z)
        e = [math.exp(v - top) for v in z]
        return 2.0 * e[0] / sum(e), 2.0 * e[1] / sum(e)

    def step(self, loss_fm: float, loss_phys: float) -> float:
        """Update the weights with this step's losses and return the physics weight to use."""
        if not (math.isfinite(loss_fm) and math.isfinite(loss_phys)):
            self.skipped, self.cap_hit = True, False
            return self.lambda_phys
        self.skipped = False
        losses = (loss_fm, loss_phys)
        if self._init is None or self._prev is None:
            self._init = self._prev = losses
            return self.lambda_phys

        rho = 1.0 if random.random() < self.beta else 0.0
        bal_start = self._balance(losses, self._init)
        bal_last = self._balance(losses, self._prev)
        previous = (self.lambda_fm, self.lambda_raw)
        new = [
            self.alpha * (rho * lam + (1.0 - rho) * b0) + (1.0 - self.alpha) * b1
            for lam, b0, b1 in zip(previous, bal_start, bal_last, strict=True)
        ]
        self.lambda_fm, self.lambda_raw = new
        self._prev = losses

        cap = self.max_lambda
        self.cap_hit = cap is not None and self.lambda_raw > cap
        self.lambda_phys = cap if cap is not None and self.cap_hit else self.lambda_raw
        return self.lambda_phys
