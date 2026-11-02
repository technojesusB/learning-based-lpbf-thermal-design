"""Split conformal prediction intervals on scalar quantities, plus coverage statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

METHODS = ("absolute", "normalized", "constant_single_draw", "raw_ensemble")
_EPS_FRACTION = 0.01  # of the median calibration spread; keeps normalized scores finite


@dataclass(frozen=True)
class Calibrated:
    method: str
    alpha: float
    q: float
    eps: float


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Rank ``ceil((n+1)(1-alpha))`` order statistic; ``inf`` when that rank exceeds n."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        raise ValueError("conformal_quantile: empty calibration scores")
    if not np.isfinite(scores).all():
        raise ValueError("conformal_quantile: non-finite calibration scores")
    n = scores.size
    rank = math.ceil(round((n + 1) * (1 - alpha), 9))
    return math.inf if rank > n else float(np.sort(scores)[rank - 1])


def run_level_supported(n_cal_runs: int, alpha: float) -> bool:
    """Whether calibration has enough runs for a finite quantile when each run counts once."""
    return math.ceil(round((n_cal_runs + 1) * (1 - alpha), 9)) <= n_cal_runs


def _check(y: np.ndarray, draws: np.ndarray) -> None:
    if draws.ndim != 2 or y.ndim != 1 or draws.shape[0] != y.shape[0]:
        raise ValueError(f"expected y (n,) and draws (n, K), got {y.shape} and {draws.shape}")


def _spread(draws: np.ndarray) -> np.ndarray:
    return draws.std(axis=1, ddof=1) if draws.shape[1] > 1 else np.zeros(draws.shape[0])


def calibrate(method: str, y: np.ndarray, draws: np.ndarray, alpha: float) -> Calibrated:
    """Fit one method on calibration data. ``raw_ensemble`` ignores *y*: it is the uncalibrated reference."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    _check(y, draws)
    if method == "raw_ensemble":
        return Calibrated(method, alpha, NormalDist().inv_cdf(1 - alpha / 2), 0.0)
    if method == "constant_single_draw":
        return Calibrated(method, alpha, conformal_quantile(np.abs(y - draws[:, 0]), alpha), 0.0)
    residual = np.abs(y - draws.mean(axis=1))
    if method == "absolute":
        return Calibrated(method, alpha, conformal_quantile(residual, alpha), 0.0)
    spread = _spread(draws)
    eps = max(_EPS_FRACTION * float(np.median(spread)), 1e-8)
    return Calibrated(method, alpha, conformal_quantile(residual / (spread + eps), alpha), eps)


def predict(method: str, cal: Calibrated, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper interval bounds for ``draws`` of shape (n, K)."""
    if method != cal.method:
        raise ValueError(f"calibration is for {cal.method!r}, not {method!r}")
    if method == "constant_single_draw":
        center, half = draws[:, 0], np.full(draws.shape[0], cal.q)
    elif method == "absolute":
        center, half = draws.mean(axis=1), np.full(draws.shape[0], cal.q)
    elif method == "normalized":
        center, half = draws.mean(axis=1), cal.q * (_spread(draws) + cal.eps)
    else:
        center, half = draws.mean(axis=1), cal.q * _spread(draws)
    return center - half, center + half


def wilson_interval(hits: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval. Assumes independent samples, which within-run data are not."""
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = hits / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = 0.0 if hits == 0 else max(0.0, center - half)
    hi = 1.0 if hits == n else min(1.0, center + half)
    return lo, hi


def coverage_stats(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    run_ids: np.ndarray,
    alpha: float,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, float]:
    """Coverage with binomial (Wilson) and run-level bootstrap CIs, widths, interval score."""
    if y.size == 0:
        raise ValueError("coverage_stats: empty evaluation set")
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    score = width + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    hits = int(covered.sum())
    binom_lo, binom_hi = wilson_interval(hits, y.size)

    runs, inverse = np.unique(run_ids, return_inverse=True)
    run_hits = np.bincount(inverse, weights=covered.astype(float), minlength=runs.size)
    run_n = np.bincount(inverse, minlength=runs.size).astype(float)
    picks = np.random.default_rng(seed).integers(0, runs.size, size=(n_boot, runs.size))
    boot = run_hits[picks].sum(axis=1) / run_n[picks].sum(axis=1)

    return {
        "n": int(y.size),
        "n_runs": int(runs.size),
        "coverage": hits / y.size,
        "binom_lo": binom_lo,
        "binom_hi": binom_hi,
        "boot_lo": float(np.quantile(boot, 0.025)),
        "boot_hi": float(np.quantile(boot, 0.975)),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "interval_score": float(score.mean()),
    }
