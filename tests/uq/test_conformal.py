"""Split conformal: finite-sample quantile, methods, coverage statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neural_pbf.uq.conformal import (
    METHODS,
    calibrate,
    conformal_quantile,
    coverage_stats,
    predict,
    run_level_supported,
    wilson_interval,
)


@pytest.mark.unit
def test_quantile_finite_sample_rank():
    scores = np.arange(1.0, 10.0)  # n=9, alpha=.1 -> rank ceil(10*.9)=9
    assert conformal_quantile(scores, 0.1) == 9.0
    assert conformal_quantile(np.arange(1.0, 20.0), 0.2) == 16.0  # ceil(20*.8)=16


@pytest.mark.unit
def test_quantile_too_few_points_is_infinite_not_silently_finite():
    assert math.isinf(conformal_quantile(np.arange(1.0, 9.0), 0.1))  # n=8 < 9
    assert math.isinf(conformal_quantile(np.array([1.0]), 0.2))


@pytest.mark.unit
def test_quantile_float_rounding_does_not_add_a_rank():
    # (n+1)(1-alpha) lands on an integer but floats may overshoot it
    assert conformal_quantile(np.arange(1.0, 20.0), 0.05) == 19.0
    for n, a in [(9, 0.1), (19, 0.05), (4, 0.2), (99, 0.01)]:
        assert math.isfinite(conformal_quantile(np.ones(n), a))


@pytest.mark.unit
@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0, np.nan]), np.array([np.inf])])
def test_quantile_rejects_empty_and_non_finite(bad):
    with pytest.raises(ValueError):
        conformal_quantile(bad, 0.1)


@pytest.mark.unit
def test_run_level_support():
    assert run_level_supported(9, 0.1) and not run_level_supported(8, 0.1)
    assert run_level_supported(4, 0.2) and not run_level_supported(3, 0.2)
    assert run_level_supported(19, 0.05) and not run_level_supported(12, 0.05)


def _synthetic(n: int, K: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    mu = rng.normal(size=n)
    s = rng.uniform(0.2, 2.0, size=n)
    y = mu + s * rng.normal(size=n)
    draws = mu[:, None] + s[:, None] * rng.normal(size=(n, K))
    return y, draws


@pytest.mark.unit
@pytest.mark.parametrize("method", ["absolute", "normalized", "constant_single_draw"])
@pytest.mark.parametrize("alpha", [0.2, 0.1, 0.05])
def test_exchangeable_data_covers_within_binomial_tolerance(method, alpha):
    rng = np.random.default_rng(0)
    hits = n = 0
    for _ in range(40):
        yc, dc = _synthetic(200, 16, rng)
        yt, dt = _synthetic(500, 16, rng)
        lo, hi = predict(method, calibrate(method, yc, dc, alpha), dt)
        hits += int(((yt >= lo) & (yt <= hi)).sum())
        n += yt.size
    cov = hits / n
    # Marginal coverage over calibration draws is 1-alpha to 1-alpha+1/(n_cal+1)
    assert 1 - alpha - 0.01 <= cov <= 1 - alpha + 1 / 201 + 0.01


@pytest.mark.unit
def test_normalized_interval_tracks_spread_and_absolute_does_not():
    rng = np.random.default_rng(1)
    yc, dc = _synthetic(300, 16, rng)
    _, dt = _synthetic(50, 16, rng)
    lo, hi = predict("normalized", calibrate("normalized", yc, dc, 0.1), dt)
    assert np.corrcoef(hi - lo, dt.std(axis=1))[0, 1] > 0.95
    lo, hi = predict("absolute", calibrate("absolute", yc, dc, 0.1), dt)
    assert np.ptp(hi - lo) < 1e-9


@pytest.mark.unit
def test_zero_spread_does_not_divide_by_zero():
    y = np.arange(30.0)
    draws = np.tile(y[:, None], (1, 4))
    cal = calibrate("normalized", y + 1.0, draws, 0.1)
    lo, hi = predict("normalized", cal, draws)
    assert cal.eps > 0 and np.isfinite(lo).all() and np.isfinite(hi).all()


@pytest.mark.unit
def test_raw_is_uncalibrated():
    y, d = _synthetic(50, 16, np.random.default_rng(2))
    a = calibrate("raw_ensemble", y, d, 0.1)
    b = calibrate("raw_ensemble", y + 1000.0, d, 0.1)
    assert a.q == b.q == pytest.approx(1.6448536, abs=1e-5)


@pytest.mark.unit
def test_methods_listed_and_unknown_rejected():
    assert set(METHODS) == {"absolute", "normalized", "constant_single_draw", "raw_ensemble"}
    with pytest.raises(ValueError):
        calibrate("nope", np.zeros(3), np.zeros((3, 2)), 0.1)


@pytest.mark.unit
def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        calibrate("absolute", np.zeros(3), np.zeros((4, 2)), 0.1)


@pytest.mark.unit
def test_wilson_interval_contains_estimate_and_clips():
    lo, hi = wilson_interval(9, 10)
    assert 0 < lo < 0.9 < hi < 1
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0


@pytest.mark.unit
def test_coverage_stats_values_and_interval_score():
    y = np.array([0.0, 0.0, 5.0, 5.0])
    lo, hi = np.full(4, -1.0), np.full(4, 1.0)
    s = coverage_stats(y, lo, hi, np.array(["a", "a", "b", "b"]), alpha=0.1, seed=0)
    assert s["n"] == 4 and s["n_runs"] == 2 and s["coverage"] == 0.5
    assert s["mean_width"] == 2.0 and s["median_width"] == 2.0
    assert s["interval_score"] == pytest.approx(2.0 + 0.5 * (2 / 0.1) * 4.0)
    assert s["boot_lo"] == 0.0 and s["boot_hi"] == 1.0


@pytest.mark.unit
def test_run_bootstrap_is_wider_than_binomial_under_within_run_dependence():
    runs = np.repeat(np.arange(10).astype(str), 50)
    y = np.repeat(np.r_[np.zeros(9), np.full(1, 5.0)], 50)  # one run misses entirely
    s = coverage_stats(y, np.full(500, -1.0), np.full(500, 1.0), runs, alpha=0.1, seed=0)
    assert (s["boot_hi"] - s["boot_lo"]) > 2 * (s["binom_hi"] - s["binom_lo"])


@pytest.mark.unit
def test_coverage_stats_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        coverage_stats(np.array([]), np.array([]), np.array([]), np.array([]), alpha=0.1, seed=0)
