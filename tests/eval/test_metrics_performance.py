"""Unit tests for performance profiling helpers."""
from __future__ import annotations

import time

import pytest

from neural_pbf.eval.metrics.performance import (
    LatencyTimer,
    reset_vram_peak,
    speedup_ratio,
    vram_peak_bytes,
)


@pytest.mark.unit
def test_latency_timer_measures_nonzero():
    with LatencyTimer() as t:
        time.sleep(0.01)
    assert t.elapsed_s > 0.005  # at least half the sleep


@pytest.mark.unit
def test_latency_timer_elapsed_monotonic():
    with LatencyTimer() as t1:
        time.sleep(0.005)
    with LatencyTimer() as t2:
        time.sleep(0.020)
    assert t2.elapsed_s > t1.elapsed_s


@pytest.mark.unit
def test_vram_peak_bytes_returns_int():
    reset_vram_peak()
    result = vram_peak_bytes()
    assert isinstance(result, int)
    assert result >= 0


@pytest.mark.unit
def test_speedup_ratio_correct():
    assert speedup_ratio(10.0, 2.0) == pytest.approx(5.0)


@pytest.mark.unit
def test_speedup_ratio_zero_candidate_returns_inf():
    assert speedup_ratio(5.0, 0.0) == float("inf")


@pytest.mark.unit
def test_speedup_ratio_negative_candidate_returns_inf():
    assert speedup_ratio(5.0, -1.0) == float("inf")
