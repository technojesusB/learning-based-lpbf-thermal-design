import pytest
import torch

from neural_pbf.diagnostics.recorder import DiagnosticsRecorder
from neural_pbf.schemas.diagnostics import DiagnosticsConfig


def test_diagnostics_nan_detection():
    cfg = DiagnosticsConfig(enabled=True, check_nan_inf=True, strict=False)
    rec = DiagnosticsRecorder(cfg)

    # Clean state
    T = torch.zeros(10, 10)
    rec.on_step_start(T, {})
    metrics = rec.on_step_end(0, T, None, {})
    assert metrics["stability/nan_count"] == 0

    # NaN state
    T[0, 0] = float("nan")
    metrics = rec.on_step_end(1, T, None, {})
    assert metrics["stability/nan_count"] == 1
    assert metrics["stability/warn_flag"] == 0  # No threshold set


def test_diagnostics_thresholds():
    cfg = DiagnosticsConfig(
        enabled=True, thresholds={"sim/temperature_max": 100.0}, strict=True
    )
    rec = DiagnosticsRecorder(cfg)

    T = torch.full((10, 10), 200.0)

    with pytest.raises(RuntimeError, match="exceeded threshold"):
        rec.on_step_end(0, T, None, {})


def test_diagnostics_energy_proxy():
    cfg = DiagnosticsConfig(enabled=True)
    rec = DiagnosticsRecorder(cfg)

    T = torch.ones(10, 10)
    meta = {"dt": 1.0, "scan_power": 50.0}

    metrics = rec.on_step_end(0, T, None, meta)
    assert metrics["energy/power_in_W"] == 50.0
    assert metrics["energy/energy_in_J"] == 50.0

    # Cumulative?
    metrics = rec.on_step_end(1, T, None, meta)
    assert metrics["energy/energy_in_J"] == 100.0
