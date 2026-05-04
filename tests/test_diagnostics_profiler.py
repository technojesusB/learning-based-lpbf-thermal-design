"""
Tests for neural_pbf.data.diagnostics module.

TDD RED phase: these tests fail before diagnostics.py is created.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from neural_pbf.data.diagnostics import (
    DiagnosticsCsvWriter,
    MLflowStepLogger,
    StepProfiler,
    StepTiming,
    format_diag_line,
)


def _timing(
    step_idx: int = 54,
    t_sim: float = 1.2,
    t_transfer: float = 0.8,
    t_io: float = 2.1,
    substeps: int = 140,
    peak_temperature: float = 1850.0,
    is_sample: bool = True,
) -> StepTiming:
    return StepTiming(
        step_idx=step_idx,
        t_sim=t_sim,
        t_transfer=t_transfer,
        t_io=t_io,
        substeps=substeps,
        peak_temperature=peak_temperature,
        is_sample=is_sample,
    )


# ── StepTiming ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_step_timing_is_frozen():
    """StepTiming must be immutable (frozen dataclass)."""
    t = _timing()
    with pytest.raises((AttributeError, TypeError)):
        t.t_sim = 9.9  # type: ignore[misc]


# ── format_diag_line ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_diag_line_exact_format():
    """Must produce the exact spec string: Step N | Sim: Xs | Xfer: Xs | IO: Xs | Steps: N | Tmax: NK."""
    result = format_diag_line(
        _timing(step_idx=54, t_sim=1.2, t_transfer=0.8, t_io=2.1, substeps=140, peak_temperature=1850.0)
    )
    assert result == "Step 54 | Sim: 1.2s | Xfer: 0.8s | IO: 2.1s | Steps: 140 | Tmax: 1850K"


@pytest.mark.unit
def test_format_diag_line_rounds_temperature():
    """Temperature value must be rounded to the nearest integer."""
    result = format_diag_line(_timing(peak_temperature=1849.6))
    assert "Tmax: 1850K" in result


@pytest.mark.unit
def test_format_diag_line_step_zero():
    """Step index 0 must format correctly (regression guard)."""
    result = format_diag_line(_timing(step_idx=0))
    assert result.startswith("Step 0 |")


# ── StepProfiler ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_step_profiler_phase_records_nonnegative_duration():
    """A phase context must record a non-negative wall-clock duration."""
    profiler = StepProfiler(sync_cuda=False)
    with profiler.phase("sim"):
        pass
    assert profiler.get("sim") >= 0.0


@pytest.mark.unit
def test_step_profiler_multiple_independent_phases():
    """Multiple phases must be tracked independently."""
    profiler = StepProfiler(sync_cuda=False)
    with profiler.phase("sim"):
        pass
    with profiler.phase("transfer"):
        pass
    with profiler.phase("io"):
        pass
    assert profiler.get("sim") >= 0.0
    assert profiler.get("transfer") >= 0.0
    assert profiler.get("io") >= 0.0


@pytest.mark.unit
def test_step_profiler_get_missing_phase_returns_default():
    """get() for an unknown phase must return the given default (0.0)."""
    profiler = StepProfiler(sync_cuda=False)
    assert profiler.get("nonexistent") == 0.0
    assert profiler.get("nonexistent", 99.0) == 99.0


@pytest.mark.unit
def test_step_profiler_build_assembles_step_timing():
    """build() must return a StepTiming with values matching recorded phases."""
    profiler = StepProfiler(sync_cuda=False)
    with profiler.phase("sim"):
        pass

    timing = profiler.build(step_idx=7, substeps=50, peak_temperature=2000.0, is_sample=True)

    assert isinstance(timing, StepTiming)
    assert timing.step_idx == 7
    assert timing.substeps == 50
    assert timing.peak_temperature == 2000.0
    assert timing.is_sample is True
    assert timing.t_sim >= 0.0
    assert timing.t_transfer == 0.0  # phase never measured
    assert timing.t_io == 0.0  # phase never measured


# ── DiagnosticsCsvWriter ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_csv_writer_creates_file_with_header(tmp_path: Path):
    """First write must create the file and include a header row."""
    csv_path = tmp_path / "diag.csv"
    DiagnosticsCsvWriter(csv_path).write(_timing())

    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.reader(f))

    assert len(rows) >= 2
    assert "step_idx" in rows[0]
    assert "t_sim" in rows[0]
    assert "peak_temperature" in rows[0]


@pytest.mark.unit
def test_csv_writer_header_appears_exactly_once(tmp_path: Path):
    """Header must appear exactly once regardless of how many rows are written."""
    csv_path = tmp_path / "diag.csv"
    writer = DiagnosticsCsvWriter(csv_path)
    for i in range(5):
        writer.write(_timing(step_idx=i))

    content = csv_path.read_text()
    assert content.count("step_idx") == 1


@pytest.mark.unit
def test_csv_writer_appends_one_row_per_write(tmp_path: Path):
    """Each write() call must add exactly one data row."""
    csv_path = tmp_path / "diag.csv"
    writer = DiagnosticsCsvWriter(csv_path)
    for i in range(3):
        writer.write(_timing(step_idx=i))

    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 4  # 1 header + 3 data rows


@pytest.mark.unit
def test_csv_writer_stores_correct_values(tmp_path: Path):
    """Written row must contain the correct step_idx and t_sim values."""
    csv_path = tmp_path / "diag.csv"
    DiagnosticsCsvWriter(csv_path).write(_timing(step_idx=99, t_sim=3.14))

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["step_idx"] == "99"
    assert abs(float(rows[0]["t_sim"]) - 3.14) < 1e-9


# ── MLflowStepLogger ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_mlflow_logger_noop_when_disabled():
    """log() must not raise when enabled=False, regardless of MLflow state."""
    MLflowStepLogger(enabled=False).log(_timing(), global_step=0)


@pytest.mark.unit
def test_mlflow_logger_noop_without_active_run():
    """log() must silently absorb the MlflowException when no run is active."""
    MLflowStepLogger(enabled=True).log(_timing(), global_step=0)
