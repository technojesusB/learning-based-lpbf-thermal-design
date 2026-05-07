"""Smoke tests for all visualisation functions.

Checks: figure object returned, file written, file is non-empty.
Does not assert pixel-perfect output.
"""
from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.adapters.identity_adapter import IdentityAdapter
from neural_pbf.eval.rollout.engine import RolloutEngine
from neural_pbf.eval.viz.report_figures import generate_report_figures
from neural_pbf.eval.viz.spatial import isotherm_overlay, profile_slices, triple_view
from neural_pbf.eval.viz.temporal import (
    error_evolution_plot,
    peak_tracking_plot,
    probe_history_plot,
)


@pytest.fixture
def two_tensors(sim_cfg):
    T_gt = torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), 800.0)
    T_pred = T_gt + 50.0
    return T_gt, T_pred


@pytest.fixture
def rollout_result(trajectory):
    return RolloutEngine().run(IdentityAdapter(), trajectory, mode="one_step")


# ── spatial ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_triple_view_returns_figure(two_tensors):
    import matplotlib.figure

    T_gt, T_pred = two_tensors
    fig = triple_view(T_gt, T_pred)
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.unit
def test_triple_view_saves_file(tmp_path, two_tensors):
    T_gt, T_pred = two_tensors
    p = tmp_path / "triple.png"
    triple_view(T_gt, T_pred, savepath=p)
    assert p.exists() and p.stat().st_size > 0


@pytest.mark.unit
def test_profile_slices_saves_file(tmp_path, two_tensors):
    T_gt, T_pred = two_tensors
    p = tmp_path / "slices.png"
    profile_slices(T_gt, T_pred, savepath=p)
    assert p.exists() and p.stat().st_size > 0


@pytest.mark.unit
def test_isotherm_overlay_saves_file(tmp_path, two_tensors):
    T_gt, T_pred = two_tensors
    p = tmp_path / "iso.png"
    isotherm_overlay(T_gt, T_pred, T_iso=700.0, savepath=p)
    assert p.exists() and p.stat().st_size > 0


# ── temporal ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_peak_tracking_plot_returns_figure(rollout_result):
    import matplotlib.figure

    fig = peak_tracking_plot(rollout_result)
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.unit
def test_peak_tracking_plot_saves_file(tmp_path, rollout_result):
    p = tmp_path / "peak.png"
    peak_tracking_plot(rollout_result, savepath=p)
    assert p.exists() and p.stat().st_size > 0


@pytest.mark.unit
def test_error_evolution_plot_saves_file(tmp_path, rollout_result):
    p = tmp_path / "err.png"
    error_evolution_plot(rollout_result, savepath=p)
    assert p.exists() and p.stat().st_size > 0


@pytest.mark.unit
def test_probe_history_plot_saves_file(tmp_path, rollout_result):
    p = tmp_path / "probes.png"
    probes = [("center", 4, 4, None)]
    probe_history_plot(rollout_result, probes, savepath=p)
    assert p.exists() and p.stat().st_size > 0


# ── report_figures ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_generate_report_figures_writes_files(tmp_path, rollout_result):
    written = generate_report_figures(rollout_result, tmp_path)
    assert len(written) >= 2
    for p in written:
        assert p.exists() and p.stat().st_size > 0
