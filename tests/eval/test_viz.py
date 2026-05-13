"""Smoke tests for all visualisation functions.

Checks: figure object returned, file written, file is non-empty.
Does not assert pixel-perfect output.
"""
from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import pytest
import torch

from neural_pbf.eval.adapters.identity_adapter import IdentityAdapter
from neural_pbf.eval.rollout.engine import RolloutEngine
from neural_pbf.eval.viz.report_figures import generate_report_figures
from neural_pbf.eval.viz.spatial import (
    gallery_evolution,
    gallery_test,
    isotherm_overlay,
    profile_slices,
    triple_view,
    val_grid_2x2,
)
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


# ── val_grid_2x2 ──────────────────────────────────────────────────────────────

@pytest.fixture
def two_tensors_3d():
    T_gt = torch.full((1, 1, 16, 8, 8), 0.3)
    T_pred = torch.full((1, 1, 16, 8, 8), 0.5)
    return T_gt, T_pred


@pytest.mark.unit
def test_val_grid_2x2_returns_figure(two_tensors):
    # two_tensors is (1,1,8,8) from conftest — treat D=1
    T_gt, T_pred = two_tensors
    fig = val_grid_2x2(T_gt, T_pred)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


@pytest.mark.unit
def test_val_grid_2x2_saves_file(tmp_path, two_tensors):
    T_gt, T_pred = two_tensors
    p = tmp_path / "grid.png"
    fig = val_grid_2x2(T_gt, T_pred, savepath=p)
    assert p.exists() and p.stat().st_size > 0
    plt.close(fig)


@pytest.mark.unit
def test_val_grid_2x2_3d_input(two_tensors_3d):
    T_gt, T_pred = two_tensors_3d
    fig = val_grid_2x2(T_gt, T_pred, epoch=5)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


@pytest.mark.unit
def test_val_grid_2x2_rope_shape():
    # RoPE model uses extra batch dim (1,1,1,D,H,W)
    T_gt = torch.full((1, 1, 1, 8, 8, 8), 0.3)
    T_pred = torch.full((1, 1, 1, 8, 8, 8), 0.5)
    fig = val_grid_2x2(T_gt, T_pred)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ── gallery_evolution ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_gallery_evolution_empty_history_raises():
    T_gt = torch.rand(1, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="empty"):
        gallery_evolution(T_gt, [])


@pytest.mark.unit
def test_gallery_evolution_single_history_returns_figure():
    T_gt = torch.rand(1, 1, 8, 8, 8)
    fig = gallery_evolution(T_gt, [torch.rand(1, 1, 8, 8, 8)])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


@pytest.mark.unit
def test_gallery_evolution_long_history_has_20_axes():
    """20-entry history → 2 rows × 10 cols = 20 axes."""
    T_gt = torch.rand(1, 1, 8, 8, 8)
    history = [torch.rand(1, 1, 8, 8, 8) for _ in range(20)]
    fig = gallery_evolution(T_gt, history)
    assert len(fig.get_axes()) == 20
    plt.close(fig)


@pytest.mark.unit
def test_gallery_evolution_saves_file(tmp_path):
    T_gt = torch.rand(1, 1, 8, 8, 8)
    history = [torch.rand(1, 1, 8, 8, 8) for _ in range(5)]
    out = tmp_path / "evolution.png"
    fig = gallery_evolution(T_gt, history, savepath=out, dpi=72)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


@pytest.mark.unit
def test_gallery_evolution_wrong_label_count_raises():
    T_gt = torch.rand(1, 1, 8, 8, 8)
    history = [torch.rand(1, 1, 8, 8, 8) for _ in range(5)]
    with pytest.raises(ValueError):
        gallery_evolution(T_gt, history, epoch_labels=["a", "b"])  # length mismatch


@pytest.mark.unit
def test_gallery_evolution_custom_labels_accepted():
    T_gt = torch.rand(1, 1, 8, 8, 8)
    history = [torch.rand(1, 1, 8, 8, 8) for _ in range(3)]
    fig = gallery_evolution(T_gt, history, epoch_labels=["E0", "E10", "E20"])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


# ── gallery_test ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_gallery_test_five_samples_returns_figure():
    samples = [(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)) for _ in range(5)]
    fig = gallery_test(samples)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


@pytest.mark.unit
def test_gallery_test_wrong_count_raises():
    # 6 samples exceeds the maximum of 5
    samples = [(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)) for _ in range(6)]
    with pytest.raises(ValueError, match="1–5"):
        gallery_test(samples)


@pytest.mark.unit
def test_gallery_test_empty_raises():
    with pytest.raises(ValueError, match="1–5"):
        gallery_test([])


@pytest.mark.unit
def test_gallery_test_fewer_than_5_samples_works():
    """gallery_test should accept 1–4 samples and hide unused columns."""
    samples = [(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)) for _ in range(3)]
    fig = gallery_test(samples)
    assert len(fig.get_axes()) == 20  # 2 rows × 10 cols always
    plt.close(fig)


@pytest.mark.unit
def test_gallery_test_has_20_axes():
    """5 samples × 2 cols each × 2 rows = 20 axes."""
    samples = [(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)) for _ in range(5)]
    fig = gallery_test(samples)
    assert len(fig.get_axes()) == 20
    plt.close(fig)


@pytest.mark.unit
def test_gallery_test_saves_file(tmp_path):
    samples = [(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)) for _ in range(5)]
    out = tmp_path / "test_gallery.png"
    fig = gallery_test(samples, savepath=out, dpi=72)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


@pytest.mark.unit
def test_gallery_test_custom_titles_accepted():
    samples = [(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 8, 8, 8)) for _ in range(5)]
    fig = gallery_test(samples, titles=["A", "B", "C", "D", "E"])
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
