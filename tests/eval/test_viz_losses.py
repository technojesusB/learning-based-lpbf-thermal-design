"""Tests for the loss analytics plot module.

Tests written FIRST (TDD) before implementation exists.
All tests should FAIL until losses.py is implemented.
"""
from __future__ import annotations

import pytest
import matplotlib


@pytest.mark.unit
def test_loss_panel_no_lambda_returns_figure():
    """Single-panel loss curve (no lambda_hist) returns a Figure."""
    from neural_pbf.eval.viz.losses import loss_panel

    train = [1.0, 0.8, 0.6, 0.4]
    val = [1.1, 0.9, 0.65, 0.45]
    fig = loss_panel(train, val)
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.unit
def test_loss_panel_with_lambda_returns_figure():
    """Two-panel figure when lambda_hist is supplied."""
    from neural_pbf.eval.viz.losses import loss_panel

    train = [1.0, 0.8, 0.6]
    val = [1.1, 0.85, 0.65]
    lam = [0.1, 0.2, 0.3]
    fig = loss_panel(train, val, lambda_hist=lam)
    assert isinstance(fig, matplotlib.figure.Figure)
    # Two-panel layout: figure must have exactly 2 axes
    assert len(fig.get_axes()) == 2


@pytest.mark.unit
def test_loss_panel_no_lambda_has_single_axes():
    """Without lambda_hist only one axes panel is created."""
    from neural_pbf.eval.viz.losses import loss_panel

    train = [1.0, 0.5]
    val = [1.2, 0.6]
    fig = loss_panel(train, val)
    assert len(fig.get_axes()) == 1


@pytest.mark.unit
def test_loss_panel_saves_file(tmp_path):
    """savepath kwarg writes a non-empty PNG file."""
    from neural_pbf.eval.viz.losses import loss_panel

    train = [1.0, 0.8, 0.6]
    val = [1.1, 0.85, 0.65]
    p = tmp_path / "losses.png"
    fig = loss_panel(train, val, savepath=p)
    assert p.exists()
    assert p.stat().st_size > 0
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.unit
def test_loss_panel_mismatched_lengths_raises():
    """Different-length train_losses and val_losses must raise ValueError."""
    from neural_pbf.eval.viz.losses import loss_panel

    with pytest.raises(ValueError, match="same length"):
        loss_panel([1.0, 0.9], [1.1, 0.85, 0.7])


@pytest.mark.unit
def test_loss_panel_empty_lists_raises():
    """Empty loss lists should raise ValueError."""
    from neural_pbf.eval.viz.losses import loss_panel

    with pytest.raises(ValueError):
        loss_panel([], [])


@pytest.mark.unit
def test_loss_panel_with_title():
    """title kwarg is accepted without error."""
    from neural_pbf.eval.viz.losses import loss_panel

    fig = loss_panel([1.0, 0.5], [1.1, 0.6], title="Training run 1")
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.unit
def test_loss_panel_saves_lambda_panel_file(tmp_path):
    """Two-panel (with lambda_hist) also writes file when savepath given."""
    from neural_pbf.eval.viz.losses import loss_panel

    train = [1.0, 0.8, 0.6]
    val = [1.1, 0.85, 0.65]
    lam = [0.1, 0.15, 0.2]
    p = tmp_path / "losses_lambda.png"
    loss_panel(train, val, lambda_hist=lam, savepath=p)
    assert p.exists()
    assert p.stat().st_size > 0


@pytest.mark.unit
def test_loss_panel_lambda_mismatched_with_train_raises():
    """lambda_hist with different length from train_losses must raise ValueError."""
    from neural_pbf.eval.viz.losses import loss_panel

    with pytest.raises(ValueError):
        loss_panel([1.0, 0.8], [1.1, 0.85], lambda_hist=[0.1])


@pytest.mark.unit
def test_loss_panel_detailed_two_panels():
    """Detailed mode (fm_losses + pde_losses) always produces exactly 2 axes."""
    from neural_pbf.eval.viz.losses import loss_panel

    train = [1.0, 0.8, 0.6]
    val = [1.1, 0.85, 0.65]
    fm = [0.9, 0.7, 0.5]
    pde = [0.1, 0.15, 0.1]
    lam = [0.5, 0.6, 0.7]
    fig = loss_panel(train, val, fm_losses=fm, pde_losses=pde, lambda_hist=lam)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.get_axes()) == 2


@pytest.mark.unit
def test_loss_panel_detailed_no_lambda():
    """Detailed mode works without lambda_hist (uses default flat lambda=1.0)."""
    from neural_pbf.eval.viz.losses import loss_panel

    train = [1.0, 0.8]
    val = [1.1, 0.9]
    fm = [0.9, 0.7]
    pde = [0.1, 0.1]
    fig = loss_panel(train, val, fm_losses=fm, pde_losses=pde)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.get_axes()) == 2


@pytest.mark.unit
def test_loss_panel_partial_subloss_raises():
    """Providing only fm_losses (without pde_losses) must raise ValueError."""
    from neural_pbf.eval.viz.losses import loss_panel

    with pytest.raises(ValueError):
        loss_panel([1.0, 0.8], [1.1, 0.9], fm_losses=[0.9, 0.7])


@pytest.mark.unit
def test_loss_panel_fm_mismatched_raises():
    """fm_losses with different length from train_losses must raise ValueError."""
    from neural_pbf.eval.viz.losses import loss_panel

    with pytest.raises(ValueError):
        loss_panel([1.0, 0.8], [1.1, 0.9], fm_losses=[0.9], pde_losses=[0.1, 0.05])


@pytest.mark.unit
def test_loss_panel_figure_not_leaked(tmp_path):
    """loss_panel must not leave open figures around (prevents resource leaks)."""
    import matplotlib.pyplot as plt
    from neural_pbf.eval.viz.losses import loss_panel

    before = plt.get_fignums()
    fig = loss_panel([1.0, 0.5], [1.1, 0.6])
    after = plt.get_fignums()
    # The returned figure may still be open; we just close it ourselves.
    plt.close(fig)
    # After closing returned fig, count should match what we started with.
    after_close = plt.get_fignums()
    assert set(after_close) == set(before)
