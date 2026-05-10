"""Tests for ReLoBRaLoWeighter — adaptive physics loss balancing."""
import math
import pytest

from neural_pbf.training.relobralo import ReLoBRaLoWeighter


# ---------------------------------------------------------------------------
# Unit tests for ReLoBRaLoWeighter
# ---------------------------------------------------------------------------


class TestReLoBRaLoWeighterInit:
    def test_default_lambda_is_one(self):
        w = ReLoBRaLoWeighter()
        assert w.lambda_phys == 1.0

    def test_custom_hyperparams_stored(self):
        w = ReLoBRaLoWeighter(alpha=0.99, beta=0.8, eps=1e-6)
        assert w.alpha == 0.99
        assert w.beta == 0.8
        assert w.eps == 1e-6

    def test_ema_buffers_start_as_none(self):
        w = ReLoBRaLoWeighter()
        assert w._ema_fm is None
        assert w._ema_phys is None
        assert w._init_fm is None
        assert w._init_phys is None


class TestReLoBRaLoWeighterFirstStep:
    """On the very first call, initialise buffers and return default lambda."""

    def test_first_step_returns_one(self):
        w = ReLoBRaLoWeighter()
        lam = w.step(loss_fm=1.0, loss_phys=1e26)
        assert lam == 1.0

    def test_first_step_sets_ema_buffers(self):
        w = ReLoBRaLoWeighter()
        w.step(loss_fm=2.5, loss_phys=1e20)
        assert w._ema_fm == pytest.approx(2.5)
        assert w._ema_phys == pytest.approx(1e20)

    def test_first_step_sets_init_buffers(self):
        w = ReLoBRaLoWeighter()
        w.step(loss_fm=2.5, loss_phys=1e20)
        assert w._init_fm == pytest.approx(2.5)
        assert w._init_phys == pytest.approx(1e20)


class TestReLoBRaLoWeighterSubsequentSteps:
    """After initialisation, lambda should adapt to balance the two losses."""

    def test_lambda_finite_after_second_step(self):
        w = ReLoBRaLoWeighter(alpha=0.9, beta=0.0)  # beta=0 → always use EMA reference
        w.step(loss_fm=1.0, loss_phys=1e26)
        lam = w.step(loss_fm=0.9, loss_phys=1e25)
        assert math.isfinite(lam)

    def test_lambda_positive(self):
        w = ReLoBRaLoWeighter(alpha=0.9, beta=0.0)
        w.step(loss_fm=1.0, loss_phys=1e26)
        for _ in range(10):
            lam = w.step(loss_fm=0.8, loss_phys=1e25)
            assert lam > 0, f"lambda went non-positive: {lam}"

    def test_lambda_scales_up_when_physics_drops_faster(self):
        """If physics loss drops much faster than FM loss, lambda should increase
        so the physics term still contributes meaningfully."""
        w = ReLoBRaLoWeighter(alpha=0.0, beta=0.0)  # alpha=0 → EMA = current loss immediately
        w.step(loss_fm=1.0, loss_phys=100.0)
        # physics drops 100x, FM stays flat
        lam = w.step(loss_fm=1.0, loss_phys=1.0)
        assert lam > 1.0, f"Expected lambda > 1.0 but got {lam}"

    def test_lambda_scales_down_when_physics_is_high(self):
        """If physics loss rises while FM is stable, lambda should decrease
        to prevent the physics term from dominating."""
        w = ReLoBRaLoWeighter(alpha=0.0, beta=0.0)
        w.step(loss_fm=1.0, loss_phys=1.0)
        # physics spikes 100x, FM stays flat
        lam = w.step(loss_fm=1.0, loss_phys=100.0)
        assert lam < 1.0, f"Expected lambda < 1.0 but got {lam}"

    def test_ema_updated_each_step(self):
        w = ReLoBRaLoWeighter(alpha=0.9, beta=0.0)
        w.step(loss_fm=1.0, loss_phys=1.0)
        prev_ema_fm = w._ema_fm
        w.step(loss_fm=2.0, loss_phys=2.0)
        assert w._ema_fm != prev_ema_fm

    def test_init_buffers_unchanged_after_first_step(self):
        w = ReLoBRaLoWeighter(alpha=0.9, beta=0.0)
        w.step(loss_fm=1.0, loss_phys=1.0)
        w.step(loss_fm=2.0, loss_phys=2.0)
        # init values must never change after first step
        assert w._init_fm == pytest.approx(1.0)
        assert w._init_phys == pytest.approx(1.0)


class TestReLoBRaLoWeighterNumericalStability:
    """Lambda must stay finite even with the extreme magnitudes seen in physics (1e26)."""

    def test_stable_with_astronomical_physics_loss(self):
        w = ReLoBRaLoWeighter(alpha=0.999, beta=0.9)
        w.step(loss_fm=0.5, loss_phys=3.2e26)
        for i in range(20):
            lam = w.step(loss_fm=0.4 - i * 0.01, loss_phys=2e26 - i * 1e24)
            assert math.isfinite(lam), f"lambda not finite at step {i}: {lam}"
            assert lam > 0

    def test_zero_physics_loss_does_not_crash(self):
        w = ReLoBRaLoWeighter(eps=1e-8)
        w.step(loss_fm=1.0, loss_phys=1e26)
        lam = w.step(loss_fm=0.9, loss_phys=0.0)
        assert math.isfinite(lam)

    def test_zero_fm_loss_does_not_crash(self):
        w = ReLoBRaLoWeighter(eps=1e-8)
        w.step(loss_fm=1.0, loss_phys=1e26)
        lam = w.step(loss_fm=0.0, loss_phys=1e26)
        assert math.isfinite(lam)
