"""ReLoBRaLoWeighter: the update of Bischof & Kraus 2021 (arXiv 2110.09813) for the data and physics terms.

    λ_i^bal(t, t') = m · softmax_i( L_i(t) / (T·L_i(t')) )                (Eq. 11, m = 2)
    λ_i^hist(t)    = ρ·λ_i(t−1) + (1−ρ)·λ_i^bal(t, 0),   ρ ~ Bernoulli(beta)
    λ_i(t)         = α·λ_i^hist(t) + (1−α)·λ_i^bal(t, t−1)

A term that progresses slowly relative to its reference gets the larger weight.
"""

import math

import pytest

from neural_pbf.training.relobralo import ReLoBRaLoWeighter


def _bal(losses: tuple[float, float], ref: tuple[float, float], T: float) -> list[float]:
    z = [loss / (T * r) for loss, r in zip(losses, ref, strict=True)]
    e = [math.exp(v - max(z)) for v in z]
    return [2 * v / sum(e) for v in e]


class TestInit:
    def test_starts_at_one(self):
        w = ReLoBRaLoWeighter()
        assert w.lambda_phys == w.lambda_raw == 1.0 and not w.cap_hit and not w.skipped

    def test_hyperparams_stored(self):
        w = ReLoBRaLoWeighter(alpha=0.99, beta=0.8, temperature=0.05, eps=1e-6, max_lambda=1.0)
        assert (w.alpha, w.beta, w.temperature, w.eps, w.max_lambda) == (0.99, 0.8, 0.05, 1e-6, 1.0)

    @pytest.mark.parametrize(
        ("kw", "name"),
        [({"alpha": 1.5}, "alpha"), ({"alpha": -0.1}, "alpha"), ({"beta": 2.0}, "beta"),
         ({"temperature": 0.0}, "temperature"), ({"temperature": math.inf}, "temperature"),
         ({"max_lambda": 0.0}, "max_lambda"), ({"max_lambda": -1.0}, "max_lambda"),
         ({"max_lambda": math.inf}, "max_lambda"), ({"max_lambda": math.nan}, "max_lambda")],
    )  # fmt: skip
    def test_invalid_hyperparams_are_refused(self, kw, name):
        with pytest.raises(ValueError, match=name):
            ReLoBRaLoWeighter(**kw)


class TestUpdate:
    def test_first_step_only_records_the_references(self):
        w = ReLoBRaLoWeighter()
        assert w.step(2.5, 1e20) == 1.0 and w.lambda_raw == 1.0

    def test_stagnating_physics_gets_the_larger_weight(self):
        w = ReLoBRaLoWeighter(alpha=0.0, beta=0.0)
        w.step(1.0, 1.0)
        assert w.step(0.01, 1.0) > 1.9  # data improves 100x, physics flat

    def test_fast_physics_gets_the_smaller_weight(self):
        w = ReLoBRaLoWeighter(alpha=0.0, beta=0.0)
        w.step(1.0, 1.0)
        assert w.step(1.0, 0.01) < 0.1  # physics improves 100x, data flat

    def test_alpha_mixes_history_and_the_last_step(self):
        """beta = 1 means rho = 1 always: λ = α·λ(t−1) + (1−α)·bal(t, t−1)."""
        w = ReLoBRaLoWeighter(alpha=0.5, beta=1.0, temperature=0.5)
        w.step(1.0, 2.0)
        w.step(0.8, 1.9)
        lam1 = 0.5 * 1.0 + 0.5 * _bal((0.8, 1.9), (1.0, 2.0), 0.5)[1]
        assert w.lambda_raw == pytest.approx(lam1)
        w.step(0.5, 1.7)
        assert w.lambda_raw == pytest.approx(0.5 * lam1 + 0.5 * _bal((0.5, 1.7), (0.8, 1.9), 0.5)[1])

    def test_lookback_uses_the_first_losses(self):
        """beta = 0 means rho = 0 always: λ = α·bal(t, 0) + (1−α)·bal(t, t−1)."""
        w = ReLoBRaLoWeighter(alpha=0.7, beta=0.0, temperature=0.5)
        w.step(1.0, 2.0)
        w.step(0.8, 1.9)
        w.step(0.5, 1.7)
        expected = 0.7 * _bal((0.5, 1.7), (1.0, 2.0), 0.5)[1] + 0.3 * _bal((0.5, 1.7), (0.8, 1.9), 0.5)[1]
        assert w.lambda_raw == pytest.approx(expected)

    def test_the_two_weights_sum_to_two(self):
        w = ReLoBRaLoWeighter(alpha=0.9, beta=0.5, temperature=0.1)
        w.step(1.0, 1.0)
        for i in range(50):
            w.step(1.0 / (i + 2), 1.0 / (0.5 * i + 2))
            assert w.lambda_fm + w.lambda_raw == pytest.approx(2.0)


class TestCap:
    @staticmethod
    def _stagnating(max_lambda: float | None) -> ReLoBRaLoWeighter:
        w = ReLoBRaLoWeighter(alpha=0.0, beta=0.0, max_lambda=max_lambda)
        w.step(1.0, 1.0)
        w.step(0.01, 1.0)
        return w

    def test_default_has_no_cap(self):
        w = self._stagnating(None)
        assert w.lambda_raw > 1.9 and w.lambda_phys == w.lambda_raw and not w.cap_hit

    def test_cap_limits_the_used_weight_and_reports_the_hit(self):
        w = self._stagnating(1.0)
        assert w.lambda_raw > 1.9 and w.lambda_phys == 1.0 and w.cap_hit

    def test_below_the_cap_nothing_changes(self):
        w = ReLoBRaLoWeighter(alpha=0.0, beta=0.0, max_lambda=1.0)
        w.step(1.0, 1.0)
        assert w.step(1.0, 0.01) == w.lambda_raw < 1.0 and not w.cap_hit

    def test_history_carries_the_uncapped_weight(self):
        """The cap acts on the weight that is used, not on the state the update builds on."""
        w = ReLoBRaLoWeighter(alpha=0.5, beta=1.0, max_lambda=1.0)
        w.step(1.0, 1.0)
        w.step(0.01, 1.0)
        raw = w.lambda_raw
        assert raw > 1.0 and w.lambda_phys == 1.0
        w.step(0.01, 1.0)  # same losses: bal(t, t−1) is 1 for both terms, up to eps in the denominator
        assert w.lambda_raw == pytest.approx(0.5 * raw + 0.5 * 1.0, rel=1e-4)


class TestNumerics:
    def test_stable_with_astronomical_physics_loss(self):
        w = ReLoBRaLoWeighter(alpha=0.999, beta=0.9)
        w.step(0.5, 3.2e26)
        for i in range(20):
            lam = w.step(0.4 - i * 0.01, 2e26 - i * 1e24)
            assert math.isfinite(lam) and lam > 0

    @pytest.mark.parametrize(("fm", "phys"), [(0.9, 0.0), (0.0, 1e26), (0.0, 0.0)])
    def test_zero_losses_do_not_crash(self, fm, phys):
        w = ReLoBRaLoWeighter()
        w.step(1.0, 1e26)
        assert math.isfinite(w.step(fm, phys))

    def test_zero_reference_does_not_crash(self):
        w = ReLoBRaLoWeighter()
        w.step(0.0, 0.0)
        assert math.isfinite(w.step(1.0, 1.0))

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_non_finite_step_holds_lambda_and_is_marked(self, bad):
        w = ReLoBRaLoWeighter(alpha=0.0, beta=0.0, max_lambda=1.0)
        w.step(1.0, 1.0)
        w.step(0.01, 1.0)
        assert w.cap_hit
        held = w.lambda_phys
        assert w.step(bad, 1.0) == held
        assert w.skipped and not w.cap_hit
        w.step(0.01, 1.0)
        assert not w.skipped
