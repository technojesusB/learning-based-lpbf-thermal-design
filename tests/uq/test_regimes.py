"""Coverage by thermal regime: solid, melt (conduction), keyhole (above boiling, outside solver validity)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_pbf.uq.config import ExploratoryConfig, PreregConfig
from neural_pbf.uq.regimes import assign_regimes, criterion, regime_rows

T_BOIL = {"SS316L": 3090.0, "Ti64": 3315.0}
PREREG = PreregConfig(alphas=(0.1,), n_boot=50)


@pytest.mark.unit
def test_regime_boundaries_per_material():
    t = np.array([1600.0, 1700.0, 3089.0, 3090.0, 3200.0])
    mat = np.array(["SS316L", "SS316L", "SS316L", "SS316L", "Ti64"])
    t_s = np.full(5, 1650.0)
    assert assign_regimes(t, mat, t_s, T_BOIL).tolist() == ["solid", "melt", "melt", "keyhole", "melt"]


@pytest.mark.unit
def test_unknown_material_fails():
    with pytest.raises(KeyError, match="IN718"):
        assign_regimes(np.array([2000.0]), np.array(["IN718"]), np.array([1500.0]), T_BOIL)


def _part(t_max: np.ndarray, noise: np.ndarray, runs: np.ndarray) -> dict[str, np.ndarray]:
    """Draws (n, K=2, 3 quantities); T_max error scaled by ``noise`` per sample."""
    n = t_max.size
    truth = np.stack([np.zeros(n), t_max, np.zeros(n)], axis=1)
    draws = np.repeat(truth[:, None, :], 2, axis=1).copy()
    draws[:, 0, 1] += noise
    draws[:, 1, 1] -= noise
    draws[:, :, 1] += noise[:, None] * np.where(np.arange(n) % 2, 1, -1)[:, None]
    return {"truth": truth, "draws": draws, "run_id": runs, "material": np.full(n, "SS316L"),
            "T_s": np.full(n, 1650.0)}  # fmt: skip


@pytest.mark.unit
def test_each_regime_gets_its_own_calibration():
    # Melt samples err by 10 K, keyhole samples by 1000 K: one global interval would be far too wide for melt.
    n = 200
    t = np.where(np.arange(n) < n // 2, 2000.0, 5000.0)
    noise = np.where(t < 3000, 10.0, 1000.0)
    runs = np.arange(n) % 20
    part = _part(t, noise, runs)
    rows = regime_rows(part, part, PREREG, T_BOIL, variant="predicted")
    width = {r["regime"]: r["mean_width"] for r in rows if r["quantity"] == "T_max_K" and r["method"] == "absolute"}
    assert width["melt"] < 100 < width["keyhole"]
    assert {"melt", "keyhole", "all"} <= set(width)


@pytest.mark.unit
def test_regime_without_calibration_samples_is_reported_not_dropped():
    cal = _part(np.full(20, 2000.0), np.full(20, 5.0), np.arange(20))
    test = _part(np.array([2000.0, 6000.0]), np.array([5.0, 5.0]), np.array([0, 1]))
    rows = regime_rows(cal, test, PREREG, T_BOIL, variant="true")
    keyhole = [r for r in rows if r["regime"] == "keyhole" and r["method"] == "absolute"][0]
    assert keyhole["n_cal"] == 0 and keyhole["n"] == 1 and np.isnan(keyhole["coverage"])


@pytest.mark.unit
def test_criterion_reads_melt_regime_at_primary_alpha():
    exp = ExploratoryConfig(criterion_min_coverage=0.85, criterion_max_width={"T_max_K": 100.0})
    rows = [{"model": "m", "variant": "predicted", "regime": "melt", "quantity": "T_max_K", "alpha": 0.1,
             "method": "absolute", "coverage": 0.9, "mean_width": 50.0}]  # fmt: skip
    assert criterion(rows, PREREG, exp, "m")["passed"]
    rows[0]["mean_width"] = 150.0
    assert not criterion(rows, PREREG, exp, "m")["passed"]
