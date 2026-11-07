"""Exploratory: coverage by thermal regime, added after the first test evaluation.

Regimes per sample: ``solid`` (T_max below the sample's solidus), ``melt`` (conduction melting, up to the
boiling point), ``keyhole`` (above boiling; the solver has no evaporation, so it is outside its validity).

Each regime gets its own split-conformal calibration (Mondrian). ``variant="predicted"`` assigns regimes
from the ensemble-mean T_max, which is known at prediction time, so it is usable in deployment.
``variant="true"`` assigns them from the true T_max: a diagnostic of where the solver is valid, unusable
in deployment because the truth is unknown then.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_pbf.uq.analysis import QUANTITIES
from neural_pbf.uq.config import ExploratoryConfig, PreregConfig
from neural_pbf.uq.conformal import METHODS, calibrate, coverage_stats, predict, run_level_supported

REGIMES = ("solid", "melt", "keyhole")
_T_MAX = QUANTITIES.index("T_max_K")


def assign_regimes(
    t_max: np.ndarray, materials: np.ndarray, t_solidus: np.ndarray, t_boil: dict[str, float]
) -> np.ndarray:
    missing = sorted(set(materials.tolist()) - set(t_boil))
    if missing:
        raise KeyError(f"no boiling point configured for {missing}")
    boil = np.array([t_boil[m] for m in materials])
    return np.where(t_max < t_solidus, "solid", np.where(t_max < boil, "melt", "keyhole"))


def _regimes(part: dict[str, np.ndarray], t_boil: dict[str, float], variant: str) -> np.ndarray:
    t_max = part["draws"][:, :, _T_MAX].mean(axis=1) if variant == "predicted" else part["truth"][:, _T_MAX]
    return assign_regimes(t_max, part["material"], part["T_s"], t_boil)


def regime_rows(
    cal: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    prereg: PreregConfig,
    t_boil: dict[str, float],
    variant: str,
) -> list[dict[str, Any]]:
    """One row per quantity, alpha, method, and regime, plus ``all`` for the regimes combined."""
    g_cal, g_test = _regimes(cal, t_boil, variant), _regimes(test, t_boil, variant)
    rows: list[dict[str, Any]] = []
    for q, quantity in enumerate(QUANTITIES):
        y = test["truth"][:, q]
        for alpha in prereg.alphas:
            for method in METHODS:
                lo, hi = np.full(y.size, np.nan), np.full(y.size, np.nan)
                for regime in REGIMES:
                    c, t = g_cal == regime, g_test == regime
                    if not t.any():
                        continue
                    n_cal_runs = int(np.unique(cal["run_id"][c]).size)
                    fixed = {"variant": variant, "regime": regime, "quantity": quantity, "alpha": alpha,
                             "method": method, "n_cal": int(c.sum()), "n_cal_runs": n_cal_runs,
                             "run_level_supported": run_level_supported(n_cal_runs, alpha)}  # fmt: skip
                    if not c.any():  # nothing to calibrate on: say so instead of borrowing another regime's width
                        rows.append({**fixed, "n": int(t.sum()), "coverage": np.nan, "mean_width": np.nan})
                        continue
                    fitted = calibrate(method, cal["truth"][c, q], cal["draws"][c][:, :, q], alpha)
                    lo[t], hi[t] = predict(method, fitted, test["draws"][t][:, :, q])
                    stats = coverage_stats(y[t], lo[t], hi[t], test["run_id"][t], alpha, prereg.bootstrap_seed,
                                           prereg.n_boot)  # fmt: skip
                    rows.append({**fixed, **stats})
                done = np.isfinite(lo)
                if done.any():
                    stats = coverage_stats(y[done], lo[done], hi[done], test["run_id"][done], alpha,
                                           prereg.bootstrap_seed, prereg.n_boot)  # fmt: skip
                    rows.append({"variant": variant, "regime": "all", "quantity": quantity, "alpha": alpha,
                                 "method": method, "n_cal": int(cal["truth"].shape[0]),
                                 "n_cal_runs": int(np.unique(cal["run_id"]).size),
                                 "run_level_supported": None, "n_uncalibrated": int((~done).sum()),
                                 **stats})  # fmt: skip
    return rows


def criterion(
    rows: list[dict[str, Any]], prereg: PreregConfig, exploratory: ExploratoryConfig, model: str
) -> dict[str, Any]:
    """The usefulness criterion fixed before run 2: melt regime, predicted regimes, primary alpha and method."""
    checks = {}
    for quantity, max_width in exploratory.criterion_max_width.items():
        row = next(
            (r for r in rows if r.get("model") == model and r["variant"] == "predicted" and r["regime"] == "melt"
             and r["quantity"] == quantity and r["alpha"] == prereg.primary_alpha
             and r["method"] == prereg.primary_method),
            None,
        )  # fmt: skip
        if row is None or not np.isfinite(row["coverage"]):
            checks[quantity] = {"passed": False, "reason": "no calibrated melt-regime samples"}
            continue
        ok = row["coverage"] >= exploratory.criterion_min_coverage and row["mean_width"] < max_width
        checks[quantity] = {"passed": bool(ok), "coverage": row["coverage"], "mean_width": row["mean_width"],
                            "max_width": max_width}  # fmt: skip
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def regime_table(results: list[dict[str, Any]], prereg: PreregConfig, t_boil: dict[str, float]) -> Any:
    """Both variants for every model, as one frame with a ``model`` column."""
    import pandas as pd

    rows = [
        {"model": r["name"], "role": r["role"], **row}
        for r in results
        for variant in ("predicted", "true")
        for row in regime_rows(r["cal"], r["test"], prereg, t_boil, variant)
    ]
    return pd.DataFrame(rows)
