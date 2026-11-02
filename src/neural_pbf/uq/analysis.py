"""Calibration and evaluation from saved draws. No GPU, no model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from neural_pbf.uq.config import PreregConfig
from neural_pbf.uq.conformal import METHODS, calibrate, coverage_stats, predict, run_level_supported

QUANTITIES = ("depth_vox", "T_max_K", "volume_vox")


def _check(cal: dict[str, np.ndarray], test: dict[str, np.ndarray], allow_shared_runs: bool) -> None:
    for name, part in (("calibration", cal), ("test", test)):
        if part["truth"].shape[0] == 0:
            raise ValueError(f"{name} set is empty")
        if not (np.isfinite(part["draws"]).all() and np.isfinite(part["truth"]).all()):
            raise ValueError(f"non-finite values in {name} draws")
    shared = set(cal["run_id"].tolist()) & set(test["run_id"].tolist())
    if shared and not allow_shared_runs:
        raise ValueError(f"runs in both calibration and test: {sorted(shared)[:3]}")


def _strata(
    test: dict[str, np.ndarray], q_index: int, quantity: str, bins: tuple[float, ...]
) -> list[tuple[str, str, np.ndarray]]:
    """Reporting groups. Conditioning on the truth is for description, never for calibration."""
    y = test["truth"][:, q_index]
    out = [("all", "all", np.ones(y.size, dtype=bool))]
    for column, kind in (("material", "material"), ("pattern", "pattern"), ("run_id", "run")):
        out += [(kind, str(v), test[column] == v) for v in np.unique(test[column])]
    if quantity == "depth_vox":
        out += [
            ("depth_stratum", "melt_only", y > 0),
            ("depth_stratum", "zero_depth", y == 0),
            ("depth_stratum", "domain_limited", y >= test["nz"]),
        ]
    if quantity == "volume_vox":
        limited = test["volume_limited"].astype(bool)
        out += [
            ("volume_stratum", "zero_volume", y == 0),
            ("volume_stratum", "patch_limited", limited),
            ("volume_stratum", "melt_unlimited", (y > 0) & ~limited),
        ]
    if quantity == "T_max_K":
        edges = [-np.inf, *bins, np.inf]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            out.append(("T_max_bin", f"{lo:g} to {hi:g} K", (y >= lo) & (y < hi)))
    return out


def evaluate(
    cal: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    prereg: PreregConfig,
    allow_shared_runs: bool = False,
) -> pd.DataFrame:
    """Coverage rows per quantity, alpha, method, and reporting stratum.

    ``allow_shared_runs`` exists only for the labelled leakage demonstration.
    """
    _check(cal, test, allow_shared_runs)
    n_cal_runs = int(np.unique(cal["run_id"]).size)
    rows: list[dict[str, Any]] = []
    for q_index, quantity in enumerate(QUANTITIES):
        strata = _strata(test, q_index, quantity, prereg.T_max_bins_K)
        for alpha in prereg.alphas:
            for method in METHODS:
                rows += _method_rows(cal, test, q_index, quantity, alpha, method, strata, n_cal_runs, prereg)
    return pd.DataFrame(rows)


def _method_rows(
    cal: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    q_index: int,
    quantity: str,
    alpha: float,
    method: str,
    strata: list[tuple[str, str, np.ndarray]],
    n_cal_runs: int,
    prereg: PreregConfig,
) -> list[dict[str, Any]]:
    """Calibrate one method on one quantity once, then score it on every reporting stratum."""
    y_cal = cal["truth"][:, q_index]
    y = test["truth"][:, q_index]
    fitted = calibrate(method, y_cal, cal["draws"][:, :, q_index], alpha)
    lo, hi = predict(method, fitted, test["draws"][:, :, q_index])
    fixed = {
        "quantity": quantity,
        "alpha": alpha,
        "method": method,
        "q": fitted.q,
        "n_cal": y_cal.size,
        "n_cal_runs": n_cal_runs,
        "run_level_supported": run_level_supported(n_cal_runs, alpha),
    }
    return [
        {
            **fixed,
            "stratum_type": kind,
            "stratum": name,
            **coverage_stats(
                y[keep], lo[keep], hi[keep], test["run_id"][keep], alpha, prereg.bootstrap_seed, prereg.n_boot
            ),
        }
        for kind, name, keep in strata
        if keep.any()
    ]


def point_errors(part: dict[str, np.ndarray]) -> pd.DataFrame:
    """Error of the ensemble mean, plus the mean ensemble spread for comparison."""
    rows = []
    for q_index, quantity in enumerate(QUANTITIES):
        error = part["draws"][:, :, q_index].mean(axis=1) - part["truth"][:, q_index]
        rows.append(
            {
                "quantity": quantity,
                "n": error.size,
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt((error**2).mean())),
                "bias": float(error.mean()),
                "mean_spread": float(part["draws"][:, :, q_index].std(axis=1, ddof=1).mean()),
            }
        )
    return pd.DataFrame(rows)


def _first_k(part: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    return {**part, "draws": part["draws"][:, :k]}


def sensitivity_table(cal: dict[str, np.ndarray], test: dict[str, np.ndarray], prereg: PreregConfig) -> pd.DataFrame:
    """Coverage and width against ensemble size, using the first K draws of each input."""
    rows = []
    usable = [k for k in prereg.sensitivity_Ks if k <= cal["draws"].shape[1]]
    for k in usable:
        table = evaluate(_first_k(cal, k), _first_k(test, k), prereg)
        table = table[table.stratum_type == "all"].assign(K=k)
        rows.append(table[["K", "quantity", "alpha", "method", "coverage", "mean_width", "n"]])
    return pd.concat(rows, ignore_index=True)


def second_seed_table(
    cal: dict[str, np.ndarray], test: dict[str, np.ndarray], second: dict[str, np.ndarray], prereg: PreregConfig
) -> pd.DataFrame:
    """Same calibration and same test inputs, draws from a second base seed."""
    keep = np.isin(test["sample_index"], second["sample_index"])
    subset = {k: v[keep] for k, v in test.items()}
    if not np.array_equal(subset["sample_index"], second["sample_index"]):
        raise ValueError("second-seed draws are not aligned with the primary test draws")
    rows = []
    for q_index, quantity in enumerate(QUANTITIES):
        for alpha in prereg.alphas:
            for method in ("absolute", "normalized"):
                fitted = calibrate(method, cal["truth"][:, q_index], cal["draws"][:, :, q_index], alpha)
                y = subset["truth"][:, q_index]
                result: dict[str, Any] = {"quantity": quantity, "alpha": alpha, "method": method, "n": y.size}
                for label, part in (("primary", subset), ("second", second)):
                    lo, hi = predict(method, fitted, part["draws"][:, :, q_index])
                    result[f"coverage_{label}_seed"] = float(((y >= lo) & (y <= hi)).mean())
                    result[f"mean_width_{label}_seed"] = float((hi - lo).mean())
                rows.append(result)
    return pd.DataFrame(rows)


def sample_points(part: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-sample ensemble spread and absolute error of the ensemble mean."""
    frames = []
    for q_index, quantity in enumerate(QUANTITIES):
        draws = part["draws"][:, :, q_index]
        frames.append(
            pd.DataFrame(
                {
                    "quantity": quantity,
                    "run_id": part["run_id"],
                    "sample_index": part["sample_index"],
                    "spread": draws.std(axis=1, ddof=1),
                    "abs_error": np.abs(draws.mean(axis=1) - part["truth"][:, q_index]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_tables(results: list[dict[str, Any]], prereg: PreregConfig) -> dict[str, pd.DataFrame]:
    """All computed tables behind the report and figures. Sensitivity covers the primary model only."""
    coverage, errors, points = [], [], []
    for result in results:
        rows = evaluate(result["cal"], result["test"], prereg, allow_shared_runs=result["role"] == "leakage_demo")
        coverage.append(rows.assign(model=result["name"], label=result["label"], role=result["role"]))
        errors.append(point_errors(result["test"]).assign(model=result["name"]))
        points.append(sample_points(result["test"]).assign(model=result["name"]))
    primary = next(r for r in results if r["role"] == "primary")
    return {
        "coverage": pd.concat(coverage, ignore_index=True),
        "sensitivity": sensitivity_table(primary["cal"], primary["test"], prereg),
        "errors": pd.concat(errors, ignore_index=True),
        "points": pd.concat(points, ignore_index=True),
    }
