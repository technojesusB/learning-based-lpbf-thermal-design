"""Evaluation tables, sensitivity analyses, and the report built from saved draws only."""

from __future__ import annotations

import numpy as np
import pytest

from neural_pbf.uq.analysis import (
    QUANTITIES,
    build_tables,
    evaluate,
    point_errors,
    second_seed_table,
    sensitivity_table,
)
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.report import build_report


def _set(n_runs=12, per_run=20, K=16, seed=0, nz=64, offset=0):
    rng = np.random.default_rng(seed)
    n = n_runs * per_run
    depth = rng.integers(0, nz + 1, size=n).astype(float)
    depth[:per_run] = 0.0
    depth[per_run : per_run * 2] = float(nz)
    t = rng.uniform(1500, 6000, size=n)
    v = rng.uniform(0, 1e5, size=n)
    truth = np.stack([depth, t, v], axis=1)
    scale = np.array([2.0, 100.0, 1000.0])
    draws = truth[:, None, :] + scale * rng.normal(size=(n, K, 3))
    runs = np.repeat([f"r{offset + i}" for i in range(n_runs)], per_run)
    return {
        "draws": draws, "truth": truth, "run_id": runs,
        "material": np.repeat(np.resize(["SS316L", "Ti64", "IN718"], n_runs), per_run),
        "pattern": np.repeat(np.resize(["raster", "zigzag"], n_runs), per_run),
        "nz": np.full(n, nz), "sample_index": np.arange(n), "volume_limited": depth >= nz,
        "T_s": np.full(n, 1650.0),  # per-sample solidus, attached from the HDF5 attrs in S6
    }


@pytest.fixture()
def cfg(tmp_path):
    return UQConfig.smoke(tmp_path).model_copy(update={"prereg": UQConfig().prereg})


@pytest.mark.unit
def test_evaluate_covers_quantities_alphas_methods_and_strata(cfg):
    df = evaluate(_set(), _set(n_runs=9, seed=1, offset=50), cfg.prereg)
    assert set(df["quantity"]) == set(QUANTITIES)
    assert set(df["alpha"]) == {0.2, 0.1, 0.05}
    assert set(df["method"]) == {"absolute", "normalized", "constant_single_draw", "raw_ensemble"}
    assert {"all", "material", "pattern", "run", "depth_stratum", "T_max_bin"} <= set(df["stratum_type"])
    assert set(df[df.stratum_type == "depth_stratum"]["quantity"]) == {"depth_vox"}
    assert set(df[df.stratum_type == "T_max_bin"]["quantity"]) == {"T_max_K"}


@pytest.mark.unit
def test_depth_strata_separate_zero_and_domain_limited(cfg):
    df = evaluate(_set(), _set(n_runs=9, seed=1, offset=50), cfg.prereg)
    s = df[(df.stratum_type == "depth_stratum") & (df.method == "absolute") & (df.alpha == 0.1)]
    strata = dict(zip(s.stratum, s.n, strict=True))
    assert strata["zero_depth"] >= 20 and strata["domain_limited"] >= 20
    assert strata["melt_only"] + strata["zero_depth"] == df[(df.stratum_type == "all")].n.iloc[0]


@pytest.mark.unit
def test_run_level_support_flag_marks_alpha_005_unsupported_with_12_runs(cfg):
    df = evaluate(_set(n_runs=12), _set(n_runs=9, seed=1, offset=50), cfg.prereg)
    flags = df[df.stratum_type == "all"].groupby("alpha")["run_level_supported"].first()
    assert flags[0.05] is np.False_ or not flags[0.05]
    assert flags[0.1] and flags[0.2]


@pytest.mark.unit
def test_evaluate_rejects_run_shared_by_cal_and_test(cfg):
    with pytest.raises(ValueError, match="both calibration and test"):
        evaluate(_set(), _set(n_runs=9, seed=1), cfg.prereg)


@pytest.mark.unit
def test_evaluate_rejects_non_finite_draws(cfg):
    bad = _set()
    bad["draws"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(bad, _set(n_runs=9, seed=1, offset=50), cfg.prereg)


@pytest.mark.unit
def test_point_errors_of_perfect_prediction_are_zero():
    s = _set(K=4)
    s["draws"] = np.repeat(s["truth"][:, None, :], 4, axis=1)
    e = point_errors(s)
    assert (e["mae"] == 0).all() and (e["rmse"] == 0).all() and (e["bias"] == 0).all()


@pytest.mark.unit
def test_sensitivity_subsamples_first_k_draws(cfg):
    df = sensitivity_table(_set(), _set(n_runs=9, seed=1, offset=50), cfg.prereg)
    assert sorted(df["K"].unique()) == [4, 8, 16]
    assert set(df["method"]) >= {"absolute", "normalized"}


@pytest.mark.unit
def test_second_seed_table_compares_same_samples(cfg):
    test = _set(n_runs=9, seed=1, offset=50)
    idx = test["sample_index"][::10]
    second = {k: v[::10] for k, v in test.items()}
    df = second_seed_table(_set(), test, second, cfg.prereg)
    assert (df["n"] == idx.size).all()
    assert {"coverage_primary_seed", "coverage_second_seed"} <= set(df.columns)


@pytest.mark.integration
def test_report_contains_only_generated_numbers_and_required_statements(cfg, tmp_path):
    results = [{
        "name": "Baseline (pre-step mask)", "label": "run-disjoint: yes (checkpoint manifest matches)",
        "cal": _set(), "test": _set(n_runs=9, seed=1, offset=50), "second": None, "role": "primary",
    }, {
        "name": "Legacy Baseline on its own held-out samples", "label": "run-disjoint: no (3 shared)",
        "cal": _set(n_runs=3, per_run=10), "test": _set(n_runs=3, per_run=5, seed=2, offset=70), "second": None,
        "role": "leakage_demo",
    }]
    meta = {
        "split_type": "run-level", "runs_per_set": {"train": 21, "val": 3, "cal": 12, "test": 9},
        "manifest_hash": "abc", "prereg_hash": "def", "access_events": [{"event": "evaluation", "allow_reeval": False}],
        "training": {"best_epoch": 3, "epochs_run": 5, "stopped_early": True},
    }
    out = tmp_path / "r.md"
    build_report(results, meta, cfg, out, tmp_path / "fig", build_tables(results, cfg.prereg))
    text = out.read_text()
    for needle in ("run-level", "n_cal", "n_test", "Wilson", "bootstrap", "run-disjoint: yes", "run-disjoint: no",
                   "not supported at run level", "simulated data only", "post-step mask", "zero-depth", "abc", "def",
                   "patch-limited", "zero-volume", "SeedSequence", "capped by the patch"):
        assert needle in text, needle
    for image in ("_nominal.png", "_spread.png", "_material.png", "coverage_per_run.png", "sensitivity_vs_k.png",
                  "leakage_demo.png"):  # fmt: skip
        assert image in text, image


@pytest.mark.integration
def test_report_records_reevaluation(cfg, tmp_path):
    results = [{"name": "M", "label": "run-disjoint: yes", "cal": _set(), "test": _set(n_runs=9, seed=1, offset=50),
                "second": None, "role": "primary"}]
    meta = {"split_type": "run-level", "runs_per_set": {}, "manifest_hash": "a", "prereg_hash": "b",
            "access_events": [{"event": "evaluation", "allow_reeval": False},
                              {"event": "evaluation", "allow_reeval": True}], "training": {}}
    out = tmp_path / "r.md"
    build_report(results, meta, cfg, out, tmp_path / "fig", build_tables(results, cfg.prereg))
    assert "evaluated 2 times" in out.read_text()


@pytest.mark.unit
def test_shared_runs_only_with_explicit_opt_in(cfg):
    assert not evaluate(_set(), _set(n_runs=9, seed=1), cfg.prereg, allow_shared_runs=True).empty


@pytest.mark.unit
def test_volume_strata_separate_zero_and_patch_limited(cfg):
    test = _set(n_runs=9, seed=1, offset=50)
    test["truth"][:7, 2] = 0.0
    test["volume_limited"][:7] = False
    df = evaluate(_set(), test, cfg.prereg)
    s = df[(df.stratum_type == "volume_stratum") & (df.method == "absolute") & (df.alpha == 0.1)]
    strata = dict(zip(s.stratum, s.n, strict=True))
    assert strata["zero_volume"] == 7
    assert strata["patch_limited"] == int(test["volume_limited"].sum())
    assert strata["melt_unlimited"] + strata["patch_limited"] + strata["zero_volume"] == test["truth"].shape[0]
    assert set(df[df.stratum_type == "volume_stratum"]["quantity"]) == {"volume_vox"}
