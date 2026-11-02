"""Pre-registration freeze, test-access guard, leakage label, dataset validation."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.guard import (
    GuardError,
    TestAccessGuard,
    check_no_shared_conditioning,
    freeze_prereg,
    leakage_label,
    verify_prereg,
)
from neural_pbf.uq.validate import ValidationError, validate_dataset, validate_run_file


def _run(path, power=100.0, n=3, mat="SS316L", pat="raster", shape=(1, 1, 4, 8, 8)):
    with h5py.File(path, "w") as f:
        f.attrs.update(Nx=8, Ny=8, Nz=4)
        grp = f.create_group("samples")
        for i in range(n):
            g = grp.create_group(f"sample_{i:06d}")
            g.create_dataset("T_in", data=np.full(shape, 300.0, dtype=np.float16))
            g.create_dataset("T_target", data=np.full(shape, 900.0, dtype=np.float16))
            g.create_dataset("Q", data=np.full(shape, 1e15, dtype=np.float32))
            post = np.zeros(shape, dtype=np.uint8)
            post[..., 0, :, :] = 1
            g.create_dataset("mask", data=post)
            g.create_dataset("mask_in", data=np.zeros(shape, dtype=np.uint8))
            g.attrs.update(
                mat_name=mat, pattern=pat, power=power, sigma=2e-5, hatch_spacing=5e-5, angle_deg=1.0,
                exposure_time=5e-5, x=float(i), y=0.0, T_l=1700.0, rho=1.0, cp=1.0, k_s=1.0, k_l=1.0, k_p=1.0,
                L=1.0, T_s=1600.0,
            )
    return str(path)


# --- pre-registration -------------------------------------------------------------------


@pytest.mark.unit
def test_prereg_freeze_then_verify(tmp_path):
    cfg = UQConfig.smoke(tmp_path)
    p = tmp_path / "prereg.json"
    h = freeze_prereg(cfg, p)
    assert verify_prereg(cfg, p) == h
    assert json.loads(p.read_text())["K"] == cfg.prereg.K


@pytest.mark.unit
def test_prereg_detects_changed_config_and_refuses_overwrite(tmp_path):
    cfg = UQConfig.smoke(tmp_path)
    p = tmp_path / "prereg.json"
    freeze_prereg(cfg, p)
    changed = cfg.model_copy(update={"prereg": cfg.prereg.model_copy(update={"K": 99})})
    with pytest.raises(GuardError, match="pre-registered"):
        verify_prereg(changed, p)
    with pytest.raises(GuardError, match="pre-registered"):
        freeze_prereg(changed, p)
    assert freeze_prereg(cfg, p)  # identical re-freeze is idempotent


@pytest.mark.unit
def test_verify_without_freeze_fails(tmp_path):
    with pytest.raises(GuardError, match="frozen"):
        verify_prereg(UQConfig.smoke(tmp_path), tmp_path / "missing.json")


# --- test access ------------------------------------------------------------------------


@pytest.mark.unit
def test_access_is_logged_and_second_evaluation_refused(tmp_path):
    g = TestAccessGuard(tmp_path / "access.jsonl", prereg_hash="abc")
    g.log_access("sampling", model="m")
    g.begin_evaluation(allow_reeval=False)
    with pytest.raises(GuardError, match="allow-reeval"):
        g.begin_evaluation(allow_reeval=False)
    assert [e["event"] for e in g.events()] == ["sampling", "evaluation"]


@pytest.mark.unit
def test_reeval_allowed_only_explicitly_and_is_recorded(tmp_path):
    g = TestAccessGuard(tmp_path / "access.jsonl", prereg_hash="abc")
    g.begin_evaluation(allow_reeval=False)
    g.begin_evaluation(allow_reeval=True)
    assert g.n_evaluations() == 2
    assert g.events()[-1]["allow_reeval"] is True


@pytest.mark.unit
def test_guard_survives_new_process_instance(tmp_path):
    TestAccessGuard(tmp_path / "a.jsonl", "abc").begin_evaluation(False)
    with pytest.raises(GuardError):
        TestAccessGuard(tmp_path / "a.jsonl", "abc").begin_evaluation(False)


@pytest.mark.unit
def test_guard_rejects_changed_prereg_hash(tmp_path):
    TestAccessGuard(tmp_path / "a.jsonl", "abc").log_access("sampling")
    with pytest.raises(GuardError, match="pre-registration changed"):
        TestAccessGuard(tmp_path / "a.jsonl", "xyz").log_access("sampling")


# --- leakage label ----------------------------------------------------------------------


def _manifest_with(run_paths_test):
    from neural_pbf.data.splits import discover_runs

    runs, _ = discover_runs(run_paths_test)
    return {"hash": "H", "sets": {"test": [{"run_id": r.run_id, "family_id": r.family_id} for r in runs]}}


@pytest.mark.unit
def test_label_yes_on_matching_hash_no_on_other_hash(tmp_path):
    m = {"hash": "H", "sets": {"test": []}}
    assert leakage_label({"manifest_hash": "H"}, m, "test").startswith("run-disjoint: yes")
    assert leakage_label({"manifest_hash": "G"}, m, "test").startswith("run-disjoint: no")


@pytest.mark.unit
def test_label_from_dataset_cfg_uses_actual_overlap(tmp_path):
    train = _run(tmp_path / "train.h5", power=100.0)
    same = _run(tmp_path / "same.h5", power=100.0)
    other = _run(tmp_path / "other.h5", power=200.0)
    ckpt = {"dataset_cfg": {"h5_paths": [train]}}
    assert leakage_label(ckpt, _manifest_with([same]), "test").startswith("run-disjoint: no")
    assert leakage_label(ckpt, _manifest_with([other]), "test").startswith("run-disjoint: yes")


@pytest.mark.unit
def test_label_unknown_without_any_information(tmp_path):
    m = {"hash": "H", "sets": {"test": []}}
    assert leakage_label({}, m, "test").startswith("run-disjoint: unknown")
    assert leakage_label({"dataset_cfg": {"h5_paths": ["/gone.h5"]}}, m, "test").startswith("run-disjoint: unknown")


@pytest.mark.unit
def test_shared_conditioning_across_sets_detected(tmp_path):
    a, b = _run(tmp_path / "a.h5", 100.0), _run(tmp_path / "b.h5", 100.0)
    m = {"sets": {
        "train": [{"sample_keys": [[a, "sample_000000"]]}],
        "test": [{"sample_keys": [[b, "sample_000000"]]}],
    }}
    with pytest.raises(GuardError, match="identical conditioning"):
        check_no_shared_conditioning(m, ("power", "sigma", "x", "y"))
    c = _run(tmp_path / "c.h5", 200.0)
    m["sets"]["test"] = [{"sample_keys": [[c, "sample_000000"]]}]
    check_no_shared_conditioning(m, ("power", "sigma", "x", "y"))


# --- validation -------------------------------------------------------------------------


@pytest.mark.unit
def test_valid_run_passes(tmp_path):
    validate_run_file(_run(tmp_path / "a.h5"), n_samples=3, deep=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda g: g["T_target"].__setitem__((0, 0, 0, 0, 0), np.nan), "non-finite"),
        (lambda g: g["Q"].__setitem__((0, 0, 0, 0, 0), np.inf), "non-finite"),
        (lambda g: g["mask_in"].__setitem__((0, 0, 3, 0, 0), 1), "subset"),
        (lambda g: g.__delitem__("mask_in"), "missing"),
        (lambda g: g.attrs.__delitem__("k_p"), "missing"),
    ],
)
def test_corrupt_run_fails_deep_validation(tmp_path, mutate, match):
    p = _run(tmp_path / "a.h5")
    with h5py.File(p, "a") as f:
        mutate(f["samples"]["sample_000001"])
    with pytest.raises(ValidationError, match=match):
        validate_run_file(p, n_samples=3, deep=True)


@pytest.mark.unit
def test_wrong_count_and_unreadable_file(tmp_path):
    with pytest.raises(ValidationError, match="samples"):
        validate_run_file(_run(tmp_path / "a.h5"), n_samples=4, deep=False)
    bad = tmp_path / "bad.h5"
    bad.write_bytes(b"not hdf5")
    with pytest.raises(ValidationError, match="unreadable"):
        validate_run_file(str(bad), n_samples=1, deep=False)


@pytest.mark.unit
def test_dataset_rejects_duplicate_tuples_and_mixed_files(tmp_path):
    a, b = _run(tmp_path / "a.h5", 100.0), _run(tmp_path / "b.h5", 100.0)
    with pytest.raises(ValidationError, match="parameter tuple"):
        validate_dataset([a, b], n_samples=3, expected_runs=2, min_per_pattern={})


@pytest.mark.unit
def test_dataset_checks_run_count_and_pattern_minimum(tmp_path):
    a, b = _run(tmp_path / "a.h5", 100.0), _run(tmp_path / "b.h5", 200.0, pat="zigzag")
    report = validate_dataset([a, b], n_samples=3, expected_runs=2, min_per_pattern={"zigzag": 1})
    assert report["n_runs"] == 2 and report["patterns"] == {"raster": 1, "zigzag": 1}
    with pytest.raises(ValidationError, match="zigzag"):
        validate_dataset([a, b], n_samples=3, expected_runs=2, min_per_pattern={"zigzag": 5})
    with pytest.raises(ValidationError, match="expected 3"):
        validate_dataset([a, b], n_samples=3, expected_runs=3, min_per_pattern={})
