"""Run-level split: no run in two sets, deterministic, manifest round-trip."""

from __future__ import annotations

import json

import h5py
import pytest

from neural_pbf.data.splits import (
    SplitError,
    discover_runs,
    load_manifest,
    make_manifest,
    save_manifest,
)

MATS = ["SS316L", "Ti64", "IN718"]
PATTERNS = ["raster", "island", "zigzag"]
FRACS = {"train": 0.45, "val": 0.1, "cal": 0.25, "test": 0.2}


def _write_run(f: h5py.File, start: int, n: int, mat: str, pat: str, power: float, t_l: float = 1700.0) -> None:
    grp = f.require_group("samples")
    for i in range(start, start + n):
        g = grp.create_group(f"sample_{i:06d}")
        g.attrs.update(
            mat_name=mat, pattern=pat, power=power, sigma=2e-5, hatch_spacing=5e-5,
            angle_deg=10.0, exposure_time=5e-5, x=float(i), y=0.0, T_l=t_l,
        )


def _dataset(tmp_path, n_runs: int = 45, n_samples: int = 2) -> list[str]:
    paths = []
    for r in range(n_runs):
        p = tmp_path / f"run_{r:03d}.h5"
        with h5py.File(p, "w") as f:
            _write_run(f, 0, n_samples, MATS[r % 3], PATTERNS[(r // 3) % 3], 100.0 + r)
        paths.append(str(p))
    return paths


@pytest.mark.unit
def test_discover_multiple_runs_in_one_file(tmp_path):
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        _write_run(f, 0, 3, "SS316L", "raster", 100.0)
        _write_run(f, 3, 2, "Ti64", "island", 200.0)
    runs, dropped = discover_runs([str(p)])
    assert [len(r.sample_keys) for r in runs] == [3, 2]
    assert [r.material for r in runs] == ["SS316L", "Ti64"]
    assert dropped == []


@pytest.mark.unit
def test_discover_drops_cross_file_duplicate_and_records_it(tmp_path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    for p in (a, b):
        with h5py.File(p, "w") as f:
            _write_run(f, 0, 2, "SS316L", "raster", 100.0)
    runs, dropped = discover_runs([str(a), str(b)])
    assert len(runs) == 1 and runs[0].h5_path == str(a)
    assert len(dropped) == 1 and dropped[0].h5_path == str(b)


@pytest.mark.unit
def test_same_scan_different_material_share_a_family(tmp_path):
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        _write_run(f, 0, 2, "SS316L", "raster", 100.0)
        _write_run(f, 2, 2, "IN718", "raster", 100.0)
    runs, _ = discover_runs([str(p)])
    assert runs[0].run_id != runs[1].run_id
    assert runs[0].family_id == runs[1].family_id


@pytest.mark.unit
def test_family_is_never_split_across_sets(tmp_path):
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        start = 0
        for r in range(12):
            _write_run(f, start, 2, "SS316L", "raster", 100.0 + r)
            _write_run(f, start + 2, 2, "IN718", "raster", 100.0 + r)
            start += 4
    m = make_manifest([str(p)], seed=1, fractions=FRACS, min_runs={"cal": 1, "test": 1})
    fam_set: dict[str, str] = {}
    for name, runs in m["sets"].items():
        for r in runs:
            assert fam_set.setdefault(r["family_id"], name) == name


@pytest.mark.unit
def test_no_run_or_sample_in_two_sets(tmp_path):
    m = make_manifest(_dataset(tmp_path), seed=7, fractions=FRACS)
    run_ids = [r["run_id"] for runs in m["sets"].values() for r in runs]
    keys = [tuple(k) for runs in m["sets"].values() for r in runs for k in r["sample_keys"]]
    assert len(run_ids) == len(set(run_ids)) == 45
    assert len(keys) == len(set(keys)) == 90


@pytest.mark.unit
def test_deterministic_and_seed_sensitive(tmp_path):
    paths = _dataset(tmp_path)
    a, b = make_manifest(paths, seed=7, fractions=FRACS), make_manifest(paths, seed=7, fractions=FRACS)
    assert a["hash"] == b["hash"]
    assert make_manifest(paths, seed=8, fractions=FRACS)["hash"] != a["hash"]


@pytest.mark.unit
def test_fractions_respected(tmp_path):
    m = make_manifest(_dataset(tmp_path), seed=7, fractions=FRACS)
    assert {k: len(v) for k, v in m["sets"].items()} == {"train": 21, "val": 3, "cal": 12, "test": 9}


@pytest.mark.unit
def test_materials_and_patterns_stratified(tmp_path):
    m = make_manifest(_dataset(tmp_path), seed=7, fractions=FRACS)
    train = {r["material"] for r in m["sets"]["train"]}
    for name in ("cal", "test"):
        assert {r["material"] for r in m["sets"][name]} == set(MATS) == train
        assert {r["pattern"] for r in m["sets"][name]} == set(PATTERNS)


@pytest.mark.unit
def test_material_absent_from_train_raises(tmp_path):
    p = tmp_path / "a.h5"
    with h5py.File(p, "w") as f:
        _write_run(f, 0, 2, "Ti64", "raster", 100.0)
    with pytest.raises(SplitError):
        make_manifest([str(p)], seed=1, fractions=FRACS)


@pytest.mark.unit
def test_too_few_runs_for_required_set_size_raises(tmp_path):
    with pytest.raises(SplitError, match="cal"):
        make_manifest(_dataset(tmp_path, n_runs=18), seed=7, fractions=FRACS, min_runs={"cal": 9})


@pytest.mark.unit
def test_bad_fractions_raise(tmp_path):
    with pytest.raises(SplitError, match="sum"):
        make_manifest(_dataset(tmp_path), seed=7, fractions={"train": 0.5, "val": 0.1, "cal": 0.1, "test": 0.1})


@pytest.mark.unit
def test_manifest_round_trip_and_tamper_detection(tmp_path):
    m = make_manifest(_dataset(tmp_path), seed=7, fractions=FRACS)
    path = tmp_path / "manifest.json"
    save_manifest(m, path)
    assert load_manifest(path) == json.loads(json.dumps(m))
    tampered = json.loads(path.read_text())
    tampered["sets"]["test"].append(tampered["sets"]["train"][0])
    path.write_text(json.dumps(tampered))
    with pytest.raises(SplitError, match="hash"):
        load_manifest(path)


@pytest.mark.unit
def test_overlap_rejected_even_with_valid_hash(tmp_path):
    from neural_pbf.data.splits import _hash_manifest

    m = make_manifest(_dataset(tmp_path), seed=7, fractions=FRACS)
    m["sets"]["test"].append(m["sets"]["train"][0])
    m["hash"] = _hash_manifest(m)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m))
    with pytest.raises(SplitError, match="more than one set"):
        load_manifest(path)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [2026, 1, 2])
def test_patterns_balanced_in_every_set_for_factorial_design(tmp_path, seed):
    from collections import Counter

    m = make_manifest(_dataset(tmp_path), seed=seed, fractions=FRACS)
    expected = {"train": 7, "val": 1, "cal": 4, "test": 3}
    for name, runs in m["sets"].items():
        assert Counter(r["pattern"] for r in runs) == dict.fromkeys(PATTERNS, expected[name]), name


@pytest.mark.unit
def test_seed_changes_which_runs_go_where(tmp_path):
    paths = _dataset(tmp_path)
    ids = lambda s: {r["run_id"] for r in make_manifest(paths, s, FRACS)["sets"]["test"]}  # noqa: E731
    assert ids(1) != ids(2)
