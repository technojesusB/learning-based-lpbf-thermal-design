"""mask_source: 'post' = legacy default, 'pre' = causal mask_in; manifest mode of build_patch_data."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.data.splits import make_manifest, save_manifest
from neural_pbf.training.factories import build_patch_data

FRACS = {"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25}


def _sample(f: h5py.File, i: int, power: float, with_mask_in: bool = True) -> None:
    g = f.require_group("samples").create_group(f"sample_{i:06d}")
    shape = (1, 1, 8, 64, 64)
    for name in ("T_in", "T_target", "Q"):
        g.create_dataset(name, data=np.full(shape, 500.0, dtype=np.float32))
    mask = np.zeros(shape, dtype=np.uint8)
    mask[..., :4, :, :] = 1
    g.create_dataset("mask", data=mask)
    if with_mask_in:
        g.create_dataset("mask_in", data=np.zeros(shape, dtype=np.uint8))
    g.attrs.update(
        mat_name="SS316L", pattern="raster", power=power, sigma=2e-5, rho=7900.0, cp=500.0, k_s=20.0, k_l=25.0,
        k_p=1.0, L=2.9e5, T_s=1658.0, T_l=1723.0, x=3e-4, y=2e-4, hatch_spacing=5e-5, angle_deg=1.0,
        exposure_time=5e-5,
    )


def _file(path, n_runs=1, per_run=2, with_mask_in=True) -> str:
    with h5py.File(path, "w") as f:
        f.attrs.update(Lx_m=1e-3, Ly_m=5e-4, Lz_m=1.25e-4, Nx=64, Ny=64, Nz=8)
        i = 0
        for r in range(n_runs):
            for _ in range(per_run):
                _sample(f, i, 100.0 + r, with_mask_in)
                i += 1
    return str(path)


@pytest.mark.unit
def test_default_is_causal_pre_mask(tmp_path):
    cfg = FMDatasetConfig(h5_paths=[_file(tmp_path / "a.h5")])
    assert cfg.mask_source == "pre"
    assert FMThermalDataset(cfg)[0]["mask"].sum() == 0


@pytest.mark.unit
def test_checkpoint_dump_without_mask_source_loads_as_post(tmp_path):
    # Checkpoints older than mask_source were trained on the post-step mask; the new default must not change them.
    p = _file(tmp_path / "a.h5")
    legacy_cfg = FMDatasetConfig(h5_paths=[p]).model_dump()
    legacy_cfg.pop("mask_source")
    cfg = FMDatasetConfig.from_checkpoint(legacy_cfg, h5_paths=[p])
    assert cfg.mask_source == "post"
    assert FMThermalDataset(cfg)[0]["mask"].sum() > 0


@pytest.mark.unit
def test_checkpoint_dump_keeps_its_own_conditioning_and_mask(tmp_path):
    p = _file(tmp_path / "a.h5")
    old = FMDatasetConfig(h5_paths=["gone.h5"], conditioning_keys=("power", "sigma"), mask_source="pre").model_dump()
    cfg = FMDatasetConfig.from_checkpoint(old, h5_paths=[p])
    assert cfg.conditioning_keys == ("power", "sigma") and cfg.mask_source == "pre" and cfg.h5_paths == [p]


@pytest.mark.unit
def test_default_conditioning_includes_exposure_time(tmp_path):
    cfg = FMDatasetConfig(h5_paths=[_file(tmp_path / "a.h5")])
    assert "exposure_time" in cfg.conditioning_keys
    # Uniform 30..80 us per run in prepare_trajectory: mean 55 us, std 50 us / sqrt(12).
    assert cfg.cond_means["exposure_time"] == pytest.approx(55e-6)
    assert cfg.cond_stds["exposure_time"] == pytest.approx(50e-6 / 12**0.5)
    cond = FMThermalDataset(cfg)[0]["conditioning"]
    i = cfg.conditioning_keys.index("exposure_time")
    assert cond[i].item() == pytest.approx((5e-5 - 55e-6) / (50e-6 / 12**0.5), rel=1e-5)


@pytest.mark.unit
def test_pre_returns_mask_in(tmp_path):
    ds = FMThermalDataset(FMDatasetConfig(h5_paths=[_file(tmp_path / "a.h5")], mask_source="pre"))
    assert ds[0]["mask"].sum() == 0


@pytest.mark.unit
def test_pre_without_mask_in_fails_loudly(tmp_path):
    p = _file(tmp_path / "a.h5", with_mask_in=False)
    ds = FMThermalDataset(FMDatasetConfig(h5_paths=[p], mask_source="pre"))
    with pytest.raises(KeyError, match="mask_in"):
        ds[0]


@pytest.mark.unit
def test_build_patch_data_with_manifest_uses_only_listed_sets(tmp_path):
    p = _file(tmp_path / "a.h5", n_runs=4)
    manifest = make_manifest([p], seed=1, fractions=FRACS)
    path = tmp_path / "m.json"
    save_manifest(manifest, path)
    split = build_patch_data([p], patch_size=64, batch_size=1, seed=1, manifest_path=str(path), mask_source="pre")
    assert len(split.train_ds) == 2 and len(split.val_ds) == 2
    assert len(split.test_ds) == 0, "training must never load calibration or test runs"
    assert split.ds_cfg.mask_source == "pre"
    keys = split.full_ds._keys
    train = {keys[i] for i in split.train_ds.indices}
    assert train == {tuple(k) for k in manifest["sets"]["train"][0]["sample_keys"]}


@pytest.mark.unit
def test_build_patch_data_rejects_manifest_with_unknown_samples(tmp_path):
    p, other = _file(tmp_path / "a.h5", n_runs=4), _file(tmp_path / "b.h5", n_runs=1)
    path = tmp_path / "m.json"
    save_manifest(make_manifest([p], seed=1, fractions=FRACS), path)
    with pytest.raises(ValueError, match="not found"):
        build_patch_data([other], patch_size=64, batch_size=1, seed=1, manifest_path=str(path))


@pytest.mark.unit
def test_build_patch_data_without_manifest_refuses_random_split(tmp_path):
    # A random split over samples puts neighbouring samples of one run in train and test (leakage).
    p = _file(tmp_path / "a.h5", n_runs=5)
    with pytest.raises(ValueError, match="manifest"):
        build_patch_data([p], patch_size=64, batch_size=1, seed=42)


@pytest.mark.unit
def test_build_patch_data_random_split_only_on_request(tmp_path):
    p = _file(tmp_path / "a.h5", n_runs=5)
    split = build_patch_data([p], patch_size=64, batch_size=1, seed=42, allow_random_split=True)
    assert (len(split.train_ds), len(split.val_ds), len(split.test_ds)) == (7, 2, 1)
    assert split.ds_cfg.mask_source == "pre"
