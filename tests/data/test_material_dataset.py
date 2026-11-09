"""Per-sample material for the physics loss: read once per run from the stored values, added to every item."""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from neural_pbf.data import material_dataset as mdmod
from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.data.material_dataset import (
    MATERIAL_DIM,
    MaterialDataset,
    exposure_time_of,
    material_from_vector,
    resolve_key,
)
from neural_pbf.data.patch_dataset import PatchFMThermalDataset
from neural_pbf.data.sampling import randomize_material
from neural_pbf.physics.material import MaterialConfig

NZ, NY, NX = 4, 16, 16
SHAPE = (1, 1, NZ, NY, NX)
COND = {"power": 250.0, "sigma": 25e-6, "x": 1.6e-5, "y": 1.6e-5}


def _attrs(mat: MaterialConfig, exposure: float) -> dict:
    return {
        "mat_name": "X", "k_s": mat.k_solid, "k_l": mat.k_liquid, "k_p": mat.k_powder, "cp": mat.cp_base,
        "rho": mat.rho, "L": mat.latent_heat_L, "T_s": mat.T_solidus, "T_l": mat.T_liquidus,
        "exposure_time": exposure, **COND,
    }  # fmt: skip


def _write_run(path: Path, mat: MaterialConfig, exposure: float, n: int = 3, attrs_override: dict | None = None):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs.update({"Lx_m": 3e-5, "Ly_m": 3e-5, "Lz_m": 6e-6, "Nx": NX, "Ny": NY, "Nz": NZ})
        for i in range(n):
            g = f.create_group(f"samples/sample_{i:06d}")
            for key in ("T_in", "T_target", "Q"):
                g.create_dataset(key, data=(rng.random(SHAPE) * 1000 + 300).astype(np.float16))
            for key in ("mask", "mask_in"):
                g.create_dataset(key, data=rng.integers(0, 2, size=SHAPE, dtype=np.uint8))
            for key in ("T_lut", "k_lut", "cp_lut"):
                g.create_dataset(key, data=np.asarray(getattr(mat, key), dtype=np.float32))
            attrs = _attrs(mat, exposure)
            if attrs_override and i == n - 1:
                attrs.update(attrs_override)
            g.attrs.update(attrs)


@pytest.fixture()
def runs(tmp_path: Path) -> tuple[list[str], list[MaterialConfig]]:
    random.seed(0)
    mats = [randomize_material(MaterialConfig.ss316l_preset()), randomize_material(MaterialConfig.ti64_preset())]
    paths = [str(tmp_path / f"run_{i:03d}.h5") for i in range(2)]
    for p, m, e in zip(paths, mats, (40e-6, 70e-6), strict=True):
        _write_run(Path(p), m, e)
    return paths, mats


def _stack(paths: list[str]) -> tuple[FMThermalDataset, Subset]:
    full = FMThermalDataset(FMDatasetConfig(h5_paths=paths))
    return full, Subset(PatchFMThermalDataset(full, patch_size=8), [5, 1, 3])


@pytest.mark.unit
def test_resolve_key_walks_subset_and_patch_wrapper_to_fm_keys(runs) -> None:
    """Pins the dependency on FMThermalDataset._keys and the wrappers' index mapping."""
    full, ds = _stack(runs[0])
    assert [resolve_key(ds, i) for i in range(3)] == [full._keys[5], full._keys[1], full._keys[3]]


@pytest.mark.unit
def test_items_carry_the_stored_material_and_keep_other_fields(runs) -> None:
    paths, mats = runs
    full, ds = _stack(paths)
    item = MaterialDataset(ds)[0]  # index 5 -> second run (Ti64, 7-point LUT)
    assert set(ds[0]) | {"material"} == set(item)
    vec = item["material"]
    assert vec.shape == (MATERIAL_DIM,) and vec.dtype == torch.float64
    rebuilt = material_from_vector(vec)
    for field in ("k_solid", "k_liquid", "k_powder", "cp_base", "rho", "latent_heat_L", "T_solidus", "T_liquidus"):
        assert getattr(rebuilt, field) == pytest.approx(getattr(mats[1], field))
    for field in ("T_lut", "k_lut", "cp_lut"):
        assert getattr(rebuilt, field) == pytest.approx(getattr(mats[1], field), rel=1e-6)
    assert len(rebuilt.T_lut) == 7
    assert rebuilt.use_T_dep and rebuilt.use_lut and rebuilt.transition_sharpness == mats[1].transition_sharpness
    assert exposure_time_of(vec) == 70e-6  # the stored attribute, not a z-score round trip


@pytest.mark.unit
def test_material_is_read_once_per_run(runs, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    real = mdmod._read_run
    monkeypatch.setattr(mdmod, "_read_run", lambda p: (opened.append(str(p)), real(p))[1])
    _, ds = _stack(runs[0])
    wrapped = MaterialDataset(ds)
    for _ in range(2):
        for i in range(len(wrapped)):
            wrapped[i]
    assert sorted(opened) == sorted(runs[0])


@pytest.mark.unit
@pytest.mark.parametrize("override", [{"rho": 1.0}, {"exposure_time": 1e-6}])
def test_a_run_whose_samples_disagree_is_refused(tmp_path: Path, override: dict) -> None:
    path = tmp_path / "run_bad.h5"
    _write_run(path, MaterialConfig.ss316l_preset(), 40e-6, attrs_override=override)
    _, ds = _stack([str(path)] * 1)
    ds = Subset(ds.dataset, [0])
    with pytest.raises(ValueError, match="differ"):
        MaterialDataset(ds)[0]


@pytest.mark.unit
def test_batches_stack_the_material(runs) -> None:
    _, ds = _stack(runs[0])
    batch = next(iter(DataLoader(MaterialDataset(ds), batch_size=3)))
    assert batch["material"].shape == (3, MATERIAL_DIM)
    # Subset indices 5, 1, 3 belong to runs 1, 0, 1.
    assert [exposure_time_of(v) for v in batch["material"]] == [70e-6, 40e-6, 70e-6]
