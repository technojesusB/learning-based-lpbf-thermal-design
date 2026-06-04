"""Unit tests for patch_dataset module."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import pytest
import torch

from neural_pbf.data.patch_dataset import _collate_with_strings, read_grid_attrs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def h5_grid(tmp_path: pytest.TempPathFactory) -> str:
    """Minimal HDF5 file with required root-level grid attributes."""
    path = str(tmp_path / "grid.h5")
    Nx, Ny, Nz = 32, 32, 8
    with h5py.File(path, "w") as f:
        f.attrs["Lx_m"] = 1.0e-3
        f.attrs["Ly_m"] = 1.0e-3
        f.attrs["Lz_m"] = 2.0e-4
        f.attrs["Nx"] = Nx
        f.attrs["Ny"] = Ny
        f.attrs["Nz"] = Nz
        samples = f.create_group("samples")
        grp = samples.create_group("s0")
        shape = (Nz, Ny, Nx)
        rng = np.random.default_rng(0)
        grp.create_dataset("T_in", data=rng.random(shape).astype(np.float16))
        grp.attrs["x"] = 5e-4
        grp.attrs["y"] = 5e-4
    return path


@pytest.fixture()
def h5_no_nz(tmp_path: pytest.TempPathFactory) -> str:
    """HDF5 without Nz — Nz should be inferred from T_in shape."""
    path = str(tmp_path / "no_nz.h5")
    Nx, Ny, Nz = 16, 16, 4
    with h5py.File(path, "w") as f:
        f.attrs["Lx_m"] = 5.0e-4
        f.attrs["Ly_m"] = 5.0e-4
        f.attrs["Nx"] = Nx
        f.attrs["Ny"] = Ny
        # No Lz_m, no Nz — isotropic assumption
        samples = f.create_group("samples")
        grp = samples.create_group("s0")
        shape = (Nz, Ny, Nx)
        rng = np.random.default_rng(1)
        grp.create_dataset("T_in", data=rng.random(shape).astype(np.float16))
        grp.attrs["x"] = 2.5e-4
        grp.attrs["y"] = 2.5e-4
    return path


# ---------------------------------------------------------------------------
# read_grid_attrs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_grid_attrs_returns_required_keys(h5_grid: str) -> None:
    attrs = read_grid_attrs(h5_grid)
    required = {"Lx_m", "Ly_m", "Lz_m", "Nx", "Ny", "Nz", "dx_m", "dy_m", "dz_m"}
    assert required <= attrs.keys()


@pytest.mark.unit
def test_read_grid_attrs_dx_computed_correctly(h5_grid: str) -> None:
    attrs = read_grid_attrs(h5_grid)
    expected_dx = 1.0e-3 / (32 - 1)
    assert abs(attrs["dx_m"] - expected_dx) < 1e-15


@pytest.mark.unit
def test_read_grid_attrs_nz_inferred_without_attribute(h5_no_nz: str) -> None:
    attrs = read_grid_attrs(h5_no_nz)
    assert attrs["Nz"] == 4


@pytest.mark.unit
def test_read_grid_attrs_isotropic_dz_when_lz_absent(h5_no_nz: str) -> None:
    attrs = read_grid_attrs(h5_no_nz)
    # Without Lz_m: dz_m = dx_m (isotropic assumption)
    assert abs(attrs["dz_m"] - attrs["dx_m"]) < 1e-15


@pytest.mark.unit
def test_read_grid_attrs_explicit_lz_used(h5_grid: str) -> None:
    attrs = read_grid_attrs(h5_grid)
    expected_dz = 2.0e-4 / (8 - 1)
    assert abs(attrs["dz_m"] - expected_dz) < 1e-15


@pytest.mark.unit
def test_read_grid_attrs_missing_required_raises(tmp_path: pytest.TempPathFactory) -> None:
    path = str(tmp_path / "bad.h5")
    with h5py.File(path, "w") as f:
        f.attrs["Lx_m"] = 1e-3
        # Missing Ly_m, Nx, Ny
    with pytest.raises(ValueError, match="missing required"):
        read_grid_attrs(path)


# ---------------------------------------------------------------------------
# _collate_with_strings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_collate_with_strings_tensor_fields_stacked() -> None:
    batch: list[dict[str, Any]] = [
        {"x": torch.tensor([1.0, 2.0]), "key": "a"},
        {"x": torch.tensor([3.0, 4.0]), "key": "b"},
    ]
    result = _collate_with_strings(batch)
    assert result["x"].shape == (2, 2)


@pytest.mark.unit
def test_collate_with_strings_preserves_string_as_list() -> None:
    batch: list[dict[str, Any]] = [
        {"x": torch.zeros(3), "name": "foo"},
        {"x": torch.ones(3), "name": "bar"},
    ]
    result = _collate_with_strings(batch)
    assert isinstance(result["name"], list)
    assert result["name"] == ["foo", "bar"]


@pytest.mark.unit
def test_collate_with_strings_multiple_string_fields() -> None:
    batch: list[dict[str, Any]] = [
        {"v": torch.zeros(2), "path": "/a/b", "key": "s1"},
        {"v": torch.ones(2), "path": "/c/d", "key": "s2"},
    ]
    result = _collate_with_strings(batch)
    assert result["path"] == ["/a/b", "/c/d"]
    assert result["key"] == ["s1", "s2"]
    assert result["v"].shape == (2, 2)


@pytest.mark.unit
def test_collate_with_strings_no_string_fields_works() -> None:
    batch: list[dict[str, Any]] = [
        {"a": torch.tensor(1.0)},
        {"a": torch.tensor(2.0)},
    ]
    result = _collate_with_strings(batch)
    assert result["a"].shape == (2,)
