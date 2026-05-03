"""Tests for FMThermalDataset and FMDatasetConfig.

Fixtures build a synthetic HDF5 file that matches the schema written by
generate_offline_dataset.py: sample groups under 'samples/', datasets shaped
(1, 1, Nz, Ny, Nx), and per-sample attributes (not a 'scalars' dataset).
Q is always returned — it is used as the 3rd spatial input channel.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NZ, NY, NX = 4, 8, 8
FIELD_SHAPE = (1, 1, NZ, NY, NX)

SAMPLE_ATTRS: dict = {
    "mat_name": "SS316L",
    "k_s": 20.0,
    "k_l": 25.0,
    "k_p": 0.5,
    "cp": 500.0,
    "rho": 7980.0,
    "L": 2.72e5,
    "T_s": 1658.0,
    "T_l": 1723.0,
    "power": 250.0,
    "sigma": 25e-6,
    "exposure_time": 50e-6,
    "point_distance": 33e-6,
    "hatch_spacing": 33e-6,
    "pattern": "zigzag",
    "angle_deg": 45.0,
    "x": 5e-4,
    "y": 2.5e-4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sample(
    grp: h5py.Group,
    shape: tuple[int, ...] = FIELD_SHAPE,
    attrs: dict | None = None,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    for key in ("T_in", "T_target", "Q"):
        data = (rng.random(shape) * 1500 + 300).astype(np.float16)
        grp.create_dataset(key, data=data, compression="gzip")
    grp.create_dataset(
        "mask",
        data=rng.integers(0, 2, size=shape, dtype=np.uint8),
        compression="gzip",
    )
    grp.attrs.update(attrs if attrs is not None else SAMPLE_ATTRS)


def _make_cfg(h5_path: str | list[str]) -> FMDatasetConfig:
    paths = [h5_path] if isinstance(h5_path, str) else h5_path
    return FMDatasetConfig(h5_paths=paths)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def h5_single(tmp_path: pytest.TempPathFactory) -> str:
    path = str(tmp_path / "single.h5")
    with h5py.File(path, "w") as f:
        grp = f.create_group("samples").create_group("sample_000000")
        _write_sample(grp, seed=42)
    return path


@pytest.fixture()
def h5_multi(tmp_path: pytest.TempPathFactory) -> str:
    path = str(tmp_path / "multi.h5")
    with h5py.File(path, "w") as f:
        samples = f.create_group("samples")
        for i in range(5):
            _write_sample(samples.create_group(f"sample_{i:06d}"), seed=i)
    return path


@pytest.fixture()
def h5_empty(tmp_path: pytest.TempPathFactory) -> str:
    path = str(tmp_path / "empty.h5")
    with h5py.File(path, "w") as f:
        f.create_group("samples")
    return path


@pytest.fixture()
def default_cfg(h5_single: str) -> FMDatasetConfig:
    return _make_cfg(h5_single)


# ---------------------------------------------------------------------------
# FMDatasetConfig — construction and defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_config_default_conditioning_keys(default_cfg: FMDatasetConfig) -> None:
    expected = ("power", "sigma", "rho", "cp", "k_s", "k_l", "k_p", "L", "T_s", "T_l", "x", "y")
    assert default_cfg.conditioning_keys == expected


@pytest.mark.unit
def test_config_frozen(default_cfg: FMDatasetConfig) -> None:
    with pytest.raises(Exception):
        default_cfg.T_ref = 9999.0  # type: ignore[misc]


@pytest.mark.unit
def test_config_default_t_ref(default_cfg: FMDatasetConfig) -> None:
    assert default_cfg.T_ref == 2000.0


@pytest.mark.unit
def test_config_default_t_ambient(default_cfg: FMDatasetConfig) -> None:
    assert default_cfg.T_ambient == 300.0


@pytest.mark.unit
def test_config_cond_means_cover_all_keys(default_cfg: FMDatasetConfig) -> None:
    for key in default_cfg.conditioning_keys:
        assert key in default_cfg.cond_means, f"cond_means missing key: {key}"


@pytest.mark.unit
def test_config_cond_stds_cover_all_keys(default_cfg: FMDatasetConfig) -> None:
    for key in default_cfg.conditioning_keys:
        assert key in default_cfg.cond_stds, f"cond_stds missing key: {key}"
        assert default_cfg.cond_stds[key] > 0.0, f"std for {key} must be positive"


# ---------------------------------------------------------------------------
# Dataset length
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_len_single_sample(h5_single: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_single))
    assert len(ds) == 1


@pytest.mark.unit
def test_len_multi_sample(h5_multi: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_multi))
    assert len(ds) == 5


@pytest.mark.unit
def test_len_empty_returns_zero(h5_empty: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_empty))
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# Return type and keys — Q is always present as the 3rd spatial channel
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_getitem_returns_dict(h5_single: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_single))
    assert isinstance(ds[0], dict)


@pytest.mark.unit
def test_getitem_required_keys(h5_single: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_single))
    item = ds[0]
    for key in ("T_in", "T_target", "mask", "Q", "conditioning"):
        assert key in item, f"Missing required key: {key}"


# ---------------------------------------------------------------------------
# Tensor dtypes
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("key", ["T_in", "T_target", "Q"])
def test_temperature_and_Q_fields_are_float32(h5_single: str, key: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_single))
    t = ds[0][key]
    assert isinstance(t, torch.Tensor)
    assert t.dtype == torch.float32, f"{key}: expected float32, got {t.dtype}"


@pytest.mark.unit
def test_mask_is_float32(h5_single: str) -> None:
    """Mask returned as float32 so it can be directly used as a model channel."""
    ds = FMThermalDataset(_make_cfg(h5_single))
    mask = ds[0]["mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.float32


@pytest.mark.unit
def test_conditioning_is_float32(h5_single: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_single))
    cond = ds[0]["conditioning"]
    assert isinstance(cond, torch.Tensor)
    assert cond.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tensor shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("key", ["T_in", "T_target", "mask", "Q"])
def test_field_shapes_match_stored(h5_single: str, key: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_single))
    t = ds[0][key]
    assert tuple(t.shape) == FIELD_SHAPE, (
        f"{key}: expected {FIELD_SHAPE}, got {tuple(t.shape)}"
    )


@pytest.mark.unit
def test_conditioning_vector_length(h5_single: str) -> None:
    cfg = _make_cfg(h5_single)
    ds = FMThermalDataset(cfg)
    cond = ds[0]["conditioning"]
    assert cond.shape == (len(cfg.conditioning_keys),), (
        f"Expected ({len(cfg.conditioning_keys)},), got {cond.shape}"
    )


# ---------------------------------------------------------------------------
# Normalization: temperature fields are O(1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_temperature_fields_are_normalized(h5_single: str) -> None:
    """T_in and T_target should be (T - T_ambient) / T_ref, values O(1)."""
    cfg = _make_cfg(h5_single)
    ds = FMThermalDataset(cfg)
    item = ds[0]
    for key in ("T_in", "T_target"):
        t = item[key]
        assert t.abs().max().item() < 5.0, (
            f"{key} max abs value {t.abs().max().item():.1f} suggests normalization missing"
        )


# ---------------------------------------------------------------------------
# Conditioning z-score normalization correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_conditioning_power_normalized(tmp_path: pytest.TempPathFactory) -> None:
    path = str(tmp_path / "norm_check.h5")
    power_val = 200.0
    attrs = dict(SAMPLE_ATTRS)
    attrs["power"] = power_val

    with h5py.File(path, "w") as f:
        grp = f.create_group("samples").create_group("sample_000000")
        _write_sample(grp, attrs=attrs, seed=0)

    cfg = _make_cfg(path)
    ds = FMThermalDataset(cfg)
    item = ds[0]
    cond = item["conditioning"]

    power_idx = list(cfg.conditioning_keys).index("power")
    expected = (power_val - cfg.cond_means["power"]) / cfg.cond_stds["power"]
    assert abs(cond[power_idx].item() - expected) < 1e-4, (
        f"power normalization failed: expected {expected:.4f}, got {cond[power_idx].item():.4f}"
    )


# ---------------------------------------------------------------------------
# Missing attribute raises a KeyError with key name in message
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_missing_attr_raises_key_error(tmp_path: pytest.TempPathFactory) -> None:
    path = str(tmp_path / "missing_attr.h5")
    incomplete_attrs = dict(SAMPLE_ATTRS)
    del incomplete_attrs["power"]

    with h5py.File(path, "w") as f:
        grp = f.create_group("samples").create_group("sample_000000")
        _write_sample(grp, attrs=incomplete_attrs, seed=0)

    ds = FMThermalDataset(_make_cfg(path))
    with pytest.raises(KeyError, match="power"):
        _ = ds[0]


# ---------------------------------------------------------------------------
# Multi-file aggregation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_file_aggregation(h5_single: str, h5_multi: str) -> None:
    ds = FMThermalDataset(FMDatasetConfig(h5_paths=[h5_single, h5_multi]))
    assert len(ds) == 6


@pytest.mark.unit
def test_multi_file_getitem_works(h5_single: str, h5_multi: str) -> None:
    ds = FMThermalDataset(FMDatasetConfig(h5_paths=[h5_single, h5_multi]))
    for i in range(len(ds)):
        item = ds[i]
        assert "T_in" in item


# ---------------------------------------------------------------------------
# Different samples produce different tensors
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_different_indices_return_different_data(h5_multi: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_multi))
    item0 = ds[0]
    item1 = ds[1]
    assert not torch.equal(item0["T_in"], item1["T_in"])


# ---------------------------------------------------------------------------
# No NaN / Inf
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("key", ["T_in", "T_target", "Q", "conditioning"])
def test_no_nan_inf(h5_multi: str, key: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_multi))
    for i in range(len(ds)):
        t = ds[i][key]
        assert not torch.isnan(t).any(), f"NaN in {key} at sample {i}"
        assert not torch.isinf(t).any(), f"Inf in {key} at sample {i}"


# ---------------------------------------------------------------------------
# Dataset is a torch Dataset subclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_torch_dataset_subclass(h5_single: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_single))
    assert isinstance(ds, Dataset)


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dataloader_batches_correctly(h5_multi: str) -> None:
    ds = FMThermalDataset(_make_cfg(h5_multi))
    loader = DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)
    batches = list(loader)
    assert len(batches) == 3
    first = batches[0]
    assert first["T_in"].shape[0] == 2
    assert first["conditioning"].shape == (2, len(_make_cfg(h5_multi).conditioning_keys))
    assert first["Q"].shape == (2, 1, 1, NZ, NY, NX)
