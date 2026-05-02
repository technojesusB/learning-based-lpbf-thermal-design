"""Tests for HDF5ThermalDataset — written TDD-first against the implementation.

The fixture creates a minimal, synthetically-generated HDF5 file that mirrors
the schema the real offline dataset generator produces:
  - f['samples']['0000'] ...
    - 'T_in'     fp16  (Nz, Ny, Nx)
    - 'Q'        fp16  (Nz, Ny, Nx)
    - 'T_target' fp16  (Nz, Ny, Nx)
    - 'T_lf'     fp16  (Nz, Ny, Nx)
    - 'mask'     uint8 (Nz, Ny, Nx)
    - 'scalars'  fp32  [t, laser_x, laser_y]

No mocking of HDF5 internals; we write real files via h5py.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from neural_pbf.data.hdf5_dataset import HDF5ThermalDataset

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIELD_SHAPE = (4, 8, 8)   # (Nz, Ny, Nx) — small for speed


def _write_sample(group: h5py.Group, shape: tuple[int, ...], seed: int = 0) -> None:
    """Write one synthetic sample into an h5py group."""
    rng = np.random.default_rng(seed)
    # Fields stored as fp16 (as the real generator does)
    for key in ("T_in", "Q", "T_target", "T_lf"):
        group.create_dataset(key, data=rng.random(shape).astype(np.float16))
    # Mask stored as uint8
    group.create_dataset(
        "mask",
        data=rng.integers(0, 2, size=shape, dtype=np.uint8),
    )
    # Scalars: [t, laser_x, laser_y]
    group.create_dataset(
        "scalars",
        data=np.array([0.001, 0.0005, 0.0003], dtype=np.float32),
    )


@pytest.fixture()
def h5_single(tmp_path: pytest.TempPathFactory) -> str:
    """HDF5 file with exactly one sample."""
    path = str(tmp_path / "single.h5")
    with h5py.File(path, "w") as f:
        samples = f.create_group("samples")
        _write_sample(samples.create_group("0000"), FIELD_SHAPE, seed=42)
    return path


@pytest.fixture()
def h5_multi(tmp_path: pytest.TempPathFactory) -> str:
    """HDF5 file with five samples."""
    path = str(tmp_path / "multi.h5")
    with h5py.File(path, "w") as f:
        samples = f.create_group("samples")
        for i in range(5):
            _write_sample(samples.create_group(f"{i:04d}"), FIELD_SHAPE, seed=i)
    return path


@pytest.fixture()
def h5_no_samples_group(tmp_path: pytest.TempPathFactory) -> str:
    """HDF5 file that has no 'samples' group at all."""
    path = str(tmp_path / "empty.h5")
    with h5py.File(path, "w") as f:
        f.create_group("metadata")   # unrelated group — samples absent
    return path


@pytest.fixture()
def h5_empty_samples(tmp_path: pytest.TempPathFactory) -> str:
    """HDF5 file whose 'samples' group exists but contains zero sample groups."""
    path = str(tmp_path / "empty_samples.h5")
    with h5py.File(path, "w") as f:
        f.create_group("samples")
    return path


# ---------------------------------------------------------------------------
# __len__ tests
# ---------------------------------------------------------------------------


def test_len_single_sample(h5_single: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    assert len(ds) == 1


def test_len_multi_sample(h5_multi: str) -> None:
    ds = HDF5ThermalDataset(h5_multi)
    assert len(ds) == 5


def test_len_no_samples_group_returns_zero(h5_no_samples_group: str) -> None:
    ds = HDF5ThermalDataset(h5_no_samples_group)
    assert len(ds) == 0


def test_len_empty_samples_group_returns_zero(h5_empty_samples: str) -> None:
    ds = HDF5ThermalDataset(h5_empty_samples)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# __getitem__ return structure
# ---------------------------------------------------------------------------


def test_getitem_returns_dict(h5_single: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    assert isinstance(item, dict)


def test_getitem_required_keys_present(h5_single: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    required = {"T_in", "Q", "T_target", "T_lf", "mask", "t", "laser_x", "laser_y"}
    assert required <= item.keys(), (
        f"Missing keys: {required - item.keys()}"
    )


def test_getitem_no_extra_unexpected_keys(h5_single: str) -> None:
    """Dataset should not return stray keys beyond the documented eight."""
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    allowed = {"T_in", "Q", "T_target", "T_lf", "mask", "t", "laser_x", "laser_y"}
    extra = item.keys() - allowed
    assert not extra, f"Unexpected extra keys returned: {extra}"


# ---------------------------------------------------------------------------
# Tensor dtype contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["T_in", "Q", "T_target", "T_lf"])
def test_field_tensors_are_float32(h5_single: str, key: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    t = item[key]
    assert isinstance(t, torch.Tensor), f"{key} is not a torch.Tensor"
    assert t.dtype == torch.float32, (
        f"{key}: expected float32, got {t.dtype}"
    )


def test_mask_tensor_is_uint8(h5_single: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    mask = item["mask"]
    assert isinstance(mask, torch.Tensor), "mask is not a torch.Tensor"
    assert mask.dtype == torch.uint8, f"mask: expected uint8, got {mask.dtype}"


@pytest.mark.parametrize("key", ["t", "laser_x", "laser_y"])
def test_scalar_tensors_are_float32(h5_single: str, key: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    val = item[key]
    assert isinstance(val, torch.Tensor), f"{key} is not a torch.Tensor"
    assert val.dtype == torch.float32, (
        f"{key}: expected float32, got {val.dtype}"
    )


# ---------------------------------------------------------------------------
# Tensor shape contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["T_in", "Q", "T_target", "T_lf", "mask"])
def test_field_tensor_shapes_match_stored_shape(h5_single: str, key: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    assert tuple(item[key].shape) == FIELD_SHAPE, (
        f"{key}: expected shape {FIELD_SHAPE}, got {tuple(item[key].shape)}"
    )


@pytest.mark.parametrize("key", ["t", "laser_x", "laser_y"])
def test_scalar_tensors_are_0d_or_1d(h5_single: str, key: str) -> None:
    """Scalars should be 0-D or 1-D tensors (single value)."""
    ds = HDF5ThermalDataset(h5_single)
    item = ds[0]
    val = item[key]
    assert val.numel() == 1, f"{key}: expected 1 element, got {val.numel()}"


# ---------------------------------------------------------------------------
# Scalar value correctness
# ---------------------------------------------------------------------------


def test_scalar_values_match_stored_data(tmp_path: pytest.TempPathFactory) -> None:
    """The t, laser_x, laser_y scalars must match what was written to disk."""
    path = str(tmp_path / "scalar_check.h5")
    t_val, lx_val, ly_val = 0.0042, 0.0012, 0.0034

    with h5py.File(path, "w") as f:
        samples = f.create_group("samples")
        grp = samples.create_group("0000")
        shape = (2, 4, 4)
        rng = np.random.default_rng(0)
        for key in ("T_in", "Q", "T_target", "T_lf"):
            grp.create_dataset(key, data=rng.random(shape).astype(np.float16))
        grp.create_dataset("mask", data=rng.integers(0, 2, size=shape, dtype=np.uint8))
        grp.create_dataset(
            "scalars",
            data=np.array([t_val, lx_val, ly_val], dtype=np.float32),
        )

    ds = HDF5ThermalDataset(path)
    item = ds[0]

    assert abs(item["t"].item() - t_val) < 1e-6, (
        f"t mismatch: expected {t_val}, got {item['t'].item()}"
    )
    assert abs(item["laser_x"].item() - lx_val) < 1e-6, (
        f"laser_x mismatch: expected {lx_val}, got {item['laser_x'].item()}"
    )
    assert abs(item["laser_y"].item() - ly_val) < 1e-6, (
        f"laser_y mismatch: expected {ly_val}, got {item['laser_y'].item()}"
    )


# ---------------------------------------------------------------------------
# fp16 → fp32 conversion: no precision loss beyond expected fp16 range
# ---------------------------------------------------------------------------


def test_float16_source_upcast_to_float32(tmp_path: pytest.TempPathFactory) -> None:
    """Verify fp16 values survive the upcast without distortion beyond fp16 precision."""
    path = str(tmp_path / "fp16_check.h5")
    shape = (2, 4, 4)

    original = np.array([300.0, 1200.5, 3500.0], dtype=np.float16).reshape(1, 1, 3)
    padded = np.zeros(shape, dtype=np.float16)
    padded[0, 0, :3] = original[0, 0, :]

    with h5py.File(path, "w") as f:
        samples = f.create_group("samples")
        grp = samples.create_group("0000")
        grp.create_dataset("T_in", data=padded)
        rng = np.random.default_rng(0)
        for key in ("Q", "T_target", "T_lf"):
            grp.create_dataset(key, data=rng.random(shape).astype(np.float16))
        grp.create_dataset("mask", data=np.zeros(shape, dtype=np.uint8))
        grp.create_dataset(
            "scalars",
            data=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )

    ds = HDF5ThermalDataset(path)
    item = ds[0]
    loaded_slice = item["T_in"][0, 0, :3].numpy()
    expected = padded[0, 0, :3].astype(np.float32)

    np.testing.assert_allclose(loaded_slice, expected, rtol=1e-3), (
        "fp16 → float32 upcast corrupted values beyond expected tolerance"
    )


# ---------------------------------------------------------------------------
# Independent file access per __getitem__ (thread-safety contract)
# ---------------------------------------------------------------------------


def test_multiple_getitem_calls_are_independent(h5_multi: str) -> None:
    """Repeated __getitem__ calls must each open/close the file independently."""
    ds = HDF5ThermalDataset(h5_multi)
    items = [ds[i] for i in range(5)]
    # Each item should be a separate dict with its own tensor data
    for i, item in enumerate(items):
        assert isinstance(item, dict), f"Item {i} is not a dict"
        assert "T_in" in item


def test_getitem_indices_return_different_data(h5_multi: str) -> None:
    """Different sample indices should return different field tensors."""
    ds = HDF5ThermalDataset(h5_multi)
    item_0 = ds[0]
    item_1 = ds[1]

    # The two samples were seeded differently so their T_in fields must differ
    assert not torch.equal(item_0["T_in"], item_1["T_in"]), (
        "Samples 0 and 1 returned identical T_in tensors — index lookup may be broken"
    )


# ---------------------------------------------------------------------------
# No NaN / Inf in loaded tensors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["T_in", "Q", "T_target", "T_lf"])
def test_no_nan_in_field_tensors(h5_multi: str, key: str) -> None:
    ds = HDF5ThermalDataset(h5_multi)
    for i in range(len(ds)):
        t = ds[i][key]
        assert not torch.isnan(t).any(), f"NaN in {key} at sample {i}"
        assert not torch.isinf(t).any(), f"Inf in {key} at sample {i}"


# ---------------------------------------------------------------------------
# Dataset is a torch Dataset subclass
# ---------------------------------------------------------------------------


def test_is_torch_dataset_subclass(h5_single: str) -> None:
    from torch.utils.data import Dataset

    ds = HDF5ThermalDataset(h5_single)
    assert isinstance(ds, Dataset), "HDF5ThermalDataset must subclass torch Dataset"


# ---------------------------------------------------------------------------
# DataLoader compatibility (basic smoke test)
# ---------------------------------------------------------------------------


def test_dataloader_single_worker(h5_multi: str) -> None:
    """DataLoader with num_workers=0 must iterate without error."""
    from torch.utils.data import DataLoader

    ds = HDF5ThermalDataset(h5_multi)
    loader = DataLoader(ds, batch_size=2, num_workers=0, shuffle=False)

    batches = list(loader)
    assert len(batches) == 3, f"Expected 3 batches (5 samples, bs=2), got {len(batches)}"

    first = batches[0]
    assert first["T_in"].shape[0] == 2, "First batch should have 2 samples"
    assert first["T_in"].dtype == torch.float32


# ---------------------------------------------------------------------------
# Mask values are binary (0 or 1)
# ---------------------------------------------------------------------------


def test_mask_values_are_binary(h5_multi: str) -> None:
    ds = HDF5ThermalDataset(h5_multi)
    for i in range(len(ds)):
        mask = ds[i]["mask"]
        unique_vals = torch.unique(mask).tolist()
        for v in unique_vals:
            assert v in (0, 1), (
                f"Sample {i}: mask contains non-binary value {v}"
            )


# ---------------------------------------------------------------------------
# h5_path attribute is stored on the dataset
# ---------------------------------------------------------------------------


def test_h5_path_attribute_stored(h5_single: str) -> None:
    ds = HDF5ThermalDataset(h5_single)
    assert hasattr(ds, "h5_path"), "Dataset should expose h5_path attribute"
    assert ds.h5_path == h5_single


# ---------------------------------------------------------------------------
# Multi-file list constructor (covers else-branch in __init__)
# ---------------------------------------------------------------------------


def test_list_constructor_aggregates_samples(
    h5_single: str, h5_multi: str
) -> None:
    """Passing a list of paths should aggregate all samples across files."""
    ds = HDF5ThermalDataset([h5_single, h5_multi])
    # h5_single has 1 sample, h5_multi has 5 → total 6
    assert len(ds) == 6, f"Expected 6 aggregated samples, got {len(ds)}"


def test_list_constructor_h5_path_set_to_first_path(
    h5_single: str, h5_multi: str
) -> None:
    """When constructed with a list, h5_path should be set to the first element."""
    ds = HDF5ThermalDataset([h5_single, h5_multi])
    assert ds.h5_path == h5_single, (
        f"h5_path expected to be first list element, got {ds.h5_path!r}"
    )


def test_list_constructor_getitem_works_across_files(
    h5_single: str, h5_multi: str
) -> None:
    """Items from both files must be individually accessible via __getitem__."""
    ds = HDF5ThermalDataset([h5_single, h5_multi])
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "T_in" in item
