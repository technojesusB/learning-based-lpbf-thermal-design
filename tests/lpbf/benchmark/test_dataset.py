"""Tests for benchmark.dataset — build_test_dataset."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset

from neural_pbf.eval.benchmark.dataset import build_test_dataset


class _SyntheticBase(Dataset):
    """Minimal in-memory dataset that mimics FMThermalDataset output."""

    def __init__(self, n: int = 8, vol_size: int = 8) -> None:
        self.n = n
        self.vol_size = vol_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        shape = (1, self.vol_size, self.vol_size, self.vol_size)
        return {
            "T_in": torch.rand(*shape),
            "T_target": torch.rand(*shape),
            "mask": torch.zeros(*shape),
            "Q": torch.zeros(*shape),
            "conditioning": torch.zeros(12),
        }


@pytest.mark.unit
def test_build_test_dataset_net_returns_dataset():
    base = _SyntheticBase()
    ds = build_test_dataset(base, patch_size=4, model_type="net")
    assert isinstance(ds, Dataset)


@pytest.mark.unit
def test_build_test_dataset_rope_returns_dataset():
    base = _SyntheticBase()
    ds = build_test_dataset(base, patch_size=4, model_type="rope")
    assert isinstance(ds, Dataset)


@pytest.mark.unit
def test_build_test_dataset_rope_wraps_with_origin_class():
    """rope/triton must use PatchFMThermalDatasetWithOrigin (has patch_origin logic)."""
    from neural_pbf.data.patch_dataset import PatchFMThermalDatasetWithOrigin

    base = _SyntheticBase(n=4, vol_size=8)
    ds = build_test_dataset(base, patch_size=4, model_type="rope")
    assert isinstance(ds, PatchFMThermalDatasetWithOrigin)


@pytest.mark.unit
def test_build_test_dataset_net_wraps_with_standard_class():
    """net/dit must use PatchFMThermalDataset (no origin overhead)."""
    from neural_pbf.data.patch_dataset import PatchFMThermalDataset

    base = _SyntheticBase(n=4, vol_size=8)
    ds = build_test_dataset(base, patch_size=4, model_type="net")
    assert isinstance(ds, PatchFMThermalDataset)


@pytest.mark.unit
def test_build_test_dataset_triton_wraps_with_origin_class():
    from neural_pbf.data.patch_dataset import PatchFMThermalDatasetWithOrigin

    base = _SyntheticBase(n=4, vol_size=8)
    ds = build_test_dataset(base, patch_size=4, model_type="triton")
    assert isinstance(ds, PatchFMThermalDatasetWithOrigin)


@pytest.mark.unit
def test_build_test_dataset_patch_size_forwarded():
    """patch_size should be stored on the returned dataset."""
    base = _SyntheticBase(n=4, vol_size=8)
    ds = build_test_dataset(base, patch_size=32, model_type="net")
    assert ds.patch_size == 32


@pytest.mark.unit
def test_build_test_dataset_unknown_type_raises():
    base = _SyntheticBase()
    with pytest.raises(ValueError, match="model_type"):
        build_test_dataset(base, patch_size=4, model_type="unknown_arch")
