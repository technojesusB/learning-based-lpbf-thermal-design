"""Integration test for test_sample_mapping traceability logic."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset, Subset, random_split


class _FakeDataset(Dataset):
    """Minimal stand-in for FMThermalDataset."""

    def __init__(self, n: int) -> None:
        self._keys: list[tuple[str, str]] = [
            (f"/data/run_{i}.h5", f"sample_{i:04d}") for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> dict[str, str]:
        h5_path, sample_key = self._keys[idx]
        return {"h5_path": h5_path, "sample_key": sample_key}


@pytest.mark.integration
def test_subset_indices_round_trip() -> None:
    """Verify that Subset.indices correctly maps local test indices to _keys."""
    n_total = 20
    full_ds = _FakeDataset(n_total)

    n_train = 14
    n_val = 3
    n_test = n_total - n_train - n_val

    _, _, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0),
    )

    test_subset_indices: list[int] = list(test_ds.indices)  # type: ignore[attr-defined]
    assert len(test_subset_indices) == n_test

    for local_idx in range(n_test):
        original_idx = test_subset_indices[local_idx]
        h5_path_expected, sample_key_expected = full_ds._keys[original_idx]

        # Simulate what the training script does
        retrieved_sample = full_ds[original_idx]
        assert retrieved_sample["h5_path"] == h5_path_expected
        assert retrieved_sample["sample_key"] == sample_key_expected


@pytest.mark.integration
def test_subset_indices_no_overlap() -> None:
    """Train, val, test splits should have non-overlapping indices."""
    n_total = 30
    full_ds = _FakeDataset(n_total)
    n_train, n_val = 21, 6
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_idx = set(train_ds.indices)  # type: ignore[attr-defined]
    val_idx = set(val_ds.indices)  # type: ignore[attr-defined]
    test_idx = set(test_ds.indices)  # type: ignore[attr-defined]

    assert len(train_idx & val_idx) == 0
    assert len(train_idx & test_idx) == 0
    assert len(val_idx & test_idx) == 0
    assert len(train_idx | val_idx | test_idx) == n_total


@pytest.mark.integration
def test_mapping_covers_all_test_samples() -> None:
    """Every test sample must appear exactly once in the mapping."""
    n_total = 15
    full_ds = _FakeDataset(n_total)
    n_train, n_val = 10, 3
    n_test = n_total - n_train - n_val

    _, _, test_ds = random_split(
        full_ds,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(7),
    )

    test_subset_indices: list[int] = list(test_ds.indices)  # type: ignore[attr-defined]
    seen_original_indices: set[int] = set()

    for local_idx in range(n_test):
        original_idx = test_subset_indices[local_idx]
        assert original_idx not in seen_original_indices, "Duplicate in mapping"
        seen_original_indices.add(original_idx)

    assert len(seen_original_indices) == n_test
