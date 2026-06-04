"""Dataset factory for the benchmark pipeline."""

from __future__ import annotations

from torch.utils.data import Dataset

from neural_pbf.eval.benchmark.constants import PATCH_SIZE


def build_test_dataset(
    base_ds: Dataset,
    patch_size: int = PATCH_SIZE,
    model_type: str = "net",
) -> Dataset:
    """Wrap *base_ds* in the correct patch dataset for *model_type*.

    - ``"net"`` / ``"dit"`` → ``PatchFMThermalDataset``
    - ``"rope"`` / ``"triton"`` → ``PatchFMThermalDatasetWithOrigin``

    Raises:
        ValueError: Unknown model_type.
    """
    if model_type in ("rope", "triton"):
        from neural_pbf.data.patch_dataset import PatchFMThermalDatasetWithOrigin

        return PatchFMThermalDatasetWithOrigin(base_ds, patch_size=patch_size)  # type: ignore[arg-type]

    if model_type in ("net", "dit"):
        from neural_pbf.data.patch_dataset import PatchFMThermalDataset

        return PatchFMThermalDataset(base_ds, patch_size=patch_size)  # type: ignore[arg-type]

    raise ValueError(f"Unknown model_type {model_type!r}. Supported: 'net', 'dit', 'rope', 'triton'.")
