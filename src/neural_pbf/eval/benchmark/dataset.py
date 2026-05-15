"""Dataset factory for the benchmark pipeline.

Wraps a base FMThermalDataset into the appropriate patch-based dataset
depending on model_type, so the caller doesn't need to branch.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from neural_pbf.eval.benchmark.constants import PATCH_SIZE


def build_test_dataset(
    base_ds: Dataset,
    patch_size: int = PATCH_SIZE,
    model_type: str = "net",
) -> Dataset:
    """Wrap *base_ds* in the correct patch dataset for *model_type*.

    - ``"net"`` / ``"dit"`` → ``PatchFMThermalDataset`` (standard patches)
    - ``"rope"`` / ``"triton"`` → ``PatchFMThermalDatasetWithOrigin``
      (same patches but also exposes ``patch_origin`` for 3D-RoPE)

    Args:
        base_ds:    An already-instantiated base dataset (e.g. FMThermalDataset).
        patch_size: Spatial edge length of each patch (default: PATCH_SIZE = 64).
        model_type: Architecture key — controls which wrapper is used.

    Returns:
        Wrapped patch dataset.

    Raises:
        ValueError: Unknown model_type.
    """
    if model_type in ("rope", "triton"):
        from experiments.train_fm_dit_rope import PatchFMThermalDatasetWithOrigin

        return PatchFMThermalDatasetWithOrigin(base_ds, patch_size=patch_size)  # type: ignore[arg-type]

    if model_type in ("net", "dit"):
        from experiments.train_fm_patches import PatchFMThermalDataset

        return PatchFMThermalDataset(base_ds, patch_size=patch_size)  # type: ignore[arg-type]

    raise ValueError(
        f"Unknown model_type {model_type!r}. Supported: 'net', 'dit', 'rope', 'triton'."
    )
