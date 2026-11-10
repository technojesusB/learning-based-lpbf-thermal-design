"""Adds each sample's material to training items, for the physics loss.

The generator randomizes the material once per run (``data.sampling.randomize_material``) and stores the result
with every sample. The physics loss needs that exact material, including the k and cp tables, and the exact
``exposure_time``; the conditioning vector only carries z-scored scalars. The vector layout is in
``physics.material_vector``.

The wrapper sits outside the hashed dataset files (``fm_dataset.py``, ``patch_dataset.py``) so the UQ pipeline's
S4 stays valid. It finds each item's ``(path, sample_key)`` through ``FMThermalDataset._keys``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from neural_pbf.physics.material_vector import LUTS, SCALARS, encode


def _unwrap(ds: Any, idx: int) -> tuple[Any, int]:
    """The FMThermalDataset under Subsets and patch wrappers, and the index of item ``idx`` in it."""
    while True:
        if isinstance(ds, Subset):
            idx, ds = int(ds.indices[idx]), ds.dataset
        elif hasattr(ds, "base_ds"):  # PatchFMThermalDataset keeps the index
            ds = ds.base_ds
        else:
            return ds, idx


def resolve_key(ds: Any, idx: int) -> tuple[str, str]:
    """(path, sample_key) of item ``idx``, through Subsets and the patch wrappers down to FMThermalDataset."""
    base, idx = _unwrap(ds, idx)
    return base._keys[idx]


def _read_run(path: str) -> Tensor:
    """The run's material vector; every sample in the run must store the same values."""
    with h5py.File(path, "r") as f:
        rows = []
        for grp in f["samples"].values():  # type: ignore[union-attr]
            scalars = {k: float(grp.attrs[k]) for k in SCALARS}
            luts = {name: np.asarray(grp[name][()], dtype=np.float64).tolist() for name in LUTS}
            rows.append((scalars, luts))
    if not rows:
        raise ValueError(f"{path}: run has no samples")
    if any(r != rows[0] for r in rows[1:]):
        raise ValueError(f"{path}: material or exposure_time differ between samples of one run")
    try:
        return encode(*rows[0])
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _material_name(path: str) -> str:
    with h5py.File(path, "r") as f:
        return str(next(iter(f["samples"].values())).attrs["mat_name"])  # type: ignore[union-attr]


class MaterialDataset(Dataset):
    """Wraps a training dataset and adds ``material``: a (MATERIAL_DIM,) float64 vector, read once per run.

    With ``t_boil`` (material name to boiling point [K]) every item also gets ``T_boil``, for the regime split of
    the val metrics. Refuses a dataset built on the post-step mask: the physics loss reads the batch mask as mask_in.
    """

    def __init__(self, base: Any, t_boil: Mapping[str, float] | None = None) -> None:
        if len(base):
            fm, _ = _unwrap(base, 0)
            if fm.cfg.mask_source != "pre":
                raise ValueError(f"the physics loss needs mask_source='pre' (mask_in), got {fm.cfg.mask_source!r}")
        self.base = base
        self.t_boil = t_boil
        self._cache: dict[str, dict[str, Tensor]] = {}

    def __len__(self) -> int:
        return len(self.base)

    def _run(self, path: str) -> dict[str, Tensor]:
        if path not in self._cache:
            extra = {"material": _read_run(path)}
            if self.t_boil is not None:
                name = _material_name(path)
                if name not in self.t_boil:
                    raise KeyError(f"{path}: no boiling point configured for material {name!r}")
                extra["T_boil"] = torch.tensor(float(self.t_boil[name]), dtype=torch.float64)
            self._cache[path] = extra
        return self._cache[path]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, _ = resolve_key(self.base, idx)
        return {**self.base[idx], **self._run(path)}
