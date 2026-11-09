"""Adds each sample's material to training items, for the physics loss.

The generator randomizes the material once per run (``data.sampling.randomize_material``) and stores the result
with every sample. The physics loss needs that exact material, including the k and cp tables, and the exact
``exposure_time``; the conditioning vector only carries z-scored scalars.

The wrapper sits outside the hashed dataset files (``fm_dataset.py``, ``patch_dataset.py``) so the UQ pipeline's
S4 stays valid. It finds each item's ``(path, sample_key)`` through ``FMThermalDataset._keys``.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from neural_pbf.physics.material import MaterialConfig

SCALARS = ("rho", "L", "T_s", "T_l", "k_p", "k_s", "k_l", "cp", "exposure_time")
LUTS = ("T_lut", "k_lut", "cp_lut")
MAX_LUT = 8
_N_LUT = len(SCALARS)
_LUT0 = _N_LUT + 1
MATERIAL_DIM = _LUT0 + len(LUTS) * MAX_LUT


def resolve_key(ds: Any, idx: int) -> tuple[str, str]:
    """(path, sample_key) of item ``idx``, through Subsets and the patch wrappers down to FMThermalDataset."""
    while True:
        if isinstance(ds, Subset):
            idx, ds = int(ds.indices[idx]), ds.dataset
        elif hasattr(ds, "base_ds"):  # PatchFMThermalDataset keeps the index
            ds = ds.base_ds
        else:
            return ds._keys[idx]


def _read_run(path: str) -> Tensor:
    """The run's material vector; every sample in the run must store the same values."""
    with h5py.File(path, "r") as f:
        rows = []
        for grp in f["samples"].values():  # type: ignore[union-attr]
            luts = [np.asarray(grp[name][()], dtype=np.float64) for name in LUTS]
            rows.append(np.concatenate([[float(grp.attrs[k]) for k in SCALARS], *luts]))
    first = rows[0]
    if any(r.shape != first.shape or not np.array_equal(r, first) for r in rows[1:]):
        raise ValueError(f"{path}: material or exposure_time differ between samples of one run")
    n = (first.size - len(SCALARS)) // len(LUTS)
    if not 2 <= n <= MAX_LUT:
        raise ValueError(f"{path}: LUT length {n} outside [2, {MAX_LUT}]")
    vec = np.zeros(MATERIAL_DIM)
    vec[: len(SCALARS)] = first[: len(SCALARS)]
    vec[_N_LUT] = n
    for i in range(len(LUTS)):
        vec[_LUT0 + i * MAX_LUT : _LUT0 + i * MAX_LUT + n] = first[len(SCALARS) + i * n : len(SCALARS) + (i + 1) * n]
    return torch.from_numpy(vec)


class MaterialDataset(Dataset):
    """Wraps a training dataset and adds ``material``: a (MATERIAL_DIM,) float64 vector, read once per run."""

    def __init__(self, base: Any) -> None:
        self.base = base
        self._cache: dict[str, Tensor] = {}

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, _ = resolve_key(self.base, idx)
        if path not in self._cache:
            self._cache[path] = _read_run(path)
        return {**self.base[idx], "material": self._cache[path]}


def exposure_time_of(vec: Tensor) -> float:
    return float(vec[SCALARS.index("exposure_time")])


def material_from_vector(vec: Tensor) -> MaterialConfig:
    """The generator's MaterialConfig for this run (presets set only these fields; the rest keep defaults)."""
    s = dict(zip(SCALARS, (float(v) for v in vec[: len(SCALARS)]), strict=True))
    n = int(vec[_N_LUT])
    T_lut, k_lut, cp_lut = ([float(v) for v in vec[_LUT0 + i * MAX_LUT : _LUT0 + i * MAX_LUT + n]] for i in range(3))
    return MaterialConfig(
        k_powder=s["k_p"], k_solid=s["k_s"], k_liquid=s["k_l"], cp_base=s["cp"], rho=s["rho"],
        T_solidus=s["T_s"], T_liquidus=s["T_l"], latent_heat_L=s["L"], use_T_dep=True, use_lut=True,
        T_lut=T_lut, k_lut=k_lut, cp_lut=cp_lut,
    )  # fmt: skip
