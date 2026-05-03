"""FMThermalDataset — PyTorch Dataset for Flow Matching surrogate training.

Reads T_in, T_target, Q, mask and per-sample attributes from the offline
HDF5 dataset produced by scripts/generate_offline_dataset.py.
Q is always returned as the 3rd spatial input channel for the FM model.
"""

from __future__ import annotations

from typing import Literal

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Default normalisation statistics (physics-derived reasonable priors)
# ---------------------------------------------------------------------------

_DEFAULT_COND_MEANS: dict[str, float] = {
    "power": 250.0,
    "sigma": 27.5e-6,
    "rho": 7900.0,
    "cp": 500.0,
    "k_s": 20.0,
    "k_l": 25.0,
    "k_p": 1.0,
    "L": 2.9e5,
    "T_s": 1658.0,
    "T_l": 1723.0,
    "x": 5.0e-4,
    "y": 2.5e-4,
}

_DEFAULT_COND_STDS: dict[str, float] = {
    "power": 60.0,
    "sigma": 8.0e-6,
    "rho": 500.0,
    "cp": 60.0,
    "k_s": 6.0,
    "k_l": 7.0,
    "k_p": 0.5,
    "L": 5.0e4,
    "T_s": 50.0,
    "T_l": 50.0,
    "x": 3.0e-4,
    "y": 1.5e-4,
}

_DEFAULT_CONDITIONING_KEYS: tuple[str, ...] = (
    "power",
    "sigma",
    "rho",
    "cp",
    "k_s",
    "k_l",
    "k_p",
    "L",
    "T_s",
    "T_l",
    "x",
    "y",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class FMDatasetConfig(BaseModel):
    """Frozen configuration for FMThermalDataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    h5_paths: list[str]

    conditioning_keys: tuple[str, ...] = Field(default=_DEFAULT_CONDITIONING_KEYS)

    cond_means: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_COND_MEANS)
    )
    cond_stds: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_COND_STDS)
    )

    @model_validator(mode="after")
    def _validate_cond_stds(self) -> FMDatasetConfig:
        for k, v in self.cond_stds.items():
            if v <= 0.0:
                raise ValueError(f"cond_stds['{k}'] must be > 0, got {v}")
        return self

    T_ref: float = Field(default=2000.0, gt=0.0)
    T_ambient: float = Field(default=300.0, ge=0.0)

    # Q_ref: normalisation scale for the volumetric heat source [W/m³]
    Q_ref: float = Field(default=1e12, gt=0.0)

    dtype: Literal["float32"] = "float32"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FMThermalDataset(Dataset):
    """Lazy-loading PyTorch Dataset for Flow Matching training.

    Returns per-sample dicts with keys:
        - T_in:         (1, 1, Nz, Ny, Nx) float32, normalised
        - T_target:     (1, 1, Nz, Ny, Nx) float32, normalised
        - Q:            (1, 1, Nz, Ny, Nx) float32, normalised
        - mask:         (1, 1, Nz, Ny, Nx) float32 in {0, 1}
        - conditioning: (D_cond,) float32, z-score normalised
    """

    def __init__(self, cfg: FMDatasetConfig) -> None:
        self.cfg = cfg
        self._keys: list[tuple[str, str]] = []

        for path in cfg.h5_paths:
            with h5py.File(path, "r") as f:
                if "samples" in f:
                    for key in f["samples"]:
                        self._keys.append((path, key))

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        path, sample_key = self._keys[idx]
        cfg = self.cfg

        with h5py.File(path, "r") as f:
            grp = f["samples"][sample_key]

            T_in = torch.from_numpy(grp["T_in"][:].astype(np.float32))
            T_target = torch.from_numpy(grp["T_target"][:].astype(np.float32))
            Q = torch.from_numpy(grp["Q"][:].astype(np.float32))
            mask = torch.from_numpy(grp["mask"][:].astype(np.float32))

            # Build conditioning vector — raises KeyError with key name if missing
            cond_values: list[float] = []
            for key in cfg.conditioning_keys:
                if key not in grp.attrs:
                    raise KeyError(
                        f"Conditioning attribute '{key}' missing in sample "
                        f"'{sample_key}' of '{path}'"
                    )
                cond_values.append(float(grp.attrs[key]))

        # Normalise temperature fields: (T - T_ambient) / T_ref → O(1)
        T_in = (T_in - cfg.T_ambient) / cfg.T_ref
        T_target = (T_target - cfg.T_ambient) / cfg.T_ref

        # Normalise heat source: Q / Q_ref → O(1)
        Q = Q / cfg.Q_ref

        # Z-score normalise conditioning vector
        cond = torch.tensor(cond_values, dtype=torch.float32)
        means = torch.tensor(
            [cfg.cond_means[k] for k in cfg.conditioning_keys], dtype=torch.float32
        )
        stds = torch.tensor(
            [cfg.cond_stds[k] for k in cfg.conditioning_keys], dtype=torch.float32
        )
        cond = (cond - means) / stds

        return {
            "T_in": T_in,
            "T_target": T_target,
            "Q": Q,
            "mask": mask,
            "conditioning": cond,
        }
