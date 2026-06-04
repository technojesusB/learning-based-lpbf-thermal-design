"""Patch-based datasets and collation utilities for FM surrogate training."""

from __future__ import annotations

from typing import Any, TypedDict

import h5py
import torch
from torch.utils.data import Dataset

from neural_pbf.data.fm_dataset import FMThermalDataset


class GridAttrs(TypedDict):
    """Spatial grid attributes read from HDF5 dataset files."""

    Lx_m: float
    Ly_m: float
    Lz_m: float
    Nx: float  # integer count stored as float for dict homogeneity
    Ny: float
    Nz: float
    dx_m: float
    dy_m: float
    dz_m: float


def read_grid_attrs(h5_path: str) -> GridAttrs:
    """Read spatial grid attributes from an HDF5 file.

    Returns a dict with keys: Lx_m, Ly_m, Lz_m, Nx, Ny, Nz, dx_m, dy_m, dz_m.
    When Lz_m is absent, assumes isotropic voxels (dz_m = dx_m).

    Raises:
        ValueError: if required attributes (Lx_m, Ly_m, Nx, Ny) are missing.
    """
    with h5py.File(h5_path, "r") as f:
        try:
            Lx_m = float(f.attrs["Lx_m"])  # type: ignore[arg-type]
            Ly_m = float(f.attrs["Ly_m"])  # type: ignore[arg-type]
            Nx = int(f.attrs["Nx"])  # type: ignore[arg-type]
            Ny = int(f.attrs["Ny"])  # type: ignore[arg-type]
        except KeyError as exc:
            raise ValueError(
                f"HDF5 file '{h5_path}' missing required root attribute: {exc}. Expected: Lx_m, Ly_m, Nx, Ny."
            ) from exc

        dx_m = Lx_m / max(Nx - 1, 1)
        dy_m = Ly_m / max(Ny - 1, 1)

        if "Nz" in f.attrs:
            Nz = int(f.attrs["Nz"])  # type: ignore[arg-type]
        else:
            samples_grp = f.get("samples", {})
            first_key = next(iter(samples_grp), None)  # type: ignore[call-overload]
            if first_key is None:
                raise ValueError(f"HDF5 '{h5_path}' has no 'Nz' attribute and no samples to infer from.")
            Nz = int(f["samples"][first_key]["T_in"].shape[-3])  # type: ignore[index]

        if "Lz_m" in f.attrs:
            Lz_m = float(f.attrs["Lz_m"])  # type: ignore[arg-type]
            dz_m = Lz_m / max(Nz - 1, 1)
        else:
            dz_m = dx_m
            Lz_m = dz_m * (Nz - 1)

    return GridAttrs(
        Lx_m=Lx_m,
        Ly_m=Ly_m,
        Lz_m=Lz_m,
        Nx=float(Nx),
        Ny=float(Ny),
        Nz=float(Nz),
        dx_m=dx_m,
        dy_m=dy_m,
        dz_m=dz_m,
    )


def _collate_with_strings(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate but preserve string fields as plain lists."""
    from torch.utils.data import default_collate

    string_keys = {k for k, v in batch[0].items() if isinstance(v, str)}
    result: dict[str, Any] = default_collate(  # type: ignore[assignment]
        [{k: v for k, v in sample.items() if k not in string_keys} for sample in batch]
    )
    for k in string_keys:
        result[k] = [sample[k] for sample in batch]
    return result


class PatchFMThermalDataset(Dataset):
    """Wraps FMThermalDataset to return 64×64×64 patches centred on the laser spot.

    Extracts the laser position from HDF5 sample attributes and crops the
    spatial dimensions of T_in, T_target, Q, and mask around that hot-spot.

    Args:
        base_ds:    FMThermalDataset (or compatible) to wrap.
        patch_size: Side length of the cubic patch in voxels (default: 64).
    """

    def __init__(self, base_ds: FMThermalDataset, patch_size: int = 64) -> None:
        self.base_ds = base_ds
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from torch.utils.data import Subset

        sample = self.base_ds[idx]
        if isinstance(self.base_ds, Subset):
            actual_ds = self.base_ds.dataset
            actual_idx = self.base_ds.indices[idx]
            path, sample_key = actual_ds._keys[actual_idx]  # type: ignore[union-attr]
        else:
            path, sample_key = self.base_ds._keys[idx]  # type: ignore[union-attr]

        with h5py.File(path, "r") as f:
            Lx: float = float(f.attrs["Lx_m"])  # type: ignore[arg-type]
            Ly: float = float(f.attrs["Ly_m"])  # type: ignore[arg-type]
            Nx: int = int(f.attrs["Nx"])  # type: ignore[arg-type]
            Ny: int = int(f.attrs["Ny"])  # type: ignore[arg-type]
            dx = Lx / max(Nx - 1, 1)
            dy = Ly / max(Ny - 1, 1)
            grp = f["samples"][sample_key]  # type: ignore[index]
            x0: float = float(grp.attrs["x"])  # type: ignore[arg-type]
            y0: float = float(grp.attrs["y"])  # type: ignore[arg-type]

        ix = int(round(x0 / dx))
        iy = int(round(y0 / dy))
        half = self.patch_size // 2
        x_start = max(0, min(Nx - self.patch_size, ix - half))
        y_start = max(0, min(Ny - self.patch_size, iy - half))

        ps = self.patch_size
        patched = {
            key: sample[key][:, :, :, y_start : y_start + ps, x_start : x_start + ps]
            for key in ["T_in", "T_target", "Q", "mask"]
        }
        return {**{k: v for k, v in sample.items() if k not in patched}, **patched}


class PatchFMThermalDatasetWithOrigin(PatchFMThermalDataset):
    """Extends PatchFMThermalDataset to expose the patch origin for 3D-RoPE.

    Adds to the returned dict:
        patch_origin: (3,) long tensor [z_start=0, y_start, x_start] in voxels.
        sample_key:   HDF5 group key string.
        h5_path:      Path to the HDF5 file string.
    """

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from torch.utils.data import Subset

        sample = self.base_ds[idx]
        if isinstance(self.base_ds, Subset):
            actual_ds = self.base_ds.dataset
            actual_idx = self.base_ds.indices[idx]
            path, sample_key = actual_ds._keys[actual_idx]  # type: ignore[union-attr]
        else:
            path, sample_key = self.base_ds._keys[idx]  # type: ignore[union-attr]

        with h5py.File(path, "r") as f:
            try:
                Lx_m = float(f.attrs["Lx_m"])  # type: ignore[arg-type]
                Ly_m = float(f.attrs["Ly_m"])  # type: ignore[arg-type]
                Nx = int(f.attrs["Nx"])  # type: ignore[arg-type]
                Ny = int(f.attrs["Ny"])  # type: ignore[arg-type]
            except KeyError as exc:
                raise ValueError(f"HDF5 '{path}' missing required root attribute: {exc}") from exc
            dx = Lx_m / max(Nx - 1, 1)
            dy = Ly_m / max(Ny - 1, 1)
            grp = f["samples"][sample_key]  # type: ignore[index]
            try:
                x0 = float(grp.attrs["x"])  # type: ignore[arg-type]
                y0 = float(grp.attrs["y"])  # type: ignore[arg-type]
            except KeyError as exc:
                raise ValueError(f"Sample '{sample_key}' in '{path}' missing laser-position attribute: {exc}") from exc

        ix = int(round(x0 / dx))
        iy = int(round(y0 / dy))
        half = self.patch_size // 2
        x_start = max(0, min(Nx - self.patch_size, ix - half))
        y_start = max(0, min(Ny - self.patch_size, iy - half))

        ps = self.patch_size
        patched = {
            key: sample[key][:, :, :, y_start : y_start + ps, x_start : x_start + ps]
            for key in ["T_in", "T_target", "Q", "mask"]
        }
        return {
            **{k: v for k, v in sample.items() if k not in patched},
            **patched,
            "patch_origin": torch.tensor([0, y_start, x_start], dtype=torch.long),
            "sample_key": sample_key,
            "h5_path": path,
        }
