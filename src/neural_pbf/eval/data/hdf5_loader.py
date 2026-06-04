"""HDF5 serialisation for Trajectory objects."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.physics.material import MaterialConfig

from .snapshot import Snapshot
from .trajectory import Trajectory

_KEY_SIM_CFG = "sim_cfg"
_KEY_MAT_CFG = "mat_cfg"
_KEY_META = "metadata"
_STEP_PREFIX = "step_"


def save_trajectory(traj: Trajectory, path: Path | str) -> None:
    """Serialise a Trajectory to an HDF5 file.

    Layout::

        /                  <- root: sim_cfg (JSON), mat_cfg (JSON), metadata (JSON)
        /step_000000/
            T              <- float32 array
            attrs: t, dt
            Q_ext          <- optional float32 array
            material_mask  <- optional uint8 array
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs[_KEY_SIM_CFG] = traj.sim_cfg.model_dump_json()
        f.attrs[_KEY_MAT_CFG] = traj.mat_cfg.model_dump_json()
        f.attrs[_KEY_META] = json.dumps(traj.metadata)
        for i, snap in enumerate(traj.snapshots):
            grp = f.create_group(f"{_STEP_PREFIX}{i:06d}")
            grp.create_dataset("T", data=snap.T.detach().cpu().float().numpy())
            grp.attrs["t"] = snap.t
            grp.attrs["dt"] = snap.dt
            if snap.Q_ext is not None:
                grp.create_dataset("Q_ext", data=snap.Q_ext.detach().cpu().float().numpy())
            if snap.material_mask is not None:
                grp.create_dataset(
                    "material_mask",
                    data=snap.material_mask.detach().cpu().numpy(),
                )


def load_trajectory(
    path: Path | str,
    device: torch.device | None = None,
) -> Trajectory:
    """Deserialise a Trajectory from an HDF5 file.
    Produced by :func:`save_trajectory`.
    """
    src = Path(path)
    target = device or torch.device("cpu")
    with h5py.File(src, "r") as f:
        sim_cfg = SimulationConfig.model_validate_json(str(f.attrs[_KEY_SIM_CFG]))
        mat_cfg = MaterialConfig.model_validate_json(str(f.attrs[_KEY_MAT_CFG]))
        metadata: dict = json.loads(str(f.attrs.get(_KEY_META, "{}")))

        keys = sorted(k for k in f if k.startswith(_STEP_PREFIX))
        snapshots: list[Snapshot] = []
        for key in keys:
            item = f[key]
            assert isinstance(item, h5py.Group), f"Expected HDF5 Group, got {type(item)}"
            grp: h5py.Group = item
            T = torch.tensor(grp["T"][()], device=target, dtype=torch.float32)  # type: ignore[arg-type]
            t = float(grp.attrs["t"])  # type: ignore[arg-type]
            dt = float(grp.attrs["dt"])  # type: ignore[arg-type]
            Q_ext: torch.Tensor | None = None
            material_mask: torch.Tensor | None = None
            if "Q_ext" in grp:
                Q_ext = torch.tensor(grp["Q_ext"][()], device=target, dtype=torch.float32)  # type: ignore[arg-type]
            if "material_mask" in grp:
                material_mask = torch.tensor(grp["material_mask"][()], device=target)  # type: ignore[arg-type]
            snapshots.append(Snapshot(T=T, t=t, dt=dt, Q_ext=Q_ext, material_mask=material_mask))

    return Trajectory(snapshots=snapshots, sim_cfg=sim_cfg, mat_cfg=mat_cfg, metadata=metadata)
