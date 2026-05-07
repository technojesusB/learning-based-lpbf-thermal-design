"""Tests for Snapshot, Trajectory and HDF5 round-trip."""
from __future__ import annotations

import pytest
import torch

from neural_pbf.eval.data.hdf5_loader import load_trajectory, save_trajectory
from neural_pbf.eval.data.snapshot import Snapshot
from neural_pbf.eval.data.trajectory import Trajectory


# ── Snapshot ──────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_snapshot_is_frozen(sim_cfg, device):
    T = torch.ones(1, 1, sim_cfg.Ny, sim_cfg.Nx)
    snap = Snapshot(T=T, t=0.0, dt=1e-5)
    with pytest.raises((AttributeError, TypeError)):
        snap.t = 99.0  # type: ignore[misc]


@pytest.mark.unit
def test_snapshot_optional_fields_default_none(sim_cfg):
    T = torch.zeros(1, 1, sim_cfg.Ny, sim_cfg.Nx)
    snap = Snapshot(T=T, t=0.0, dt=1e-5)
    assert snap.Q_ext is None
    assert snap.material_mask is None


@pytest.mark.unit
def test_snapshot_stores_tensors_by_reference(sim_cfg):
    T = torch.ones(1, 1, sim_cfg.Ny, sim_cfg.Nx)
    snap = Snapshot(T=T, t=0.0, dt=1e-5)
    assert snap.T is T


# ── Trajectory ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_trajectory_len(trajectory):
    assert len(trajectory) == 6


@pytest.mark.unit
def test_trajectory_getitem(trajectory):
    snap = trajectory[0]
    assert isinstance(snap, Snapshot)


@pytest.mark.unit
def test_trajectory_metadata_default_empty(sim_cfg, mat_cfg):
    traj = Trajectory(snapshots=[], sim_cfg=sim_cfg, mat_cfg=mat_cfg)
    assert traj.metadata == {}


# ── HDF5 round-trip ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_hdf5_roundtrip_tensor_equality(tmp_path, trajectory):
    path = tmp_path / "traj.h5"
    save_trajectory(trajectory, path)
    loaded = load_trajectory(path)

    assert len(loaded) == len(trajectory)
    for orig, load in zip(trajectory.snapshots, loaded.snapshots):
        assert torch.allclose(orig.T, load.T), "T tensors not equal after round-trip"
        assert abs(orig.t - load.t) < 1e-12
        assert abs(orig.dt - load.dt) < 1e-12


@pytest.mark.unit
def test_hdf5_roundtrip_configs(tmp_path, trajectory):
    path = tmp_path / "traj.h5"
    save_trajectory(trajectory, path)
    loaded = load_trajectory(path)

    assert loaded.sim_cfg.Nx == trajectory.sim_cfg.Nx
    assert loaded.sim_cfg.Ny == trajectory.sim_cfg.Ny
    assert abs(loaded.mat_cfg.T_solidus - trajectory.mat_cfg.T_solidus) < 1e-6


@pytest.mark.unit
def test_hdf5_roundtrip_with_qext(tmp_path, sim_cfg, mat_cfg):
    shape = (1, 1, sim_cfg.Ny, sim_cfg.Nx)
    snaps = [
        Snapshot(
            T=torch.full(shape, 300.0 + i * 10.0),
            t=i * 1e-5,
            dt=1e-5,
            Q_ext=torch.ones(shape) * i,
        )
        for i in range(3)
    ]
    traj = Trajectory(snapshots=snaps, sim_cfg=sim_cfg, mat_cfg=mat_cfg)
    path = tmp_path / "traj_q.h5"
    save_trajectory(traj, path)
    loaded = load_trajectory(path)

    for orig, load in zip(traj.snapshots, loaded.snapshots):
        assert load.Q_ext is not None
        assert torch.allclose(orig.Q_ext, load.Q_ext)  # type: ignore[arg-type]


@pytest.mark.unit
def test_hdf5_roundtrip_metadata(tmp_path, trajectory):
    trajectory.metadata["scan_speed_m_s"] = 0.5
    path = tmp_path / "traj_meta.h5"
    save_trajectory(trajectory, path)
    loaded = load_trajectory(path)
    assert loaded.metadata.get("scan_speed_m_s") == 0.5
