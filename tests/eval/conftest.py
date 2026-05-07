"""Shared fixtures for the eval test suite."""
from __future__ import annotations

import matplotlib
import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.eval.data.snapshot import Snapshot
from neural_pbf.eval.data.trajectory import Trajectory
from neural_pbf.physics.material import MaterialConfig

matplotlib.use("Agg")


@pytest.fixture
def sim_cfg() -> SimulationConfig:
    return SimulationConfig(Lx=1.0, Ly=1.0, Nx=8, Ny=8)


@pytest.fixture
def sim_cfg_3d() -> SimulationConfig:
    return SimulationConfig(Lx=1.0, Ly=1.0, Lz=0.5, Nx=8, Ny=8, Nz=4)


@pytest.fixture
def mat_cfg() -> MaterialConfig:
    return MaterialConfig.ss316l_preset()


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def make_snapshot(
    sim_cfg: SimulationConfig,
    fill: float = 300.0,
    t: float = 0.0,
    dt: float = 1e-5,
    device: torch.device | None = None,
) -> Snapshot:
    device = device or torch.device("cpu")
    if sim_cfg.is_3d:
        shape = (1, 1, sim_cfg.Nz, sim_cfg.Ny, sim_cfg.Nx)
    else:
        shape = (1, 1, sim_cfg.Ny, sim_cfg.Nx)
    T = torch.full(shape, fill, dtype=torch.float32, device=device)
    return Snapshot(T=T, t=t, dt=dt)


@pytest.fixture
def snap(sim_cfg: SimulationConfig, device: torch.device) -> Snapshot:
    return make_snapshot(sim_cfg, fill=300.0, device=device)


@pytest.fixture
def trajectory(
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
    device: torch.device,
) -> Trajectory:
    """Trajectory with increasing temperature (+50 K/step) for general tests."""
    snaps = [
        make_snapshot(sim_cfg, fill=300.0 + i * 50.0, t=i * 1e-5, dt=1e-5, device=device)
        for i in range(6)
    ]
    return Trajectory(snapshots=snaps, sim_cfg=sim_cfg, mat_cfg=mat_cfg)


@pytest.fixture
def constant_trajectory(
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
    device: torch.device,
) -> Trajectory:
    """Trajectory with constant temperature across all snapshots.

    Using a constant field makes the IdentityAdapter trivially correct:
    it returns GT[i] which equals GT[i+1], giving zero MAE.
    """
    snaps = [
        make_snapshot(sim_cfg, fill=300.0, t=i * 1e-5, dt=1e-5, device=device)
        for i in range(6)
    ]
    return Trajectory(snapshots=snaps, sim_cfg=sim_cfg, mat_cfg=mat_cfg)
