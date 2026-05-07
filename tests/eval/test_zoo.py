"""Tests for MaterialZoo and zoo benchmark driver."""
from __future__ import annotations

import pytest
import pandas as pd

from neural_pbf.eval.adapters.identity_adapter import IdentityAdapter
from neural_pbf.eval.zoo.benchmark import run_zoo_benchmark
from neural_pbf.eval.zoo.materials import MaterialZoo, ZooEntry
from neural_pbf.physics.material import MaterialConfig


# ── MaterialZoo ───────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_default_zoo_has_five_entries():
    zoo = MaterialZoo.default()
    assert len(zoo.all()) == 5


@pytest.mark.unit
def test_default_zoo_has_id_and_ood():
    zoo = MaterialZoo.default()
    assert len(zoo.id_materials()) > 0
    assert len(zoo.ood_materials()) > 0


@pytest.mark.unit
def test_zoo_register_and_get():
    zoo = MaterialZoo()
    cfg = MaterialConfig.ss316l_preset()
    zoo.register("test_mat", cfg, in_distribution=True)
    entry = zoo.get("test_mat")
    assert isinstance(entry, ZooEntry)
    assert entry.name == "test_mat"
    assert entry.in_distribution


@pytest.mark.unit
def test_zoo_get_raises_on_unknown():
    zoo = MaterialZoo()
    with pytest.raises(KeyError):
        zoo.get("nonexistent")


@pytest.mark.unit
def test_zoo_id_ood_partition(mat_cfg):
    zoo = MaterialZoo()
    zoo.register("a", mat_cfg, in_distribution=True)
    zoo.register("b", mat_cfg, in_distribution=False)
    assert len(zoo.id_materials()) == 1
    assert len(zoo.ood_materials()) == 1


# ── benchmark driver ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_benchmark_returns_dataframe(trajectory):
    df = run_zoo_benchmark(
        stepper=IdentityAdapter(),
        trajectories={"ss316l": trajectory},
        mode="one_step",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


@pytest.mark.unit
def test_benchmark_dataframe_columns(trajectory):
    df = run_zoo_benchmark(
        stepper=IdentityAdapter(),
        trajectories={"ss316l": trajectory},
    )
    expected_cols = {"material_name", "stepper_name", "mean_mae", "diverged"}
    assert expected_cols.issubset(set(df.columns))


@pytest.mark.unit
def test_benchmark_two_materials(trajectory, sim_cfg, mat_cfg):
    from neural_pbf.eval.data.snapshot import Snapshot
    from neural_pbf.eval.data.trajectory import Trajectory
    import torch

    # Second trajectory with a different mat_cfg
    snaps = [
        Snapshot(
            T=torch.full((1, 1, sim_cfg.Ny, sim_cfg.Nx), 300.0 + i * 10.0),
            t=i * 1e-5,
            dt=1e-5,
        )
        for i in range(4)
    ]
    traj2 = Trajectory(snaps, sim_cfg=sim_cfg, mat_cfg=MaterialConfig.ti64_preset())

    zoo = MaterialZoo()
    zoo.register("ss316l", mat_cfg, in_distribution=True)
    zoo.register("ti64", MaterialConfig.ti64_preset(), in_distribution=False)

    df = run_zoo_benchmark(
        stepper=IdentityAdapter(),
        trajectories={"ss316l": trajectory, "ti64": traj2},
        zoo=zoo,
    )
    assert len(df) == 2
    id_row = df[df["material_name"] == "ss316l"].iloc[0]
    ood_row = df[df["material_name"] == "ti64"].iloc[0]
    assert id_row["in_distribution"]
    assert not ood_row["in_distribution"]
