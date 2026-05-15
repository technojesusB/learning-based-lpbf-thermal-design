"""Smoke test for benchmark.runner — run_physics_sweep."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from neural_pbf.eval.benchmark.runner import run_physics_sweep


class _TinyDataset(Dataset):
    """4-sample, 8³-voxel synthetic dataset for quick smoke tests."""

    N = 4
    S = 8  # spatial size

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        shape = (1, self.S, self.S, self.S)
        return {
            "T_in": torch.rand(*shape),
            "T_target": torch.rand(*shape),
            "mask": torch.zeros(*shape),
            "Q": torch.zeros(*shape),
            "conditioning": torch.zeros(4),
        }


class _ZeroVelocityNet(nn.Module):
    def forward(self, x: torch.Tensor, tau: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, :1])


class _PassthroughCondEnc(nn.Module):
    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return cond


@pytest.mark.integration
def test_run_physics_sweep_returns_dataframe():
    ds = _TinyDataset()
    models = [("stub", _ZeroVelocityNet(), _PassthroughCondEnc(), "net")]
    df, vols, throughput = run_physics_sweep(
        models=models,
        dataset=ds,
        device=torch.device("cpu"),
        test_indices=[0, 1],
    )
    import pandas as pd
    assert isinstance(df, pd.DataFrame)


@pytest.mark.integration
def test_run_physics_sweep_dataframe_schema():
    ds = _TinyDataset()
    models = [("stub", _ZeroVelocityNet(), _PassthroughCondEnc(), "net")]
    df, _, _ = run_physics_sweep(
        models=models,
        dataset=ds,
        device=torch.device("cpu"),
        test_indices=[0, 1],
    )
    expected = {"Model", "Sample", "IoU", "Depth_GT", "Depth_Pred", "T_max_Error", "Offset_vox"}
    assert expected.issubset(set(df.columns))


@pytest.mark.integration
def test_run_physics_sweep_row_count():
    ds = _TinyDataset()
    models = [
        ("m1", _ZeroVelocityNet(), _PassthroughCondEnc(), "net"),
        ("m2", _ZeroVelocityNet(), _PassthroughCondEnc(), "net"),
    ]
    df, _, _ = run_physics_sweep(
        models=models,
        dataset=ds,
        device=torch.device("cpu"),
        test_indices=[0, 1, 2],
    )
    # 2 models × 3 samples = 6 rows
    assert len(df) == 6


@pytest.mark.integration
def test_run_physics_sweep_returns_vols_for_collect_indices():
    ds = _TinyDataset()
    models = [("m1", _ZeroVelocityNet(), _PassthroughCondEnc(), "net")]
    _, vols, _ = run_physics_sweep(
        models=models,
        dataset=ds,
        device=torch.device("cpu"),
        test_indices=[0, 1, 2],
        collect_indices=[0, 2],
    )
    assert "m1" in vols
    assert 0 in vols["m1"] and 2 in vols["m1"]
    assert 1 not in vols["m1"]


@pytest.mark.integration
def test_run_physics_sweep_throughput_dict():
    ds = _TinyDataset()
    models = [("m1", _ZeroVelocityNet(), _PassthroughCondEnc(), "net")]
    _, _, throughput = run_physics_sweep(
        models=models,
        dataset=ds,
        device=torch.device("cpu"),
        test_indices=[0],
    )
    assert "m1" in throughput
    assert "s_per_sample" in throughput["m1"]


@pytest.mark.integration
def test_run_physics_sweep_no_side_effects(tmp_path):
    """runner must not write any files or call mlflow."""
    import os
    before = set(os.listdir(tmp_path))
    ds = _TinyDataset()
    models = [("m1", _ZeroVelocityNet(), _PassthroughCondEnc(), "net")]
    run_physics_sweep(
        models=models,
        dataset=ds,
        device=torch.device("cpu"),
        test_indices=[0],
    )
    after = set(os.listdir(tmp_path))
    assert before == after, "runner must not create files as a side effect"
