"""Regressions found while restoring CI: each test failed before its fix."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import mlflow
import pytest
import torch

from neural_pbf.models.generative.fm.dit import VelocityDiTRoPE

REPO = Path(__file__).resolve().parent.parent


def _rope() -> VelocityDiTRoPE:
    return VelocityDiTRoPE(
        depth=1, embed_dim=288, num_heads=8, patch_size=8, input_size=64, in_channels=3, cond_embed_dim=16
    )


def test_rope_loads_checkpoints_that_still_hold_the_unused_pos_embed() -> None:
    state = {**_rope().state_dict(), "pos_embed": torch.zeros(1, 512, 288)}
    assert _rope().load_state_dict(state).unexpected_keys == []
    assert "pos_embed" in state, "the caller's dict must not be mutated"


def test_old_final_layer_name_loads_into_unpatchify_with_identical_output() -> None:
    new = _rope().eval()
    state = dict(new.state_dict())
    state["final_layer.weight"] = state.pop("unpatchify.weight")
    state["final_layer.bias"] = state.pop("unpatchify.bias")
    old = _rope().eval()
    old.load_state_dict(state)
    x, t, c = torch.rand(1, 3, 64, 64, 64), torch.tensor([0.5]), torch.rand(1, 16)
    coords = torch.rand(1, 512, 3)
    with torch.no_grad():
        assert torch.equal(old(x, t, c, coords), new(x, t, c, coords))


def test_rope_still_rejects_other_unexpected_keys() -> None:
    state = {**_rope().state_dict(), "not_a_param": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="not_a_param"):
        _rope().load_state_dict(state)


def test_tracker_survives_a_zombie_run_from_another_store(tmp_path: Path) -> None:
    from neural_pbf.tracking.backends.mlflow_backend import MLflowTracker

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/old.db")
    mlflow.set_experiment("old")
    mlflow.start_run()
    try:
        tracker = MLflowTracker(tracking_uri=f"sqlite:///{tmp_path}/new.db", experiment_name="new")
        with tracker.start_run("r", {}, {}):
            assert mlflow.active_run() is not None
        assert mlflow.active_run() is None
    finally:
        while mlflow.active_run():
            mlflow.end_run()


# train_fm_dit.py no longer draws its snapshot itself: run_training logs it through log_val_image_rope (training/loops.py).
@pytest.mark.parametrize("script", ["train_fm_patches.py"])
def test_snapshot_plot_helper_is_imported(script: str) -> None:
    tree = ast.parse((REPO / "experiments" / script).read_text())
    imported = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "val_grid_2x2" in imported


def test_generator_uses_triton_only_on_gpu() -> None:
    from argparse import Namespace

    from experiments.generate_offline_dataset import _should_use_triton

    assert not _should_use_triton(Namespace(no_triton=False), torch.device("cpu"))
    assert _should_use_triton(Namespace(no_triton=False), torch.device("cuda"))
    assert not _should_use_triton(Namespace(no_triton=True), torch.device("cuda"))


def test_generator_exits_nonzero_when_a_run_failed() -> None:
    from experiments import generate_offline_dataset as gen

    with (
        patch.object(gen, "_orchestrate", return_value=[0]),
        patch.object(gen, "_resolve_wsl_safe", return_value=False),
        patch("sys.argv", ["gen"]),
        pytest.raises(SystemExit) as exc,
    ):
        gen.main()
    assert exc.value.code == 1


def test_dit_v2_points_at_its_real_checkpoint() -> None:
    from experiments.benchmark_suite import _MODEL_REGISTRY

    assert _MODEL_REGISTRY["DiT v2"][0] == "checkpoints/dit/best.pt"


def test_benchmark_exits_nonzero_when_a_model_failed() -> None:
    from experiments import benchmark_suite as bench

    with patch.object(bench, "run_benchmark", return_value=["DiT v1"]), pytest.raises(SystemExit) as exc:
        bench.main([])
    assert exc.value.code == 1
