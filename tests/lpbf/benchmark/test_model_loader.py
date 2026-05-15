"""Tests for neural_pbf.eval.benchmark.model_loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_pbf.eval.benchmark.model_loader import load_model_adaptive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_net_ckpt(tmp_path: Path, device: torch.device) -> Path:
    """Write a minimal VelocityNet checkpoint that load_model_adaptive can read."""
    from neural_pbf.models.generative.fm.config import FMConfig
    from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
    from neural_pbf.models.generative.fm.velocity_net import VelocityNet

    cfg = FMConfig()
    model = VelocityNet(cfg)
    enc = ConditioningEncoder(cfg.cond_dim, cfg.cond_embed_dim)
    ckpt_path = tmp_path / "net_best.pt"
    torch.save(
        {
            "fm_cfg": cfg.model_dump(),
            "model_state": model.state_dict(),
            "cond_encoder_state": enc.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


# ---------------------------------------------------------------------------
# Tests — net model
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_load_net_returns_four_tuple(tmp_path: Path) -> None:
    device = torch.device("cpu")
    ckpt_path = _make_net_ckpt(tmp_path, device)
    result = load_model_adaptive("test-net", ckpt_path, "net", device)
    assert len(result) == 4, "Expected (model, cond_enc, patch_size, grid_attrs)"


@pytest.mark.unit
def test_load_net_models_in_eval_mode(tmp_path: Path) -> None:
    device = torch.device("cpu")
    ckpt_path = _make_net_ckpt(tmp_path, device)
    model, cond_enc, _, _ = load_model_adaptive("test-net", ckpt_path, "net", device)
    assert not model.training
    assert not cond_enc.training


@pytest.mark.unit
def test_load_net_patch_size_is_four(tmp_path: Path) -> None:
    device = torch.device("cpu")
    ckpt_path = _make_net_ckpt(tmp_path, device)
    _, _, patch_size, _ = load_model_adaptive("test-net", ckpt_path, "net", device)
    assert patch_size == 4


@pytest.mark.unit
def test_load_net_grid_attrs_fallback_when_absent(tmp_path: Path) -> None:
    device = torch.device("cpu")
    ckpt_path = _make_net_ckpt(tmp_path, device)
    _, _, _, grid_attrs = load_model_adaptive("test-net", ckpt_path, "net", device)
    expected = {"dx_m": 1.5e-05, "dy_m": 1.5e-05, "dz_m": 1.5e-05}
    assert grid_attrs == expected


@pytest.mark.unit
def test_load_net_grid_attrs_returned_when_present(tmp_path: Path) -> None:
    device = torch.device("cpu")
    # Write a checkpoint that includes grid_attrs
    from neural_pbf.models.generative.fm.config import FMConfig
    from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
    from neural_pbf.models.generative.fm.velocity_net import VelocityNet

    cfg = FMConfig()
    model = VelocityNet(cfg)
    enc = ConditioningEncoder(cfg.cond_dim, cfg.cond_embed_dim)
    ckpt_path = tmp_path / "net_with_grid.pt"
    expected = {"dx_m": 1.5e-5, "dy_m": 1.5e-5, "dz_m": 1.5e-5}
    torch.save(
        {
            "fm_cfg": cfg.model_dump(),
            "model_state": model.state_dict(),
            "cond_encoder_state": enc.state_dict(),
            "grid_attrs": expected,
        },
        ckpt_path,
    )
    _, _, _, grid_attrs = load_model_adaptive("test-grid", ckpt_path, "net", device)
    assert grid_attrs == expected


# ---------------------------------------------------------------------------
# Tests — error paths
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_missing_checkpoint_raises_file_not_found(tmp_path: Path) -> None:
    device = torch.device("cpu")
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_model_adaptive("x", tmp_path / "nope.pt", "net", device)


@pytest.mark.unit
def test_unknown_model_type_raises_value_error(tmp_path: Path) -> None:
    device = torch.device("cpu")
    ckpt_path = _make_net_ckpt(tmp_path, device)
    with pytest.raises(ValueError, match="Unknown model_type"):
        load_model_adaptive("x", ckpt_path, "unknown_type", device)
