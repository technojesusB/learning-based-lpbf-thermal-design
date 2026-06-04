"""Tests for neural_pbf.training.checkpointing — save_checkpoint / load_checkpoint."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from neural_pbf.training.checkpointing import load_checkpoint, save_checkpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture()
def models() -> tuple[_TinyNet, _TinyNet]:
    return _TinyNet(), _TinyNet()


@pytest.fixture()
def fake_args():
    class _Args:
        lr = 1e-3
        epochs = 10
    return _Args()


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_creates_best_pt_file(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        save_checkpoint(model, cond_encoder, tmp_path, epoch=3, val_loss=0.25, args=fake_args)
        assert (tmp_path / "best.pt").exists()

    def test_saved_dict_has_required_keys(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        save_checkpoint(model, cond_encoder, tmp_path, epoch=7, val_loss=0.1, args=fake_args)
        ckpt = torch.load(tmp_path / "best.pt", weights_only=True)
        for key in ("model_state", "cond_encoder_state", "epoch", "val_loss", "args"):
            assert key in ckpt

    def test_saved_epoch_and_loss_values(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        save_checkpoint(model, cond_encoder, tmp_path, epoch=5, val_loss=0.42, args=fake_args)
        ckpt = torch.load(tmp_path / "best.pt", weights_only=True)
        assert ckpt["epoch"] == 5
        assert ckpt["val_loss"] == pytest.approx(0.42)

    def test_saves_plain_state_dicts_without_accelerator(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        expected_model_keys = set(model.state_dict().keys())
        save_checkpoint(model, cond_encoder, tmp_path, epoch=0, val_loss=1.0,
                        args=fake_args, accelerator=None)
        ckpt = torch.load(tmp_path / "best.pt", weights_only=True)
        assert set(ckpt["model_state"].keys()) == expected_model_keys

    def test_noop_when_accelerator_not_main_process(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        accel = MagicMock()
        accel.is_main_process = False
        save_checkpoint(model, cond_encoder, tmp_path, epoch=0, val_loss=0.5,
                        args=fake_args, accelerator=accel)
        assert not (tmp_path / "best.pt").exists()

    def test_calls_unwrap_model_with_accelerator(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        accel = MagicMock()
        accel.is_main_process = True
        accel.unwrap_model.side_effect = lambda m: m  # pass-through
        save_checkpoint(model, cond_encoder, tmp_path, epoch=1, val_loss=0.3,
                        args=fake_args, accelerator=accel)
        assert accel.unwrap_model.call_count == 2

    def test_overwrites_existing_checkpoint(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        save_checkpoint(model, cond_encoder, tmp_path, epoch=0, val_loss=0.9, args=fake_args)
        save_checkpoint(model, cond_encoder, tmp_path, epoch=5, val_loss=0.1, args=fake_args)
        ckpt = torch.load(tmp_path / "best.pt", weights_only=True)
        assert ckpt["epoch"] == 5


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def _save(self, tmp_path, model, cond_encoder, fake_args):
        save_checkpoint(model, cond_encoder, tmp_path, epoch=3, val_loss=0.15, args=fake_args)
        return tmp_path / "best.pt"

    def test_restores_model_weights(self, tmp_path, fake_args):
        src_model, src_cond = _TinyNet(), _TinyNet()
        ckpt_path = self._save(tmp_path, src_model, src_cond, fake_args)

        new_model, new_cond = _TinyNet(), _TinyNet()
        with torch.no_grad():
            for p in new_model.parameters():
                p.fill_(99.0)

        load_checkpoint(new_model, new_cond, ckpt_path, device="cpu")

        for p_src, p_loaded in zip(src_model.parameters(), new_model.parameters()):
            assert torch.allclose(p_src, p_loaded)

    def test_returns_raw_checkpoint_dict(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        ckpt_path = self._save(tmp_path, model, cond_encoder, fake_args)
        result = load_checkpoint(model, cond_encoder, ckpt_path, device="cpu")
        assert isinstance(result, dict)
        assert "epoch" in result

    def test_raises_file_not_found_for_missing_path(self, tmp_path, models):
        model, cond_encoder = models
        with pytest.raises(FileNotFoundError):
            load_checkpoint(model, cond_encoder, tmp_path / "missing.pt", device="cpu")

    def test_delegates_to_unwrap_model_when_accelerator_provided(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        ckpt_path = self._save(tmp_path, model, cond_encoder, fake_args)

        accel = MagicMock()
        accel.unwrap_model.side_effect = lambda m: m
        load_checkpoint(model, cond_encoder, ckpt_path, device="cpu", accelerator=accel)
        assert accel.unwrap_model.call_count == 2

    def test_accepts_path_object_and_string(self, tmp_path, models, fake_args):
        model, cond_encoder = models
        ckpt_path = self._save(tmp_path, model, cond_encoder, fake_args)
        load_checkpoint(model, cond_encoder, str(ckpt_path), device="cpu")  # str path
        load_checkpoint(model, cond_encoder, ckpt_path, device="cpu")       # Path object
