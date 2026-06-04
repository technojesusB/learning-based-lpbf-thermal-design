"""Checkpoint save/load utilities shared across experiment scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    cond_encoder: nn.Module,
    ckpt_dir: Path,
    epoch: int,
    val_loss: float,
    args: Any,
    accelerator: Any = None,
) -> None:
    """Save best-checkpoint dict to ckpt_dir/best.pt.

    When *accelerator* is provided the call is a no-op on non-main processes and
    state dicts are extracted via accelerator.unwrap_model before saving.
    """
    if accelerator is not None and not accelerator.is_main_process:
        return
    if accelerator is not None:
        model_state = accelerator.unwrap_model(model).state_dict()
        cond_state = accelerator.unwrap_model(cond_encoder).state_dict()
    else:
        model_state = model.state_dict()
        cond_state = cond_encoder.state_dict()
    torch.save(
        {
            "model_state": model_state,
            "cond_encoder_state": cond_state,
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args) if hasattr(args, "__dict__") else args,
        },
        ckpt_dir / "best.pt",
    )
    logger.debug("Checkpoint saved to %s (epoch=%d, val_loss=%.6f)", ckpt_dir, epoch, val_loss)


def load_checkpoint(
    model: nn.Module,
    cond_encoder: nn.Module,
    ckpt_path: Path,
    device: torch.device | str,
    accelerator: Any = None,
) -> dict:
    """Load model and cond_encoder weights from *ckpt_path* in-place.

    Returns the raw checkpoint dict for any extra fields callers may need.
    Raises FileNotFoundError if the path does not exist.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt: dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if accelerator is not None:
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        accelerator.unwrap_model(cond_encoder).load_state_dict(ckpt["cond_encoder_state"])
    else:
        model.load_state_dict(ckpt["model_state"])
        cond_encoder.load_state_dict(ckpt["cond_encoder_state"])
    return ckpt
