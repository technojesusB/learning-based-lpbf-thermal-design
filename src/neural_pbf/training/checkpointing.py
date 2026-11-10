"""Checkpoint save/load utilities shared across experiment scripts."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def atomic_torch_save(obj: Any, path: Path) -> None:
    """Write to a temporary file next to ``path``, then rename: a crash never leaves a half-written checkpoint."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_checkpoint(
    model: nn.Module,
    cond_encoder: nn.Module,
    ckpt_dir: Path,
    epoch: int,
    val_loss: float,
    args: Any,
    accelerator: Any = None,
    dataset_cfg: dict[str, Any] | None = None,
    fm_cfg: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Save best-checkpoint dict to ckpt_dir/best.pt.

    ``dataset_cfg`` and ``fm_cfg`` let evaluation rebuild exactly the inputs the model was trained on.
    ``provenance`` (``tracking.RunTracker.provenance``) makes the checkpoint traceable without MLflow: run name,
    pinned flag, git commit, manifest and data hashes, run ids and every parameter.

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
    atomic_torch_save(
        {
            "model_state": model_state,
            "cond_encoder_state": cond_state,
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args) if hasattr(args, "__dict__") else args,
            **({"dataset_cfg": dataset_cfg} if dataset_cfg is not None else {}),
            **({"fm_cfg": fm_cfg} if fm_cfg is not None else {}),
            **({"provenance": provenance} if provenance is not None else {}),
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
