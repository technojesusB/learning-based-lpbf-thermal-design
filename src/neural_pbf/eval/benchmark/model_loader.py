"""Adaptive model loader — detects patch_size automatically.

Public API:
    load_model_adaptive  -- load any checkpoint, return (model, cond_enc, patch_size)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_model_adaptive(
    name: str,
    ckpt_path: str | Path,
    model_type: str,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, int, dict[str, Any] | None]:
    """Load a model checkpoint with automatic patch_size detection.

    Tries patch sizes [4, 8] for DiT / RoPE / Triton variants.  Raises
    ``ValueError`` if no patch size succeeds — never silently falls back.

    Args:
        name:       Display name used only for log messages.
        ckpt_path:  Path to the ``best.pt`` checkpoint file.
        model_type: One of ``"net"``, ``"dit"``, ``"rope"``, ``"triton"``.
        device:     Target device.

    Returns:
        (model, cond_enc, patch_size, grid_attrs) where ``grid_attrs`` is the
        dict stored under key ``"grid_attrs"`` in the checkpoint, or ``None``
        if the key is absent.  Required for ``"rope"`` / ``"triton"`` rollouts.

    Raises:
        FileNotFoundError: Checkpoint file missing.
        ValueError: Unknown model_type or no patch_size worked.
    """
    from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder

    ckpt_path = Path(ckpt_path)
    ckpt: dict[str, Any] = torch.load(str(ckpt_path), map_location=device, weights_only=True)

    model: nn.Module | None = None
    cond_enc: nn.Module
    patch_size = 4
    # Default fallback for grid_attrs if missing in checkpoint
    grid_attrs: dict[str, Any] | None = ckpt.get("grid_attrs")
    if grid_attrs is None:
        grid_attrs = {"dx_m": 1.5e-05, "dy_m": 1.5e-05, "dz_m": 1.5e-05}
        logger.warning("Checkpoint for %r lacks 'grid_attrs'; using fallback: %s", name, grid_attrs)

    if model_type == "net":
        from neural_pbf.models.generative.fm.config import FMConfig
        from neural_pbf.models.generative.fm.velocity_net import VelocityNet

        fm_cfg = FMConfig(**ckpt["fm_cfg"])
        model = VelocityNet(fm_cfg).to(device)
        cond_enc = ConditioningEncoder(fm_cfg.cond_dim, fm_cfg.cond_embed_dim).to(device)
        model.load_state_dict(ckpt["model_state"])
        cond_enc.load_state_dict(ckpt["cond_encoder_state"])

    elif model_type == "dit":
        from neural_pbf.models.generative.fm.dit import VelocityDiT

        cond_enc = ConditioningEncoder(12, 128).to(device)
        for p_size in [4, 8]:
            try:
                candidate = VelocityDiT(
                    depth=6,
                    embed_dim=256,
                    num_heads=8,
                    patch_size=p_size,
                    input_size=64,
                    in_channels=3,
                    cond_embed_dim=128,
                ).to(device)
                candidate.load_state_dict(ckpt["model_state"])
                model = candidate
                patch_size = p_size
                break
            except Exception as exc:
                logger.debug("DiT patch_size=%d failed for %r: %s", p_size, name, exc)
                continue
        if model is None:
            raise ValueError(f"Could not load DiT {name!r} with any patch_size in [4, 8]")
        cond_enc.load_state_dict(ckpt["cond_encoder_state"])

    elif model_type in ("rope", "triton"):
        from neural_pbf.models.generative.fm.dit import VelocityDiTRoPE

        cond_enc = ConditioningEncoder(12, 128).to(device)
        for p_size in [4, 8]:
            try:
                candidate = VelocityDiTRoPE(
                    depth=6,
                    embed_dim=288,
                    num_heads=8,
                    patch_size=p_size,
                    input_size=64,
                    in_channels=3,
                    cond_embed_dim=128,
                ).to(device)
                candidate.load_state_dict(ckpt["model_state"])
                model = candidate
                patch_size = p_size
                break
            except Exception as exc:
                logger.debug("RoPE patch_size=%d failed for %r: %s", p_size, name, exc)
                continue
        if model is None:
            raise ValueError(f"Could not load RoPE {name!r} with any patch_size in [4, 8]")
        cond_enc.load_state_dict(ckpt["cond_encoder_state"])

    else:
        raise ValueError(f"Unknown model_type {model_type!r} for {name!r}")

    model.eval()
    cond_enc.eval()
    logger.info("Loaded %s (type=%s, patch_size=%d)", name, model_type, patch_size)
    return model, cond_enc, patch_size, grid_attrs
