"""Spatial input layout of FM velocity models; training and sampling both build it here."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def net_inputs(x: Tensor, batch: dict[str, Any], in_channels: int) -> Tensor:
    """``[x_tau, mask, Q]`` for 3 channels, ``[x_tau, mask, Q, T_in]`` for 4.

    T_in is the temperature before the step. Without it the model cannot see heat left by earlier tracks.
    """
    parts = [x, batch["mask"], batch["Q"]]
    if in_channels == 4:
        parts.append(batch["T_in"])
    elif in_channels != 3:
        raise ValueError(f"in_channels must be 3 or 4, got {in_channels}")
    return torch.cat(parts, dim=1)


def model_in_channels(model: nn.Module) -> int:
    """Input channel count of a VelocityNet (``cfg``) or a DiT (``patch_embed``); 3 for anything else.

    The fallback is safe: a real 4-channel model fed 3 channels fails loudly in its first convolution.
    """
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "in_channels"):
        return int(cfg.in_channels)
    embed = getattr(model, "patch_embed", None)
    if isinstance(embed, nn.Conv3d):
        return int(embed.in_channels)
    return 3
