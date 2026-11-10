"""The 3D-RoPE DiT of the RoPE scripts: arguments, model and call for ``train_run.run_training``.

The scripts differ in precision and Accelerate, the default recipe, the physics form and backend, and overlapping
unpatchify. RoPE needs each token's patch-centre index in the full domain, from the patch origin.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import partial
from typing import Any

from torch import Tensor, nn

from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.dit import VelocityDiTRoPE, patch_center_coords_idx
from neural_pbf.training.script_args import add_data_args, add_run_args
from neural_pbf.training.train_run import ModelSpec, add_common_args, with_tv


def build_rope_parser(description: str, use_accelerate: bool, **defaults: Any) -> argparse.ArgumentParser:
    """The RoPE scripts' arguments; ``defaults`` overrides the defaults below for one script."""
    parser = argparse.ArgumentParser(description=description)
    add_data_args(parser)
    add_run_args(parser)
    add_common_args(parser, use_accelerate=use_accelerate)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=288)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--cond_embed_dim", type=int, default=128)
    parser.set_defaults(**defaults)
    return parser


def build_rope(
    args: argparse.Namespace, ds_cfg: Any, grid_attrs: Mapping[str, float], overlap: bool
) -> tuple[nn.Module, nn.Module, dict]:
    cfg = {"model": "VelocityDiTRoPE", "patch_size": args.patch_size, "in_channels": args.in_channels,
           "embed_dim": args.embed_dim, "depth": args.depth, "num_heads": args.num_heads,
           "cond_embed_dim": args.cond_embed_dim, "overlap_unpatchify": overlap}  # fmt: skip
    model = VelocityDiTRoPE(**{k: v for k, v in cfg.items() if k != "model"})
    return model, ConditioningEncoder(len(ds_cfg.conditioning_keys), args.cond_embed_dim), cfg


def call_rope(
    model: nn.Module, inputs: Tensor, tau: Tensor, cond_emb: Tensor, batch: Mapping[str, Any],
    grid_attrs: Mapping[str, float], patch_size: int,
) -> Tensor:  # fmt: skip
    """RoPE needs each token's patch-centre index in the full domain, from the patch origin."""
    D, H, W = inputs.shape[2:]
    coords = patch_center_coords_idx(batch["patch_origin"], patch_size, grid_attrs,  # type: ignore[arg-type]
                                     (D // patch_size, H // patch_size, W // patch_size), inputs.device)  # fmt: skip
    return model(inputs, tau, cond_emb, coords)


def rope_spec(
    args: argparse.Namespace, *, overlap: bool, use_accelerate: bool, schedule: str = "onecycle_pct0.1",
    tv_weight: float | None = None,
) -> ModelSpec:  # fmt: skip
    """``schedule`` and ``tv_weight``: train_fm_dit_rope.py trains with a constant lr and TV 0.001."""
    fixed: dict[str, Any] = {"overlap_unpatchify": True} if overlap else {}
    extra: dict[str, Any] = {}
    if tv_weight is not None:
        fixed["tv_weight"] = tv_weight
        extra["data_loss"] = with_tv(tv_weight)
    return ModelSpec(
        build=partial(build_rope, overlap=overlap), call=partial(call_rope, patch_size=args.patch_size),
        schedule=schedule, use_accelerate=use_accelerate, fixed=fixed, **extra,
    )  # fmt: skip
