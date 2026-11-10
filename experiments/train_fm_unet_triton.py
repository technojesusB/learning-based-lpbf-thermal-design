"""train_fm_unet_triton.py — 3D U-Net flow matching, bf16 under Accelerate, optionally with the physics loss.

The model is the U-Net VelocityNet; everything else is the shared training run (training/train_run.py). Run on its
own, it trains with its defaults and logs to the MLflow experiment lpbf_experiments, unpinned.

Run:
    uv run accelerate launch experiments/train_fm_unet_triton.py
    uv run accelerate launch experiments/train_fm_unet_triton.py --physics_form tin --physics_backend triton

In the ablation: experiments unet_none, unet_physics_coupled, unet_physics_tin (training/ablation.py).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from torch import nn

from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.velocity_net import VelocityNet
from neural_pbf.training.script_args import add_data_args, add_run_args
from neural_pbf.training.tracking import RunHooks
from neural_pbf.training.train_run import ModelSpec, add_common_args, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D U-Net flow matching")
    add_data_args(parser)
    add_run_args(parser)
    add_common_args(parser, use_accelerate=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--cond_embed_dim", type=int, default=128)
    return parser


def _build(args: argparse.Namespace, ds_cfg: Any, grid_attrs: Mapping[str, float]) -> tuple[nn.Module, nn.Module, dict]:
    cfg = FMConfig(
        in_channels=args.in_channels, out_channels=1, base_channels=args.base_channels, depth=args.depth,
        cond_dim=len(ds_cfg.conditioning_keys), cond_embed_dim=args.cond_embed_dim, tau_embed_dim=args.cond_embed_dim,
        T_ref=ds_cfg.T_ref, T_ambient=ds_cfg.T_ambient,
    )  # fmt: skip
    return VelocityNet(cfg), ConditioningEncoder(len(ds_cfg.conditioning_keys), args.cond_embed_dim), cfg.model_dump()


def spec(args: argparse.Namespace) -> ModelSpec:
    return ModelSpec(build=_build, schedule="onecycle_pct0.1", use_accelerate=True)


def main(argv: list[str] | None = None, *, hooks: RunHooks | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args, __file__, spec(args), hooks)


if __name__ == "__main__":
    main()
