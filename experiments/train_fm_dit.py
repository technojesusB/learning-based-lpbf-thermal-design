"""train_fm_dit.py — the DiT with an absolute position embedding: learned (APE) or fixed 3D sinusoidal.

The learned embedding is models/dit_variants.py (rebuilt from the journal), the sinusoidal one the DiT of
models/generative/fm/dit.py. fp32 without Accelerate, constant learning rate, and TV regularization of the velocity
field (weight 0.001) in the training loss; validation uses the plain FM loss. Run on its own, it logs to the MLflow
experiment lpbf_experiments, unpinned.

Run:
    uv run python experiments/train_fm_dit.py --pos_embed sincos
    uv run python experiments/train_fm_dit.py --pos_embed learned

In the ablation: experiments dit_v1_ape and dit_v2_sincos (training/ablation.py).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from torch import nn

from neural_pbf.models.dit_variants import VelocityDiTAPE
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.dit import VelocityDiT
from neural_pbf.training.script_args import add_data_args, add_run_args
from neural_pbf.training.tracking import RunHooks
from neural_pbf.training.train_run import ModelSpec, add_common_args, run_training, with_tv

TV_WEIGHT = 0.001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DiT flow matching with an absolute position embedding")
    add_data_args(parser)
    add_run_args(parser)
    add_common_args(parser, use_accelerate=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos_embed", choices=("sincos", "learned"), required=True)
    parser.add_argument("--patch_size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--embed_dim", type=int, default=288)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--cond_embed_dim", type=int, default=128)
    return parser


def _build(args: argparse.Namespace, ds_cfg: Any, grid_attrs: Mapping[str, float]) -> tuple[nn.Module, nn.Module, dict]:
    model_cls = VelocityDiTAPE if args.pos_embed == "learned" else VelocityDiT
    cfg = {"patch_size": args.patch_size, "in_channels": args.in_channels, "embed_dim": args.embed_dim,
           "depth": args.depth, "num_heads": args.num_heads, "cond_embed_dim": args.cond_embed_dim}  # fmt: skip
    model = model_cls(**cfg)
    return (
        model,
        ConditioningEncoder(len(ds_cfg.conditioning_keys), args.cond_embed_dim),
        {"model": model_cls.__name__, "pos_embed": args.pos_embed, **cfg},
    )


def spec(args: argparse.Namespace) -> ModelSpec:
    return ModelSpec(build=_build, schedule="constant", use_accelerate=False, data_loss=with_tv(TV_WEIGHT),
                    fixed={"tv_weight": TV_WEIGHT})  # fmt: skip


def main(argv: list[str] | None = None, *, hooks: RunHooks | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args, __file__, spec(args), hooks)


if __name__ == "__main__":
    main()
