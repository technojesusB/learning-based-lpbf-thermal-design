"""train_fm_dit_rope.py — the 3D-RoPE DiT in fp32, without Accelerate.

Model: training/rope_model.py (VelocityDiTRoPE with physical 3D rotary position embeddings); the rest is
the shared training run (training/train_run.py). Constant learning rate and TV regularization of the
velocity field (weight 0.001); defaults: 8³ patches, batch 4. Run on its own, it logs to the MLflow
experiment lpbf_experiments, unpinned.

Run:
    uv run python experiments/train_fm_dit_rope.py

In the ablation: experiment dit_v3_rope (training/ablation.py).
"""

from __future__ import annotations

import argparse

from neural_pbf.training.rope_model import build_rope_parser, rope_spec
from neural_pbf.training.tracking import RunHooks
from neural_pbf.training.train_run import ModelSpec, run_training

TV_WEIGHT = 0.001


def build_parser() -> argparse.ArgumentParser:
    return build_rope_parser("3D-RoPE DiT flow matching in fp32", use_accelerate=False, batch_size=4, patch_size=8)


def spec(args: argparse.Namespace) -> ModelSpec:
    return rope_spec(args, overlap=False, use_accelerate=False, schedule="constant", tv_weight=TV_WEIGHT)


def main(argv: list[str] | None = None, *, hooks: RunHooks | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args, __file__, spec(args), hooks)


if __name__ == "__main__":
    main()
