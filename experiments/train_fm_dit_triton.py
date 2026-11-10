"""train_fm_dit_triton.py — the 3D-RoPE DiT in bf16 under Accelerate, with the Triton physics loss.

Model: training/rope_model.py; the rest is the shared training run (training/train_run.py). OneCycle
schedule; the physics form and backend come from the command line. Run on its own, it logs to the
MLflow experiment lpbf_experiments, unpinned.

Run:
    uv run accelerate launch experiments/train_fm_dit_triton.py
    uv run accelerate launch experiments/train_fm_dit_triton.py --physics_form coupled --physics_backend triton

In the ablation: experiments dit_v5_none_triton, dit_v5_coupled_triton, dit_v5_tin_triton (training/ablation.py).
"""

from __future__ import annotations

import argparse

from neural_pbf.training.rope_model import build_rope_parser, rope_spec
from neural_pbf.training.tracking import RunHooks
from neural_pbf.training.train_run import ModelSpec, run_training


def build_parser() -> argparse.ArgumentParser:
    return build_rope_parser("3D-RoPE DiT flow matching in bf16 under Accelerate", use_accelerate=True)


def spec(args: argparse.Namespace) -> ModelSpec:
    return rope_spec(args, overlap=False, use_accelerate=True)


def main(argv: list[str] | None = None, *, hooks: RunHooks | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args, __file__, spec(args), hooks)


if __name__ == "__main__":
    main()
