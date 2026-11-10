"""train_fm_dit_accelerate.py — the 3D-RoPE DiT in bf16 under Accelerate, with the PyTorch physics loss.

The same model and run as train_fm_dit_triton.py; meant for the PyTorch reference implementation of the
physics loss (--physics_backend reference), which runs on any device. Before 2026-11-09 this script always
trained with the old residual (dt fixed at 5 µs, scalar k) and had no switch. Run on its own, it logs to
the MLflow experiment lpbf_experiments, unpinned.

Run:
    uv run accelerate launch experiments/train_fm_dit_accelerate.py --physics_form coupled --physics_backend reference

In the ablation: experiment dit_v4_coupled_reference (training/ablation.py).
"""

from __future__ import annotations

import argparse

from neural_pbf.training.rope_model import build_rope_parser, rope_spec
from neural_pbf.training.tracking import RunHooks
from neural_pbf.training.train_run import ModelSpec, run_training


def build_parser() -> argparse.ArgumentParser:
    return build_rope_parser("3D-RoPE DiT flow matching with the reference physics loss", use_accelerate=True)


def spec(args: argparse.Namespace) -> ModelSpec:
    return rope_spec(args, overlap=False, use_accelerate=True)


def main(argv: list[str] | None = None, *, hooks: RunHooks | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args, __file__, spec(args), hooks)


if __name__ == "__main__":
    main()
