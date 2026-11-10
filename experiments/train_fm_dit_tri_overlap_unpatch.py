"""train_fm_dit_tri_overlap_unpatch.py — the 3D-RoPE DiT with overlapping unpatchify.

As train_fm_dit_triton.py, but the unpatchify ends in a 3×3×3 convolution over neighbouring patches
instead of the zero-overlap transposed conv, against checkerboard noise. Run on its own, it logs to the
MLflow experiment lpbf_experiments, unpinned.

Run:
    uv run accelerate launch experiments/train_fm_dit_tri_overlap_unpatch.py \\
        --physics_form coupled --physics_backend triton

In the ablation: experiment dit_v6_overlap_coupled_triton (training/ablation.py).
"""

from __future__ import annotations

import argparse

from neural_pbf.training.rope_model import build_rope_parser, rope_spec
from neural_pbf.training.tracking import RunHooks
from neural_pbf.training.train_run import ModelSpec, run_training


def build_parser() -> argparse.ArgumentParser:
    return build_rope_parser("3D-RoPE DiT flow matching with overlapping unpatchify", use_accelerate=True)


def spec(args: argparse.Namespace) -> ModelSpec:
    return rope_spec(args, overlap=True, use_accelerate=True)


def main(argv: list[str] | None = None, *, hooks: RunHooks | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_training(args, __file__, spec(args), hooks)


if __name__ == "__main__":
    main()
