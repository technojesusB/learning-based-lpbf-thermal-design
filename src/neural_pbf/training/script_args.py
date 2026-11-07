"""Data arguments shared by every training script, so none can drift back to a leaky default."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_pbf.data.splits import load_manifest

DEFAULT_MANIFEST = "artifacts/uq/manifest.json"


def add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Run-level split manifest; the HDF5 files are read from it (UQ pipeline, stage S3).",
    )
    group.add_argument(
        "--h5",
        nargs="+",
        default=None,
        help="HDF5 files; only with --allow-random-split, or to restrict the manifest's files.",
    )
    group.add_argument(
        "--allow-random-split",
        action="store_true",
        help="Split over samples instead of runs. Leaks neighbouring samples of one run; tests only.",
    )
    group.add_argument("--in-channels", type=int, choices=(3, 4), default=4,
                       help="4 adds T_in, the temperature before the step, as input; 3 is the ablation.")  # fmt: skip


def resolve_data(args: argparse.Namespace) -> tuple[list[str], str | None]:
    """``(h5_paths, manifest_path)``; ``manifest_path`` is None only for an explicit random split."""
    if args.allow_random_split:
        if not args.h5:
            raise SystemExit("--allow-random-split needs --h5")
        return list(args.h5), None
    if not args.manifest:
        raise SystemExit("no manifest given; pass --manifest, or --allow-random-split with --h5")
    if not Path(args.manifest).exists():
        raise SystemExit(f"manifest {args.manifest} not found; run the UQ pipeline up to S3 or pass --manifest")
    manifest = load_manifest(args.manifest)
    paths = sorted({p for runs in manifest["sets"].values() for r in runs for p, _ in r["sample_keys"]})
    if args.h5:
        paths = [p for p in paths if p in set(args.h5)]
    return paths, args.manifest
