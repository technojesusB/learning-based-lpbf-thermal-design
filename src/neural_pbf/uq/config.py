"""Frozen configuration of the UQ pipeline; `prereg` is hashed before any test access."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from neural_pbf.data.fm_dataset import _DEFAULT_CONDITIONING_KEYS

METHOD_NAMES = Literal["absolute", "normalized", "constant_single_draw", "raw_ensemble"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GenConfig(_Frozen):
    n_runs: int = 45  # 5 runs per material x pattern cell
    base_seed: int = 1000  # legacy data used 42..51
    materials: str = "SS316L,Ti64,IN718"
    patterns: str = "raster,island,zigzag"
    nx: int = 512
    ny: int = 256
    nz: int = 64
    samples_per_run: int = 50
    min_per_pattern: dict[str, int] = {"zigzag": 5}
    max_retries_per_run: int = 3
    backoff_s: int = 60
    run_timeout_s: int = 10800  # worst planned run: 411 steps at the slowest legacy 8.4 s/step is about 1 h
    wsl_safe: bool = True


class SplitConfig(_Frozen):
    seed: int = 2026
    fractions: dict[str, float] = {"train": 0.45, "val": 0.1, "cal": 0.25, "test": 0.2}
    # 9 calibration runs is the minimum for a finite run-level quantile at alpha=0.1.
    min_runs: dict[str, int] = {"train": 10, "val": 2, "cal": 9, "test": 6}


class TrainConfig(_Frozen):
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 4
    lr: float = 2e-4
    seed: int = 42
    patch_size: int = 64
    base_channels: int = 32
    depth: int = 3
    cond_embed_dim: int = 128
    mask_source: Literal["pre"] = "pre"
    # Run 2: run 1 lacked T_in and exposure_time, so the model could not see residual heat or energy dose.
    model_name: str = "baseline_tin_exp"
    in_channels: Literal[3, 4] = 4
    conditioning_keys: tuple[str, ...] = _DEFAULT_CONDITIONING_KEYS


class ExploratoryConfig(_Frozen):
    """Analyses added after the first test evaluation; reported as exploratory, never as pre-registered.

    T_boil at 1 atm, model values from LPBF simulation literature (not measured; alloys have no sharp
    boiling point): SS316L 3090 K (3086-3090; arXiv 2201.02614, 2603.28811), IN718 3188 K (3120-3223;
    Mater. Des. S0264127523009504), Ti64 3315 K (3533-3560 outliers, pure Ti about 3560; arXiv 2109.09480).
    The solver has no evaporation, so temperatures above T_boil are outside its validity (keyhole regime).
    """

    T_boil_K: dict[str, float] = {"SS316L": 3090.0, "IN718": 3188.0, "Ti64": 3315.0}
    # Usefulness criterion fixed before run 2, in the melt regime at the primary alpha.
    criterion_min_coverage: float = 0.85
    criterion_max_width: dict[str, float] = {"depth_vox": 25.0, "T_max_K": 1000.0}


class PreregConfig(_Frozen):
    K: int = 16
    alphas: tuple[float, ...] = (0.2, 0.1, 0.05)
    primary_method: METHOD_NAMES = "absolute"
    primary_alpha: float = 0.1
    sampling_seed: int = 7001
    second_sampling_seed: int = 7002
    bootstrap_seed: int = 7003
    n_boot: int = 2000
    n_euler_steps: int = 25
    sensitivity_Ks: tuple[int, ...] = (4, 8, 16)
    second_seed_fraction: float = 0.1
    T_max_bins_K: tuple[float, ...] = (2000.0, 3000.0, 5000.0)


class RunConfig(_Frozen):
    draw_batch: int = 4
    chunk_size: int = 25
    min_free_gb_generate: float = 60.0  # 45 runs x about 0.75 GB, plus margin
    min_free_gb_sampling: float = 5.0
    max_stage_retries: int = 3
    retry_backoff_s: int = 60
    stall_timeout_s: int = 7200  # no progress for this long in S4 or S5 counts as a hung GPU
    device: str = "cuda"
    legacy_checkpoint: str = "checkpoints/fm/best.pt"
    legacy_sample_limit: int | None = None  # None = every sample of each set
    mlflow_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "uq_conformal"


class UQConfig(_Frozen):
    mode: Literal["full", "smoke"] = "full"
    root: str = "artifacts/uq"
    data_dir: str = "data/uq_v1"
    report_path: str = "docs/uq_report.md"
    figure_dir: str = "docs/assets/uq"
    log_dir: str = "logs/uq"
    gen: GenConfig = GenConfig()
    split: SplitConfig = SplitConfig()
    train: TrainConfig = TrainConfig()
    prereg: PreregConfig = PreregConfig()
    exploratory: ExploratoryConfig = ExploratoryConfig()
    run: RunConfig = RunConfig()

    @classmethod
    def full(cls) -> UQConfig:
        return cls()

    @classmethod
    def smoke(cls, root: str | Path | None = None) -> UQConfig:
        """Whole chain on a tiny grid.

        Uses 4 runs of one material: four disjoint sets need at least 4 runs, and every
        calibration or test material must also be in train. UQ_SMOKE_ROOT moves the output, also for
        stage children, so tests leave artifacts/uq_smoke alone.
        """
        root = str(root or os.environ.get("UQ_SMOKE_ROOT", "artifacts/uq_smoke"))
        return cls(
            mode="smoke",
            root=root,
            data_dir=f"{root}/data",
            report_path=f"{root}/uq_report.md",
            figure_dir=f"{root}/figures",
            log_dir=f"{root}/logs",
            gen=GenConfig(
                n_runs=4,
                base_seed=9000,
                materials="SS316L",
                patterns="raster,zigzag",
                nx=96,
                ny=64,
                nz=16,
                samples_per_run=6,
                min_per_pattern={},
                max_retries_per_run=1,
                backoff_s=2,
                wsl_safe=True,
            ),
            split=SplitConfig(
                seed=1,
                fractions={"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25},
                min_runs={"train": 1, "val": 1, "cal": 1, "test": 1},
            ),
            train=TrainConfig(max_epochs=2, patience=2, batch_size=2, base_channels=8, cond_embed_dim=32),
            prereg=PreregConfig(K=2, sensitivity_Ks=(2,), n_euler_steps=4, n_boot=100, second_seed_fraction=0.2),
            run=RunConfig(
                chunk_size=3,
                min_free_gb_generate=1.0,
                min_free_gb_sampling=0.2,
                max_stage_retries=1,
                retry_backoff_s=2,
                legacy_sample_limit=3,
                mlflow_uri=f"sqlite:///{root}/mlflow.db",
                mlflow_experiment="uq_conformal_smoke",
            ),
        )
