"""The ablation layer: the fixed ablation experiments and how the queue runner runs their scripts pinned.

The experiment scripts (experiments/train_fm_*.py) are standalone training entry points and know nothing of this
module (tests/test_import_layering.py). An ablation experiment is a script plus a pinned recipe, the physics form and
backend, the values that define it (``required``) and the chapter through which it enters the ablation. The queue
runner builds the script's arguments from the recipe (``argv_for``) and passes ``hooks_for(name)``, which validates the
parsed arguments against the recipe and pins the run: MLflow experiment lpbf_ablation_rerun, run name and provenance
key ``ablation_experiment``. A standalone run is never pinned, and the ablation evaluation refuses unpinned
checkpoints, so it can never slip into the study.

Each step of the chain changes exactly one thing, except the two named recipe steps (decided 2026-11-10). The
historical recipes were inconsistent (other data sizes, v1 partly unrecoverable), so they are kept in ``note`` and in
the report as "historical recipe, not used". All experiments train 100 epochs: on today's 1040 train samples that is
about twice the sample-steps of the old 500-epoch runs on 105 samples.

    dit_v1_ape -> dit_v2_sincos -> dit_v3_rope      the position encoding only (learned APE, sincos, RoPE)
    dit_v3_rope -> dit_v5_none_triton               named step "scale_up": patch 4, bf16 + Accelerate, OneCycle,
                                                    batch 2, no tv
    dit_v5_none_triton -> dit_v5_coupled / _tin     the physics loss, then its form
    dit_v5_coupled_triton -> dit_v6_overlap_*       overlapping unpatchify
    dit_v5_coupled_triton vs dit_v4_coupled_reference   the implementation of the same loss (PyTorch vs Triton)
    unet_baseline (UQ run 2) -> unet_none           named step "recipe": depth, batch, lr, schedule, precision,
                                                    early stop
    unet_none -> unet_physics_coupled / _tin        the physics loss, then its form
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any

from neural_pbf.training.tracking import ABLATION_EXPERIMENT, DEBUG_EXPERIMENT, RunHooks

ROOT = Path(__file__).resolve().parents[3]

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class AblationExperiment:
    name: str
    chapter: str
    script: str
    physics_form: str = "none"
    physics_backend: str | None = None
    required: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    recipe: Mapping[str, Any] = field(default_factory=lambda: _EMPTY)
    note: str = ""


_ONECYCLE = "onecycle_pct0.1"
_UNET = {"epochs": 100, "batch_size": 2, "lr": 1e-4, "seed": 42, "mixed_precision": "bf16", "schedule": _ONECYCLE,
         "base_channels": 32, "depth": 4, "cond_embed_dim": 128}  # fmt: skip
_DIT_ROPE = {"epochs": 100, "batch_size": 2, "lr": 1e-4, "seed": 42, "schedule": _ONECYCLE, "embed_dim": 288,
             "depth": 6, "num_heads": 8, "cond_embed_dim": 128, "patch_size": 4}  # fmt: skip
_DIT_V5 = {**_DIT_ROPE, "mixed_precision": "bf16"}
# One recipe for v1, v2 and v3, from v2's logged 100-epoch run; 288 because 3D-RoPE needs head_dim % 6 == 0.
_DIT_POS = {"epochs": 100, "batch_size": 4, "lr": 1e-4, "seed": 42, "mixed_precision": "no", "schedule": "constant",
            "tv_weight": 0.001, "embed_dim": 288, "depth": 6, "num_heads": 8, "cond_embed_dim": 128,
            "patch_size": 8}  # fmt: skip
_UQ_BASELINE = {"max_epochs": 100, "batch_size": 4, "lr": 2e-4, "base_channels": 32, "depth": 3,
                "cond_embed_dim": 128, "early_stopping": True}  # fmt: skip


_HIST = "historical recipe, not used: "
_HIST_V5 = _HIST + "fm_dit_triton(_tin) 2026-06-05, 500 epochs on 105 samples, otherwise as pinned"
_HIST_UNET = _HIST + "fm_unet_triton(_tin) 2026-06-04/05, 500 epochs on 105 samples, otherwise as pinned"


def _experiment(
    name: str, chapter: str, script: str, recipe: Mapping[str, Any], form: str = "none",
    backend: str | None = None, note: str = "", **required: Any,
) -> AblationExperiment:  # fmt: skip
    return AblationExperiment(name, chapter, f"experiments/{script}", form, backend, MappingProxyType(required),
               MappingProxyType(dict(recipe)), note)  # fmt: skip


EXPERIMENTS: Mapping[str, AblationExperiment] = MappingProxyType({
    a.name: a for a in (
        AblationExperiment("unet_baseline", "baseline", "experiments/uq_conformal.py",
            recipe=MappingProxyType(_UQ_BASELINE),
            note="trained by the UQ pipeline (S4, run 2); not retrained by a training script"),
        _experiment("unet_none", "recipe", "train_fm_unet_triton.py", _UNET, note=_HIST_UNET),
        _experiment("unet_physics_coupled", "physics_loss", "train_fm_unet_triton.py", _UNET, "coupled", "triton",
             note=_HIST_UNET),
        _experiment("unet_physics_tin", "physics_loss", "train_fm_unet_triton.py", _UNET, "tin", "triton",
                    note=_HIST_UNET),
        _experiment("dit_v1_ape", "position_encoding", "train_fm_dit.py", _DIT_POS, pos_embed="learned",
             note=_HIST + "benchmark_dit_v1_150samples 2026-05-08: batch 2, lr 1e-4, 100 epochs, 4^3 patches; model "
             "size, tv, schedule and precision were not logged and the code predates git"),
        _experiment("dit_v2_sincos", "position_encoding", "train_fm_dit.py", _DIT_POS, pos_embed="sincos",
             note=_HIST + "benchmark_dit_v2_150samples 2026-05-09: embed 256, otherwise as pinned"),
        _experiment("dit_v3_rope", "position_encoding", "train_fm_dit_rope.py", _DIT_POS,
             note=_HIST + "benchmark_dit_v3_150samples 2026-05-09: batch 4, lr 3e-4, 500 epochs (best at 247), "
             "tv 0.001, embed 288, patch 4, OneCycle, fp32"),
        _experiment("dit_v4_coupled_reference", "implementation", "train_fm_dit_accelerate.py", _DIT_V5, "coupled",
             "reference", note=_HIST + "fm_dit_accelerate 2026-05-10: batch 4, 250 epochs, otherwise as pinned"),
        _experiment("dit_v5_none_triton", "scale_up", "train_fm_dit_triton.py", _DIT_V5, note=_HIST_V5),
        _experiment("dit_v5_coupled_triton", "physics_loss", "train_fm_dit_triton.py", _DIT_V5, "coupled", "triton",
             note=_HIST_V5),
        _experiment("dit_v5_tin_triton", "physics_loss", "train_fm_dit_triton.py", _DIT_V5, "tin", "triton",
             note=_HIST_V5),
        _experiment("dit_v6_overlap_coupled_triton", "architecture", "train_fm_dit_tri_overlap_unpatch.py",
             {**_DIT_V5, "overlap_unpatchify": True}, "coupled", "triton"),
    )
})  # fmt: skip


# The queue order: the U-Net chapter, the physics chapter, then the position-encoding chapter; v4 (the
# implementation comparison) and v1 last, as the least informative if GPU time runs out.
QUEUE: tuple[str, ...] = (
    "unet_none", "unet_physics_coupled", "unet_physics_tin", "dit_v5_none_triton", "dit_v5_coupled_triton",
    "dit_v5_tin_triton", "dit_v6_overlap_coupled_triton", "dit_v3_rope", "dit_v2_sincos", "dit_v4_coupled_reference",
    "dit_v1_ape",
)  # fmt: skip


def resolve_experiment(
    name: str, script: str | Path, args: argparse.Namespace, fixed: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AblationExperiment:  # fmt: skip
    """The ablation experiment, if this script and these arguments train it; otherwise ValueError.

    ``fixed`` holds the recipe values the script does not take from the command line (schedule, precision of a
    script without Accelerate, regularization weights); every recipe key must come from ``args`` or ``fixed``.
    ``overrides``: the queue runner's test mode replaces recipe values (tiny models on the CPU).
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown ablation experiment {name!r}; registered: {sorted(EXPERIMENTS)}")
    exp = EXPERIMENTS[name]
    if Path(script).name != Path(exp.script).name:
        raise ValueError(f"{name!r} is trained by the script {exp.script}, not by {Path(script).name}")
    identity = {"physics_form": exp.physics_form, "physics_backend": exp.physics_backend, **exp.required}
    for key, value in identity.items():
        got = getattr(args, key, None)
        if got != value:
            raise ValueError(f"{name!r} needs {key}={value!r}, got {got!r}")
    fixed = fixed or {}
    for key, value in {**exp.recipe, **(overrides or {})}.items():
        if hasattr(args, key):
            got = getattr(args, key)
        elif key in fixed:
            got = fixed[key]
        else:
            raise ValueError(f"{name!r} pins {key}={value!r}, but the script provides no {key}")
        if got != value:
            raise ValueError(f"{name!r} pins {key}={value!r}, got {got!r}")
    return exp


def load_script(script: str | Path) -> ModuleType:
    """Import an experiment script (``experiments/...``) as a module: ``build_parser``, ``spec``, ``main``."""
    path = Path(script) if Path(script).is_absolute() else ROOT / script
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def argv_for(name: str, parser: argparse.ArgumentParser, overrides: Mapping[str, Any] | None = None) -> list[str]:
    """The script arguments of an ablation experiment: its identity and recipe, as far as the script takes them.

    Values the script fixes itself (schedule, TV weight, precision without Accelerate) are not options; the pinned
    check compares them through ``spec(args).recipe_fixed()``.
    """
    exp = EXPERIMENTS[name]
    options = {s for action in parser._actions for s in action.option_strings}
    values = {"physics_form": exp.physics_form, "physics_backend": exp.physics_backend, **exp.required, **exp.recipe,
              **(overrides or {})}  # fmt: skip
    return [x for k, v in values.items() if v is not None and f"--{k}" in options for x in (f"--{k}", str(v))]


def hooks_for(name: str, overrides: Mapping[str, Any] | None = None) -> RunHooks:
    """Pin a script run to the ablation experiment ``name``.

    With ``overrides`` (the queue runner's test mode) the run is still checked against the recipe it was given, but
    marked: provenance key ``test_overrides`` and the MLflow experiment lpbf_debug, never lpbf_ablation_rerun.
    """
    exp = EXPERIMENTS[name]
    record: dict[str, Any] = {
        "ablation_experiment": name, "chapter": exp.chapter, "physics_form": exp.physics_form,
        "physics_backend": exp.physics_backend, "required": dict(exp.required), "recipe": dict(exp.recipe),
    }  # fmt: skip

    if overrides:
        record["test_overrides"] = dict(overrides)

    def validate(args: argparse.Namespace, fixed: Mapping[str, Any]) -> None:
        resolve_experiment(name, exp.script, args, fixed, overrides)

    experiment = DEBUG_EXPERIMENT if overrides else ABLATION_EXPERIMENT
    return RunHooks(name=name, mlflow_experiment=experiment, pinned=True, record=record, validate=validate)
