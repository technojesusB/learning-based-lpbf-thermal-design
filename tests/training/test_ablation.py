"""The ablation layer: the fixed experiments, their pinned recipes, and the arguments and hooks it runs a script with.

The scripts know nothing of it (tests/test_import_layering.py); here every queued experiment's argv is built from its
recipe and passed through the script's own parser and the pinned check, as the queue runner's --dry-run does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from neural_pbf.training.ablation import EXPERIMENTS, QUEUE, argv_for, hooks_for, load_script, resolve_experiment
from neural_pbf.training.tracking import ABLATION_EXPERIMENT

EXPECTED = {
    "unet_baseline", "unet_none", "unet_physics_coupled", "unet_physics_tin",
    "dit_v1_ape", "dit_v2_sincos", "dit_v3_rope", "dit_v4_coupled_reference",
    "dit_v5_none_triton", "dit_v5_coupled_triton", "dit_v5_tin_triton", "dit_v6_overlap_coupled_triton",
}  # fmt: skip

UNET = {"epochs": 100, "batch_size": 2, "lr": 1e-4, "seed": 42, "mixed_precision": "bf16",
        "schedule": "onecycle_pct0.1", "base_channels": 32, "depth": 4, "cond_embed_dim": 128}
DIT_ROPE = {"epochs": 100, "batch_size": 2, "lr": 1e-4, "seed": 42, "schedule": "onecycle_pct0.1",
            "embed_dim": 288, "depth": 6, "num_heads": 8, "cond_embed_dim": 128, "patch_size": 4}
DIT_POS = {"epochs": 100, "batch_size": 4, "lr": 1e-4, "seed": 42, "mixed_precision": "no", "schedule": "constant",
           "tv_weight": 0.001, "embed_dim": 288, "depth": 6, "num_heads": 8, "cond_embed_dim": 128, "patch_size": 8}
ROOT = Path(__file__).resolve().parents[2]


def _args(recipe: dict | None = None, **kw) -> argparse.Namespace:
    base = {"physics_form": "none", "physics_backend": None}
    return argparse.Namespace(**{**base, **(recipe or {}), **kw})


def _cli(recipe: dict) -> dict:
    """The recipe values a script takes from the command line; the rest it declares as fixed."""
    return {k: v for k, v in recipe.items() if k not in ("schedule", "mixed_precision", "tv_weight")}


def _fixed(recipe: dict) -> dict:
    return {k: v for k, v in recipe.items() if k in ("schedule", "mixed_precision", "tv_weight")}


@pytest.mark.unit
def test_registry_has_exactly_the_planned_experiments() -> None:
    assert set(EXPERIMENTS) == EXPECTED
    assert all(exp.name == name for name, exp in EXPERIMENTS.items())


@pytest.mark.unit
def test_experiments_are_consistent() -> None:
    for exp in EXPERIMENTS.values():
        assert exp.physics_form in ("none", "coupled", "tin")
        assert (exp.physics_backend is None) == (exp.physics_form == "none")
        assert exp.physics_backend in (None, "reference", "triton")
        assert exp.chapter


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "script", "form", "backend"),
    [
        ("unet_none", "train_fm_unet_triton.py", "none", None),
        ("dit_v5_none_triton", "train_fm_dit_triton.py", "none", None),
        ("dit_v3_rope", "train_fm_dit_rope.py", "none", None),
        ("dit_v4_coupled_reference", "train_fm_dit_accelerate.py", "coupled", "reference"),
        ("dit_v5_coupled_triton", "train_fm_dit_triton.py", "coupled", "triton"),
        ("dit_v5_tin_triton", "train_fm_dit_triton.py", "tin", "triton"),
        ("dit_v6_overlap_coupled_triton", "train_fm_dit_tri_overlap_unpatch.py", "coupled", "triton"),
        ("unet_physics_coupled", "train_fm_unet_triton.py", "coupled", "triton"),
        ("unet_physics_tin", "train_fm_unet_triton.py", "tin", "triton"),
    ],
)
def test_the_planned_mapping(name: str, script: str, form: str, backend: str | None) -> None:
    exp = EXPERIMENTS[name]
    assert (exp.script, exp.physics_form, exp.physics_backend) == (f"experiments/{script}", form, backend)


@pytest.mark.unit
def test_v1_v2_v3_differ_only_in_the_position_encoding() -> None:
    v1, v2, v3 = EXPERIMENTS["dit_v1_ape"], EXPERIMENTS["dit_v2_sincos"], EXPERIMENTS["dit_v3_rope"]
    assert v1.script == v2.script == "experiments/train_fm_dit.py"
    assert dict(v1.required) == {"pos_embed": "learned"} and dict(v2.required) == {"pos_embed": "sincos"}
    assert dict(v3.required) == {}
    assert dict(v1.recipe) == dict(v2.recipe) == dict(v3.recipe) == DIT_POS
    assert {a.chapter for a in (v1, v2, v3)} == {"position_encoding"}


@pytest.mark.unit
def test_the_named_recipe_steps_and_the_implementation_pair() -> None:
    assert EXPERIMENTS["dit_v5_none_triton"].chapter == "scale_up"  # v3 -> v5_none: patch 4, bf16, OneCycle, batch 2, no tv
    assert EXPERIMENTS["unet_none"].chapter == "recipe"  # unet_baseline -> unet_none
    assert dict(EXPERIMENTS["dit_v4_coupled_reference"].recipe) == dict(EXPERIMENTS["dit_v5_coupled_triton"].recipe)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["dit_v1_ape", "dit_v2_sincos", "dit_v3_rope", "dit_v4_coupled_reference",
                                  "dit_v5_coupled_triton", "unet_none"])  # fmt: skip
def test_historical_recipes_are_recorded_as_not_used(name: str) -> None:
    assert "historical recipe, not used" in EXPERIMENTS[name].note


QUEUE_ORDER = ("unet_none", "unet_physics_coupled", "unet_physics_tin", "dit_v5_none_triton", "dit_v5_coupled_triton",
               "dit_v5_tin_triton", "dit_v6_overlap_coupled_triton", "dit_v3_rope", "dit_v2_sincos",
               "dit_v4_coupled_reference", "dit_v1_ape")  # fmt: skip


@pytest.mark.unit
def test_the_queue_order_covers_every_retrained_experiment() -> None:
    assert QUEUE == QUEUE_ORDER
    assert set(QUEUE) == set(EXPERIMENTS) - {"unet_baseline"}


def _parse(name: str):
    module = load_script(EXPERIMENTS[name].script)
    parser = module.build_parser()
    args = parser.parse_args([*argv_for(name, parser), "--manifest", "m.json"])
    return module, args


@pytest.mark.unit
@pytest.mark.parametrize("name", QUEUE_ORDER)
def test_every_queued_experiment_runs_its_script_with_the_pinned_recipe(name: str) -> None:
    module, args = _parse(name)
    hooks = hooks_for(name)
    assert hooks.validate is not None
    hooks.validate(args, module.spec(args).recipe_fixed())  # the pinned check passes for the built argv
    assert hooks.pinned and hooks.name == name and hooks.mlflow_experiment == ABLATION_EXPERIMENT
    assert hooks.record["ablation_experiment"] == name and hooks.record["chapter"] == EXPERIMENTS[name].chapter
    for key, value in {**EXPERIMENTS[name].required, **EXPERIMENTS[name].recipe}.items():
        if hasattr(args, key):
            assert getattr(args, key) == value, key


@pytest.mark.unit
def test_the_hooks_refuse_args_that_leave_the_recipe() -> None:
    module, args = _parse("dit_v3_rope")
    args.lr = 3e-4
    with pytest.raises(ValueError, match="lr"):
        hooks_for("dit_v3_rope").validate(args, module.spec(args).recipe_fixed())  # type: ignore[misc]


@pytest.mark.unit
def test_the_argv_holds_only_options_the_script_has() -> None:
    parser = load_script(EXPERIMENTS["dit_v3_rope"].script).build_parser()
    argv = argv_for("dit_v3_rope", parser)
    assert "--schedule" not in argv and "--tv_weight" not in argv and "--mixed_precision" not in argv
    assert argv[argv.index("--batch_size") + 1] == "4" and argv[argv.index("--patch_size") + 1] == "8"


@pytest.mark.unit
def test_the_pinned_recipes() -> None:
    for name in ("unet_none", "unet_physics_coupled", "unet_physics_tin"):
        assert dict(EXPERIMENTS[name].recipe) == UNET
    for name in ("dit_v4_coupled_reference", "dit_v5_none_triton", "dit_v5_coupled_triton", "dit_v5_tin_triton"):
        assert dict(EXPERIMENTS[name].recipe) == {**DIT_ROPE, "mixed_precision": "bf16"}
    assert dict(EXPERIMENTS["dit_v6_overlap_coupled_triton"].recipe) == {**DIT_ROPE, "mixed_precision": "bf16",
                                                                  "overlap_unpatchify": True}  # fmt: skip
    assert EXPERIMENTS["unet_baseline"].recipe["depth"] == 3  # the UQ pipeline's recipe, for the report only


@pytest.mark.unit
def test_resolve_accepts_a_matching_run() -> None:
    recipe = {**DIT_ROPE, "mixed_precision": "bf16"}
    args = _args(_cli(recipe), physics_form="tin", physics_backend="triton")
    exp = resolve_experiment("dit_v5_tin_triton", "/any/where/experiments/train_fm_dit_triton.py", args, _fixed(recipe))
    assert exp is EXPERIMENTS["dit_v5_tin_triton"]


@pytest.mark.unit
@pytest.mark.parametrize(("key", "value"), [("lr", 2e-4), ("epochs", 50), ("depth", 2), ("batch_size", 8)])
def test_a_deviating_recipe_is_refused(key: str, value) -> None:
    with pytest.raises(ValueError, match=key):
        resolve_experiment("unet_none", "experiments/train_fm_unet_triton.py",
                    _args({**_cli(UNET), key: value}), fixed=_fixed(UNET))  # fmt: skip


@pytest.mark.unit
def test_a_script_declaring_another_schedule_is_refused() -> None:
    with pytest.raises(ValueError, match="schedule"):
        resolve_experiment("unet_none", "experiments/train_fm_unet_triton.py", _args(_cli(UNET)),
                    fixed={**_fixed(UNET), "schedule": "constant"})  # fmt: skip


@pytest.mark.unit
def test_a_recipe_value_the_script_does_not_provide_is_an_error() -> None:
    with pytest.raises(ValueError, match="mixed_precision"):
        resolve_experiment("unet_none", "experiments/train_fm_unet_triton.py", _args(_cli(UNET)),
                    fixed={"schedule": "onecycle_pct0.1"})  # fmt: skip


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "script", "kw", "match"),
    [
        ("dit_v9", "train_fm_dit_triton.py", {}, "unknown ablation experiment"),
        ("dit_v5_tin_triton", "train_fm_unet_triton.py", {"physics_form": "tin", "physics_backend": "triton"},
         "script"),
        ("dit_v5_tin_triton", "train_fm_dit_triton.py", {"physics_form": "coupled", "physics_backend": "triton"},
         "physics_form"),
        ("dit_v5_tin_triton", "train_fm_dit_triton.py", {"physics_form": "tin", "physics_backend": "reference"},
         "physics_backend"),
        ("dit_v1_ape", "train_fm_dit.py", {"pos_embed": "sincos"}, "pos_embed"),
        ("unet_baseline", "train_fm_unet_triton.py", {}, "script"),
    ],
)  # fmt: skip
def test_resolve_refuses_a_mismatch(name: str, script: str, kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_experiment(name, f"experiments/{script}", _args(**kw), fixed={})


@pytest.mark.unit
@pytest.mark.parametrize("name", ["dit_v2_sincos", "dit_v3_rope"])
def test_the_position_encoding_experiments_train_with_the_tv_term(name: str) -> None:
    import torch

    from neural_pbf.models.generative.fm.flow import fm_loss
    from neural_pbf.training.train_run import tv_loss

    module, args = _parse(name)
    spec = module.spec(args)
    v, noise, target = (torch.randn(2, 1, 8, 8, 8, generator=torch.Generator().manual_seed(i)) for i in range(3))
    expected = fm_loss(v, noise, target) + 0.001 * tv_loss(v)
    torch.testing.assert_close(spec.data_loss(v, noise, target), expected)
    assert spec.schedule == "constant" and not spec.use_accelerate
