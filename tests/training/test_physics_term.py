"""The physics term of the training scripts: loss, ReLoBRaLo combination, val residual, loaders, pre-flight."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.data.patch_dataset import PatchFMThermalDataset
from neural_pbf.data.sampling import randomize_material
from neural_pbf.physics import triton_pde_loss
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.physics.material_vector import encode
from neural_pbf.training.physics_term import PhysicsTerm, ValResidual, preflight, with_material
from tests.data.test_material_dataset import _write_run

DS = SimpleNamespace(T_ref=2000.0, T_ambient=300.0, Q_ref=1e12)
GRID = {"dx_m": 2e-6, "dy_m": 2.5e-6, "dz_m": 3e-6}


def _args(form: str = "tin", backend: str | None = "reference", **kw) -> argparse.Namespace:
    base = {"physics_form": form, "physics_backend": backend, "relobralo_alpha": 0.0, "relobralo_beta": 0.0,
            "relobralo_temperature": 0.1, "relobralo_epsilon": 1e-8, "max_lambda": 1.0}  # fmt: skip
    return argparse.Namespace(**{**base, **kw})


def _vec(mat: MaterialConfig) -> torch.Tensor:
    return encode(
        {"rho": mat.rho, "L": mat.latent_heat_L, "T_s": mat.T_solidus, "T_l": mat.T_liquidus, "k_p": mat.k_powder,
         "k_s": mat.k_solid, "k_l": mat.k_liquid, "cp": mat.cp_base, "exposure_time": 50e-6},
        {"T_lut": mat.T_lut, "k_lut": mat.k_lut, "cp_lut": mat.cp_lut},
    )  # fmt: skip


def _batch(T_peak: tuple[float, float] = (2000.0, 3500.0)) -> dict[str, torch.Tensor]:
    """Two SS316L samples (T_s about 1653 K, T_boil 3090 K): normalized fields, material, boiling point."""
    random.seed(0)
    g = torch.Generator().manual_seed(0)
    shape = (2, 1, 5, 6, 7)
    mat = randomize_material(MaterialConfig.ss316l_preset())
    T_in = 0.1 + 0.2 * torch.rand(shape, generator=g, dtype=torch.float64)
    peak = torch.tensor([(p - DS.T_ambient) / DS.T_ref for p in T_peak], dtype=torch.float64).view(2, 1, 1, 1, 1)
    T_target = T_in + (peak - T_in) * torch.rand(shape, generator=g, dtype=torch.float64)
    T_target[:, 0, 2, 3, 3] = peak.flatten()  # the true T_max of each sample
    return {
        "T_in": T_in, "T_target": T_target, "Q": 1e3 * torch.rand(shape, generator=g, dtype=torch.float64),
        "mask": (torch.rand(shape, generator=g) > 0.5).double(), "material": torch.stack([_vec(mat), _vec(mat)]),
        "T_boil": torch.tensor([3090.0, 3090.0], dtype=torch.float64),
    }  # fmt: skip


def _model_outputs(batch: dict, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(batch["T_in"].shape, generator=g, dtype=torch.float64)
    x_tau = torch.randn(batch["T_in"].shape, generator=g, dtype=torch.float64)
    tau = torch.rand(2, generator=g, dtype=torch.float64)
    return v, x_tau, tau


# --- PhysicsTerm --------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_form_none_never_calls_the_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    from neural_pbf.training import physics_term as pt

    monkeypatch.setattr(pt, "physics_loss", lambda *a, **k: pytest.fail("physics loss called with form none"))
    term = PhysicsTerm.from_args(_args("none", None), DS, GRID)
    b = _batch()
    v, x_tau, tau = _model_outputs(b)
    loss_phys = term.loss(v, x_tau, tau, b)
    total, metrics = term.combine(torch.tensor(0.5, dtype=torch.float64), loss_phys)
    assert float(total) == 0.5 and metrics["loss_phys"] == 0.0 and metrics["lambda_phys"] == 0.0


@pytest.mark.unit
def test_loss_uses_the_batch_material_and_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from neural_pbf.training import physics_term as pt

    seen = {}
    monkeypatch.setattr(pt, "physics_loss", lambda *a, **k: (seen.update(k, form=a[8]), torch.tensor(1.0))[1])
    b = _batch()
    PhysicsTerm.from_args(_args("tin", "reference"), DS, GRID).loss(*_model_outputs(b), b)
    assert seen["form"] == "tin" and seen["backend"] == "reference"


@pytest.mark.unit
def test_combine_applies_the_capped_relobralo_weight() -> None:
    term = PhysicsTerm.from_args(_args(), DS, GRID)
    one = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    term.combine(one, one)  # first step: references
    loss_fm = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
    loss_phys = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)  # stagnates -> raw weight near 2
    total, m = term.combine(loss_fm, loss_phys)
    assert m["lambda_raw"] > 1.9 and m["lambda_phys"] == 1.0 and m["cap_hit"] == 1.0
    torch.testing.assert_close(total, loss_fm + loss_phys)
    total.backward()
    assert loss_phys.grad is not None and float(loss_phys.grad) == 1.0


@pytest.mark.unit
def test_combine_reduces_before_weighting() -> None:
    """Every process must feed ReLoBRaLo the same (gathered) losses, or the weights drift apart."""
    seen = []
    term = PhysicsTerm.from_args(_args(), DS, GRID)
    term.combine(torch.tensor(1.0), torch.tensor(2.0), reduce=lambda t: (seen.append(float(t)), t * 10)[1])
    assert seen == [1.0, 2.0] and term.weighter._init == (10.0, 20.0)


@pytest.mark.unit
def test_physics_on_without_backend_is_refused() -> None:
    with pytest.raises(ValueError, match="backend"):
        PhysicsTerm.from_args(_args("tin", None), DS, GRID)


# --- ValResidual ----------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_val_residual_logs_both_forms_overall_and_melt() -> None:
    """Sample 0 peaks at 2000 K (melt), sample 1 at 3500 K (keyhole); melt pools only sample 0."""
    from neural_pbf.physics.fm_physics import physics_residual_sums

    b = _batch()
    v, x_tau, tau = _model_outputs(b)
    T_pred = x_tau + (1 - tau.view(-1, 1, 1, 1, 1)) * v
    val = ValResidual(DS, GRID)
    val.update(v, x_tau, tau, b)
    out = val.compute()
    assert set(out) == {"val_phys_tin", "val_phys_tin_melt", "val_phys_coupled", "val_phys_coupled_melt"}
    for form in ("tin", "coupled"):
        fields = (b["T_in"], b["T_target"], b["Q"], b["mask"], b["material"])
        consts = (DS.T_ref, DS.T_ambient, DS.Q_ref, *GRID.values())
        num, den = physics_residual_sums(T_pred, *fields, form, *consts, backend="reference")
        assert out[f"val_phys_{form}"] == pytest.approx(float(num.sum() / den.sum()))
        assert out[f"val_phys_{form}_melt"] == pytest.approx(float(num[0] / den[0]))


@pytest.mark.unit
def test_val_residual_pools_over_batches_and_is_nan_without_melt() -> None:
    b = _batch(T_peak=(3500.0, 3600.0))  # both keyhole
    val = ValResidual(DS, GRID)
    val.update(*_model_outputs(b, 1), b)
    val.update(*_model_outputs(b, 2), b)
    out = val.compute()
    assert math.isfinite(out["val_phys_tin"]) and math.isnan(out["val_phys_tin_melt"])


# --- loaders --------------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_with_material_keeps_the_loader_settings(tmp_path: Path) -> None:
    path = tmp_path / "run_000.h5"
    _write_run(path, MaterialConfig.ss316l_preset(), 40e-6, n=4)
    full = FMThermalDataset(FMDatasetConfig(h5_paths=[str(path)]))
    ds = Subset(PatchFMThermalDataset(full, patch_size=8), [0, 1, 2, 3])
    loader = DataLoader(ds, batch_size=3, shuffle=True, drop_last=True, num_workers=0)
    wrapped = with_material(loader, t_boil={"X": 3090.0})
    assert wrapped.batch_size == 3 and wrapped.drop_last and isinstance(wrapped.sampler, RandomSampler)
    batch = next(iter(wrapped))
    assert batch["material"].shape[0] == 3
    assert torch.equal(batch["T_boil"], torch.full((3,), 3090.0, dtype=torch.float64))


# --- pre-flight -----------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_preflight_is_skipped_for_the_reference_backend() -> None:
    assert preflight("reference", torch.device("cpu"))["preflight"] == "not needed"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_preflight_passes_on_this_gpu() -> None:
    out = preflight("triton", torch.device("cuda"))
    assert out["preflight"] == "passed" and out["preflight_max_rel_err"] < 1e-8


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_preflight_aborts_on_a_wrong_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    real = triton_pde_loss.PDEResidualLoss.apply
    monkeypatch.setattr(triton_pde_loss.PDEResidualLoss, "apply", lambda *a: real(*a) * 1.001)
    with pytest.raises(RuntimeError, match="pre-flight"):
        preflight("triton", torch.device("cuda"))
