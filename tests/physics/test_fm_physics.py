"""Unit tests for fm_physics: the physics_loss entry point and denorm_cond_batch."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from neural_pbf.data.sampling import randomize_material
from neural_pbf.physics import triton_pde_loss
from neural_pbf.physics.fm_physics import denorm_cond_batch, physics_loss, physics_loss_reference
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.physics.material_vector import encode

T_REF, T_AMB, Q_REF = 2000.0, 300.0, 1e12
GRID = (2e-6, 2.5e-6, 3e-6)


def _inputs(device: str = "cpu") -> dict[str, Any]:
    random.seed(0)
    g = torch.Generator().manual_seed(0)
    mat = randomize_material(MaterialConfig.ss316l_preset())
    vec = encode(
        {"rho": mat.rho, "L": mat.latent_heat_L, "T_s": mat.T_solidus, "T_l": mat.T_liquidus, "k_p": mat.k_powder,
         "k_s": mat.k_solid, "k_l": mat.k_liquid, "cp": mat.cp_base, "exposure_time": 50e-6},
        {"T_lut": mat.T_lut, "k_lut": mat.k_lut, "cp_lut": mat.cp_lut},
    )  # fmt: skip
    shape = (2, 1, 5, 6, 7)
    T_in = 0.2 + 0.5 * torch.rand(shape, generator=g, dtype=torch.float64)
    return {
        "v_pred": torch.randn(shape, generator=g, dtype=torch.float64),
        "x_tau": torch.randn(shape, generator=g, dtype=torch.float64),
        "tau": torch.rand(2, generator=g, dtype=torch.float64),
        "T_in": T_in,
        "T_target": T_in + 0.3 * torch.rand(shape, generator=g, dtype=torch.float64),
        "Q": 1e3 * torch.rand(shape, generator=g, dtype=torch.float64),
        "mask_in": (torch.rand(shape, generator=g) > 0.5).double(),
        "material": torch.stack([vec, vec]),
    }


def _call(i: dict[str, Any], form: str) -> torch.Tensor:
    return physics_loss(
        i["v_pred"], i["x_tau"], i["tau"], i["T_in"], i["T_target"], i["Q"], i["mask_in"], i["material"], form,
        T_REF, T_AMB, Q_REF, *GRID,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# physics_loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("form", ["coupled", "tin"])
def test_physics_loss_on_cpu_is_the_reference_on_si_fields(form: str) -> None:
    i = _inputs()
    T_pred = (i["x_tau"] + (1 - i["tau"].view(-1, 1, 1, 1, 1)) * i["v_pred"]) * T_REF + T_AMB
    expected = physics_loss_reference(
        T_pred, i["T_in"] * T_REF + T_AMB, i["T_target"] * T_REF + T_AMB, i["Q"] * Q_REF, i["mask_in"],
        i["material"], form, *GRID,
    )  # fmt: skip
    torch.testing.assert_close(_call(i, form), expected)


@pytest.mark.unit
def test_physics_loss_gradient_reaches_v_pred_but_not_x_tau() -> None:
    i = _inputs()
    i["v_pred"].requires_grad_(True)
    i["x_tau"].requires_grad_(True)
    _call(i, "coupled").backward()
    assert i["v_pred"].grad is not None and i["v_pred"].grad.abs().sum() > 0
    assert i["x_tau"].grad is None


@pytest.mark.unit
def test_physics_loss_refuses_an_unknown_form() -> None:
    with pytest.raises(ValueError, match="form"):
        _call(_inputs(), "none")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_physics_loss_on_cuda_uses_the_triton_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = triton_pde_loss.PDEResidualLoss.apply
    monkeypatch.setattr(triton_pde_loss.PDEResidualLoss, "apply", lambda *a: (calls.append(1), real(*a))[1])
    cpu = _inputs()
    gpu = {k: v.cuda() for k, v in cpu.items()}
    torch.testing.assert_close(_call(gpu, "coupled").cpu(), _call(cpu, "coupled"), rtol=1e-9, atol=0.0)
    assert calls == [1]


# ---------------------------------------------------------------------------
# denorm_cond_batch
# ---------------------------------------------------------------------------


def _fake_ds_cfg() -> Any:
    """SimpleNamespace mimicking FMDatasetConfig normalisation attributes."""
    return SimpleNamespace(
        conditioning_keys=("rho", "cp", "k_s"),
        cond_means={"rho": 7900.0, "cp": 500.0, "k_s": 20.0},
        cond_stds={"rho": 500.0, "cp": 60.0, "k_s": 6.0},
    )


@pytest.mark.unit
def test_denorm_cond_batch_output_shape() -> None:
    B = 4
    cfg = _fake_ds_cfg()
    cond = torch.randn(B, 3)
    out = denorm_cond_batch(cond, cfg, "rho")
    assert out.shape == (B, 1, 1, 1, 1)


@pytest.mark.unit
def test_denorm_cond_batch_roundtrip() -> None:
    """De-normalising a z-score of 0 should return the mean."""
    B = 3
    cfg = _fake_ds_cfg()
    cond = torch.zeros(B, 3)  # all z-scores = 0
    out = denorm_cond_batch(cond, cfg, "rho")
    expected = 7900.0
    assert abs(out.mean().item() - expected) < 1e-3


@pytest.mark.unit
def test_denorm_cond_batch_correct_index_selected() -> None:
    B = 2
    cfg = _fake_ds_cfg()
    # Set only cp channel (idx=1) to z-score 1.0
    cond = torch.zeros(B, 3)
    cond[:, 1] = 1.0
    cp_out = denorm_cond_batch(cond, cfg, "cp")
    # z=1 → cp = 500 + 60*1 = 560
    assert abs(cp_out.mean().item() - 560.0) < 1e-2


@pytest.mark.unit
def test_denorm_cond_batch_broadcast_shape() -> None:
    B = 5
    cfg = _fake_ds_cfg()
    cond = torch.randn(B, 3)
    out = denorm_cond_batch(cond, cfg, "k_s")
    # Shape must be broadcast-compatible with (B, 1, D, H, W) tensors
    dummy_field = torch.randn(B, 1, 8, 8, 8)
    result = dummy_field * out
    assert result.shape == (B, 1, 8, 8, 8)
