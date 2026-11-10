"""Unit tests for fm_physics: the physics_loss entry point."""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch

from neural_pbf.data.sampling import randomize_material
from neural_pbf.physics import triton_pde_loss
from neural_pbf.physics.fm_physics import (
    physics_loss,
    physics_loss_reference,
    physics_residual_sums,
)
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


def _call(i: dict[str, Any], form: str, backend: str = "reference") -> torch.Tensor:
    return physics_loss(
        i["v_pred"], i["x_tau"], i["tau"], i["T_in"], i["T_target"], i["Q"], i["mask_in"], i["material"], form,
        T_REF, T_AMB, Q_REF, *GRID, backend=backend,
    )  # fmt: skip


def _sums(i: dict[str, Any], form: str, backend: str = "reference") -> tuple[torch.Tensor, torch.Tensor]:
    T_pred = i["x_tau"] + (1 - i["tau"].view(-1, 1, 1, 1, 1)) * i["v_pred"]
    return physics_residual_sums(
        T_pred, i["T_in"], i["T_target"], i["Q"], i["mask_in"], i["material"], form, T_REF, T_AMB, Q_REF, *GRID,
        backend=backend,
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


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["auto", None, "pytorch"])
def test_physics_loss_refuses_an_unknown_backend(backend: Any) -> None:
    with pytest.raises(ValueError, match="backend"):
        _call(_inputs(), "tin", backend)


@pytest.mark.unit
def test_triton_backend_on_cpu_tensors_is_refused() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        _call(_inputs(), "tin", "triton")


@pytest.mark.unit
@pytest.mark.parametrize("form", ["coupled", "tin"])
def test_residual_sums_pool_to_the_loss(form: str) -> None:
    """Per-sample sums for the val metric: pooled over the batch they give the loss value."""
    i = _inputs()
    num, den = _sums(i, form)
    assert num.shape == den.shape == (2,) and not num.requires_grad
    torch.testing.assert_close(num.sum() / den.sum(), _call(i, form))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_physics_loss_on_cuda_uses_the_triton_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = triton_pde_loss.PDEResidualLoss.apply
    monkeypatch.setattr(triton_pde_loss.PDEResidualLoss, "apply", lambda *a: (calls.append(1), real(*a))[1])
    cpu = _inputs()
    gpu = {k: v.cuda() for k, v in cpu.items()}
    torch.testing.assert_close(_call(gpu, "coupled", "triton").cpu(), _call(cpu, "coupled"), rtol=1e-9, atol=0.0)
    assert calls == [1]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_reference_backend_stays_on_pytorch_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """v4 trains the reference on the GPU: the backend, not the device, chooses the implementation."""
    monkeypatch.setattr(triton_pde_loss.PDEResidualLoss, "apply", lambda *a: pytest.fail("Triton was called"))
    gpu = {k: v.cuda() for k, v in _inputs().items()}
    assert torch.isfinite(_call(gpu, "coupled", "reference"))


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("form", ["coupled", "tin"])
def test_triton_residual_sums_match_the_reference(form: str) -> None:
    cpu = _inputs()
    gpu = {k: v.cuda() for k, v in cpu.items()}
    for got, want in zip(_sums(gpu, form, "triton"), _sums(cpu, form), strict=True):
        torch.testing.assert_close(got.cpu(), want, rtol=1e-9, atol=0.0)
