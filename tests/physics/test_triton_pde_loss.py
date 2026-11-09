"""Triton kernels for the physics loss, checked against the PyTorch reference and plain autograd.

GPU only (marked ``slow``): ``uv run pytest tests/physics/test_triton_pde_loss.py -m slow``.
Temperatures are drawn away from table nodes and T_solidus, where the derivative jumps.
"""

from __future__ import annotations

import random

import pytest
import torch

from neural_pbf.data.sampling import randomize_material
from neural_pbf.physics.fm_physics import enthalpy
from neural_pbf.physics.material import MaterialConfig, cp_eff, k_eff
from neural_pbf.physics.material_vector import encode, material_from_vector

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a CUDA device"),
]
CUDA = torch.device("cuda")


def _vector(mat: MaterialConfig, exposure: float) -> torch.Tensor:
    scalars = {"rho": mat.rho, "L": mat.latent_heat_L, "T_s": mat.T_solidus, "T_l": mat.T_liquidus,
               "k_p": mat.k_powder, "k_s": mat.k_solid, "k_l": mat.k_liquid, "cp": mat.cp_base,
               "exposure_time": exposure}  # fmt: skip
    return encode(scalars, {"T_lut": mat.T_lut, "k_lut": mat.k_lut, "cp_lut": mat.cp_lut})


def _materials(seed: int = 0) -> torch.Tensor:
    """SS316L (6-point tables) and Ti64 (7-point tables), randomized as the generator does."""
    random.seed(seed)
    mats = [randomize_material(MaterialConfig.ss316l_preset()), randomize_material(MaterialConfig.ti64_preset())]
    return torch.stack([_vector(mats[0], 40e-6), _vector(mats[1], 70e-6)])


def _away_from_kinks(T: torch.Tensor, material: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """Shift temperatures that sit within `margin` K of a table node or T_solidus."""
    out = T.clone()
    for b, vec in enumerate(material):
        mat = material_from_vector(vec)
        for kink in [*(mat.T_lut or []), mat.T_solidus]:
            near = (out[b] - kink).abs() < margin
            out[b][near] += 2 * margin
    return out


def _temperatures(material: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    T = 250.0 + 3750.0 * torch.rand((material.shape[0], *shape), generator=g, dtype=torch.float64)
    return _away_from_kinks(T, material).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_probe_matches_the_material_functions(dtype: torch.dtype) -> None:
    from neural_pbf.physics.triton_pde_loss import probe_material

    material = _materials()
    T = _temperatures(material, (4096,), dtype)
    g = torch.Generator().manual_seed(1)
    mask = (torch.rand(T.shape, generator=g) > 0.5).to(dtype)
    H, cp, k, dk = probe_material(T.to(CUDA), mask.to(CUDA), material.to(CUDA, dtype))

    tol = {"rtol": 1e-10, "atol": 1e-8} if dtype == torch.float64 else {"rtol": 2e-5, "atol": 1e-2}
    for b, vec in enumerate(material):
        mat = material_from_vector(vec)
        Tb = T[b].double().requires_grad_(True)
        solid = (mask[b].double().bool() | (mat.T_solidus <= Tb.detach())).double()
        k_ref = k_eff(Tb, mat, solid)
        (dk_ref,) = torch.autograd.grad(k_ref.sum(), Tb)
        torch.testing.assert_close(H[b].cpu().double(), enthalpy(Tb.detach(), mat), **tol)
        torch.testing.assert_close(cp[b].cpu().double(), cp_eff(Tb.detach(), mat), **tol)
        torch.testing.assert_close(k[b].cpu().double(), k_ref.detach(), **tol)
        torch.testing.assert_close(dk[b].cpu().double(), dk_ref, **tol)


def test_probe_covers_powder_promotion_and_both_table_ends() -> None:
    """Powder below T_s uses k_p (dk 0); powder above T_s is solid; above the table k is flat (dk 0)."""
    from neural_pbf.physics.triton_pde_loss import probe_material

    material = _materials()[:1].double()
    mat = material_from_vector(material[0])
    T = torch.tensor([[mat.T_solidus - 300.0, mat.T_solidus + 50.0, mat.T_lut[-1] + 900.0, 100.0]], dtype=torch.float64)
    powder = torch.zeros_like(T)
    _, _, k, dk = probe_material(T.to(CUDA), powder.to(CUDA), material.to(CUDA))
    assert k[0, 0].item() == pytest.approx(mat.k_powder) and dk[0, 0].item() == 0.0
    assert k[0, 1].item() == pytest.approx(float(k_eff(T[0, 1:2], mat, torch.ones(1, dtype=torch.float64))))
    assert dk[0, 2].item() == 0.0 and k[0, 2].item() == pytest.approx(mat.k_lut[-1])


# --- the loss ---------------------------------------------------------------------------------------------

T_REF, T_AMB, Q_REF = 2000.0, 300.0, 1e12
DX, DY, DZ = 2e-6, 2.5e-6, 3e-6  # unequal, so a mixed-up axis shows


def _case(shape: tuple[int, int, int], dtype: torch.dtype, seed: int = 0, wide: bool = False) -> dict:
    """Normalized model inputs whose SI temperatures avoid table nodes and T_s.

    The default spans powder, solid and the melt range; ``wide`` reaches keyhole temperatures above every table end.
    """
    material = _materials(seed)
    B = material.shape[0]
    g = torch.Generator().manual_seed(seed)

    def rand(*s: int) -> torch.Tensor:
        return torch.rand((B, 1, *s), generator=g, dtype=torch.float64)

    T_in = _away_from_kinks(300.0 + (3000.0 if wide else 1700.0) * rand(*shape), material)
    T_target = _away_from_kinks(T_in + (1500.0 if wide else 900.0) * rand(*shape), material)
    T_pred = _away_from_kinks(T_target + 60.0 * (rand(*shape) - 0.5), material)
    tau = torch.rand(B, generator=g, dtype=torch.float64) * 0.9
    x_tau = torch.randn((B, 1, *shape), generator=g, dtype=torch.float64)
    v = ((T_pred - T_AMB) / T_REF - x_tau) / (1.0 - tau.view(-1, 1, 1, 1, 1))
    case = {
        "v": v, "x_tau": x_tau, "tau": tau, "T_in": (T_in - T_AMB) / T_REF, "T_target": (T_target - T_AMB) / T_REF,
        "Q": 1e15 * rand(*shape) / Q_REF, "mask_in": (rand(*shape) > 0.5).double(), "material": material,
    }  # fmt: skip
    return {k: t.to(CUDA, dtype) if k != "material" else t.to(CUDA) for k, t in case.items()}


def _triton_loss(c: dict, form: str, v: torch.Tensor | None = None) -> torch.Tensor:
    from neural_pbf.physics.triton_pde_loss import PDEResidualLoss

    return PDEResidualLoss.apply(
        c["v"] if v is None else v, c["x_tau"], c["tau"], c["T_in"], c["T_target"], c["Q"], c["mask_in"],
        c["material"], form, T_REF, T_AMB, Q_REF, DX, DY, DZ,
    )  # fmt: skip


def _reference_loss(c: dict, form: str, v: torch.Tensor) -> torch.Tensor:
    from neural_pbf.physics.fm_physics import physics_loss_reference

    def si(t: torch.Tensor) -> torch.Tensor:
        return t.double() * T_REF + T_AMB

    T_pred = si(c["x_tau"] + (1.0 - c["tau"].view(-1, 1, 1, 1, 1)) * v)
    return physics_loss_reference(
        T_pred, si(c["T_in"]), si(c["T_target"]), c["Q"].double() * Q_REF, c["mask_in"], c["material"].double(),
        form, DX, DY, DZ,
    )  # fmt: skip


@pytest.mark.parametrize("form", ["tin", "coupled"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_loss_matches_the_reference(form: str, dtype: torch.dtype) -> None:
    c = _case((8, 9, 10), dtype)
    expected = _reference_loss(c, form, c["v"].double())
    rtol = 1e-10 if dtype == torch.float64 else 1e-4
    torch.testing.assert_close(_triton_loss(c, form).double(), expected, rtol=rtol, atol=0.0)


@pytest.mark.parametrize("form", ["tin", "coupled"])
def test_gradient_matches_autograd_of_the_reference(form: str) -> None:
    c = _case((8, 9, 10), torch.float64)
    v_ref = c["v"].clone().requires_grad_(True)
    (g_ref,) = torch.autograd.grad(_reference_loss(c, form, v_ref), v_ref)
    v = c["v"].clone().requires_grad_(True)
    (g,) = torch.autograd.grad(_triton_loss(c, form, v), v)
    torch.testing.assert_close(g, g_ref, rtol=1e-8, atol=1e-14)
    face = g[..., 0, :, :].abs().sum()
    assert (face > 0) if form == "coupled" else (face == 0)  # only coupled reaches the face layer


@pytest.mark.parametrize("form", ["tin", "coupled"])
def test_gradcheck_float64(form: str) -> None:
    c = _case((5, 6, 7), torch.float64, seed=3)
    v = c["v"].clone().requires_grad_(True)
    # The loss is about 1e4, so a central difference with eps 1e-6 carries about 2e-6 of rounding noise, at the
    # tolerance. eps 1e-5 moves T by at most 0.02 K, still 100x inside the 2 K margin around the kinks.
    assert torch.autograd.gradcheck(lambda x: _triton_loss(c, form, x), (v,), eps=1e-5, atol=1e-6, rtol=1e-4)


def test_unknown_form_is_refused() -> None:
    with pytest.raises(ValueError, match="form"):
        _triton_loss(_case((5, 6, 7), torch.float64), "implicit")


@pytest.mark.parametrize("form", ["tin", "coupled"])
def test_mixed_batch_up_to_keyhole_temperatures(form: str) -> None:
    """Two materials and dts, every regime from powder to above the table ends, value and gradient."""
    c = _case((8, 9, 10), torch.float64, seed=5, wide=True)
    T_target = c["T_target"] * T_REF + T_AMB
    assert float(T_target.max()) > 4000.0 and float(T_target.min()) < 800.0
    v_ref = c["v"].clone().requires_grad_(True)
    expected = _reference_loss(c, form, v_ref)
    (g_ref,) = torch.autograd.grad(expected, v_ref)
    v = c["v"].clone().requires_grad_(True)
    loss = _triton_loss(c, form, v)
    (g,) = torch.autograd.grad(loss, v)
    torch.testing.assert_close(loss, expected.detach(), rtol=1e-10, atol=0.0)
    torch.testing.assert_close(g, g_ref, rtol=1e-8, atol=1e-14)
