"""PyTorch reference for the physics loss: two forms, per-sample material and dt, batch-level normalization."""

from __future__ import annotations

import random

import pytest
import torch

from neural_pbf.data.sampling import randomize_material
from neural_pbf.physics.fm_physics import enthalpy, heat_residual_si, physics_loss_reference
from neural_pbf.physics.material import MaterialConfig, cp_eff
from neural_pbf.physics.material_vector import MATERIAL_DIM, SCALARS, material_from_vector

SHAPE = (1, 1, 6, 7, 8)
DX = DY = DZ = 2e-6


def _vector(mat: MaterialConfig, exposure: float) -> torch.Tensor:
    vec = torch.zeros(MATERIAL_DIM, dtype=torch.float64)
    vals = {"rho": mat.rho, "L": mat.latent_heat_L, "T_s": mat.T_solidus, "T_l": mat.T_liquidus, "k_p": mat.k_powder,
            "k_s": mat.k_solid, "k_l": mat.k_liquid, "cp": mat.cp_base, "exposure_time": exposure}  # fmt: skip
    vec[: len(SCALARS)] = torch.tensor([vals[k] for k in SCALARS], dtype=torch.float64)
    n = len(mat.T_lut or [])
    vec[len(SCALARS)] = n
    for i, name in enumerate(("T_lut", "k_lut", "cp_lut")):
        start = len(SCALARS) + 1 + i * 8
        vec[start : start + n] = torch.tensor(getattr(mat, name), dtype=torch.float64)
    return vec


def _batch(seed: int = 0) -> dict[str, torch.Tensor]:
    """Two samples with different materials and exposure times; temperatures across the melt range."""
    random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    mats = [randomize_material(MaterialConfig.ss316l_preset()), randomize_material(MaterialConfig.ti64_preset())]
    B = len(mats)
    T_in = 400.0 + 1200.0 * torch.rand((B, *SHAPE[1:]), generator=g, dtype=torch.float64)
    T_target = T_in + 600.0 * torch.rand((B, *SHAPE[1:]), generator=g, dtype=torch.float64)
    return {
        "T_in": T_in,
        "T_target": T_target,
        "T_pred": T_target + 30.0 * torch.randn((B, *SHAPE[1:]), generator=g, dtype=torch.float64),
        "Q": 1e15 * torch.rand((B, *SHAPE[1:]), generator=g, dtype=torch.float64),
        "mask_in": (torch.rand((B, *SHAPE[1:]), generator=g) > 0.5).to(torch.uint8),
        "material": torch.stack([_vector(mats[0], 40e-6), _vector(mats[1], 70e-6)]),
    }


def _loss(b: dict[str, torch.Tensor], form: str) -> torch.Tensor:
    return physics_loss_reference(
        b["T_pred"], b["T_in"], b["T_target"], b["Q"], b["mask_in"], b["material"], form, DX, DY, DZ
    )


def _by_hand(b: dict[str, torch.Tensor], form: str) -> torch.Tensor:
    num = den = torch.zeros((), dtype=torch.float64)
    for i in range(b["T_in"].shape[0]):
        mat, dt = material_from_vector(b["material"][i]), float(b["material"][i][SCALARS.index("exposure_time")])
        sl = slice(i, i + 1)
        R = heat_residual_si(b["T_in"][sl], b["T_pred"][sl], b["Q"][sl], b["mask_in"][sl], mat, dt, DX, DY, DZ,
                             "enthalpy", "implicit" if form == "coupled" else "explicit")  # fmt: skip
        stored = mat.rho * (enthalpy(b["T_target"][sl], mat) - enthalpy(b["T_in"][sl], mat)) / dt
        num = num + R.pow(2).sum()
        den = den + stored[..., 1:-1, 1:-1, 1:-1].pow(2).sum()
    return num / den


@pytest.mark.unit
@pytest.mark.parametrize("form", ["coupled", "tin"])
def test_loss_is_the_batch_level_ratio_of_the_reference_residual(form: str) -> None:
    b = _batch()
    torch.testing.assert_close(_loss(b, form), _by_hand(b, form))


@pytest.mark.unit
def test_the_two_forms_differ() -> None:
    b = _batch()
    assert not torch.allclose(_loss(b, "coupled"), _loss(b, "tin"))


@pytest.mark.unit
def test_a_barely_heating_sample_does_not_blow_up_the_loss() -> None:
    """Pooled over the batch: a near-zero denominator of one sample is outweighed by the other."""
    b = _batch()
    b["T_target"][1] = b["T_in"][1] + 1e-6
    b["T_pred"][1] = b["T_in"][1] + 1e-6
    loss = _loss(b, "tin")
    assert torch.isfinite(loss) and float(loss) < 1e6


@pytest.mark.unit
@pytest.mark.parametrize("form", ["coupled", "tin"])
def test_gradient_flows_to_the_prediction_only(form: str) -> None:
    b = _batch()
    for key in ("T_pred", "T_target", "T_in"):
        b[key].requires_grad_(True)
    _loss(b, form).backward()
    assert b["T_pred"].grad is not None and b["T_pred"].grad.abs().sum() > 0
    assert b["T_target"].grad is None or b["T_target"].grad.abs().sum() == 0
    assert b["T_in"].grad is None or b["T_in"].grad.abs().sum() == 0


@pytest.mark.unit
def test_tin_gradient_is_local_per_voxel() -> None:
    """With diffusion at T_in, dL/dT_pred = 2·R·ρ·cp_eff(T_pred)/dt / Σden² in every interior voxel, 0 on faces."""
    b = _batch()
    b["T_pred"].requires_grad_(True)
    _loss(b, "tin").backward()
    grad = b["T_pred"].grad
    den = sum(
        (material_from_vector(m).rho * (enthalpy(b["T_target"][i], material_from_vector(m))
                                        - enthalpy(b["T_in"][i], material_from_vector(m)))
         / float(m[SCALARS.index("exposure_time")]))[..., 1:-1, 1:-1, 1:-1].pow(2).sum()
        for i, m in enumerate(b["material"])
    )  # fmt: skip
    for i, m in enumerate(b["material"]):
        mat, dt = material_from_vector(m), float(m[SCALARS.index("exposure_time")])
        sl = slice(i, i + 1)
        R = heat_residual_si(b["T_in"][sl], b["T_pred"][sl].detach(), b["Q"][sl], b["mask_in"][sl], mat, dt,
                             DX, DY, DZ, "enthalpy", "explicit")  # fmt: skip
        expected = 2 * R * mat.rho * cp_eff(b["T_pred"][sl].detach(), mat)[..., 1:-1, 1:-1, 1:-1] / dt / den
        torch.testing.assert_close(grad[sl][..., 1:-1, 1:-1, 1:-1], expected)
        assert grad[sl][..., 0, :, :].abs().sum() == 0


@pytest.mark.unit
def test_each_sample_uses_its_own_exposure_time() -> None:
    b = _batch()
    base = _loss(b, "tin")
    b["material"][1][SCALARS.index("exposure_time")] *= 2.0
    assert not torch.allclose(_loss(b, "tin"), base)


@pytest.mark.unit
@pytest.mark.parametrize("form", ["none", "bogus", "implicit"])
def test_unknown_or_off_form_is_refused(form: str) -> None:
    with pytest.raises(ValueError, match="form"):
        _loss(_batch(), form)
