"""TDD tests for triton_pde_loss.PDEResidualLoss.

RED before implementation — all tests must fail.
GREEN after implementation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(B: int, D: int, H: int, W: int, *, requires_grad: bool = False):
    """Canonical small batch for unit tests."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    v_pred = torch.randn(B, 1, D, H, W, device=device, requires_grad=requires_grad)
    x_tau = torch.randn(B, 1, D, H, W, device=device)
    tau = torch.rand(B, device=device)
    T_in = torch.rand(B, 1, D, H, W, device=device) * 0.5
    Q = torch.rand(B, 1, D, H, W, device=device) * 0.1
    rho = torch.full((B, 1, 1, 1, 1), 7900.0, device=device)
    cp = torch.full((B, 1, 1, 1, 1), 500.0, device=device)
    k = torch.full((B, 1, 1, 1, 1), 20.0, device=device)
    return v_pred, x_tau, tau, T_in, Q, rho, cp, k


_PHYS = dict(dx=1e-5, dy=1e-5, dz=1e-5, T_ref=2000.0, T_ambient=300.0, Q_ref=1.35e15, dt_s=5e-6)


# ---------------------------------------------------------------------------
# Import guard — the module does not exist yet (RED)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pde_module():
    return pytest.importorskip("neural_pbf.physics.triton_pde_loss")


# ---------------------------------------------------------------------------
# A. PyTorch reference (_pde_residual_pytorch)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pytorch_reference_returns_scalar(pde_module) -> None:
    """_pde_residual_pytorch must return a 0-d tensor."""
    B, D, H, W = 2, 4, 4, 4
    v_pred, x_tau, tau, T_in, Q, rho, cp, k = _make_batch(B, D, H, W)
    loss = pde_module._pde_residual_pytorch(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k, **_PHYS
    )
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"


@pytest.mark.unit
def test_pytorch_reference_matches_rope_physics_residual(pde_module) -> None:
    """_pde_residual_pytorch must agree with the existing physics_heat_residual."""
    from neural_pbf.data.fm_dataset import FMDatasetConfig
    from neural_pbf.physics.fm_physics import physics_heat_residual

    B, D, H, W = 1, 8, 8, 8
    v_pred, x_tau, tau, T_in, Q, rho, cp, k = _make_batch(B, D, H, W)

    ds_cfg = FMDatasetConfig(h5_paths=[], Q_ref=_PHYS["Q_ref"])

    ref = physics_heat_residual(
        v_pred, x_tau.detach(), tau, T_in, Q,
        rho, cp, k,
        _PHYS["dx"], _PHYS["dy"], _PHYS["dz"],
        ds_cfg, dt_s=_PHYS["dt_s"],
    )
    ours = pde_module._pde_residual_pytorch(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k, **_PHYS
    )
    assert torch.allclose(ref.float(), ours.float(), rtol=1e-4), (
        f"PyTorch reference diverges from rope implementation: {ref.item():.6e} vs {ours.item():.6e}"
    )


@pytest.mark.unit
def test_pytorch_reference_zero_for_exact_solution(pde_module) -> None:
    """When v_pred satisfies the PDE exactly, residual must be near zero."""
    from neural_pbf.physics.ops import div_k_grad

    B, D, H, W = 1, 8, 8, 8
    dx = dy = dz = 1.0
    T_ref, T_ambient, Q_ref, dt_s = 1.0, 0.0, 1.0, 1.0
    rho_v, cp_v, k_v = 1.0, 1.0, 1.0

    tau = torch.zeros(B)
    x_tau = torch.zeros(B, 1, D, H, W)
    T_in = torch.zeros(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)
    rho = torch.full((B, 1, 1, 1, 1), rho_v)
    cp = torch.full((B, 1, 1, 1, 1), cp_v)
    k = torch.full((B, 1, 1, 1, 1), k_v)

    # exact solution: rho*cp*(T_pred_SI - T_in_SI)/dt_s == k*Laplacian(T_pred_SI) + Q_SI
    # With tau=0, T_pred_SI = (x_tau + v_pred)*T_ref + T_ambient = v_pred (since T_ref=1, T_ambient=0, x_tau=0)
    # So: rho*cp*v_pred/dt_s == k*Laplacian(v_pred) + Q => v_pred = k*Laplacian(v_pred)*dt_s
    # Use v_pred = 0, Q = 0 → residual = 0.
    v_pred = torch.zeros(B, 1, D, H, W, requires_grad=False)

    loss = pde_module._pde_residual_pytorch(
        v_pred, x_tau, tau, T_in, Q, rho, cp, k,
        dx=dx, dy=dy, dz=dz,
        T_ref=T_ref, T_ambient=T_ambient, Q_ref=Q_ref, dt_s=dt_s,
    )
    assert loss.item() < 1e-12, f"Expected ~0 for zero fields, got {loss.item():.3e}"


@pytest.mark.unit
def test_pytorch_reference_nonzero_for_bad_v(pde_module) -> None:
    B, D, H, W = 1, 4, 4, 4
    v_pred = torch.ones(B, 1, D, H, W) * 1e3
    x_tau = torch.zeros(B, 1, D, H, W)
    tau = torch.zeros(B)
    T_in = torch.zeros(B, 1, D, H, W)
    Q = torch.zeros(B, 1, D, H, W)
    rho = torch.full((B, 1, 1, 1, 1), 1.0)
    cp = torch.full((B, 1, 1, 1, 1), 1.0)
    k = torch.full((B, 1, 1, 1, 1), 1.0)
    loss = pde_module._pde_residual_pytorch(
        v_pred, x_tau, tau, T_in, Q, rho, cp, k,
        dx=1.0, dy=1.0, dz=1.0,
        T_ref=1.0, T_ambient=0.0, Q_ref=1.0, dt_s=1.0,
    )
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# B. PDEResidualLoss autograd Function — forward
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pde_loss_apply_returns_scalar(pde_module) -> None:
    B, D, H, W = 2, 4, 4, 4
    v_pred, x_tau, tau, T_in, Q, rho, cp, k = _make_batch(B, D, H, W, requires_grad=True)
    loss = pde_module.PDEResidualLoss.apply(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k,
        _PHYS["dx"], _PHYS["dy"], _PHYS["dz"],
        _PHYS["T_ref"], _PHYS["T_ambient"], _PHYS["Q_ref"], _PHYS["dt_s"],
    )
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"


@pytest.mark.unit
def test_pde_loss_apply_matches_pytorch_reference(pde_module) -> None:
    """PDEResidualLoss.apply must agree with _pde_residual_pytorch within 1%."""
    B, D, H, W = 2, 4, 4, 4
    v_pred, x_tau, tau, T_in, Q, rho, cp, k = _make_batch(B, D, H, W)
    kwargs = _PHYS

    ref = pde_module._pde_residual_pytorch(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k, **kwargs
    ).item()
    got = pde_module.PDEResidualLoss.apply(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k,
        kwargs["dx"], kwargs["dy"], kwargs["dz"],
        kwargs["T_ref"], kwargs["T_ambient"], kwargs["Q_ref"], kwargs["dt_s"],
    ).item()

    if ref == 0.0:
        assert abs(got) < 1e-10
    else:
        rel_err = abs(got - ref) / (abs(ref) + 1e-30)
        assert rel_err < 0.01, f"Forward relative error {rel_err:.4f} exceeds 1%: ref={ref:.6e} got={got:.6e}"


@pytest.mark.unit
def test_pde_loss_apply_finite_for_typical_inputs(pde_module) -> None:
    """Typical LPBF magnitudes (Q_ref=1.35e15) must not produce inf/nan."""
    B, D, H, W = 2, 4, 4, 4
    v_pred, x_tau, tau, T_in, Q, rho, cp, k = _make_batch(B, D, H, W, requires_grad=True)
    loss = pde_module.PDEResidualLoss.apply(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k,
        _PHYS["dx"], _PHYS["dy"], _PHYS["dz"],
        _PHYS["T_ref"], _PHYS["T_ambient"], _PHYS["Q_ref"], _PHYS["dt_s"],
    )
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# C. PDEResidualLoss — backward
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pde_loss_backward_gradient_flows(pde_module) -> None:
    """backward() must produce a gradient w.r.t. v_pred."""
    B, D, H, W = 1, 4, 4, 4
    v_pred, x_tau, tau, T_in, Q, rho, cp, k = _make_batch(B, D, H, W, requires_grad=True)
    loss = pde_module.PDEResidualLoss.apply(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k,
        _PHYS["dx"], _PHYS["dy"], _PHYS["dz"],
        _PHYS["T_ref"], _PHYS["T_ambient"], _PHYS["Q_ref"], _PHYS["dt_s"],
    )
    loss.backward()
    assert v_pred.grad is not None, "No gradient for v_pred after backward()"
    assert v_pred.grad.shape == v_pred.shape
    assert torch.isfinite(v_pred.grad).all(), "Gradient contains inf/nan"


@pytest.mark.unit
def test_pde_loss_backward_no_gradient_through_x_tau(pde_module) -> None:
    """x_tau is always passed detached — confirm it never receives a gradient."""
    B, D, H, W = 1, 4, 4, 4
    v_pred, x_tau, tau, T_in, Q, rho, cp, k = _make_batch(B, D, H, W, requires_grad=True)
    x_tau_leaf = x_tau.detach().requires_grad_(True)
    loss = pde_module.PDEResidualLoss.apply(
        v_pred, x_tau_leaf, tau, T_in, Q, rho, cp, k,
        _PHYS["dx"], _PHYS["dy"], _PHYS["dz"],
        _PHYS["T_ref"], _PHYS["T_ambient"], _PHYS["Q_ref"], _PHYS["dt_s"],
    )
    loss.backward()
    assert x_tau_leaf.grad is None or torch.all(x_tau_leaf.grad == 0), (
        "x_tau should not receive non-zero gradients"
    )


@pytest.mark.unit
def test_pde_loss_backward_matches_finite_difference(pde_module) -> None:
    """Analytical backward gradient must match finite-difference approximation within 2%.

    Uses float64 throughout for numerical accuracy.
    """
    B, D, H, W = 1, 4, 4, 4
    eps = 1e-4

    torch.manual_seed(42)
    v_pred = torch.randn(B, 1, D, H, W, dtype=torch.float64) * 0.1
    x_tau = torch.randn(B, 1, D, H, W, dtype=torch.float64) * 0.1
    tau = torch.rand(B, dtype=torch.float64) * 0.5
    T_in = torch.rand(B, 1, D, H, W, dtype=torch.float64) * 0.5
    Q = torch.zeros(B, 1, D, H, W, dtype=torch.float64)
    rho = torch.full((B, 1, 1, 1, 1), 1.0, dtype=torch.float64)
    cp = torch.full((B, 1, 1, 1, 1), 1.0, dtype=torch.float64)
    k = torch.full((B, 1, 1, 1, 1), 1.0, dtype=torch.float64)
    phys64 = dict(dx=1e-3, dy=1e-3, dz=1e-3, T_ref=1.0, T_ambient=0.0, Q_ref=1.0, dt_s=1.0)

    # Analytical gradient
    v_pred_ad = v_pred.clone().requires_grad_(True)
    loss_ad = pde_module._pde_residual_pytorch(
        v_pred_ad, x_tau, tau, T_in, Q, rho, cp, k, **phys64
    )
    loss_ad.backward()
    grad_analytical = v_pred_ad.grad.clone()

    # Finite-difference gradient (random probe direction)
    torch.manual_seed(7)
    direction = torch.randn_like(v_pred)
    direction = direction / direction.norm()

    loss_plus = pde_module._pde_residual_pytorch(
        v_pred + eps * direction, x_tau, tau, T_in, Q, rho, cp, k, **phys64
    )
    loss_minus = pde_module._pde_residual_pytorch(
        v_pred - eps * direction, x_tau, tau, T_in, Q, rho, cp, k, **phys64
    )
    grad_fd_proj = ((loss_plus - loss_minus) / (2 * eps)).item()
    grad_ad_proj = (grad_analytical * direction).sum().item()

    if abs(grad_fd_proj) < 1e-12:
        assert abs(grad_ad_proj) < 1e-10
    else:
        rel_err = abs(grad_ad_proj - grad_fd_proj) / (abs(grad_fd_proj) + 1e-30)
        assert rel_err < 0.02, (
            f"Gradient FD mismatch: analytical={grad_ad_proj:.6e} fd={grad_fd_proj:.6e} "
            f"rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# D. PDEResidualLoss — Triton path (GPU-gated)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_triton_forward_matches_pytorch_reference_on_gpu(pde_module) -> None:
    """On GPU, Triton forward must agree with PyTorch reference within 1%."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("triton")

    device = torch.device("cuda")
    B, D, H, W = 2, 4, 4, 4
    torch.manual_seed(0)
    v_pred = torch.randn(B, 1, D, H, W, device=device)
    x_tau = torch.randn(B, 1, D, H, W, device=device)
    tau = torch.rand(B, device=device)
    T_in = torch.rand(B, 1, D, H, W, device=device) * 0.5
    Q = torch.rand(B, 1, D, H, W, device=device) * 0.1
    rho = torch.full((B, 1, 1, 1, 1), 7900.0, device=device)
    cp = torch.full((B, 1, 1, 1, 1), 500.0, device=device)
    k = torch.full((B, 1, 1, 1, 1), 20.0, device=device)

    ref = pde_module._pde_residual_pytorch(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k, **_PHYS
    ).item()
    got = pde_module.PDEResidualLoss.apply(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k,
        _PHYS["dx"], _PHYS["dy"], _PHYS["dz"],
        _PHYS["T_ref"], _PHYS["T_ambient"], _PHYS["Q_ref"], _PHYS["dt_s"],
    ).item()

    if ref == 0.0:
        assert abs(got) < 1e-10
    else:
        rel_err = abs(got - ref) / (abs(ref) + 1e-30)
        assert rel_err < 0.01, f"GPU Triton forward rel_err={rel_err:.4f}: ref={ref:.6e} got={got:.6e}"


@pytest.mark.unit
def test_triton_gradient_matches_pytorch_on_gpu(pde_module) -> None:
    """GPU backward gradient must agree with PyTorch autograd within 2%."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("triton")

    device = torch.device("cuda")
    B, D, H, W = 1, 4, 4, 4
    torch.manual_seed(42)
    v_pred_ref = torch.randn(B, 1, D, H, W, device=device, dtype=torch.float32) * 0.1
    x_tau = torch.randn(B, 1, D, H, W, device=device)
    tau = torch.rand(B, device=device) * 0.5
    T_in = torch.rand(B, 1, D, H, W, device=device) * 0.5
    Q = torch.zeros(B, 1, D, H, W, device=device)
    rho = torch.full((B, 1, 1, 1, 1), 1.0, device=device)
    cp = torch.full((B, 1, 1, 1, 1), 1.0, device=device)
    k = torch.full((B, 1, 1, 1, 1), 1.0, device=device)
    phys = dict(dx=1e-3, dy=1e-3, dz=1e-3, T_ref=1.0, T_ambient=0.0, Q_ref=1.0, dt_s=1.0)

    # PyTorch reference gradient
    v_ref = v_pred_ref.clone().requires_grad_(True)
    pde_module._pde_residual_pytorch(v_ref, x_tau.detach(), tau, T_in, Q, rho, cp, k, **phys).backward()
    grad_ref = v_ref.grad.clone()

    # Triton gradient
    v_tri = v_pred_ref.clone().requires_grad_(True)
    pde_module.PDEResidualLoss.apply(
        v_tri, x_tau.detach(), tau, T_in, Q, rho, cp, k,
        phys["dx"], phys["dy"], phys["dz"],
        phys["T_ref"], phys["T_ambient"], phys["Q_ref"], phys["dt_s"],
    ).backward()
    grad_tri = v_tri.grad.clone()

    rel_err = (grad_tri - grad_ref).norm() / (grad_ref.norm() + 1e-12)
    assert rel_err.item() < 0.02, f"GPU gradient rel_err={rel_err.item():.4f}"


@pytest.mark.unit
def test_triton_accepts_bf16_input(pde_module) -> None:
    """PDEResidualLoss.apply must not raise on BF16 inputs (GPU-gated)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    pytest.importorskip("triton")

    device = torch.device("cuda")
    B, D, H, W = 1, 4, 4, 4
    torch.manual_seed(0)
    v_pred = torch.randn(B, 1, D, H, W, device=device, dtype=torch.bfloat16) * 0.1
    x_tau = torch.randn(B, 1, D, H, W, device=device, dtype=torch.bfloat16) * 0.1
    tau = torch.rand(B, device=device, dtype=torch.float32)
    T_in = torch.rand(B, 1, D, H, W, device=device, dtype=torch.bfloat16) * 0.5
    Q = torch.zeros(B, 1, D, H, W, device=device, dtype=torch.bfloat16)
    rho = torch.full((B, 1, 1, 1, 1), 7900.0, device=device, dtype=torch.float32)
    cp = torch.full((B, 1, 1, 1, 1), 500.0, device=device, dtype=torch.float32)
    k = torch.full((B, 1, 1, 1, 1), 20.0, device=device, dtype=torch.float32)
    v_pred.requires_grad_(True)

    loss = pde_module.PDEResidualLoss.apply(
        v_pred, x_tau.detach(), tau, T_in, Q, rho, cp, k,
        _PHYS["dx"], _PHYS["dy"], _PHYS["dz"],
        _PHYS["T_ref"], _PHYS["T_ambient"], _PHYS["Q_ref"], _PHYS["dt_s"],
    )
    assert torch.isfinite(loss), f"BF16 loss not finite: {loss.item()}"
    loss.backward()
    assert v_pred.grad is not None
