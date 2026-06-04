"""Custom Triton kernels for the heat-PDE residual loss in physics-informed FM.

Mathematical derivation
-----------------------
The loss is MSE(lhs_scaled, rhs_scaled) over all N = B*Nz*Ny*Nx voxels.

    T_pred_SI  = (x_tau.detach + (1-τ)·v_pred)·T_ref + T_ambient
    lhs_scaled = ρ·cp·(T_pred_SI − T_in_SI) / (dt_s · Q_ref)
    rhs_scaled = k · ∇²(T_pred_SI) / Q_ref + Q_SI / Q_ref
    residual   = lhs_scaled − rhs_scaled
    L          = (1/N) Σ residual²

Since x_tau is always detached, ∂T_pred_SI/∂v_pred = (1−τ)·T_ref.
The Laplacian is self-adjoint, so the explicit backward gradient is:

    ∂L/∂v_pred_j = (2/N) · (1−τ) · T_ref/Q_ref · [ρ·cp/dt_s · res_j − k · ∇²(res)_j]

On CUDA + Triton: fused kernels handle forward and backward.
On CPU or without Triton: falls back to div_k_grad-based PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from neural_pbf.physics.ops import div_k_grad

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernels (compiled only when triton is installed)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def _pde_fwd_kernel(
        v_ptr,
        x_tau_ptr,
        T_in_ptr,
        Q_ptr,
        res_ptr,
        Nx,
        Ny,
        Nz,
        NyNx,
        tau_f,
        one_minus_tau_f,
        T_ref_f,
        T_ambient_f,
        Q_ref_f,
        inv_Q_ref_f,
        rho_f,
        cp_f,
        k_f,
        inv_dx2,
        inv_dy2,
        inv_dz2,
        inv_dt_s_Q_ref_f,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Per-voxel: residual = lhs_scaled − rhs_scaled.

        All input pointers address flat (Nz*Ny*Nx,) float32 arrays for a
        single batch element.  Neumann BCs are enforced by clamping neighbor
        indices to [0, Nmax-1].
        """
        pid = tl.program_id(0)
        flat = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        N_spatial = Nx * Ny * Nz
        mask = flat < N_spatial

        ix = flat % Nx
        iy = (flat // Nx) % Ny
        iz = flat // NyNx

        idx_c = iz * NyNx + iy * Nx + ix
        v_c = tl.load(v_ptr + idx_c, mask=mask, other=0.0).to(tl.float32)
        x_c = tl.load(x_tau_ptr + idx_c, mask=mask, other=0.0).to(tl.float32)
        T_in_c = tl.load(T_in_ptr + idx_c, mask=mask, other=0.0).to(tl.float32)
        Q_c = tl.load(Q_ptr + idx_c, mask=mask, other=0.0).to(tl.float32)

        T_pred_c = (x_c + one_minus_tau_f * v_c) * T_ref_f + T_ambient_f
        T_in_SI_c = T_in_c * T_ref_f + T_ambient_f
        lhs = rho_f * cp_f * (T_pred_c - T_in_SI_c) * inv_dt_s_Q_ref_f

        ix_xp = tl.minimum(ix + 1, Nx - 1)
        ix_xm = tl.maximum(ix - 1, 0)
        iy_yp = tl.minimum(iy + 1, Ny - 1)
        iy_ym = tl.maximum(iy - 1, 0)
        iz_zp = tl.minimum(iz + 1, Nz - 1)
        iz_zm = tl.maximum(iz - 1, 0)

        # Load neighbors and compute T_pred_SI at each neighbor
        idx_xp = iz * NyNx + iy * Nx + ix_xp
        T_xp = (
            tl.load(x_tau_ptr + idx_xp, mask=mask, other=0.0).to(tl.float32)
            + one_minus_tau_f * tl.load(v_ptr + idx_xp, mask=mask, other=0.0).to(tl.float32)
        ) * T_ref_f + T_ambient_f

        idx_xm = iz * NyNx + iy * Nx + ix_xm
        T_xm = (
            tl.load(x_tau_ptr + idx_xm, mask=mask, other=0.0).to(tl.float32)
            + one_minus_tau_f * tl.load(v_ptr + idx_xm, mask=mask, other=0.0).to(tl.float32)
        ) * T_ref_f + T_ambient_f

        idx_yp = iz * NyNx + iy_yp * Nx + ix
        T_yp = (
            tl.load(x_tau_ptr + idx_yp, mask=mask, other=0.0).to(tl.float32)
            + one_minus_tau_f * tl.load(v_ptr + idx_yp, mask=mask, other=0.0).to(tl.float32)
        ) * T_ref_f + T_ambient_f

        idx_ym = iz * NyNx + iy_ym * Nx + ix
        T_ym = (
            tl.load(x_tau_ptr + idx_ym, mask=mask, other=0.0).to(tl.float32)
            + one_minus_tau_f * tl.load(v_ptr + idx_ym, mask=mask, other=0.0).to(tl.float32)
        ) * T_ref_f + T_ambient_f

        idx_zp = iz_zp * NyNx + iy * Nx + ix
        T_zp = (
            tl.load(x_tau_ptr + idx_zp, mask=mask, other=0.0).to(tl.float32)
            + one_minus_tau_f * tl.load(v_ptr + idx_zp, mask=mask, other=0.0).to(tl.float32)
        ) * T_ref_f + T_ambient_f

        idx_zm = iz_zm * NyNx + iy * Nx + ix
        T_zm = (
            tl.load(x_tau_ptr + idx_zm, mask=mask, other=0.0).to(tl.float32)
            + one_minus_tau_f * tl.load(v_ptr + idx_zm, mask=mask, other=0.0).to(tl.float32)
        ) * T_ref_f + T_ambient_f

        lap = (
            (T_xp - 2.0 * T_pred_c + T_xm) * inv_dx2
            + (T_yp - 2.0 * T_pred_c + T_ym) * inv_dy2
            + (T_zp - 2.0 * T_pred_c + T_zm) * inv_dz2
        )
        rhs = (k_f * lap + Q_c * Q_ref_f) * inv_Q_ref_f
        tl.store(res_ptr + idx_c, (lhs - rhs).to(tl.float32), mask=mask)

    @triton.jit
    def _pde_bwd_kernel(
        res_ptr,
        grad_v_ptr,
        Nx,
        Ny,
        Nz,
        NyNx,
        coeff_lhs_f,
        coeff_rhs_f,
        inv_dx2,
        inv_dy2,
        inv_dz2,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Per-voxel: grad_v = coeff_lhs·res − coeff_rhs·∇²(res).

        coeff_lhs = 2*grad_out/N · (1−τ)·T_ref/Q_ref · ρ·cp/dt_s
        coeff_rhs = 2*grad_out/N · (1−τ)·T_ref/Q_ref · k
        """
        pid = tl.program_id(0)
        flat = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        N_spatial = Nx * Ny * Nz
        mask = flat < N_spatial

        ix = flat % Nx
        iy = (flat // Nx) % Ny
        iz = flat // NyNx

        idx_c = iz * NyNx + iy * Nx + ix
        res_c = tl.load(res_ptr + idx_c, mask=mask, other=0.0).to(tl.float32)

        ix_xp = tl.minimum(ix + 1, Nx - 1)
        ix_xm = tl.maximum(ix - 1, 0)
        iy_yp = tl.minimum(iy + 1, Ny - 1)
        iy_ym = tl.maximum(iy - 1, 0)
        iz_zp = tl.minimum(iz + 1, Nz - 1)
        iz_zm = tl.maximum(iz - 1, 0)

        res_xp = tl.load(res_ptr + (iz * NyNx + iy * Nx + ix_xp), mask=mask, other=0.0).to(tl.float32)
        res_xm = tl.load(res_ptr + (iz * NyNx + iy * Nx + ix_xm), mask=mask, other=0.0).to(tl.float32)
        res_yp = tl.load(res_ptr + (iz * NyNx + iy_yp * Nx + ix), mask=mask, other=0.0).to(tl.float32)
        res_ym = tl.load(res_ptr + (iz * NyNx + iy_ym * Nx + ix), mask=mask, other=0.0).to(tl.float32)
        res_zp = tl.load(res_ptr + (iz_zp * NyNx + iy * Nx + ix), mask=mask, other=0.0).to(tl.float32)
        res_zm = tl.load(res_ptr + (iz_zm * NyNx + iy * Nx + ix), mask=mask, other=0.0).to(tl.float32)

        lap_res = (
            (res_xp - 2.0 * res_c + res_xm) * inv_dx2
            + (res_yp - 2.0 * res_c + res_ym) * inv_dy2
            + (res_zp - 2.0 * res_c + res_zm) * inv_dz2
        )
        grad_v = coeff_lhs_f * res_c - coeff_rhs_f * lap_res
        tl.store(grad_v_ptr + idx_c, grad_v.to(tl.float32), mask=mask)


# ---------------------------------------------------------------------------
# Triton launcher helpers
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 256


def _triton_forward_single(
    v: Tensor,
    x_tau: Tensor,
    T_in: Tensor,
    Q: Tensor,
    Nz: int,
    Ny: int,
    Nx: int,
    tau_b: float,
    T_ref: float,
    T_ambient: float,
    Q_ref: float,
    dt_s: float,
    rho_b: float,
    cp_b: float,
    k_b: float,
    dx: float,
    dy: float,
    dz: float,
) -> Tensor:
    """Launch forward kernel for one batch element. Returns (Nz, Ny, Nx) residual."""
    N_spatial = Nz * Ny * Nx
    res = torch.empty(N_spatial, device=v.device, dtype=torch.float32)
    grid = ((N_spatial + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _pde_fwd_kernel[grid](
        v.view(-1).float(),
        x_tau.view(-1).float(),
        T_in.view(-1).float(),
        Q.view(-1).float(),
        res,
        Nx,
        Ny,
        Nz,
        Ny * Nx,
        tau_b,
        1.0 - tau_b,
        T_ref,
        T_ambient,
        Q_ref,
        1.0 / Q_ref,
        rho_b,
        cp_b,
        k_b,
        1.0 / (dx * dx),
        1.0 / (dy * dy),
        1.0 / (dz * dz),
        1.0 / (dt_s * Q_ref),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return res.view(Nz, Ny, Nx)


def _triton_backward_single(
    res: Tensor,
    Nz: int,
    Ny: int,
    Nx: int,
    coeff_lhs: float,
    coeff_rhs: float,
    dx: float,
    dy: float,
    dz: float,
) -> Tensor:
    """Launch backward kernel for one batch element. Returns (Nz, Ny, Nx) grad_v."""
    N_spatial = Nz * Ny * Nx
    grad_v = torch.empty(N_spatial, device=res.device, dtype=torch.float32)
    grid = ((N_spatial + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _pde_bwd_kernel[grid](
        res.view(-1).float(),
        grad_v,
        Nx,
        Ny,
        Nz,
        Ny * Nx,
        coeff_lhs,
        coeff_rhs,
        1.0 / (dx * dx),
        1.0 / (dy * dy),
        1.0 / (dz * dz),
        BLOCK_SIZE=_BLOCK_SIZE,
    )
    return grad_v.view(Nz, Ny, Nx)


# ---------------------------------------------------------------------------
# PyTorch reference (CPU-compatible, used for testing and CPU fallback)
# ---------------------------------------------------------------------------


def _compute_residual_field(
    v_pred: Tensor,
    x_tau: Tensor,
    tau: Tensor,
    T_in: Tensor,
    Q: Tensor,
    rho: Tensor,
    cp: Tensor,
    k: Tensor,
    dx: float,
    dy: float,
    dz: float,
    T_ref: float,
    T_ambient: float,
    Q_ref: float,
    dt_s: float,
) -> Tensor:
    """Return (B, 1, Nz, Ny, Nx) residual field (lhs_scaled − rhs_scaled)."""
    tau_v = tau.view(-1, 1, 1, 1, 1).to(v_pred.dtype)
    T_pred_norm = x_tau + (1.0 - tau_v) * v_pred
    T_pred_SI = T_pred_norm * T_ref + T_ambient
    T_in_SI = T_in * T_ref + T_ambient
    Q_SI = Q * Q_ref

    lhs = rho * cp * (T_pred_SI - T_in_SI) / dt_s
    k_field = k.expand_as(T_pred_SI)
    rhs = div_k_grad(T_pred_SI.float(), k_field.float(), dx, dy, dz) + Q_SI

    return (lhs - rhs) / Q_ref


def _pde_residual_pytorch(
    v_pred: Tensor,
    x_tau: Tensor,
    tau: Tensor,
    T_in: Tensor,
    Q: Tensor,
    rho: Tensor,
    cp: Tensor,
    k: Tensor,
    dx: float,
    dy: float,
    dz: float,
    T_ref: float,
    T_ambient: float,
    Q_ref: float,
    dt_s: float,
) -> Tensor:
    """PyTorch PDE residual loss — matches train_fm_dit_rope.py."""
    res = _compute_residual_field(
        v_pred,
        x_tau,
        tau,
        T_in,
        Q,
        rho,
        cp,
        k,
        dx,
        dy,
        dz,
        T_ref,
        T_ambient,
        Q_ref,
        dt_s,
    )
    return F.mse_loss(res, torch.zeros_like(res))


# ---------------------------------------------------------------------------
# Analytical backward (PyTorch ops — used on CPU and as reference)
# ---------------------------------------------------------------------------


def _gradient_pytorch(
    res_field: Tensor,
    tau: Tensor,
    T_ref: float,
    Q_ref: float,
    rho: Tensor,
    cp: Tensor,
    k: Tensor,
    dx: float,
    dy: float,
    dz: float,
    dt_s: float,
    N: int,
    grad_out: float,
) -> Tensor:
    """Compute ∂L/∂v_pred analytically from the saved residual field.

    Returns (B, 1, Nz, Ny, Nx) gradient.
    """
    B = res_field.shape[0]
    res = res_field.unsqueeze(1)  # (B, 1, Nz, Ny, Nx)
    k_field = k.expand_as(res).float()
    # div_k_grad with uniform k = k * Laplacian(res)
    lap_k_res = div_k_grad(res.float(), k_field, dx, dy, dz)

    tau_v = tau.view(B, 1, 1, 1, 1).to(res.dtype)
    coeff = (2.0 * grad_out / N) * (1.0 - tau_v) * (T_ref / Q_ref)
    return coeff * (rho * cp / dt_s * res - lap_k_res)


# ---------------------------------------------------------------------------
# PDEResidualLoss — custom autograd Function
# ---------------------------------------------------------------------------


class PDEResidualLoss(torch.autograd.Function):
    """Heat-PDE residual loss with Triton-accelerated forward and backward.

    Usage::

        loss = PDEResidualLoss.apply(
            v_pred, x_tau.detach(), tau, T_in, Q,
            rho, cp, k, dx, dy, dz, T_ref, T_ambient, Q_ref, dt_s,
        )
        loss.backward()
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        v_pred: Tensor,  # (B,1,Nz,Ny,Nx) — differentiable
        x_tau: Tensor,  # (B,1,Nz,Ny,Nx) — always detached by caller
        tau: Tensor,  # (B,)
        T_in: Tensor,  # (B,1,Nz,Ny,Nx)
        Q: Tensor,  # (B,1,Nz,Ny,Nx)
        rho: Tensor,  # (B,1,1,1,1)
        cp: Tensor,  # (B,1,1,1,1)
        k: Tensor,  # (B,1,1,1,1)
        dx: float,
        dy: float,
        dz: float,
        T_ref: float,
        T_ambient: float,
        Q_ref: float,
        dt_s: float,
    ) -> Tensor:
        B, _, Nz, Ny, Nx = v_pred.shape
        device = v_pred.device
        use_triton = _TRITON_AVAILABLE and device.type == "cuda"

        rho_f = rho.float()
        cp_f = cp.float()
        k_f = k.float()
        tau_f = tau.float()

        if use_triton:
            res_list = []
            for b in range(B):
                res_b = _triton_forward_single(
                    v_pred[b, 0].contiguous(),
                    x_tau[b, 0].contiguous(),
                    T_in[b, 0].contiguous(),
                    Q[b, 0].contiguous(),
                    Nz,
                    Ny,
                    Nx,
                    tau_f[b].item(),
                    T_ref,
                    T_ambient,
                    Q_ref,
                    dt_s,
                    rho_f[b, 0, 0, 0, 0].item(),
                    cp_f[b, 0, 0, 0, 0].item(),
                    k_f[b, 0, 0, 0, 0].item(),
                    dx,
                    dy,
                    dz,
                )
                res_list.append(res_b)
            res_stack = torch.stack(res_list)  # (B, Nz, Ny, Nx)
        else:
            res_stack = (
                _compute_residual_field(
                    v_pred.float(),
                    x_tau.float(),
                    tau_f,
                    T_in.float(),
                    Q.float(),
                    rho_f,
                    cp_f,
                    k_f,
                    dx,
                    dy,
                    dz,
                    T_ref,
                    T_ambient,
                    Q_ref,
                    dt_s,
                )
                .squeeze(1)
                .detach()
            )  # (B, Nz, Ny, Nx) — detach: no double-backprop

        # Tensors go through save_for_backward so the autograd engine tracks mutations.
        # Scalar ints/floats and bools are stored directly on ctx.
        ctx.save_for_backward(res_stack, tau_f, rho_f, cp_f, k_f)
        ctx.dx, ctx.dy, ctx.dz = dx, dy, dz
        ctx.dt_s, ctx.T_ref, ctx.Q_ref = dt_s, T_ref, Q_ref
        ctx.use_triton = use_triton
        ctx.B, ctx.Nz, ctx.Ny, ctx.Nx = B, Nz, Ny, Nx

        return res_stack.pow(2).mean()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ):
        res_stack, tau, rho, cp, k = ctx.saved_tensors
        B, Nz, Ny, Nx = ctx.B, ctx.Nz, ctx.Ny, ctx.Nx
        T_ref, Q_ref = ctx.T_ref, ctx.Q_ref
        dx, dy, dz, dt_s = ctx.dx, ctx.dy, ctx.dz, ctx.dt_s
        use_triton = ctx.use_triton
        N = B * Nz * Ny * Nx
        grad_out = grad_output.item()

        if use_triton:
            grad_list = []
            for b in range(B):
                tau_b = tau[b].item()
                rho_b = rho[b, 0, 0, 0, 0].item()
                cp_b = cp[b, 0, 0, 0, 0].item()
                k_b = k[b, 0, 0, 0, 0].item()
                shared = 2.0 * grad_out / N * (1.0 - tau_b) * T_ref / Q_ref
                coeff_lhs = shared * rho_b * cp_b / dt_s
                coeff_rhs = shared * k_b
                grad_b = _triton_backward_single(
                    res_stack[b].contiguous(),
                    Nz,
                    Ny,
                    Nx,
                    coeff_lhs,
                    coeff_rhs,
                    dx,
                    dy,
                    dz,
                )
                grad_list.append(grad_b)
            grad_v = torch.stack(grad_list).unsqueeze(1)
        else:
            grad_v = _gradient_pytorch(
                res_stack,
                tau,
                T_ref,
                Q_ref,
                rho,
                cp,
                k,
                dx,
                dy,
                dz,
                dt_s,
                N,
                grad_out,
            )

        # (v_pred, x_tau, tau, T_in, Q, rho, cp, k, dx, dy, dz, T_ref,
        #  T_ambient, Q_ref, dt_s)
        return (
            grad_v,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
