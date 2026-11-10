"""Triton kernels for the physics loss, in the two forms ``coupled`` and ``tin``.

The loss is ``Σ R² / Σ S²`` over the interiors of all patches in the batch, with

    R = ρ·(H(T_pred) − H(T_in))/dt − ∇·(k∇T) − Q          (T = T_pred for coupled, T_in for tin)
    S = ρ·(H(T_target) − H(T_in))/dt                        (ground truth, no gradient)

using each sample's own material and dt (``physics.material_vector``), enthalpy storage with latent heat, and the
solver's conductivity: the k table blended with k_powder by the mask, powder promoted above T_solidus, harmonic
means at the faces. ``fm_physics.physics_loss_reference`` is the plain PyTorch definition; these kernels must give
the same value and the same gradient as autograd of it.

The original version of this file (tag ``v0-random-split-eval``) used dt fixed at 5 µs, one scalar k per sample,
constant-cp storage, and diffusion at the prediction.
"""

from __future__ import annotations

import torch
from torch import Tensor

from neural_pbf.physics.material import MaterialConfig
from neural_pbf.physics.material_vector import LUTS, MAX_LUT, SCALARS

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only installs
    _TRITON_AVAILABLE = False

SHARPNESS = float(MaterialConfig.model_fields["transition_sharpness"].default)  # randomize_material keeps it
_BLOCK = 256

# Per-sample table, one row of _ROW values per sample.
_RHO, _L, _TS, _TL, _KP, _INV_DT, _N = range(7)
_T0, _K0, _C0, _H0 = 8, 8 + MAX_LUT, 8 + 2 * MAX_LUT, 8 + 3 * MAX_LUT
_ROW = 8 + 4 * MAX_LUT


def material_table(material: Tensor, dtype: torch.dtype) -> Tensor:
    """(B, _ROW) table for the kernels from (B, MATERIAL_DIM) material vectors, on the same device."""
    m = material.to(torch.float64)
    col = {k: m[:, i] for i, k in enumerate(SCALARS)}
    n_col = len(SCALARS)
    n = m[:, n_col].round().long()
    luts = {name: m[:, n_col + 1 + i * MAX_LUT : n_col + 1 + (i + 1) * MAX_LUT] for i, name in enumerate(LUTS)}
    t, cp = luts["T_lut"], luts["cp_lut"]
    seg = torch.arange(MAX_LUT - 1, device=m.device)
    valid = seg.unsqueeze(0) + 1 < n.unsqueeze(1)
    area = torch.where(valid, 0.5 * (cp[:, 1:] + cp[:, :-1]) * t.diff(dim=1), torch.zeros_like(t[:, 1:]))
    H_nodes = torch.nn.functional.pad(area.cumsum(dim=1), (1, 0))
    tab = torch.zeros(m.shape[0], _ROW, dtype=torch.float64, device=m.device)
    tab[:, _RHO], tab[:, _L], tab[:, _TS], tab[:, _TL] = col["rho"], col["L"], col["T_s"], col["T_l"]
    tab[:, _KP], tab[:, _INV_DT], tab[:, _N] = col["k_p"], 1.0 / col["exposure_time"], n.to(torch.float64)
    tab[:, _T0 : _T0 + MAX_LUT], tab[:, _K0 : _K0 + MAX_LUT] = t, luts["k_lut"]
    tab[:, _C0 : _C0 + MAX_LUT], tab[:, _H0 : _H0 + MAX_LUT] = cp, H_nodes
    return tab.to(dtype).contiguous()


def _params(
    dtype: torch.dtype, device: torch.device, T_ref: float = 1.0, T_amb: float = 0.0, Q_ref: float = 1.0,
    dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
) -> Tensor:  # fmt: skip
    """Constants as a tensor in the field dtype: Triton passes Python floats as fp32, too coarse for float64 checks."""
    vals = [T_ref, T_amb, Q_ref, 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2, 0.0, SHARPNESS]
    return torch.tensor(vals, dtype=dtype, device=device)


if _TRITON_AVAILABLE:

    @triton.jit
    def _material_at(T, solid_in, tab, SHARP, MAXL: tl.constexpr):  # SHARP: a value loaded in the field dtype
        """H, cp_eff, k and dk/dT at T, as enthalpy, cp_eff, k_eff and autograd of k_eff in PyTorch."""
        n = tl.load(tab + 6).to(tl.int32)
        L = tl.load(tab + 1)
        T_s = tl.load(tab + 2)
        T_l = tl.load(tab + 3)
        k_p = tl.load(tab + 4)
        t_first = tl.load(tab + 8)
        t_last = tl.load(tab + 8 + n - 1)
        k_val = tl.zeros_like(T) + tl.load(tab + 8 + MAXL)
        cp_val = tl.zeros_like(T) + tl.load(tab + 8 + 2 * MAXL)
        k_slope = tl.zeros_like(T)
        H_s = tl.zeros_like(T)
        Tc = tl.minimum(tl.maximum(T, t_first), t_last)
        for i in tl.static_range(MAXL - 1):
            valid = (i + 1) < n
            t0 = tl.load(tab + 8 + i)
            t1 = tl.load(tab + 8 + i + 1)
            k0 = tl.load(tab + 8 + MAXL + i)
            k1 = tl.load(tab + 8 + MAXL + i + 1)
            c0 = tl.load(tab + 8 + 2 * MAXL + i)
            c1 = tl.load(tab + 8 + 2 * MAXL + i + 1)
            h0 = tl.load(tab + 8 + 3 * MAXL + i)
            span = tl.where(valid, t1 - t0, 1.0)
            # interpolate_1d: segment (t0, t1], linear inside.
            inside = valid & (t0 < T) & (t1 >= T)
            w = (T - t0) / span
            k_val = tl.where(inside, k0 + w * (k1 - k0), k_val)
            k_slope = tl.where(inside, (k1 - k0) / span, k_slope)
            cp_val = tl.where(inside, c0 + w * (c1 - c0), cp_val)
            # Exact integral of the piecewise-linear cp table, on T clamped to the table.
            in_h = valid & (t0 < Tc) & (t1 >= Tc)
            s = Tc - t0
            H_s = tl.where(in_h, h0 + c0 * s + 0.5 * (c1 - c0) / span * s * s, H_s)
        above = t_last <= T
        k_val = tl.where(above, tl.load(tab + 8 + MAXL + n - 1), k_val)
        k_slope = tl.where(above, 0.0, k_slope)
        cp_last = tl.load(tab + 8 + 2 * MAXL + n - 1)
        cp_val = tl.where(above, cp_last, cp_val)
        cp_first = tl.load(tab + 8 + 2 * MAXL)
        H_s = H_s + cp_first * tl.minimum(T - t_first, 0.0) + cp_last * tl.maximum(T - t_last, 0.0)
        # Latent heat: L·φ(T) with the sigmoid of melt_fraction / cp_eff.
        width = T_l - T_s
        width = tl.where(width < 1e-6, 1.0, width)
        half = 0.5 * width
        lat = tl.where(L > 1e-9, L, 0.0)
        phi = 1.0 / (1.0 + tl.exp(-SHARP * (T - (T_s + half)) / half))
        H = H_s + lat * phi
        cp = cp_val + lat * SHARP * phi * (1.0 - phi) / half
        solid = (solid_in > 0.5) | (T_s <= T)
        k = tl.where(solid, k_val, k_p)
        dk = tl.where(solid, k_slope, 0.0)
        return H, cp, k, dk

    @triton.jit
    def _probe_kernel(T_ptr, m_ptr, tab_ptr, H_ptr, cp_ptr, k_ptr, dk_ptr, N, prm_ptr,
                      ROW: tl.constexpr, MAXL: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
        b = tl.program_id(0)
        SHARP = tl.load(prm_ptr + 7)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < N
        base = b * N + offs
        T = tl.load(T_ptr + base, mask=ok, other=300.0)
        m = tl.load(m_ptr + base, mask=ok, other=0.0)
        H, cp, k, dk = _material_at(T, m, tab_ptr + b * ROW, SHARP, MAXL)
        tl.store(H_ptr + base, H, mask=ok)
        tl.store(cp_ptr + base, cp, mask=ok)
        tl.store(k_ptr + base, k, mask=ok)
        tl.store(dk_ptr + base, dk, mask=ok)


def probe_material(T: Tensor, mask: Tensor, material: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """H, cp_eff, k and dk/dT for (B, N) temperatures and masks: the kernels' material functions, for testing."""
    T, mask = T.contiguous(), mask.to(T.dtype).contiguous()
    tab = material_table(material, T.dtype)
    out = [torch.empty_like(T) for _ in range(4)]
    B, N = T.shape
    _probe_kernel[(B, triton.cdiv(N, _BLOCK))](
        T, mask, tab, *out, N, _params(T.dtype, T.device), ROW=_ROW, MAXL=MAX_LUT, BLOCK=_BLOCK
    )
    return out[0], out[1], out[2], out[3]


FORMS = ("coupled", "tin")
_EPS_HARMONIC = 1e-12  # ops.harmonic_mean

if _TRITON_AVAILABLE:

    @triton.jit
    def _field_T(idx, ok, v_ptr, x_ptr, tin_ptr, omt, T_ref, T_amb, COUPLED: tl.constexpr):
        """SI temperature of the field the diffusion term uses: the prediction (coupled) or T_in (tin)."""
        if COUPLED:
            x = tl.load(x_ptr + idx, mask=ok, other=0.0)
            v = tl.load(v_ptr + idx, mask=ok, other=0.0)
            return (x + omt * v) * T_ref + T_amb
        else:
            return tl.load(tin_ptr + idx, mask=ok, other=0.0) * T_ref + T_amb

    @triton.jit
    def _harmonic(a, b):
        return 2.0 * a * b / (a + b + 1e-12)

    @triton.jit
    def _fwd_kernel(v_ptr, x_ptr, tin_ptr, tgt_ptr, q_ptr, m_ptr, omt_ptr, tab_ptr, R_ptr, S_ptr,
                    Nx, Ny, Nz, prm_ptr,
                    COUPLED: tl.constexpr, ROW: tl.constexpr, MAXL: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
        b = tl.program_id(0)
        T_ref, T_amb, Q_ref = tl.load(prm_ptr + 0), tl.load(prm_ptr + 1), tl.load(prm_ptr + 2)
        inv_dx2, inv_dy2, inv_dz2 = tl.load(prm_ptr + 3), tl.load(prm_ptr + 4), tl.load(prm_ptr + 5)
        SHARP = tl.load(prm_ptr + 7)
        N = Nx * Ny * Nz
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < N
        ix = offs % Nx
        iy = (offs // Nx) % Ny
        iz = offs // (Nx * Ny)
        inner = ok & (ix > 0) & (ix < Nx - 1) & (iy > 0) & (iy < Ny - 1) & (iz > 0) & (iz < Nz - 1)
        base = b * N
        c = base + offs
        tab = tab_ptr + b * ROW
        rho = tl.load(tab + 0)
        inv_dt = tl.load(tab + 5)
        omt = tl.load(omt_ptr + b)

        T_in = tl.load(tin_ptr + c, mask=inner, other=0.0) * T_ref + T_amb
        T_tg = tl.load(tgt_ptr + c, mask=inner, other=0.0) * T_ref + T_amb
        T_pr = _field_T(c, inner, v_ptr, x_ptr, tin_ptr, omt, T_ref, T_amb, True)
        m_c = tl.load(m_ptr + c, mask=inner, other=0.0)
        H_in, _, _, _ = _material_at(T_in, m_c, tab, SHARP, MAXL)
        H_tg, _, _, _ = _material_at(T_tg, m_c, tab, SHARP, MAXL)
        H_pr, _, _, _ = _material_at(T_pr, m_c, tab, SHARP, MAXL)

        T_c = _field_T(c, inner, v_ptr, x_ptr, tin_ptr, omt, T_ref, T_amb, COUPLED)
        _, _, k_c, _ = _material_at(T_c, m_c, tab, SHARP, MAXL)
        div = tl.zeros_like(T_c)
        for axis in tl.static_range(3):
            if axis == 0:
                step = 1
                inv_h2 = inv_dx2
            elif axis == 1:
                step = Nx
                inv_h2 = inv_dy2
            else:
                step = Nx * Ny
                inv_h2 = inv_dz2
            n_lo = c - step
            n_hi = c + step
            T_lo = _field_T(n_lo, inner, v_ptr, x_ptr, tin_ptr, omt, T_ref, T_amb, COUPLED)
            T_hi = _field_T(n_hi, inner, v_ptr, x_ptr, tin_ptr, omt, T_ref, T_amb, COUPLED)
            _, _, k_lo, _ = _material_at(T_lo, tl.load(m_ptr + n_lo, mask=inner, other=0.0), tab, SHARP, MAXL)
            _, _, k_hi, _ = _material_at(T_hi, tl.load(m_ptr + n_hi, mask=inner, other=0.0), tab, SHARP, MAXL)
            div += (_harmonic(k_c, k_hi) * (T_hi - T_c) - _harmonic(k_c, k_lo) * (T_c - T_lo)) * inv_h2

        Q = tl.load(q_ptr + c, mask=inner, other=0.0) * Q_ref
        R = rho * (H_pr - H_in) * inv_dt - div - Q
        S = rho * (H_tg - H_in) * inv_dt
        tl.store(R_ptr + c, tl.where(inner, R, 0.0), mask=ok)
        tl.store(S_ptr + c, tl.where(inner, S, 0.0), mask=ok)

    @triton.jit
    def _bwd_kernel(v_ptr, x_ptr, m_ptr, omt_ptr, tab_ptr, R_ptr, coeff_ptr, G_ptr,
                    Nx, Ny, Nz, prm_ptr,
                    COUPLED: tl.constexpr, ROW: tl.constexpr, MAXL: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
        """dL/dv for every voxel. g = coeff·R is zero on the face layer, where R is not defined.

        Storage: g_j·ρ·cp_eff(T_j)/dt. Coupled diffusion, summed over the faces (j, n) of voxel j:
        −(g_n − g_j)·[K_jn − (T_n − T_j)·k'_j·∂K/∂k_j]/h², with K the harmonic mean.
        """
        b = tl.program_id(0)
        T_ref, T_amb = tl.load(prm_ptr + 0), tl.load(prm_ptr + 1)
        inv_dx2, inv_dy2, inv_dz2 = tl.load(prm_ptr + 3), tl.load(prm_ptr + 4), tl.load(prm_ptr + 5)
        SHARP = tl.load(prm_ptr + 7)
        N = Nx * Ny * Nz
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = offs < N
        base = b * N
        j = base + offs
        tab = tab_ptr + b * ROW
        rho = tl.load(tab + 0)
        inv_dt = tl.load(tab + 5)
        omt = tl.load(omt_ptr + b)
        coeff = tl.load(coeff_ptr)

        T_j = _field_T(j, ok, v_ptr, x_ptr, x_ptr, omt, T_ref, T_amb, True)
        m_j = tl.load(m_ptr + j, mask=ok, other=0.0)
        _, cp_j, k_j, dk_j = _material_at(T_j, m_j, tab, SHARP, MAXL)
        g_j = coeff * tl.load(R_ptr + j, mask=ok, other=0.0)
        grad = g_j * rho * cp_j * inv_dt
        if COUPLED:
            ix = offs % Nx
            iy = (offs // Nx) % Ny
            iz = offs // (Nx * Ny)
            for face in tl.static_range(6):
                if face == 0:
                    has = ix > 0
                    step = -1
                    inv_h2 = inv_dx2
                elif face == 1:
                    has = ix < Nx - 1
                    step = 1
                    inv_h2 = inv_dx2
                elif face == 2:
                    has = iy > 0
                    step = -Nx
                    inv_h2 = inv_dy2
                elif face == 3:
                    has = iy < Ny - 1
                    step = Nx
                    inv_h2 = inv_dy2
                elif face == 4:
                    has = iz > 0
                    step = -Nx * Ny
                    inv_h2 = inv_dz2
                else:
                    has = iz < Nz - 1
                    step = Nx * Ny
                    inv_h2 = inv_dz2
                nb = ok & has
                n = j + step
                T_n = _field_T(n, nb, v_ptr, x_ptr, x_ptr, omt, T_ref, T_amb, True)
                _, _, k_n, _ = _material_at(T_n, tl.load(m_ptr + n, mask=nb, other=0.0), tab, SHARP, MAXL)
                g_n = coeff * tl.load(R_ptr + n, mask=nb, other=0.0)
                den = k_j + k_n + 1e-12
                K = 2.0 * k_j * k_n / den
                dK = 2.0 * k_n * (k_n + 1e-12) / (den * den)
                term = (g_n - g_j) * (K - (T_n - T_j) * dk_j * dK) * inv_h2
                grad -= tl.where(nb, term, 0.0)
        tl.store(G_ptr + j, grad * omt * T_ref, mask=ok)


def _forward(
    v: Tensor, x: Tensor, omt: Tensor, tin: Tensor, tgt: Tensor, q: Tensor, m: Tensor, material: Tensor, form: str,
    T_ref: float, T_ambient: float, Q_ref: float, dx: float, dy: float, dz: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[int, int]]:  # fmt: skip
    """Launch the forward kernel on contiguous (B, 1, D, H, W) fields of one dtype: R and S, zero on the faces."""
    B, _, Nz, Ny, Nx = v.shape
    tab = material_table(material.to(v.device), v.dtype)
    prm = _params(v.dtype, v.device, T_ref, T_ambient, Q_ref, dx, dy, dz)
    R, S = torch.empty_like(v), torch.empty_like(v)
    grid = (B, triton.cdiv(Nx * Ny * Nz, _BLOCK))
    _fwd_kernel[grid](
        v, x, tin, tgt, q, m, omt, tab, R, S, Nx, Ny, Nz, prm,
        COUPLED=form == "coupled", ROW=_ROW, MAXL=MAX_LUT, BLOCK=_BLOCK,
    )  # fmt: skip
    return R, S, tab, prm, grid


@torch.no_grad()
def residual_sums(
    T_pred: Tensor, T_in: Tensor, T_target: Tensor, Q: Tensor, mask_in: Tensor, material: Tensor, form: str,
    T_ref: float, T_ambient: float, Q_ref: float, dx: float, dy: float, dz: float,
) -> tuple[Tensor, Tensor]:  # fmt: skip
    """Per-sample (Σ R², Σ S²) for a predicted temperature, in normalized units; no gradient. For val metrics."""
    if form not in FORMS:
        raise ValueError(f"physics form must be one of {FORMS}, got {form!r}")
    dtype = torch.float64 if T_pred.dtype == torch.float64 else torch.float32
    v, tin, tgt, q, m = (t.to(dtype).contiguous() for t in (T_pred, T_in, T_target, Q, mask_in))
    x = torch.zeros_like(v)
    omt = torch.ones(v.shape[0], dtype=dtype, device=v.device)
    R, S, *_ = _forward(v, x, omt, tin, tgt, q, m, material, form, T_ref, T_ambient, Q_ref, dx, dy, dz)
    return R.pow(2).flatten(1).sum(1), S.pow(2).flatten(1).sum(1)


class PDEResidualLoss(torch.autograd.Function):
    """Physics loss Σ R² / Σ S² with Triton forward and backward; gradient only for ``v_pred``.

    Fields are (B, 1, D, H, W) in the dataset's normalized units: T as (T_SI − T_ambient)/T_ref, Q as Q_SI/Q_ref.
    ``x_tau`` must be detached by the caller. ``material`` is (B, MATERIAL_DIM).

    Usage::

        loss = PDEResidualLoss.apply(v_pred, x_tau.detach(), tau, T_in, T_target, Q, mask_in, material, form,
                                     T_ref, T_ambient, Q_ref, dx, dy, dz)
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        v_pred: Tensor, x_tau: Tensor, tau: Tensor, T_in: Tensor, T_target: Tensor, Q: Tensor, mask_in: Tensor,
        material: Tensor, form: str, T_ref: float, T_ambient: float, Q_ref: float, dx: float, dy: float, dz: float,
    ) -> Tensor:  # fmt: skip
        if form not in FORMS:
            raise ValueError(f"physics form must be one of {FORMS}, got {form!r}")
        if not (v_pred.ndim == 5 and v_pred.shape[1] == 1 and v_pred.is_cuda):
            raise ValueError(f"v_pred must be a CUDA tensor of shape (B, 1, D, H, W), got {tuple(v_pred.shape)}")
        dtype = torch.float64 if v_pred.dtype == torch.float64 else torch.float32
        v, x, tin, tgt, q, m = (t.detach().to(dtype).contiguous() for t in (v_pred, x_tau, T_in, T_target, Q, mask_in))
        omt = (1.0 - tau.detach().to(dtype)).contiguous()
        R, S, tab, prm, grid = _forward(v, x, omt, tin, tgt, q, m, material, form, T_ref, T_ambient, Q_ref, dx, dy, dz)
        Nz, Ny, Nx = v.shape[2:]
        den = S.pow(2).sum()
        if not float(den) > 0.0:
            raise ValueError("no stored heat in the batch: every patch interior has T_target == T_in")
        ctx.save_for_backward(v, x, m, omt, tab, R, den, prm)
        ctx.dims, ctx.coupled, ctx.grid = (Nx, Ny, Nz), form == "coupled", grid
        ctx.out_dtype = v_pred.dtype
        return (R.pow(2).sum() / den).to(v_pred.dtype)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: Tensor):  # type: ignore[override]
        v, x, m, omt, tab, R, den, prm = ctx.saved_tensors  # type: ignore[attr-defined]
        coeff = (2.0 * grad_out.to(v.dtype) / den).reshape(1).contiguous()
        G = torch.empty_like(v)
        _bwd_kernel[ctx.grid](  # type: ignore[attr-defined]
            v,
            x,
            m,
            omt,
            tab,
            R,
            coeff,
            G,
            *ctx.dims,
            prm,  # type: ignore[attr-defined]
            COUPLED=ctx.coupled,
            ROW=_ROW,
            MAXL=MAX_LUT,
            BLOCK=_BLOCK,  # type: ignore[attr-defined]
        )
        return (G.to(ctx.out_dtype),) + (None,) * 14  # type: ignore[attr-defined]
