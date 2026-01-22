import torch
import triton
import triton.language as tl

@triton.jit
def _thermal_step_3d_kernel(
    T_ptr, mask_ptr, Q_ptr, T_new_ptr,
    Nx, Ny, Nz,
    stride_x, stride_y, stride_z,
    dx, dy, dz, dt,
    loss_h, T_ambient,
    k_powder, k_solid, k_liquid,
    cp_base, rho,
    T_sol, T_liq, L,
    sharpness,
    use_t_dep: tl.constexpr, T_ref,
    ks_coeff, kl_coeff, cp_coeff,
    # LUT
    use_lut: tl.constexpr, T_lut_ptr, k_lut_ptr, cp_lut_ptr, n_lut: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    pid_x, pid_y, pid_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)[:, None, None]
    o_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)[None, :, None]
    o_z = pid_z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)[None, None, :]
    m_all = (o_x < Nx) & (o_y < Ny) & (o_z < Nz)
    idx_c = o_x * stride_x + o_y * stride_y + o_z * stride_z
    
    Tc = tl.load(T_ptr + idx_c, mask=m_all)
    mc = tl.load(mask_ptr + idx_c, mask=m_all).to(tl.float32)

    # Neighbors Indices
    xl, xr = tl.maximum(0, o_x - 1), tl.minimum(Nx - 1, o_x + 1)
    yu, yd = tl.maximum(0, o_y - 1), tl.minimum(Ny - 1, o_y + 1)
    zf, zb = tl.maximum(0, o_z - 1), tl.minimum(Nz - 1, o_z + 1)

    # Loads
    Txl = tl.load(T_ptr + (xl*stride_x+o_y*stride_y+o_z*stride_z), mask=m_all)
    mxl = tl.load(mask_ptr + (xl*stride_x+o_y*stride_y+o_z*stride_z), mask=m_all).to(tl.float32)
    Txr = tl.load(T_ptr + (xr*stride_x+o_y*stride_y+o_z*stride_z), mask=m_all)
    mxr = tl.load(mask_ptr + (xr*stride_x+o_y*stride_y+o_z*stride_z), mask=m_all).to(tl.float32)
    Tyu = tl.load(T_ptr + (o_x*stride_x+yu*stride_y+o_z*stride_z), mask=m_all)
    myu = tl.load(mask_ptr + (o_x*stride_x+yu*stride_y+o_z*stride_z), mask=m_all).to(tl.float32)
    Tyd = tl.load(T_ptr + (o_x*stride_x+yd*stride_y+o_z*stride_z), mask=m_all)
    myd = tl.load(mask_ptr + (o_x*stride_x+yd*stride_y+o_z*stride_z), mask=m_all).to(tl.float32)
    Tzf = tl.load(T_ptr + (o_x*stride_x+o_y*stride_y+zf*stride_z), mask=m_all)
    mzf = tl.load(mask_ptr + (o_x*stride_x+o_y*stride_y+zf*stride_z), mask=m_all).to(tl.float32)
    Tzb = tl.load(T_ptr + (o_x*stride_x+o_y*stride_y+zb*stride_z), mask=m_all)
    mzb = tl.load(mask_ptr + (o_x*stride_x+o_y*stride_y+zb*stride_z), mask=m_all).to(tl.float32)

    mid = 0.5 * (T_sol + T_liq)
    inv_hw = 1.0 / (0.5 * (T_liq - T_sol) + 1e-9)

    if use_lut:
        k_lo_v = tl.load(k_lut_ptr + 0)
        k_lo_t = tl.load(T_lut_ptr + 0)
        k_hi_v = tl.load(k_lut_ptr + n_lut - 1)
        k_hi_t = tl.load(T_lut_ptr + n_lut - 1)
        cp_lo_v = tl.load(cp_lut_ptr + 0)
        cp_hi_v = tl.load(cp_lut_ptr + n_lut - 1)

    # 1. K-Center & CP-Center
    phi_c = tl.sigmoid((Tc - mid) * inv_hw * sharpness)
    if use_lut:
        rk = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), k_lo_v, dtype=tl.float32)
        rc = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), cp_lo_v, dtype=tl.float32)
        for i in range(16):
            if i < n_lut - 1:
                t0, t1 = tl.load(T_lut_ptr + i), tl.load(T_lut_ptr + i + 1)
                vk0, vk1 = tl.load(k_lut_ptr + i), tl.load(k_lut_ptr + i + 1)
                vc0, vc1 = tl.load(cp_lut_ptr + i), tl.load(cp_lut_ptr + i + 1)
                m_in = (Tc >= t0) & (Tc < t1)
                al = (Tc - t0) / (t1 - t0 + 1e-9)
                rk = tl.where(m_in, vk0 + al * (vk1 - vk0), rk)
                rc = tl.where(m_in, vc0 + al * (vc1 - vc0), rc)
        kc = (1.0-mc)*k_powder + mc*tl.where(Tc <= k_lo_t, k_lo_v, tl.where(Tc >= k_hi_t, k_hi_v, rk))
        cp_base_c = tl.where(Tc <= k_lo_t, cp_lo_v, tl.where(Tc >= k_hi_t, cp_hi_v, rc))
    else:
        ks_c = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Tc - T_ref)), k_solid)
        kl_c = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Tc - T_ref)), k_liquid)
        kc = (1.0 - mc) * k_powder + mc * ((1.0 - phi_c) * ks_c + phi_c * kl_c)
        cp_base_c = tl.where(use_t_dep, cp_base * (1.0 + cp_coeff * (Tc - T_ref)), cp_base)

    cp_total = cp_base_c + L * (sharpness * phi_c * (1.0 - phi_c)) * inv_hw

    # Macros for neighbor K... wait, can't use python macros easily. 
    # I'll just write them carefully.
    
    # XL
    if use_lut:
        r = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), k_lo_v, dtype=tl.float32)
        for i in range(16):
            if i < n_lut - 1:
                t0, t1 = tl.load(T_lut_ptr+i), tl.load(T_lut_ptr+i+1)
                v0, v1 = tl.load(k_lut_ptr+i), tl.load(k_lut_ptr+i+1)
                r = tl.where((Txl >= t0) & (Txl < t1), v0 + (Txl-t0)/(t1-t0+1e-9)*(v1-v0), r)
        k_xl = (1.0-mxl)*k_powder + mxl*tl.where(Txl <= k_lo_t, k_lo_v, tl.where(Txl >= k_hi_t, k_hi_v, r))
    else:
        p = tl.sigmoid((Txl-mid)*inv_hw*sharpness)
        k_xl = (1.0-mxl)*k_powder + mxl*((1.0-p)*tl.where(use_t_dep, k_solid*(1.0+ks_coeff*(Txl-T_ref)), k_solid) + p*tl.where(use_t_dep, k_liquid*(1.0+kl_coeff*(Txl-T_ref)), k_liquid))

    # XR
    if use_lut:
        r = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), k_lo_v, dtype=tl.float32)
        for i in range(16):
            if i < n_lut - 1:
                t0, t1 = tl.load(T_lut_ptr+i), tl.load(T_lut_ptr+i+1)
                v0, v1 = tl.load(k_lut_ptr+i), tl.load(k_lut_ptr+i+1)
                r = tl.where((Txr >= t0) & (Txr < t1), v0 + (Txr-t0)/(t1-t0+1e-9)*(v1-v0), r)
        k_xr = (1.0-mxr)*k_powder + mxr*tl.where(Txr <= k_lo_t, k_lo_v, tl.where(Txr >= k_hi_t, k_hi_v, r))
    else:
        p = tl.sigmoid((Txr-mid)*inv_hw*sharpness)
        k_xr = (1.0-mxr)*k_powder + mxr*((1.0-p)*tl.where(use_t_dep, k_solid*(1.0+ks_coeff*(Txr-T_ref)), k_solid) + p*tl.where(use_t_dep, k_liquid*(1.0+kl_coeff*(Txr-T_ref)), k_liquid))

    # YU
    if use_lut:
        r = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), k_lo_v, dtype=tl.float32)
        for i in range(16):
            if i < n_lut - 1:
                t0, t1 = tl.load(T_lut_ptr+i), tl.load(T_lut_ptr+i+1)
                v0, v1 = tl.load(k_lut_ptr+i), tl.load(k_lut_ptr+i+1)
                r = tl.where((Tyu >= t0) & (Tyu < t1), v0 + (Tyu-t0)/(t1-t0+1e-9)*(v1-v0), r)
        k_yu = (1.0-myu)*k_powder + myu*tl.where(Tyu <= k_lo_t, k_lo_v, tl.where(Tyu >= k_hi_t, k_hi_v, r))
    else:
        p = tl.sigmoid((Tyu-mid)*inv_hw*sharpness)
        k_yu = (1.0-myu)*k_powder + myu*((1.0-p)*tl.where(use_t_dep, k_solid*(1.0+ks_coeff*(Tyu-T_ref)), k_solid) + p*tl.where(use_t_dep, k_liquid*(1.0+kl_coeff*(Tyu-T_ref)), k_liquid))

    # YD
    if use_lut:
        r = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), k_lo_v, dtype=tl.float32)
        for i in range(16):
            if i < n_lut - 1:
                t0, t1 = tl.load(T_lut_ptr+i), tl.load(T_lut_ptr+i+1)
                v0, v1 = tl.load(k_lut_ptr+i), tl.load(k_lut_ptr+i+1)
                r = tl.where((Tyd >= t0) & (Tyd < t1), v0 + (Tyd-t0)/(t1-t0+1e-9)*(v1-v0), r)
        k_yd = (1.0-myd)*k_powder + myd*tl.where(Tyd <= k_lo_t, k_lo_v, tl.where(Tyd >= k_hi_t, k_hi_v, r))
    else:
        p = tl.sigmoid((Tyd-mid)*inv_hw*sharpness)
        k_yd = (1.0-myd)*k_powder + myd*((1.0-p)*tl.where(use_t_dep, k_solid*(1.0+ks_coeff*(Tyd-T_ref)), k_solid) + p*tl.where(use_t_dep, k_liquid*(1.0+kl_coeff*(Tyd-T_ref)), k_liquid))

    # ZF
    if use_lut:
        r = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), k_lo_v, dtype=tl.float32)
        for i in range(16):
            if i < n_lut - 1:
                t0, t1 = tl.load(T_lut_ptr+i), tl.load(T_lut_ptr+i+1)
                v0, v1 = tl.load(k_lut_ptr+i), tl.load(k_lut_ptr+i+1)
                r = tl.where((Tzf >= t0) & (Tzf < t1), v0 + (Tzf-t0)/(t1-t0+1e-9)*(v1-v0), r)
        k_zf = (1.0-mzf)*k_powder + mzf*tl.where(Tzf <= k_lo_t, k_lo_v, tl.where(Tzf >= k_hi_t, k_hi_v, r))
    else:
        p = tl.sigmoid((Tzf-mid)*inv_hw*sharpness)
        k_zf = (1.0-mzf)*k_powder + mzf*((1.0-p)*tl.where(use_t_dep, k_solid*(1.0+ks_coeff*(Tzf-T_ref)), k_solid) + p*tl.where(use_t_dep, k_liquid*(1.0+kl_coeff*(Tzf-T_ref)), k_liquid))

    # ZB
    if use_lut:
        r = tl.full((BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z), k_lo_v, dtype=tl.float32)
        for i in range(16):
            if i < n_lut - 1:
                t0, t1 = tl.load(T_lut_ptr+i), tl.load(T_lut_ptr+i+1)
                v0, v1 = tl.load(k_lut_ptr+i), tl.load(k_lut_ptr+i+1)
                r = tl.where((Tzb >= t0) & (Tzb < t1), v0 + (Tzb-t0)/(t1-t0+1e-9)*(v1-v0), r)
        k_zb = (1.0-mzb)*k_powder + mzb*tl.where(Tzb <= k_lo_t, k_lo_v, tl.where(Tzb >= k_hi_t, k_hi_v, r))
    else:
        p = tl.sigmoid((Tzb-mid)*inv_hw*sharpness)
        k_zb = (1.0-mzb)*k_powder + mzb*((1.0-p)*tl.where(use_t_dep, k_solid*(1.0+ks_coeff*(Tzb-T_ref)), k_solid) + p*tl.where(use_t_dep, k_liquid*(1.0+kl_coeff*(Tzb-T_ref)), k_liquid))

    # Divergence
    div_x = ( (2.0*kc*k_xr)/(kc+k_xr+1e-12)*(Txr-Tc) - (2.0*kc*k_xl)/(kc+k_xl+1e-12)*(Tc-Txl) ) / (dx*dx)
    div_y = ( (2.0*kc*k_yd)/(kc+k_yd+1e-12)*(Tyd-Tc) - (2.0*kc*k_yu)/(kc+k_yu+1e-12)*(Tc-Tyu) ) / (dy*dy)
    div_z = ( (2.0*kc*k_zb)/(kc+k_zb+1e-12)*(Tzb-Tc) - (2.0*kc*k_zf)/(kc+k_zf+1e-12)*(Tc-Tzf) ) / (dz*dz)

    Qc = tl.load(Q_ptr + idx_c, mask=m_all)
    rhs = div_x + div_y + div_z + Qc - loss_h * (Tc - T_ambient)
    tl.store(T_new_ptr + idx_c, Tc + (dt/rho)*rhs/(cp_total+1e-9), mask=m_all)

def run_thermal_step_3d_triton(T, mask, Q, sim_cfg, mat_cfg, dt):
    nx, ny, nz = T.shape[-3], T.shape[-2], T.shape[-1]
    T_new = torch.empty_like(T)
    stride_x, stride_y, stride_z = ny * nz, nz, 1
    BLOCK_X, BLOCK_Y, BLOCK_Z = 8, 8, 4 
    grid = ((nx+BLOCK_X-1)//BLOCK_X, (ny+BLOCK_Y-1)//BLOCK_Y, (nz+BLOCK_Z-1)//BLOCK_Z)

    if mat_cfg.use_lut and mat_cfg.T_lut is not None:
        T_lut = torch.tensor(mat_cfg.T_lut, device=T.device, dtype=T.dtype)
        k_lut = torch.tensor(mat_cfg.k_lut, device=T.device, dtype=T.dtype)
        cp_lut = torch.tensor(mat_cfg.cp_lut, device=T.device, dtype=T.dtype)
        n_lut = len(mat_cfg.T_lut)
        use_lut = True
    else:
        T_lut = k_lut = cp_lut = torch.zeros(1, device=T.device, dtype=T.dtype)
        n_lut = 1 
        use_lut = False

    _thermal_step_3d_kernel[grid](
        T, mask, Q, T_new, nx, ny, nz, stride_x, stride_y, stride_z,
        sim_cfg.dz, sim_cfg.dy, sim_cfg.dx, dt,
        sim_cfg.loss_h, sim_cfg.T_ambient,
        mat_cfg.k_powder, mat_cfg.k_solid, mat_cfg.k_liquid,
        mat_cfg.cp_base, mat_cfg.rho,
        mat_cfg.T_solidus, mat_cfg.T_liquidus, mat_cfg.latent_heat_L,
        mat_cfg.transition_sharpness,
        use_t_dep=mat_cfg.use_T_dep, T_ref=mat_cfg.T_ref,
        ks_coeff=mat_cfg.k_solid_T_coeff, kl_coeff=mat_cfg.k_liquid_T_coeff, cp_coeff=mat_cfg.cp_T_coeff,
        use_lut=use_lut, T_lut_ptr=T_lut, k_lut_ptr=k_lut, cp_lut_ptr=cp_lut, n_lut=n_lut,
        BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y, BLOCK_SIZE_Z=BLOCK_Z,
        num_warps=4
    )
    return T_new
