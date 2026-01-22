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
    use_t_dep, T_ref,
    ks_coeff, kl_coeff, cp_coeff,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    o_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)[:, None, None]
    o_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)[None, :, None]
    o_z = pid_z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)[None, None, :]

    m_all = (o_x < Nx) & (o_y < Ny) & (o_z < Nz)
    id_c = o_x * stride_x + o_y * stride_y + o_z * stride_z
    
    Tc = tl.load(T_ptr + id_c, mask=m_all)
    mc = tl.load(mask_ptr + id_c, mask=m_all).to(tl.float32)

    # Neighbors
    xl = tl.maximum(0, o_x - 1)
    xr = tl.minimum(Nx - 1, o_x + 1)
    yu = tl.maximum(0, o_y - 1)
    yd = tl.minimum(Ny - 1, o_y + 1)
    zf = tl.maximum(0, o_z - 1)
    zb = tl.minimum(Nz - 1, o_z + 1)

    Txl = tl.load(T_ptr + (xl*stride_x + o_y*stride_y + o_z*stride_z), mask=m_all)
    mxl = tl.load(mask_ptr + (xl*stride_x + o_y*stride_y + o_z*stride_z), mask=m_all).to(tl.float32)
    Txr = tl.load(T_ptr + (xr*stride_x + o_y*stride_y + o_z*stride_z), mask=m_all)
    mxr = tl.load(mask_ptr + (xr*stride_x + o_y*stride_y + o_z*stride_z), mask=m_all).to(tl.float32)

    Tyu = tl.load(T_ptr + (o_x*stride_x + yu*stride_y + o_z*stride_z), mask=m_all)
    myu = tl.load(mask_ptr + (o_x*stride_x + yu*stride_y + o_z*stride_z), mask=m_all).to(tl.float32)
    Tyd = tl.load(T_ptr + (o_x*stride_x + yd*stride_y + o_z*stride_z), mask=m_all)
    myd = tl.load(mask_ptr + (o_x*stride_x + yd*stride_y + o_z*stride_z), mask=m_all).to(tl.float32)

    Tzf = tl.load(T_ptr + (o_x*stride_x + o_y*stride_y + zf*stride_z), mask=m_all)
    mzf = tl.load(mask_ptr + (o_x*stride_x + o_y*stride_y + zf*stride_z), mask=m_all).to(tl.float32)
    Tzb = tl.load(T_ptr + (o_x*stride_x + o_y*stride_y + zb*stride_z), mask=m_all)
    mzb = tl.load(mask_ptr + (o_x*stride_x + o_y*stride_y + zb*stride_z), mask=m_all).to(tl.float32)

    # Phys constants
    mid = 0.5 * (T_sol + T_liq)
    inv_hw = 1.0 / (0.5 * (T_liq - T_sol) + 1e-9)

    # Thermal Conductivity Kernel (Inlined)
    # Center
    phi_c = tl.sigmoid((Tc - mid) * inv_hw * sharpness)
    ks_c = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Tc - T_ref)), k_solid)
    kl_c = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Tc - T_ref)), k_liquid)
    kc = (1.0 - mc) * k_powder + mc * ((1.0 - phi_c) * ks_c + phi_c * kl_c)

    # X
    phi_xl = tl.sigmoid((Txl - mid) * inv_hw * sharpness)
    ks_xl = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Txl - T_ref)), k_solid)
    kl_xl = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Txl - T_ref)), k_liquid)
    k_xl = (1.0 - mxl) * k_powder + mxl * ((1.0 - phi_xl) * ks_xl + phi_xl * kl_xl)

    phi_xr = tl.sigmoid((Txr - mid) * inv_hw * sharpness)
    ks_xr = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Txr - T_ref)), k_solid)
    kl_xr = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Txr - T_ref)), k_liquid)
    k_xr = (1.0 - mxr) * k_powder + mxr * ((1.0 - phi_xr) * ks_xr + phi_xr * kl_xr)

    # Y
    phi_yu = tl.sigmoid((Tyu - mid) * inv_hw * sharpness)
    ks_yu = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Tyu - T_ref)), k_solid)
    kl_yu = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Tyu - T_ref)), k_liquid)
    k_yu = (1.0 - myu) * k_powder + myu * ((1.0 - phi_yu) * ks_yu + phi_yu * kl_yu)

    phi_yd = tl.sigmoid((Tyd - mid) * inv_hw * sharpness)
    ks_yd = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Tyd - T_ref)), k_solid)
    kl_yd = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Tyd - T_ref)), k_liquid)
    k_yd = (1.0 - myd) * k_powder + myd * ((1.0 - phi_yd) * ks_yd + phi_yd * kl_yd)

    # Z
    phi_zf = tl.sigmoid((Tzf - mid) * inv_hw * sharpness)
    ks_zf = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Tzf - T_ref)), k_solid)
    kl_zf = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Tzf - T_ref)), k_liquid)
    k_zf = (1.0 - mzf) * k_powder + mzf * ((1.0 - phi_zf) * ks_zf + phi_zf * kl_zf)

    phi_zb = tl.sigmoid((Tzb - mid) * inv_hw * sharpness)
    ks_zb = tl.where(use_t_dep, k_solid * (1.0 + ks_coeff * (Tzb - T_ref)), k_solid)
    kl_zb = tl.where(use_t_dep, k_liquid * (1.0 + kl_coeff * (Tzb - T_ref)), k_liquid)
    k_zb = (1.0 - mzb) * k_powder + mzb * ((1.0 - phi_zb) * ks_zb + phi_zb * kl_zb)

    # Divergence
    div_x = ( (2.0*kc*k_xr)/(kc+k_xr+1e-12)*(Txr-Tc) - (2.0*kc*k_xl)/(kc+k_xl+1e-12)*(Tc-Txl) ) / (dx*dx)
    div_y = ( (2.0*kc*k_yd)/(kc+k_yd+1e-12)*(Tyd-Tc) - (2.0*kc*k_yu)/(kc+k_yu+1e-12)*(Tc-Tyu) ) / (dy*dy)
    div_z = ( (2.0*kc*k_zb)/(kc+k_zb+1e-12)*(Tzb-Tc) - (2.0*kc*k_zf)/(kc+k_zf+1e-12)*(Tc-Tzf) ) / (dz*dz)

    Q = tl.load(Q_ptr + id_c, mask=m_all)
    # CP
    d_phi_dT = (sharpness * phi_c * (1.0 - phi_c)) * inv_hw
    cp = tl.where(use_t_dep, cp_base * (1.0 + cp_coeff * (Tc - T_ref)), cp_base) + L * d_phi_dT
    
    T_new = Tc + (dt / rho) * (div_x + div_y + div_z + Q - loss_h * (Tc - T_ambient)) / (cp + 1e-9)
    tl.store(T_new_ptr + id_c, T_new, mask=m_all)

def run_thermal_step_3d_triton(T, mask, Q, sim_cfg, mat_cfg, dt):
    nx, ny, nz = T.shape[-3], T.shape[-2], T.shape[-1]
    T_new = torch.empty_like(T)
    stride_x, stride_y, stride_z = ny * nz, nz, 1
    BLOCK_X, BLOCK_Y, BLOCK_Z = 8, 8, 4 
    grid = ((nx+BLOCK_X-1)//BLOCK_X, (ny+BLOCK_Y-1)//BLOCK_Y, (nz+BLOCK_Z-1)//BLOCK_Z)
    _thermal_step_3d_kernel[grid](
        T, mask, Q, T_new, nx, ny, nz, stride_x, stride_y, stride_z,
        sim_cfg.dx, sim_cfg.dy, sim_cfg.dz, dt,
        sim_cfg.loss_h, sim_cfg.T_ambient,
        mat_cfg.k_powder, mat_cfg.k_solid, mat_cfg.k_liquid,
        mat_cfg.cp_base, mat_cfg.rho,
        mat_cfg.T_solidus, mat_cfg.T_liquidus, mat_cfg.latent_heat_L,
        mat_cfg.transition_sharpness,
        mat_cfg.use_T_dep, mat_cfg.T_ref,
        mat_cfg.k_solid_T_coeff, mat_cfg.k_liquid_T_coeff, mat_cfg.cp_T_coeff,
        BLOCK_SIZE_X=BLOCK_X, BLOCK_SIZE_Y=BLOCK_Y, BLOCK_SIZE_Z=BLOCK_Z,
        num_warps=4
    )
    return T_new
