# data/single_dot.py
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
import torch

from physics.material import MaterialConfig, k_eff, cp_eff
from physics.operators import div_k_grad_2d
from scan.heat_source import PulseConfig, pulse_Q
from utils.history import make_smooth_preheat_field
from utils.CUDA_check import cuda_check
from utils.grid import make_xy_grid

class SingleDotSimConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    H: int = Field(default = 128, ge=8)
    W: int = Field(default = 128, ge=8)

    # simulation horizon and stepping
    T_steps: int = Field(default = 120, ge = 2)
    dt: float = Field(default = 0.005, gt = 0.0)

    # substepping inside each dt for better stability/temporal resolution of pulse
    substeps: int = Field(default = 4, ge = 1)

    # ambient / initial conditions
    T_ambient: float = Field(default = 0.0)
    # optional preheat offset added to initial field
    preheat: float = Field(default = 0.0)

    # simple linear cooling to ambient (toy convective loss)
    loss_h: float = Field(default = 0.0, ge = 0.0)

    print_stability: bool = Field(False)


@torch.no_grad()
def simulate_single_dot(
    sim: SingleDotSimConfig,
    mat: MaterialConfig,
    pulse: PulseConfig,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    T0: torch.Tensor | None = None,
):
    if device is None:
        device = cuda_check()
    else:
        device = device

    H, W = sim.H, sim.W

    if T0 is None:
        T = torch.full((1, 1, H, W), sim.T_ambient + sim.preheat, device=device, dtype=dtype)
    else:
        # expect shape [1,1,H,W]
        assert T0.shape == (1, 1, H, W), f"T0 must be [1,1,{H},{W}]"
        T = T0.to(device=device, dtype=dtype).clone()

    dx = 1.0 / (W - 1)
    dy = 1.0 / (H - 1)

    # crude stability estimate (uses max possible k/cp)
    if sim.print_stability:
        k_max = max(mat.k_powder, mat.k_solid, mat.k_liquid)
        cp_min = sim.T_ambient + 0.0  # not used; use cfg.cp_base as floor
        alpha_est = k_max / (mat.rho * max(mat.cp_base, 1e-6))
        dt_max = (dx * dx) / (4.0 * alpha_est + 1e-12)
        print(f"[stability] dx={dx:.4e}, alpha_est={alpha_est:.4e}, dt_sub={sim.dt/sim.substeps:.4e}, dt_max≈{dt_max:.4e}")
    
    
    X, Y = make_xy_grid(H, W, device, dtype)

    # initial temperature field (can later be replaced with a randomized history map)
    T = torch.full((1, 1, H, W), sim.T_ambient + sim.preheat, device=device, dtype=dtype)

    # Track peak and peak time index
    T_peak = T.clone()
    t_peak_idx = torch.zeros((1, 1, H, W), device=device, dtype=torch.long)

    # Store history for a simple cooling-rate estimate
    T_hist = torch.empty((sim.T_steps, 1, 1, H, W), device=device, dtype=dtype)

    dt_sub = sim.dt / sim.substeps

    for n in range(sim.T_steps):
        # substeps improve stability and resolve pulse gating better
        for s in range(sim.substeps):
            t = torch.tensor((n + s / sim.substeps) * sim.dt, device=device, dtype=dtype)

            k = k_eff(T, mat)               # [1,1,H,W]
            cp = cp_eff(T, mat)             # [1,1,H,W]
            
            k = torch.clamp(k, min = 1e-6, max = 1.0)
            cp = torch.clamp(cp, min = 1e-3)

            div_term = div_k_grad_2d(T, k, dx=dx, dy=dy)

            Q = pulse_Q(X, Y, t, pulse)

            # heat equation: rho * cp(T) * dT/dt = div(k grad T) + Q - h*(T-Tamb)
            rhs = div_term + Q - sim.loss_h * (T - sim.T_ambient)

            dT = (dt_sub / (mat.rho)) * (rhs / (cp + 1e-12))
            T = T + dT

        T_hist[n] = T

        better = T > T_peak
        T_peak = torch.where(better, T, T_peak)
        t_peak_idx = torch.where(better, torch.full_like(t_peak_idx, n), t_peak_idx)

    # Cooling rate estimate: (T_peak - T(t_peak+1)) / dt
    idx_next = torch.clamp(t_peak_idx + 1, max=sim.T_steps - 1)

    Th = T_hist[:, 0, 0].reshape(sim.T_steps, -1)   # [T, HW]
    inx = idx_next[0, 0].reshape(-1)                # [HW]
    cols = torch.arange(inx.numel(), device=device)
    T_next = Th[inx, cols].reshape(1, 1, H, W)

    R_cool = (T_peak - T_next) / sim.dt

    return {
        "T_peak": T_peak,     # [1,1,H,W]
        "R_cool": R_cool,     # [1,1,H,W]
        "T_final": T,         # [1,1,H,W]
        "t_peak_idx": t_peak_idx,
    }


if __name__ == "__main__":
    
    device = cuda_check()

    sim = SingleDotSimConfig(H=128, 
                             W=128, 
                             T_steps=200, 
                             dt=0.002, 
                             substeps=4, 
                             preheat=0.05, 
                             loss_h=0.5)
    
    T0 = make_smooth_preheat_field(
                                H=sim.H, W=sim.W,
                                device=device,
                                dtype=torch.float32,
                                ambient=sim.T_ambient,
                                amplitude=0.05,       # Stärke der Vorwärmung
                                kernel_size=51,
                                sigma=12.0,
                                clamp_min=sim.T_ambient,   # nie unter ambient (optional)   
                            )
    
    mat = MaterialConfig(
        k_powder=0.0012,
        k_solid=0.018,
        k_liquid=0.012,
        cp_base=1.0,
        T_solidus=0.60,
        T_liquidus=0.70,
        latent_heat_L=0.35,
        latent_width=0.03,
    )

    pulse = PulseConfig(x0=0.5, 
                        y0=0.5, 
                        power=2.0, 
                        t_on=0.02, 
                        t_off=0.10, 
                        sigma=0.02, 
                        eta=1.0)

    out = simulate_single_dot(sim, mat, pulse, T0=T0)
    T_peak = out["T_peak"]
    R_cool = out["R_cool"]

    print("T_peak:", T_peak.shape, float(T_peak.min()), float(T_peak.max()))
    print("R_cool:", R_cool.shape, float(R_cool.min()), float(R_cool.max()))
