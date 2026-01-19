# tests/lpbf/test_physics.py
import pytest
import torch
import math
from lpbf.physics.material import MaterialConfig, melt_fraction, cp_eff, k_eff
from lpbf.physics.ops import div_k_grad

@pytest.fixture
def mat_cfg():
    return MaterialConfig(
        k_powder=0.1,
        k_solid=1.0,
        k_liquid=1.2,
        cp_base=1.0,
        rho=1.0,
        T_solidus=100.0,
        T_liquidus=110.0,
        latent_heat_L=50.0,
        transition_sharpness=5.0
    )

def test_melt_fraction(mat_cfg):
    """Test transitions from 0 to 1."""
    T_cold = torch.tensor([50.0])
    T_hot = torch.tensor([150.0])
    T_mid = torch.tensor([105.0])
    
    phi_cold = melt_fraction(T_cold, mat_cfg).item()
    phi_hot = melt_fraction(T_hot, mat_cfg).item()
    phi_mid = melt_fraction(T_mid, mat_cfg).item()
    
    assert phi_cold < 0.01
    assert phi_hot > 0.99
    assert abs(phi_mid - 0.5) < 0.05

def test_latent_heat_integration(mat_cfg):
    """
    Integrate cp(T) from 80K to 130K.
    Expected: (130-80)*cp_base + L
    """
    T = torch.linspace(80, 130, 2000)
    dT = (130 - 80) / 2000
    
    cp_vals = cp_eff(T, mat_cfg)
    H_total = torch.sum(cp_vals) * dT
    
    expected_sensible = (130 - 80) * mat_cfg.cp_base
    expected_total = expected_sensible + mat_cfg.latent_heat_L
    
    # Tolerances: The Gaussian is truncated, so it might be slightly off
    assert abs(H_total.item() - expected_total) / expected_total < 0.05

def test_ops_laplacian_2d():
    """
    Test div(k grad T) for constant k=1.
    T = x^2 + y^2 => grad T = (2x, 2y) => div grad T = 2+2=4.
    """
    H, W = 20, 20
    dx = 0.1
    dy = 0.1
    
    y = torch.arange(H, dtype=torch.float64) * dy
    x = torch.arange(W, dtype=torch.float64) * dx
    Y, X = torch.meshgrid(y, x, indexing='ij') # [H, W]
    
    T = (X**2 + Y**2).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    k = torch.ones_like(T)
    
    print(f"DEBUG: dx={dx}, T[0,0,0,:5] = {T[0,0,0,:5]}")
    
    # For interior points, result should be close to 4.0
    lap = div_k_grad(T, k, dx, dy)
    
    # check center region to avoid boundary effects
    lap_inner = lap[0, 0, 2:-2, 2:-2]
    
    # Error should be small (finite difference error)
    # central difference of x^2 is exact, so error should be effectively 0
    assert torch.allclose(lap_inner, torch.tensor(4.0, dtype=torch.float64), atol=1e-5)

def test_ops_conservation_2d():
    """
    Test that sum(div term) is ~0 for a closed system (Neumann BC).
    Total heat change should be 0.
    """
    H, W = 30, 30
    T = torch.rand(1, 1, H, W, dtype=torch.float64)
    k = torch.ones_like(T)
    dx = 0.1
    dy = 0.1
    
    lap = div_k_grad(T, k, dx, dy)
    
    # Sum of divergence over the whole domain should be 0 (if flux cancels out)
    # Actually, discrete conservation depends on the scheme.
    # Our flux formulation F_r - F_l should sum to (F_boundary) which is 0.
    total_change = lap.sum()
    assert abs(total_change) < 1e-5
