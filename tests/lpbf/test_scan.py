# tests/lpbf/test_scan.py
import torch

from lpbf.scan.engine import ScanPathGenerator
from lpbf.scan.sources import GaussianBeam, GaussianSourceConfig


def test_gaussian_source_integral_2d():
    """
    Integrate Gaussian source over 2D plane. Should roughly equal Power * Eta.
    """
    P = 100.0
    sigma = 0.5e-3  # 0.5 mm
    eta = 0.8

    cfg = GaussianSourceConfig(power=P, eta=eta, sigma=sigma)
    source = GaussianBeam(cfg)

    # Grid covering +/- 4 sigma
    L = 8 * sigma
    N = 400
    dx = L / N

    x = torch.linspace(-L / 2, L / 2, N)
    y = torch.linspace(-L / 2, L / 2, N)
    Y, X = torch.meshgrid(y, x, indexing="ij")

    Q_flux = source.intensity(X, Y, None, x0=0.0, y0=0.0)

    # Integrate: sum * dx * dy
    # Q_flux is W/m^2 (surface flux mode)
    full_power = torch.sum(Q_flux) * dx * dx

    expected = P * eta

    # Should be very close (within 1%)
    assert abs(full_power.item() - expected) / expected < 0.01


def test_gaussian_source_volumetric_3d():
    """
    Integrate 3D source with depth.
    Energy conservation: Integral(Q dV) = Power * Eta.
    """
    P = 100.0
    sigma = 1.0  # m
    depth = 2.0  # m
    eta = 1.0

    cfg = GaussianSourceConfig(power=P, eta=eta, sigma=sigma, depth=depth)
    source = GaussianBeam(cfg)

    # Grid
    # XY coverage
    L = 6 * sigma
    N = 50
    dx = L / N
    x = torch.linspace(-L / 2, L / 2, N)
    y = torch.linspace(-L / 2, L / 2, N)

    # Z coverage: 0 to 5 * depth
    D = 5 * depth
    Nz = 50
    dz = D / Nz
    z = torch.linspace(0, -D, Nz)  # assume z goes negative into material

    # Meshgrid 3D: [Z, Y, X]
    # torch.meshgrid returns tuple of tensors
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")

    # Source at origin, surface at z=0
    Q_vol = source.intensity(grid_x, grid_y, grid_z, x0=0.0, y0=0.0, z0=0.0)

    # Integrate
    total_power = torch.sum(Q_vol) * dx * dx * dz

    # Tolerances: 3D grid is coarse, boundaries cut off tails. Maybe 5% error.
    assert abs(total_power.item() - P * eta) / (P * eta) < 0.05


def test_scan_hatch_generation():
    """
    Verify event generation for a simple hatch.
    """
    # 2x2 mm square
    events = ScanPathGenerator.hatch(
        corner_start=(0.0, 0.0),
        width=2e-3,
        height=2e-3,
        spacing=0.5e-3,  # 0.5 mm -> should be 5 lines: y=0, 0.5, 1.0, 1.5, 2.0
        power=100.0,
        speed=1.0,
        angle_deg=0.0,
    )

    # Expected lines:
    # 1. (0,0)->(2,0)
    # 2. Travel to (2, 0.5)
    # 3. (2, 0.5)->(0, 0.5)
    # 4. Travel to (0, 1.0)
    # 5. (0, 1.0)->(2, 1.0)
    # 6. Travel to (2, 1.5)
    # 7. (2, 1.5)->(0, 1.5)
    # 8. Travel to (0, 2.0)
    # 9. (0, 2.0)->(2, 2.0)

    # Total events: 5 * Scan + 4 * Travel = 9 events
    assert len(events) == 9

    # Check first event
    e0 = events[0]
    assert e0.laser_on
    assert e0.x_start == 0.0 and e0.x_end == 2e-3
    assert e0.y_start == 0.0 and e0.y_end == 0.0

    # Check first travel
    e1 = events[1]
    assert not e1.laser_on
    assert e1.x_start == 2e-3 and e1.x_end == 2e-3
    assert e1.y_start == 0.0 and e1.y_end == 0.5e-3
