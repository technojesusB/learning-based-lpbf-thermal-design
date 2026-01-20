# src/lpbf/physics/ops.py
import torch
import torch.nn.functional as F


def harmonic_mean(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the harmonic mean of two tensors.

    H(a, b) = 2 * a * b / (a + b)

    The harmonic mean is standard for computing effective thermal conductivity
    at the interface between two control volumes in Finite Volume / Difference schemes.
    It ensures flux continuity.

    Args:
        a (torch.Tensor): First value.
        b (torch.Tensor): Second value.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        torch.Tensor: Harmonic mean.
    """
    return (2.0 * a * b) / (a + b + eps)


def div_k_grad(
    T: torch.Tensor, k: torch.Tensor, dx: float, dy: float, dz: float | None = None
) -> torch.Tensor:
    r"""
    Compute the divergence of the conductive heat flux: Div(k * Grad(T)) [W/m^3].

    Operator:
        \nabla \cdot (k \nabla T) = \partial_x (k \partial_x T) + \partial_y (k \partial_y T) + \partial_z (k \partial_z T)

    Method:
        Standard 2nd-order Central Finite Difference on a dense grid (5-point stencil in 2D, 7-point in 3D).
        The conductivity k is averaged at cell faces using the harmonic mean.

    Boundary Conditions:
        Applies Homogeneous Neumann Boundary Conditions (Zero Flux) at all domain boundaries.
        Implemented via 'replicate' padding, effectively mirroring the temperature at the boundary.

    Args:
        T (torch.Tensor): Temperature field [K]. Shape (B, C, [D], H, W).
        k (torch.Tensor): Thermal conductivity field [W/(m K)]. Shape matches T.
        dx (float): Grid spacing in X [m].
        dy (float): Grid spacing in Y [m].
        dz (float | None): Grid spacing in Z [m]. Required if T is 3D (5-dim).

    Returns:
        torch.Tensor: The Laplacian term values [W/m^3]. Shape matches T valid region (boundary padding removed).
    """
    dims = T.ndim
    is_3d = dims == 5

    # Pad for Neumann (Reflect) boundary conditions
    # pad args are (left, right, top, bottom, front, back)
    if is_3d:
        assert dz is not None, "dz must be provided for 3D tensors"
        pad = (1, 1, 1, 1, 1, 1)
        # T: B, C, D, H, W
        T_p = F.pad(T, pad, mode="replicate")
        k_p = F.pad(k, pad, mode="replicate")

        # Slices
        # Center: 1:-1
        # Left/Right (X): 0:-2 / 2:
        # Up/Down (Y): 0:-2 / 2:
        # Front/Back (Z): 0:-2 / 2:

        Tc = T_p[..., 1:-1, 1:-1, 1:-1]
        kc = k_p[..., 1:-1, 1:-1, 1:-1]

        # X neighbors
        Tx_l = T_p[..., 1:-1, 1:-1, 0:-2]
        Tx_r = T_p[..., 1:-1, 1:-1, 2:]
        kx_l = harmonic_mean(kc, k_p[..., 1:-1, 1:-1, 0:-2])
        kx_r = harmonic_mean(kc, k_p[..., 1:-1, 1:-1, 2:])

        flux_x_r = kx_r * (Tx_r - Tc) / dx
        flux_x_l = kx_l * (Tc - Tx_l) / dx
        div_x = (flux_x_r - flux_x_l) / dx

        # Y neighbors
        Ty_u = T_p[..., 1:-1, 0:-2, 1:-1]  # Y-1
        Ty_d = T_p[..., 1:-1, 2:, 1:-1]  # Y+1
        ky_u = harmonic_mean(kc, k_p[..., 1:-1, 0:-2, 1:-1])
        ky_d = harmonic_mean(kc, k_p[..., 1:-1, 2:, 1:-1])

        flux_y_d = ky_d * (Ty_d - Tc) / dy
        flux_y_u = ky_u * (Tc - Ty_u) / dy
        div_y = (flux_y_d - flux_y_u) / dy

        # Z neighbors
        Tz_f = T_p[..., 0:-2, 1:-1, 1:-1]  # Z-1
        Tz_b = T_p[..., 2:, 1:-1, 1:-1]  # Z+1
        kz_f = harmonic_mean(kc, k_p[..., 0:-2, 1:-1, 1:-1])
        kz_b = harmonic_mean(kc, k_p[..., 2:, 1:-1, 1:-1])

        flux_z_b = kz_b * (Tz_b - Tc) / dz
        flux_z_f = kz_f * (Tc - Tz_f) / dz
        div_z = (flux_z_b - flux_z_f) / dz

        return div_x + div_y + div_z

    else:
        # 2D case: B, C, H, W
        pad = (1, 1, 1, 1)
        T_p = F.pad(T, pad, mode="replicate")
        k_p = F.pad(k, pad, mode="replicate")

        Tc = T_p[..., 1:-1, 1:-1]
        kc = k_p[..., 1:-1, 1:-1]

        # X
        Tx_l = T_p[..., 1:-1, 0:-2]
        Tx_r = T_p[..., 1:-1, 2:]
        kx_l = harmonic_mean(kc, k_p[..., 1:-1, 0:-2])
        kx_r = harmonic_mean(kc, k_p[..., 1:-1, 2:])

        flux_x_r = kx_r * (Tx_r - Tc) / dx
        flux_x_l = kx_l * (Tc - Tx_l) / dx
        div_x = (flux_x_r - flux_x_l) / dx

        # Y
        Ty_u = T_p[..., 0:-2, 1:-1]
        Ty_d = T_p[..., 2:, 1:-1]
        ky_u = harmonic_mean(kc, k_p[..., 0:-2, 1:-1])
        ky_d = harmonic_mean(kc, k_p[..., 2:, 1:-1])

        flux_y_d = ky_d * (Ty_d - Tc) / dy
        flux_y_u = ky_u * (Tc - Ty_u) / dy
        div_y = (flux_y_d - flux_y_u) / dy

        return div_x + div_y
