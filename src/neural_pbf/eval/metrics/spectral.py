"""Spectral fidelity metrics for LPBF thermal surrogate evaluation.

Public API:
    compute_radial_psd              -- radially averaged 3-D PSD
    compute_axis_psd                -- per-axis 1-D PSD slices
    calculate_total_variation       -- L1 TV (spatial smoothness)
    calculate_boundary_discontinuity -- patch-boundary artefact ratio (PBD)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_pbf.schemas.viz import THEME

MODEL_PALETTE = THEME.spectral.model_palette


def compute_radial_psd(
    volume: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radially averaged Power Spectral Density of a 3-D volume.

    Args:
        volume: 3-D tensor (D, H, W).

    Returns:
        (freqs, radial_profile) — both 1-D arrays of the same length.

    Raises:
        ValueError: volume is not 3-D.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape}")

    vol_np = volume.detach().cpu().numpy()
    f_coeffs = np.fft.fftn(vol_np)
    f_shifted = np.fft.fftshift(f_coeffs)
    psd_3d = np.abs(f_shifted) ** 2

    nz, ny, nx = psd_3d.shape
    z, y, x = np.indices(psd_3d.shape)
    center = (nz // 2, ny // 2, nx // 2)
    r = np.sqrt(
        (x - center[2]) ** 2 + (y - center[1]) ** 2 + (z - center[0]) ** 2
    ).astype(int)

    tbin = np.bincount(r.ravel(), psd_3d.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = np.divide(
        tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr > 0
    )

    freqs_radial = np.linspace(0, 0.5, len(radial_profile))
    return freqs_radial, radial_profile


def compute_axis_psd(
    volume: torch.Tensor,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Compute PSD slices along the X, Y, and Z axes through the centre.

    Args:
        volume: 3-D tensor (D, H, W).

    Returns:
        Three (freqs, psd) tuples for the X, Y, and Z axes respectively.
    """
    vol_np = volume.detach().cpu().numpy()
    f_coeffs = np.fft.fftn(vol_np)
    f_shifted = np.fft.fftshift(f_coeffs)
    psd_3d = np.abs(f_shifted) ** 2

    nz, ny, nx = psd_3d.shape
    cz, cy, cx = nz // 2, ny // 2, nx // 2

    psd_x = psd_3d[cz, cy, cx:]
    psd_y = psd_3d[cz, cy:, cx]
    psd_z = psd_3d[cz:, cy, cx]

    return (
        (np.linspace(0, 0.5, len(psd_x)), psd_x),
        (np.linspace(0, 0.5, len(psd_y)), psd_y),
        (np.linspace(0, 0.5, len(psd_z)), psd_z),
    )


def calculate_total_variation(volume: torch.Tensor) -> float:
    """Compute the Total Variation (L1 norm of gradients) of a 3-D volume.

    Args:
        volume: 3-D tensor (D, H, W).

    Returns:
        Scalar TV value.
    """
    diff_z = torch.abs(volume[1:, :, :] - volume[:-1, :, :]).mean()
    diff_y = torch.abs(volume[:, 1:, :] - volume[:, :-1, :]).mean()
    diff_x = torch.abs(volume[:, :, 1:] - volume[:, :, :-1]).mean()
    return (diff_z + diff_y + diff_x).item()


def calculate_boundary_discontinuity(
    volume: torch.Tensor,
    patch_size: int = 8,
) -> float:
    """Measure patch-boundary artefacts (PBD = boundary grad / interior grad).

    A PBD of 1.0 means boundary gradients match interior gradients (ideal).
    Values > 1.0 indicate visible patch-seam artefacts.

    Args:
        volume:     3-D tensor (D, H, W).
        patch_size: Spatial patch edge length used during inference.

    Returns:
        PBD ratio.  Returns 1.0 when no patch boundaries exist.
    """
    nz, ny, nx = volume.shape
    grad_z = torch.abs(volume[1:, :, :] - volume[:-1, :, :])
    grad_y = torch.abs(volume[:, 1:, :] - volume[:, :-1, :])
    grad_x = torch.abs(volume[:, :, 1:] - volume[:, :, :-1])

    def get_boundary_mask(dim_size: int, p_size: int) -> torch.Tensor:
        mask = torch.zeros(dim_size - 1, dtype=torch.bool)
        for i in range(p_size - 1, dim_size - 1, p_size):
            mask[i] = True
        return mask

    mask_z = get_boundary_mask(nz, patch_size)
    mask_y = get_boundary_mask(ny, patch_size)
    mask_x = get_boundary_mask(nx, patch_size)

    b_grads = []
    i_grads = []
    if mask_z.any():
        b_grads.append(grad_z[mask_z, :, :].mean())
        i_grads.append(grad_z[~mask_z, :, :].mean())
    if mask_y.any():
        b_grads.append(grad_y[:, mask_y, :].mean())
        i_grads.append(grad_y[:, ~mask_y, :].mean())
    if mask_x.any():
        b_grads.append(grad_x[:, :, mask_x].mean())
        i_grads.append(grad_x[:, :, ~mask_x].mean())

    if not b_grads:
        return 1.0
    mean_b = torch.stack(b_grads).mean()
    mean_i = torch.stack(i_grads).mean()
    eps = 1e-8
    return (mean_b / (mean_i + eps)).item()


def plot_spectral_full_analysis(
    model_results: dict,
    gt_result: dict,
    output_path: str,
    patch_sizes: list[int] | None = None,
    color_map: dict[str, str] | None = None,
) -> None:
    """Generate a 2×2 PSD ratio grid (model/GT) and save to *output_path*.

    Panels: Radial Average | X-Axis Slice | Y-Axis Slice | Z-Axis Slice.
    """
    if patch_sizes is None:
        patch_sizes = [4, 8]

    plt.style.use("dark_background")
    # Use schema-defined figsize and DPI
    fig, axes = plt.subplots(
        2, 2, figsize=THEME.spectral.figsize, dpi=THEME.spectral.dpi
    )
    fig.patch.set_facecolor(THEME.spectral.bg_figure)
    eps = 1e-12

    titles = ["Radial Average", "X-Axis Slice", "Y-Axis Slice", "Z-Axis Slice"]

    # Dynamic palette assignment using color_map if provided
    model_names = sorted(model_results.keys())
    if color_map:
        pal_map = {name: color_map[name] for name in model_names if name in color_map}
    else:
        palette = THEME.spectral.color_palette
        pal_map = {
            name: palette[i % len(palette)] for i, name in enumerate(model_names)
        }

    gt_radial = gt_result["radial"]
    gt_axes = gt_result["axes"]

    for idx, ax in enumerate(axes.flatten()):
        ax.set_facecolor(THEME.spectral.bg_axis)
        ax.grid(
            THEME.spectral.grid.enabled,
            which=THEME.spectral.grid.which,
            axis=THEME.spectral.grid.axis,
            linestyle=THEME.spectral.grid.linestyle,
            alpha=THEME.spectral.grid.alpha,
            color=THEME.spectral.grid.color,
        )
        ax.axhline(
            y=1.0,
            color=THEME.spectral.color_gt,
            linestyle="-",
            linewidth=THEME.spectral.line_width_ref,
            label="GT (Ref)",
            zorder=10,
        )
        ax.set_yscale("log")
        ax.set_title(
            titles[idx], fontsize=THEME.spectral.font_size_label + 1, color="white"
        )
        ax.set_xlabel("Frequency [px⁻¹]", fontsize=THEME.spectral.font_size_label)
        ax.set_ylabel("PSD Ratio (Model / GT)", fontsize=THEME.spectral.font_size_label)
        ax.tick_params(labelsize=THEME.spectral.font_size_tick)

        cur_gt_f, cur_gt_p = gt_radial if idx == 0 else gt_axes[idx - 1]

        for name, m_res in model_results.items():
            color = pal_map.get(name, "#888888")
            cur_m_f, cur_m_p = m_res["radial"] if idx == 0 else m_res["axes"][idx - 1]

            min_len = min(len(cur_m_f), len(cur_gt_f), len(cur_m_p), len(cur_gt_p))
            ratio = (cur_m_p[:min_len] + eps) / (cur_gt_p[:min_len] + eps)

            ax.plot(
                cur_m_f[:min_len],
                ratio,
                label=name,
                color=color,
                alpha=THEME.spectral.alpha_line,
            )

        for p in patch_sizes:
            ax.axvline(
                x=1.0 / p,
                color=THEME.spectral.color_aliasing,
                linestyle=":",
                alpha=THEME.spectral.alpha_aliasing,
            )
            # Label the aliasing lines
            ax.text(
                1.0 / p,
                ax.get_ylim()[1] * 0.8,
                f"{p}³ Patch",
                color=THEME.spectral.color_aliasing,
                rotation=90,
                va="top",
                ha="right",
                fontsize=THEME.spectral.font_size_tick - 1,
                alpha=0.7,
            )

    axes[0, 0].legend(loc="upper left", fontsize=THEME.spectral.font_size_legend)
    fig.suptitle(
        "Full Spectral Aliasing Analysis: PSD Ratios (Model / GT)",
        fontsize=THEME.spectral.font_size_title + 2,
        fontweight=THEME.spectral.font_weight_title,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, facecolor=THEME.spectral.bg_figure, dpi=THEME.spectral.dpi)
    plt.close()


# Alias for backward compatibility
plot_spectral_comparison = plot_spectral_full_analysis
