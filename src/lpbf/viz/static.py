# src/lpbf/viz/static.py
from __future__ import annotations

import matplotlib.pyplot as plt

from lpbf.core.config import SimulationConfig
from lpbf.core.state import SimulationState


def plot_temperature_field(
    state: SimulationState, sim: SimulationConfig, save_path: str | None = None
):
    """
    Plot the 2D temperature field using Matplotlib.

    If the simulation is 3D, this function extracts the top-most layer (Z=0 index usually).

    Args:
        state (SimulationState): The current simulation state containing the T field.
        sim (SimulationConfig): Configuration object (used for checking dimensionality).
        save_path (str | None): If provided, saves the figure to this path instead of showing it.
    """
    # Extract T
    T = state.T
    if T.ndim == 5:  # B, C, D, H, W
        # Assuming Top Layer is index 0 or -1?
        # Usually index 0 in Z if Z increases into depth? Or 0 is surface?
        # Let's assume index 0 is surface.
        T_slice = T[0, 0, 0, :, :].detach().cpu().numpy()
    else:
        # 2D case: B, C, H, W
        T_slice = T[0, 0, :, :].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(T_slice, origin="upper", cmap="inferno")
    plt.colorbar(im, label="Temperature [K]")
    plt.title(f"T Field t={state.t * 1000:.2f} ms")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_cooling_rate(state: SimulationState, save_path: str | None = None):
    """
    Plot the map of Maximum Cooling Rates captured during solidification.

    Args:
        state (SimulationState): The state containing the `cooling_rate` field.
        save_path (str | None): File path to save the plot.
    """
    if state.cooling_rate is None:
        print("No cooling rate data.")
        return

    CR = state.cooling_rate
    if CR.ndim == 5:
        # Top layer
        CR_slice = CR[0, 0, 0, :, :].detach().cpu().numpy()
    else:
        CR_slice = CR[0, 0, :, :].detach().cpu().numpy()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(CR_slice, origin="upper", cmap="viridis")
    plt.colorbar(im, label="Cooling Rate [K/s]")
    plt.title("Captured Cooling Rate")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
