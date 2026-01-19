# src/lpbf/viz/static.py
from __future__ import annotations
import matplotlib.pyplot as plt
import torch
import numpy as np
from lpbf.state import SimulationState
from lpbf.config import SimulationConfig

def plot_temperature_field(state: SimulationState, sim: SimulationConfig, save_path: str | None = None):
    """
    Plot 2D temperature field (using z=0 slice if 3D).
    """
    # Extract T
    T = state.T
    if T.ndim == 5: # B, C, D, H, W
        T_slice = T[0, 0, 0, :, :].detach().cpu().numpy() # Top layer
    else:
        T_slice = T[0, 0, :, :].detach().cpu().numpy()
        
    plt.figure(figsize=(6, 5))
    im = plt.imshow(T_slice, origin='upper', cmap='inferno')
    plt.colorbar(im, label='Temperature [K]')
    plt.title(f"T Field t={state.t*1000:.2f} ms")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_cooling_rate(state: SimulationState, save_path: str | None = None):
    """
    Plot Max Cooling Rate map.
    """
    if state.cooling_rate is None:
        print("No cooling rate data.")
        return
        
    CR = state.cooling_rate
    if CR.ndim == 5:
        CR_slice = CR[0, 0, 0, :, :].detach().cpu().numpy()
    else:
        CR_slice = CR[0, 0, :, :].detach().cpu().numpy()
        
    plt.figure(figsize=(6, 5))
    im = plt.imshow(CR_slice, origin='upper', cmap='viridis')
    plt.colorbar(im, label='Cooling Rate [K/s]')
    plt.title("Captured Cooling Rate")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
