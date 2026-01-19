# src/lpbf/viz/interactive.py
from __future__ import annotations
import plotly.graph_objects as go
import torch
import numpy as np
from lpbf.state import SimulationState

def figure_temperature_field(state: SimulationState) -> go.Figure:
    """
    Create an interactive Plotly Heatmap for the temperature field.

    Args:
        state (SimulationState): Current simulation state.

    Returns:
        go.Figure: Plotly figure object containing the heatmap.
    """
    T = state.T
    if T.ndim == 5:
        T_slice = T[0, 0, 0, :, :].detach().cpu().numpy()
    else:
        T_slice = T[0, 0, :, :].detach().cpu().numpy()
        
    fig = go.Figure(data=go.Heatmap(
        z=T_slice,
        colorscale='Inferno',
        colorbar=dict(title='Temperature [K]')
    ))
    
    fig.update_layout(
        title=f"Temperature Field at t={state.t*1000:.2f} ms",
        xaxis_title="X [px]",
        yaxis_title="Y [px]",
        template="plotly_dark"
    )
    return fig

def figure_cooling_rate(state: SimulationState) -> go.Figure:
    """
    Create an interactive Plotly Heatmap for the captured cooling rate.

    Args:
        state (SimulationState): Current simulation state.

    Returns:
        go.Figure: Plotly figure object (empty if cooling rate is None).
    """
    if state.cooling_rate is None:
        return go.Figure()
        
    CR = state.cooling_rate
    if CR.ndim == 5:
        CR_slice = CR[0, 0, 0, :, :].detach().cpu().numpy()
    else:
        CR_slice = CR[0, 0, :, :].detach().cpu().numpy()
        
    # Log scale might be useful?
    fig = go.Figure(data=go.Heatmap(
        z=CR_slice,
        colorscale='Viridis',
        colorbar=dict(title='Cooling Rate [K/s]')
    ))
    
    fig.update_layout(
        title="Captured Cooling Rate",
        template="plotly_dark"
    )
    return fig
