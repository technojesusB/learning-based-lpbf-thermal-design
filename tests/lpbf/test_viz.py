# tests/lpbf/test_viz.py
import pytest
import torch
import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
from lpbf.state import SimulationState
from lpbf.config import SimulationConfig, LengthUnit
from lpbf.viz.static import plot_temperature_field, plot_cooling_rate
from lpbf.viz.interactive import figure_temperature_field, figure_cooling_rate

@pytest.fixture
def dummy_state():
    T = torch.rand((1, 1, 10, 10)) * 1000.0
    CR = torch.rand((1, 1, 10, 10)) * 50.0
    return SimulationState(T=T, t=0.1, cooling_rate=CR)

@pytest.fixture
def sim_config():
    return SimulationConfig(Lx=0.01, Ly=0.01, Nx=10, Ny=10, length_unit=LengthUnit.METERS)

def test_static_plots(dummy_state, sim_config, tmp_path):
    # Smoke test: check if they run and save file
    bg_path = tmp_path / "temp.png"
    plot_temperature_field(dummy_state, sim_config, save_path=str(bg_path))
    assert bg_path.exists()
    
    bg_path2 = tmp_path / "cr.png"
    plot_cooling_rate(dummy_state, save_path=str(bg_path2))
    assert bg_path2.exists()

def test_interactive_plots(dummy_state):
    # Smoke test for Plotly
    fig = figure_temperature_field(dummy_state)
    assert fig is not None
    # Check if we can build it (serialization check)
    json_str = fig.to_json()
    assert len(json_str) > 0

    fig2 = figure_cooling_rate(dummy_state)
    assert fig2 is not None
