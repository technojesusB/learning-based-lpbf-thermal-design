"""Visual smoke tests for the three new plot modules.

Each test builds a minimal synthetic DataFrame / dict, calls the plot
function, and asserts a non-empty PNG was written — no pixel-level checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neural_pbf.eval.viz.physical_dashboard import plot_physical_dashboard
from neural_pbf.eval.viz.structural import plot_tv_pbd_comparison
from neural_pbf.eval.viz.system_dashboard import plot_system_dashboard


def _physics_df(n_models: int = 2, n_samples: int = 5) -> pd.DataFrame:
    rows = []
    for m in range(n_models):
        for s in range(n_samples):
            rows.append({
                "Model": f"Model {m}",
                "Sample": f"s{s:04d}",
                "IoU": 0.7 + m * 0.05,
                "Depth_GT": float(5 + s),
                "Depth_Pred": float(5 + s + m * 0.2),
                "T_max_Error": float(50 + s * 10),
                "Offset_vox": float(m + 0.5),
            })
    return pd.DataFrame(rows)


def _system_train_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Version": ["Baseline", "DiT v3", "Hero Run (v5)"],
        "Duration [h]": [2.0, 4.0, 6.0],
        "GPU Util [%]": [85.0, 90.0, 92.0],
        "GPU Mem [MB]": [8000.0, 10000.0, 12000.0],
        "s/sample": [0.5, 0.8, 1.2],
        "Final Loss": [0.05, 0.03, 0.01],
        "Inf GPU Util [%]": [30.0, 40.0, 50.0],
        "Inf GPU Mem [MB]": [4000.0, 5000.0, 6000.0],
    })


def _inference_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Model": ["Baseline", "DiT v3", "Hero Run (v5)"],
        "s_per_sample": [0.10, 0.25, 0.40],
    })


def _structural_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Model": ["Baseline", "DiT v3", "Hero Run (v5)"],
        "TV": [0.02, 0.015, 0.012],
        "PBD": [1.5, 1.2, 1.05],
    })


# ---------------------------------------------------------------------------
# plot_physical_dashboard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_plot_physical_dashboard_creates_file(tmp_path):
    df = _physics_df()
    out = tmp_path / "dashboard.png"
    plot_physical_dashboard(df, out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_physical_dashboard_empty_df_still_writes(tmp_path):
    df = pd.DataFrame(columns=["Model", "Sample", "IoU", "Depth_GT",
                                "Depth_Pred", "T_max_Error", "Offset_vox"])
    out = tmp_path / "empty_dashboard.png"
    plot_physical_dashboard(df, out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_physical_dashboard_creates_parent_dirs(tmp_path):
    df = _physics_df()
    out = tmp_path / "deep" / "nested" / "dash.png"
    plot_physical_dashboard(df, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# plot_system_dashboard
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_plot_system_dashboard_creates_file(tmp_path):
    out = tmp_path / "system.png"
    plot_system_dashboard(_system_train_df(), _inference_df(), out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_system_dashboard_empty_data_still_writes(tmp_path):
    empty_train = pd.DataFrame(columns=["Version", "Duration [h]", "GPU Util [%]",
                                         "GPU Mem [MB]", "s/sample", "Final Loss",
                                         "Inf GPU Util [%]", "Inf GPU Mem [MB]"])
    empty_inf = pd.DataFrame(columns=["Model", "s_per_sample"])
    out = tmp_path / "empty_sys.png"
    plot_system_dashboard(empty_train, empty_inf, out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_system_dashboard_creates_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "system.png"
    plot_system_dashboard(_system_train_df(), _inference_df(), out)
    assert out.exists()


# ---------------------------------------------------------------------------
# plot_tv_pbd_comparison
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_plot_tv_pbd_comparison_creates_file(tmp_path):
    out = tmp_path / "structural.png"
    plot_tv_pbd_comparison(_structural_df(), out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_tv_pbd_comparison_empty_df_still_writes(tmp_path):
    df = pd.DataFrame(columns=["Model", "TV", "PBD"])
    out = tmp_path / "empty_structural.png"
    plot_tv_pbd_comparison(df, out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.unit
def test_plot_tv_pbd_comparison_creates_parent_dirs(tmp_path):
    out = tmp_path / "a" / "b" / "structural.png"
    plot_tv_pbd_comparison(_structural_df(), out)
    assert out.exists()
