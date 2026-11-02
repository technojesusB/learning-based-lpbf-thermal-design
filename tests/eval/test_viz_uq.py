"""UQ figures render from saved tables alone, using THEME.uq for every style value."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from neural_pbf.eval.viz import uq as viz
from neural_pbf.schemas.viz import THEME

QUANTITIES = ("depth_vox", "T_max_K", "volume_vox")
METHODS = tuple(THEME.uq.method_colors)
MODELS = ("Baseline (pre-step mask)", "Legacy Baseline on the new test runs")
CANONICAL = {
    "coverage_nominal_vs_empirical.png", "spread_vs_error.png", "coverage_per_material.png",
    "coverage_per_run.png", "sensitivity_vs_k.png", "leakage_demo.png",
}  # fmt: skip


@pytest.fixture(autouse=True)
def low_dpi(monkeypatch):
    """Rendering at full DPI takes seconds per figure and tests nothing extra."""
    monkeypatch.setattr(THEME.uq, "dpi", 40)


def _stratum_rows(model: str, kind: str, names: list[str]) -> list[dict]:
    rows = []
    for quantity in QUANTITIES:
        for alpha in (0.2, 0.1, 0.05):
            for method in METHODS:
                for name in names:
                    rows.append({
                        "model": model, "role": "primary" if model == MODELS[0] else "leakage_demo",
                        "label": f"run-disjoint: {model[:6]}", "quantity": quantity, "alpha": alpha,
                        "method": method, "stratum_type": kind, "stratum": name, "coverage": 0.9, "boot_lo": 0.85,
                        "boot_hi": 0.95, "mean_width": 2.0, "run_level_supported": alpha > 0.05,
                    })  # fmt: skip
    return rows


@pytest.fixture()
def coverage() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        rows += _stratum_rows(model, "all", ["all"])
        rows += _stratum_rows(model, "material", ["SS316L", "Ti64"])
        rows += _stratum_rows(model, "run", [f"r{i}" for i in range(5)])
    return pd.DataFrame(rows)


@pytest.fixture()
def points() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"model": m, "quantity": q, "run_id": "r0", "sample_index": i, "spread": rng.random(), "abs_error": rng.random()}
        for m in MODELS for q in QUANTITIES for i in range(5)
    ])  # fmt: skip


@pytest.fixture()
def sensitivity() -> pd.DataFrame:
    return pd.DataFrame([
        {"K": k, "quantity": q, "alpha": 0.1, "method": m, "coverage": 0.9, "mean_width": 2.0 + k, "n": 10}
        for k in (4, 8, 16) for q in QUANTITIES for m in METHODS
    ])  # fmt: skip


@pytest.fixture()
def errors() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": m, "quantity": q, "n": 5, "mae": 1.0, "rmse": 2.0, "bias": 0.1, "mean_spread": 0.5}
        for m in MODELS for q in QUANTITIES
    ])  # fmt: skip


@pytest.fixture()
def data_dir(tmp_path, coverage, points, sensitivity, errors) -> Path:
    out = tmp_path / "data"
    out.mkdir()
    coverage.to_csv(out / viz.COVERAGE_FILE, index=False)
    points.to_csv(out / viz.POINTS_FILE, index=False)
    sensitivity.to_csv(out / viz.SENSITIVITY_FILE, index=False)
    errors.to_csv(out / viz.ERRORS_FILE, index=False)
    return out


def _one(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.model == MODELS[0]]


@pytest.mark.unit
def test_coverage_vs_nominal_writes_png(tmp_path, coverage):
    viz.plot_coverage_vs_nominal(_one(coverage), tmp_path / "sub" / "a.png")
    assert (tmp_path / "sub" / "a.png").exists()


@pytest.mark.unit
def test_spread_vs_error_writes_png(tmp_path, points):
    viz.plot_spread_vs_error(_one(points), tmp_path / "a.png")
    assert (tmp_path / "a.png").exists()


@pytest.mark.unit
def test_coverage_per_material_writes_png(tmp_path, coverage):
    viz.plot_coverage_per_material(_one(coverage), tmp_path / "a.png", "absolute", 0.1)
    assert (tmp_path / "a.png").exists()


@pytest.mark.unit
def test_material_figure_survives_estimate_outside_bootstrap_interval(tmp_path, coverage):
    outside = coverage.assign(coverage=0.5, boot_lo=0.6, boot_hi=0.7)
    viz.plot_coverage_per_material(_one(outside), tmp_path / "a.png", "absolute", 0.1)
    assert (tmp_path / "a.png").exists()


@pytest.mark.unit
def test_run_coverage_spread_writes_png(tmp_path, coverage):
    viz.plot_run_coverage_spread(_one(coverage), tmp_path / "a.png", "absolute", 0.1)
    assert (tmp_path / "a.png").exists()


@pytest.mark.unit
def test_sensitivity_vs_k_writes_png(tmp_path, sensitivity):
    viz.plot_sensitivity_vs_k(sensitivity, tmp_path / "a.png", 0.1)
    assert (tmp_path / "a.png").exists()


@pytest.mark.unit
def test_sensitivity_tolerates_infinite_width(tmp_path, sensitivity):
    viz.plot_sensitivity_vs_k(sensitivity.assign(mean_width=np.inf), tmp_path / "a.png", 0.1)
    assert (tmp_path / "a.png").exists()


@pytest.mark.unit
def test_leakage_demo_writes_png(tmp_path, coverage, errors):
    viz.plot_leakage_demo(coverage, errors, tmp_path / "a.png", "absolute", 0.1)
    assert (tmp_path / "a.png").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "call",
    [
        lambda p: viz.plot_coverage_vs_nominal(pd.DataFrame({"x": [1]}), p),
        lambda p: viz.plot_spread_vs_error(pd.DataFrame({"x": [1]}), p),
        lambda p: viz.plot_coverage_per_material(pd.DataFrame({"x": [1]}), p, "absolute", 0.1),
        lambda p: viz.plot_run_coverage_spread(pd.DataFrame({"x": [1]}), p, "absolute", 0.1),
        lambda p: viz.plot_sensitivity_vs_k(pd.DataFrame({"x": [1]}), p, 0.1),
        lambda p: viz.plot_leakage_demo(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1]}), p, "absolute", 0.1),
    ],
)
def test_missing_columns_are_rejected_by_name(tmp_path, call):
    with pytest.raises(ValueError, match="missing columns"):
        call(tmp_path / "a.png")
    assert not (tmp_path / "a.png").exists()


@pytest.mark.unit
def test_empty_selection_is_rejected(tmp_path, coverage):
    with pytest.raises(ValueError, match="no rows"):
        viz.plot_coverage_per_material(coverage, tmp_path / "a.png", "absolute", 0.123)


@pytest.mark.unit
def test_plots_do_not_mutate_input(tmp_path, coverage):
    before = coverage.copy()
    viz.plot_coverage_vs_nominal(_one(coverage), tmp_path / "a.png")
    pd.testing.assert_frame_equal(coverage, before)


@pytest.mark.unit
def test_background_and_dpi_come_from_theme(tmp_path, coverage):
    from PIL import Image

    path = tmp_path / "a.png"
    viz.plot_coverage_vs_nominal(_one(coverage), path)
    corner = mpimg.imread(path)[0, 0, :3]
    assert np.allclose(corner, [10 / 255] * 3, atol=0.01)
    assert round(Image.open(path).info["dpi"][0]) == THEME.uq.dpi


@pytest.mark.unit
def test_tracker_pass_logs_a_copy_and_leaves_the_saved_png_alone(tmp_path, coverage):
    logged: list[tuple[str, str, bool]] = []

    class Tracker:
        def log_artifact(self, path, artifact_path=None):
            logged.append((Path(path).name, artifact_path, Path(path).exists()))

    path = tmp_path / "a.png"
    path.write_bytes(b"saved")
    viz.plot_coverage_vs_nominal(_one(coverage), path, tracker=Tracker())
    assert logged == [("a.png", "figures", True)]
    assert path.read_bytes() == b"saved"


@pytest.mark.unit
def test_failed_save_keeps_the_old_png_and_leaves_no_temp_file(tmp_path, coverage, monkeypatch):
    path = tmp_path / "a.png"
    path.write_bytes(b"saved")

    def fail(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", fail)
    with pytest.raises(OSError):
        viz.plot_coverage_vs_nominal(_one(coverage), path)
    assert [p.name for p in tmp_path.iterdir()] == ["a.png"]
    assert path.read_bytes() == b"saved"


@pytest.mark.unit
def test_figures_name_their_model_method_and_alpha(tmp_path, coverage, points, sensitivity, errors, monkeypatch):
    titles = []

    def capture(fig, *args, **kwargs):
        titles.append(fig._suptitle.get_text())
        viz.plt.close(fig)

    monkeypatch.setattr(viz, "_finish", capture)
    out = tmp_path / "a.png"
    viz.plot_coverage_vs_nominal(_one(coverage), out)
    viz.plot_spread_vs_error(_one(points), out)
    viz.plot_coverage_per_material(_one(coverage), out, "absolute", 0.1)
    viz.plot_run_coverage_spread(_one(coverage), out, "absolute", 0.1)
    viz.plot_sensitivity_vs_k(sensitivity, out, 0.1)
    viz.plot_leakage_demo(coverage, errors, out, "absolute", 0.1)
    for title in titles[:4]:
        assert MODELS[0] in title
    assert "Absolute" in titles[2] and "0.1" in titles[2]
    assert "Absolute" in titles[3] and "0.1" in titles[3]
    assert "0.1" in titles[4]
    assert "Absolute" in titles[5] and "0.1" in titles[5]


@pytest.mark.integration
def test_canonical_figures_use_the_primary_role_not_row_order(tmp_path, data_dir, coverage, monkeypatch):
    coverage.iloc[::-1].to_csv(data_dir / viz.COVERAGE_FILE, index=False)
    seen = []
    monkeypatch.setattr(viz, "plot_run_coverage_spread", lambda df, *a, **k: seen.append(set(df["model"])))
    viz.render_uq_figures(data_dir, tmp_path / "fig", "absolute", 0.1)
    assert seen == [{MODELS[0]}]


@pytest.mark.integration
def test_failed_rerender_removes_the_old_png(tmp_path, data_dir, monkeypatch):
    fig_dir = tmp_path / "fig"
    fig_dir.mkdir()
    (fig_dir / viz.LEAKAGE_FILE).write_bytes(b"old result")

    def boom(*args, **kwargs):
        raise RuntimeError("broken plot")

    monkeypatch.setattr(viz, "plot_leakage_demo", boom)
    viz.render_uq_figures(data_dir, fig_dir, "absolute", 0.1)
    assert not (fig_dir / viz.LEAKAGE_FILE).exists()


@pytest.mark.integration
def test_failed_tracker_pass_keeps_saved_pngs(tmp_path, data_dir):
    fig_dir = tmp_path / "fig"
    viz.render_uq_figures(data_dir, fig_dir, "absolute", 0.1)
    before = (fig_dir / viz.LEAKAGE_FILE).read_bytes()

    class Tracker:
        def log_artifact(self, path, artifact_path=None):
            raise RuntimeError("store down")

    assert viz.render_uq_figures(data_dir, fig_dir, "absolute", 0.1, tracker=Tracker())
    assert (fig_dir / viz.LEAKAGE_FILE).read_bytes() == before


@pytest.mark.unit
def test_published_figure_names_are_pinned():
    """The report keeps these links after the run, and this code is outside the S6 hash."""
    assert {viz.NOMINAL_FILE, viz.SPREAD_FILE, viz.MATERIAL_FILE, viz.RUN_SPREAD_FILE, viz.SENSITIVITY_FIG_FILE,
            viz.LEAKAGE_FILE} == CANONICAL  # fmt: skip
    assert viz.model_figure_name("Baseline (pre-step mask)", "spread") == "baseline__pre_step_mask__spread.png"
    assert viz.MODEL_FIGURES == ("nominal", "spread", "material")


@pytest.mark.integration
def test_render_writes_every_figure_from_saved_tables(tmp_path, data_dir):
    fig_dir = tmp_path / "fig"
    assert viz.render_uq_figures(data_dir, fig_dir, "absolute", 0.1) == {}
    names = {p.name for p in fig_dir.glob("*.png")}
    assert CANONICAL <= names
    for model in MODELS:
        for kind in viz.MODEL_FIGURES:
            assert viz.model_figure_name(model, kind) in names


@pytest.mark.integration
def test_render_reports_a_failed_figure_and_keeps_the_rest(tmp_path, data_dir, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("broken plot")

    monkeypatch.setattr(viz, "plot_leakage_demo", boom)
    fig_dir = tmp_path / "fig"
    failures = viz.render_uq_figures(data_dir, fig_dir, "absolute", 0.1)
    assert failures == {"leakage_demo.png": "RuntimeError: broken plot"}
    assert (fig_dir / "sensitivity_vs_k.png").exists()


@pytest.mark.integration
def test_render_reports_unreadable_data_instead_of_raising(tmp_path):
    failures = viz.render_uq_figures(tmp_path, tmp_path / "fig", "absolute", 0.1)
    assert list(failures) == ["all figures"]


@pytest.mark.integration
def test_render_needs_no_gpu_and_does_not_initialize_cuda(tmp_path, data_dir):
    code = (
        "import sys, torch\n"
        "from neural_pbf.eval.viz.uq import render_uq_figures\n"
        f"failures = render_uq_figures({str(data_dir)!r}, {str(tmp_path / 'fig')!r}, 'absolute', 0.1)\n"
        "sys.exit(1 if failures or torch.cuda.is_initialized() else 0)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    assert subprocess.run([sys.executable, "-c", code], env=env, check=False).returncode == 0


@pytest.mark.unit
def test_module_does_not_import_uq_or_load_models():
    tree = ast.parse(Path(viz.__file__).read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not any(name.startswith(("neural_pbf.uq", "torch")) for name in imported if name)
