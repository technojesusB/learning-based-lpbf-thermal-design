"""S6 saves figure data first, a failed figure never fails the stage, and re-rendering stays guard-free."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neural_pbf.eval.viz import uq as viz
from neural_pbf.schemas.viz import THEME
from neural_pbf.uq import pipeline, stages
from neural_pbf.uq.analysis import QUANTITIES, build_tables
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.guard import freeze_prereg
from neural_pbf.uq.report import save_tables, update_missing_figures
from tests.uq.test_analysis import _set


def _results() -> list[dict]:
    primary = {"name": "Baseline", "label": "run-disjoint: yes", "role": "primary", "second": None,
               "cal": _set(K=2, per_run=4), "test": _set(n_runs=9, K=2, per_run=4, seed=1, offset=50)}  # fmt: skip
    demo = {"name": "Legacy", "label": "run-disjoint: no", "role": "leakage_demo", "second": None,
            "cal": _set(n_runs=3, K=2, per_run=4), "test": _set(n_runs=3, K=2, per_run=4, seed=2, offset=70)}  # fmt: skip
    return [primary, demo]


@pytest.fixture(autouse=True)
def low_dpi(monkeypatch):
    monkeypatch.setattr(THEME.uq, "dpi", 40)


@pytest.fixture()
def cfg(tmp_path):
    return UQConfig.smoke(tmp_path)


@pytest.fixture()
def ctx(cfg, monkeypatch):
    context = stages.Context("S6", cfg, allow_reeval=False)
    freeze_prereg(cfg, context.prereg_path)
    (context.root / "train_result.json").write_text(
        json.dumps({"best_epoch": 1, "epochs_run": 2, "stopped_early": False})
    )
    (context.root / "status.json").write_text(json.dumps({"attempts": [], "stages": {}}))
    monkeypatch.setattr("neural_pbf.data.splits.load_manifest", lambda _: {"hash": "h", "sets": {"test": []}})
    monkeypatch.setattr(stages, "_load_results", lambda *_: _results())
    monkeypatch.setattr(stages, "_log_evaluation", lambda *_: {})
    return context


def _boom(*args, **kwargs):
    raise RuntimeError("broken plot")


@pytest.mark.unit
def test_build_tables_keeps_the_rows_figures_need(cfg):
    tables = build_tables(_results(), cfg.prereg)
    assert set(tables) == {"coverage", "sensitivity", "errors", "points"}
    assert list(dict.fromkeys(tables["coverage"]["model"])) == ["Baseline", "Legacy"]
    assert {"model", "label", "role"} <= set(tables["coverage"].columns)
    assert {"model", "quantity", "mae"} <= set(tables["errors"].columns)
    assert set(tables["points"]["quantity"]) == set(QUANTITIES)
    assert (tables["points"]["spread"] >= 0).all() and (tables["points"]["abs_error"] >= 0).all()


@pytest.mark.unit
def test_save_tables_round_trips_infinite_widths(cfg, tmp_path):
    tables = build_tables(_results(), cfg.prereg)
    tables = {**tables, "sensitivity": tables["sensitivity"].assign(mean_width=np.inf)}
    save_tables(tables, tmp_path)
    assert np.isinf(pd.read_csv(tmp_path / viz.SENSITIVITY_FILE)["mean_width"]).all()


@pytest.mark.integration
def test_s6_renders_all_figures_without_a_failure_notice(ctx, cfg):
    stages.s6_evaluate(ctx)
    assert (ctx.root / viz.COVERAGE_FILE).exists()
    assert (ctx.root / viz.LEAKAGE_FILE).exists() is False
    assert (ctx.root / "figures" / viz.LEAKAGE_FILE).exists()
    assert json.loads((ctx.root / stages.FIGURE_FAILURES_FILE).read_text()) == {}
    assert "--render-figures" not in (ctx.root / "uq_report.md").read_text()


@pytest.mark.integration
def test_figure_failure_keeps_the_stage_and_is_stated_in_report_and_summary(ctx, cfg, monkeypatch, capsys):
    monkeypatch.setattr(viz, "plot_leakage_demo", _boom)
    output = stages.s6_evaluate(ctx)
    assert output["output_hash"]
    report = (ctx.root / "uq_report.md").read_text()
    assert viz.LEAKAGE_FILE in report.split("## Setup")[0]
    assert "scripts/run_uq_pipeline.sh --render-figures --smoke" in report
    assert "broken plot" in report

    stages.s7_summary(stages.Context("S7", cfg, allow_reeval=False))
    summary = (ctx.root / "summary.md").read_text()
    assert viz.LEAKAGE_FILE in summary
    assert "scripts/run_uq_pipeline.sh --render-figures --smoke" in summary
    assert "--render-figures" in capsys.readouterr().out
    assert "--render-figures" in (ctx.root / "uq_report.md").read_text()


@pytest.mark.integration
def test_rerender_recovers_and_clears_the_notice_without_touching_the_guard(ctx, cfg, monkeypatch):
    with monkeypatch.context() as broken:
        broken.setattr(viz, "plot_leakage_demo", _boom)
        stages.s6_evaluate(ctx)
        stages.s7_summary(stages.Context("S7", cfg, allow_reeval=False))
    guard_log = (ctx.root / "test_access.jsonl").read_bytes()
    coverage = (ctx.root / viz.COVERAGE_FILE).read_bytes()

    assert stages.render_figures(cfg) == {}

    assert (ctx.root / "figures" / viz.LEAKAGE_FILE).exists()
    assert "--render-figures" not in (ctx.root / "uq_report.md").read_text()
    assert "--render-figures" not in (ctx.root / "summary.md").read_text()
    assert (ctx.root / "test_access.jsonl").read_bytes() == guard_log
    assert (ctx.root / viz.COVERAGE_FILE).read_bytes() == coverage


@pytest.mark.integration
def test_rerender_needs_only_saved_tables(ctx, cfg, monkeypatch):
    stages.s6_evaluate(ctx)
    monkeypatch.setattr(stages, "_load_results", _boom)
    assert stages.render_figures(cfg) == {}


@pytest.mark.unit
def test_rerender_refuses_before_evaluation(cfg):
    with pytest.raises(stages.PipelineError, match="S6"):
        stages.render_figures(cfg)


@pytest.fixture()
def smoke_root(tmp_path, monkeypatch):
    cfg = UQConfig.smoke(tmp_path)
    monkeypatch.setattr(pipeline.UQConfig, "smoke", classmethod(lambda _: cfg))
    return tmp_path


@pytest.mark.unit
@pytest.mark.parametrize(("failures", "code"), [({}, 0), ({"a.png": "boom"}, 1)])
def test_render_figures_flag_maps_failures_to_exit_code(monkeypatch, smoke_root, failures, code):
    monkeypatch.setattr(pipeline, "render_figures", lambda _: failures)
    assert pipeline.main(["--render-figures", "--smoke"]) == code
    assert not (smoke_root / "driver.pid").exists()


@pytest.mark.unit
def test_render_figures_refuses_while_a_driver_runs(monkeypatch, smoke_root):
    monkeypatch.setattr(pipeline, "_running_driver_pid", lambda _: 123)
    monkeypatch.setattr(pipeline, "render_figures", _boom)
    assert pipeline.main(["--render-figures", "--smoke"]) == pipeline.EXIT_REFUSED


@pytest.mark.unit
def test_render_figures_warns_about_uncommitted_code(monkeypatch, smoke_root, capsys):
    monkeypatch.setattr(pipeline, "dirty_code_paths", lambda: ["src/x.py"])
    monkeypatch.setattr(pipeline, "render_figures", lambda _: {})
    assert pipeline.main(["--render-figures", "--smoke"]) == 0
    assert "src/x.py" in capsys.readouterr().err


@pytest.mark.integration
def test_mlflow_failure_keeps_the_stage_and_is_stated_in_the_summary(ctx, cfg, monkeypatch, capsys):
    monkeypatch.setattr(stages, "_log_evaluation", _boom)
    assert stages.s6_evaluate(ctx)["output_hash"]
    stages.s7_summary(stages.Context("S7", cfg, allow_reeval=False))
    summary = (ctx.root / "summary.md").read_text()
    assert "MLflow" in summary and "broken plot" in summary
    assert "MLflow" in capsys.readouterr().out


@pytest.mark.integration
def test_incomplete_mlflow_logging_is_stated_in_the_summary(ctx, cfg, monkeypatch):
    monkeypatch.setattr(stages, "_log_evaluation", lambda *_: {"a.png": "store down"})
    stages.s6_evaluate(ctx)
    stages.s7_summary(stages.Context("S7", cfg, allow_reeval=False))
    assert "store down" in (ctx.root / "summary.md").read_text()


@pytest.mark.integration
def test_clean_mlflow_run_adds_nothing_to_the_summary(ctx, cfg):
    stages.s6_evaluate(ctx)
    stages.s7_summary(stages.Context("S7", cfg, allow_reeval=False))
    assert "MLflow" not in (ctx.root / "summary.md").read_text()


@pytest.mark.integration
def test_mlflow_is_logged_for_the_primary_role(ctx, cfg, monkeypatch):
    seen = []
    monkeypatch.setattr(stages, "_load_results", lambda *_: _results()[::-1])
    monkeypatch.setattr(stages, "_log_evaluation", lambda _c, _r, result, *_: seen.append(result["role"]) or {})
    stages.s6_evaluate(ctx)
    assert seen == ["primary"]


@pytest.mark.unit
def test_missing_notice_is_replaced_not_stacked(tmp_path):
    path = tmp_path / "r.md"
    path.write_text("# Title\nbody\n")
    update_missing_figures(path, {"a.png": "first"}, "cmd")
    update_missing_figures(path, {"b.png": "second"}, "cmd")
    text = path.read_text()
    assert "b.png" in text and "a.png" not in text and text.startswith("# Title\n")
    update_missing_figures(path, {}, "cmd")
    assert path.read_text() == "# Title\nbody\n"


@pytest.mark.unit
def test_multi_line_error_stays_inside_the_notice(tmp_path):
    path = tmp_path / "r.md"
    path.write_text("# Title\nbody\n")
    update_missing_figures(path, {"a.png": "first line\nsecond line"}, "cmd")
    text = path.read_text()
    assert "first line" in text and "second line" not in text


@pytest.mark.unit
def test_figure_code_is_outside_the_s6_hash():
    code = stages.STAGE_DEFS["S6"]["code"]
    assert not any("eval/viz" in path or "schemas/viz" in path for path in code)
