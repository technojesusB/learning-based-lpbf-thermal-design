"""Orchestrator mechanics: markers, stale detection, retries, GPU-loss handling, status, exit codes."""

from __future__ import annotations

import json
import sys

import pytest

from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.pipeline import (
    EXIT_FAILED,
    EXIT_GPU_LOST,
    PipelineError,
    StageState,
    classify_failure,
    decide_stage,
    disk_guard,
    format_status,
    load_status,
    read_marker,
    run_child_with_retries,
    select_stages,
    write_marker,
)
from neural_pbf.uq.stages import STAGE_IDS


@pytest.mark.unit
def test_stage_order():
    assert STAGE_IDS == ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")


@pytest.mark.unit
def test_marker_round_trip_is_atomic(tmp_path):
    write_marker(tmp_path, "S1", "in", "out", {"seed": 1})
    m = read_marker(tmp_path, "S1")
    assert m["input_hash"] == "in" and m["output_hash"] == "out" and m["provenance"]["seed"] == 1
    assert not list(tmp_path.rglob("*.tmp"))
    assert read_marker(tmp_path, "S2") is None


@pytest.mark.unit
def test_decide_runs_when_no_marker_and_skips_when_matching(tmp_path):
    assert decide_stage(tmp_path, "S1", "h", force=False, upstream_reran=False) == StageState.RUN
    write_marker(tmp_path, "S1", "h", "o", {})
    assert decide_stage(tmp_path, "S1", "h", force=False, upstream_reran=False) == StageState.SKIP


@pytest.mark.unit
def test_stale_marker_fails_loudly_instead_of_silently_rerunning(tmp_path):
    write_marker(tmp_path, "S1", "old", "o", {})
    with pytest.raises(PipelineError, match="--force-stage S1"):
        decide_stage(tmp_path, "S1", "new", force=False, upstream_reran=False)


@pytest.mark.unit
def test_force_and_upstream_rerun_allow_stale(tmp_path):
    write_marker(tmp_path, "S1", "old", "o", {})
    assert decide_stage(tmp_path, "S1", "new", force=True, upstream_reran=False) == StageState.RUN
    assert decide_stage(tmp_path, "S1", "new", force=False, upstream_reran=True) == StageState.RUN
    assert decide_stage(tmp_path, "S1", "old", force=True, upstream_reran=False) == StageState.RUN


@pytest.mark.unit
def test_partial_stage_with_other_inputs_refused(tmp_path):
    from neural_pbf.uq.pipeline import mark_started

    mark_started(tmp_path, "S5", "old")
    with pytest.raises(PipelineError, match="partial"):
        decide_stage(tmp_path, "S5", "new", force=False, upstream_reran=False)
    assert decide_stage(tmp_path, "S5", "old", force=False, upstream_reran=False) == StageState.RUN


@pytest.mark.unit
def test_select_stages():
    assert select_stages(None, None) == list(STAGE_IDS)
    assert select_stages("S4", None) == ["S4", "S5", "S6", "S7"]
    assert select_stages(None, "S3") == ["S3"]
    with pytest.raises(PipelineError):
        select_stages("S9", None)
    with pytest.raises(PipelineError):
        select_stages("S1", "S2")


@pytest.mark.unit
@pytest.mark.parametrize(
    "tail,code,expected",
    [
        ("RuntimeError: CUDA error: unknown error", 1, "gpu"),
        ("nvidia-smi has failed because it couldn't communicate with the NVIDIA driver", 1, "gpu"),
        ("torch.cuda.OutOfMemoryError: CUDA out of memory", 1, "oom"),
        ("", -11, "gpu"),
        ("ValueError: bad", 1, "fatal"),
        ("anything", 2, "fatal"),
        ("anything", 3, "gpu"),
    ],
)
def test_classify_failure(tail, code, expected):
    assert classify_failure(code, tail) == expected


def _script(tmp_path, body: str) -> list[str]:
    p = tmp_path / "child.py"
    p.write_text(body)
    return [sys.executable, str(p)]


def _common(tmp_path, **kw):
    return dict(log_path=tmp_path / "s.log", max_retries=2, backoff_s=0, gpu_alive=lambda: True, **kw)


@pytest.mark.integration
def test_child_success_logs_output(tmp_path):
    rc = run_child_with_retries(_script(tmp_path, "print('hello')"), **_common(tmp_path))
    assert rc == 0 and "hello" in (tmp_path / "s.log").read_text()


@pytest.mark.integration
def test_gpu_failure_retries_then_succeeds(tmp_path):
    flag = tmp_path / "flag"
    body = f"""
import pathlib, sys
f = pathlib.Path({str(flag)!r})
if not f.exists():
    f.write_text("x")
    print("RuntimeError: CUDA error: device lost")
    sys.exit(1)
"""
    assert run_child_with_retries(_script(tmp_path, body), **_common(tmp_path)) == 0


@pytest.mark.integration
def test_fatal_failure_does_not_retry(tmp_path):
    counter = tmp_path / "n"
    body = f"""
import pathlib, sys
f = pathlib.Path({str(counter)!r}); f.write_text(f.read_text() + "x" if f.exists() else "x")
print("ValueError: broken"); sys.exit(1)
"""
    assert run_child_with_retries(_script(tmp_path, body), **_common(tmp_path)) == EXIT_FAILED
    assert counter.read_text() == "x"


@pytest.mark.integration
def test_retries_exhausted_returns_gpu_lost(tmp_path, capsys):
    body = "import sys; print('CUDA error: x'); sys.exit(1)"
    assert run_child_with_retries(_script(tmp_path, body), **_common(tmp_path)) == EXIT_GPU_LOST
    assert "wsl --shutdown" in capsys.readouterr().out


@pytest.mark.integration
def test_dead_gpu_stops_immediately(tmp_path, capsys):
    counter = tmp_path / "n"
    body = f"""
import pathlib, sys
f = pathlib.Path({str(counter)!r}); f.write_text(f.read_text() + "x" if f.exists() else "x")
print("CUDA error: x"); sys.exit(1)
"""
    kw = _common(tmp_path)
    kw["gpu_alive"] = lambda: False
    assert run_child_with_retries(_script(tmp_path, body), **kw) == EXIT_GPU_LOST
    assert counter.read_text() == "x"
    assert "restart" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_disk_guard(tmp_path):
    disk_guard(tmp_path, min_free_gb=0.0, purpose="x")
    with pytest.raises(PipelineError, match="free"):
        disk_guard(tmp_path, min_free_gb=1e12, purpose="x")


@pytest.mark.unit
def test_status_format_shows_progress_eta_and_stale_heartbeat(tmp_path):
    status = {
        "stage": "S4", "state": "running", "heartbeat": "2020-01-01T00:00:00", "progress": {"done": 5, "total": 100},
        "eta_s": 3600, "stages": {"S0": {"state": "done", "duration_s": 3}}, "timeline": {"projected_total_h": 12.5},
    }
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status))
    text = format_status(load_status(path))
    assert "S4" in text and "5/100" in text and "1.0 h" in text and "12.5" in text
    assert "no heartbeat" in text.lower()


@pytest.mark.unit
def test_full_config_meets_design_constraints():
    cfg = UQConfig.full()
    assert cfg.gen.n_runs == 45 and cfg.gen.base_seed > 51
    assert cfg.split.min_runs["cal"] == 9
    assert cfg.prereg.K == 16 and 16 in cfg.prereg.sensitivity_Ks
    assert cfg.train.mask_source == "pre" and cfg.train.lr == 2e-4 and cfg.train.batch_size == 4


@pytest.mark.unit
def test_draws_from_another_checkpoint_are_refused(tmp_path):
    from neural_pbf.uq.stages import bind_draws_to_checkpoint

    d = tmp_path / "draws"
    bind_draws_to_checkpoint(d, "ckpt-a", "manifest-a")
    bind_draws_to_checkpoint(d, "ckpt-a", "manifest-a")  # same binding on resume is fine
    with pytest.raises(PipelineError, match="--force-stage S5"):
        bind_draws_to_checkpoint(d, "ckpt-b", "manifest-a")


@pytest.mark.integration
def test_stop_request_is_not_mistaken_for_gpu_failure(tmp_path):
    import neural_pbf.uq.pipeline as mod

    counter = tmp_path / "n"
    body = f"""
import os, pathlib, signal
f = pathlib.Path({str(counter)!r}); f.write_text(f.read_text() + "x" if f.exists() else "x")
os.kill(os.getpid(), signal.SIGTERM)
"""
    mod.STOP_REQUESTED.set()
    try:
        assert run_child_with_retries(_script(tmp_path, body), **_common(tmp_path)) == mod.EXIT_STOPPED
    finally:
        mod.STOP_REQUESTED.clear()
    assert counter.read_text() == "x", "a stopped stage must not be retried"


@pytest.mark.unit
def test_no_gpus_message_counts_as_gpu_failure():
    assert classify_failure(1, "RuntimeError: No CUDA GPUs are available") == "gpu"
