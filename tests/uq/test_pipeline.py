"""Orchestrator mechanics: markers, stale detection, retries, GPU-loss handling, status, exit codes."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

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


@pytest.mark.unit
def test_each_model_binds_its_own_draws(tmp_path):
    # Retraining the new model must not invalidate the legacy draws, which come from an unchanged checkpoint.
    from neural_pbf.uq.harness import ModelSpec
    from neural_pbf.uq.stages import bind_model_draws

    new, legacy = tmp_path / "new.pt", tmp_path / "legacy.pt"
    new.write_bytes(b"a")
    legacy.write_bytes(b"l")
    specs = [ModelSpec("baseline_tin_exp", str(new), "net"), ModelSpec("legacy_baseline", str(legacy), "net")]
    bind_model_draws(tmp_path / "draws", specs, "m")
    new.write_bytes(b"b")
    with pytest.raises(PipelineError, match="baseline_tin_exp"):
        bind_model_draws(tmp_path / "draws", specs, "m")
    bind_model_draws(tmp_path / "draws", specs[1:], "m")  # legacy alone is still valid


@pytest.mark.unit
def test_forcing_s5_wipes_only_the_trained_models_draws():
    from neural_pbf.uq.stages import STAGE_DEFS

    cfg = UQConfig.full()
    assert STAGE_DEFS["S5"]["wipe"](cfg) == [Path(cfg.root) / "draws" / cfg.train.model_name]
    assert STAGE_DEFS["S4"]["wipe"](cfg) == [Path(cfg.root) / "ckpt"]


@pytest.mark.integration
def test_stop_request_is_not_mistaken_for_gpu_failure(tmp_path):
    import neural_pbf.uq.pipeline as mod

    counter = tmp_path / "n"
    body = f"""
import os, pathlib, signal, time
f = pathlib.Path({str(counter)!r}); f.write_text(f.read_text() + "x" if f.exists() else "x")
time.sleep(1)
os.kill(os.getpid(), signal.SIGTERM)
"""
    threading.Timer(0.3, mod.STOP_REQUESTED.set).start()  # while the child runs
    try:
        assert run_child_with_retries(_script(tmp_path, body), **_common(tmp_path)) == mod.EXIT_STOPPED
    finally:
        mod.STOP_REQUESTED.clear()
    assert counter.read_text() == "x", "a stopped stage must not be retried"


@pytest.mark.unit
def test_no_gpus_message_counts_as_gpu_failure():
    assert classify_failure(1, "RuntimeError: No CUDA GPUs are available") == "gpu"


@pytest.mark.integration
def test_stalled_child_is_killed_and_treated_as_gpu_failure(tmp_path, capsys):
    progress = tmp_path / "p.json"
    progress.write_text("{}")
    kw = _common(tmp_path)
    kw["max_retries"] = 0
    body = "import time; time.sleep(120)"
    code = run_child_with_retries(_script(tmp_path, body), **kw, progress_path=progress, stall_timeout_s=1)
    assert code == EXIT_GPU_LOST
    assert "stalled" in (tmp_path / "s.log").read_text()


@pytest.mark.unit
def test_cal_and_test_use_different_noise_streams():
    from neural_pbf.uq.stages import set_seed

    assert set_seed(7, "cal") != set_seed(7, "test") and set_seed(7, "cal") == set_seed(7, "cal")


@pytest.mark.unit
def test_timeline_projection_scales_with_runs():
    from neural_pbf.uq.stages import _timeline

    t = _timeline(UQConfig.full(), 800.0)
    assert t["projected_generation_h"] == pytest.approx(10.0)
    assert t["projected_total_h"] > t["projected_generation_h"]


# --- review fixes: stale PID file ------------------------------------------------------


def _write_pid(root, pid):
    root.mkdir(parents=True, exist_ok=True)
    (root / "driver.pid").write_text(str(pid))


@pytest.mark.integration
def test_stale_pid_of_unrelated_process_does_not_block_or_get_killed(tmp_path):
    import subprocess

    from neural_pbf.uq.pipeline import _claim_lock, _stop

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _write_pid(tmp_path, other.pid)
        assert _stop(UQConfig.smoke(tmp_path)) == 0
        assert other.poll() is None, "--stop must not signal an unrelated process"
        assert not (tmp_path / "driver.pid").exists()
        _write_pid(tmp_path, other.pid)
        assert _claim_lock(tmp_path).read_text() != str(other.pid)
    finally:
        other.kill()
        other.wait()


@pytest.mark.unit
@pytest.mark.parametrize("content", ["999999", "", "garbage"])
def test_stop_with_dead_or_broken_pid_does_not_crash(tmp_path, content):
    from neural_pbf.uq.pipeline import _stop

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "driver.pid").write_text(content)
    assert _stop(UQConfig.smoke(tmp_path)) == 0


@pytest.mark.integration
def test_live_driver_is_still_recognized(tmp_path):
    import subprocess

    from neural_pbf.uq.pipeline import _claim_lock

    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "neural_pbf.uq.pipeline"])
    try:
        _write_pid(tmp_path, fake.pid)
        with pytest.raises(PipelineError, match="already running"):
            _claim_lock(tmp_path)
    finally:
        fake.kill()
        fake.wait()


# --- review fixes: timeline, hashing, failure classes -----------------------------------


@pytest.mark.unit
def test_time_accumulates_across_restarts_including_hard_crash(tmp_path):
    from neural_pbf.uq.pipeline import Status
    from neural_pbf.uq.stages import accumulated_seconds

    first = Status(tmp_path, "smoke")
    first.begin_attempt("S4")
    first.data["attempts"][-1].update(start=100.0, last=3700.0)  # 1 h, then the process dies without cleanup
    first.update()
    second = Status(tmp_path, "smoke")
    second.begin_attempt("S4")
    second.data["attempts"][-1].update(start=9000.0, last=9060.0)
    second.begin_attempt("S5")
    second.data["attempts"][-1].update(start=9060.0, last=9070.0)
    totals = accumulated_seconds(second.data)
    assert totals["S4"] == {"seconds": 3660.0, "executions": 2}
    assert totals["S5"] == {"seconds": 10.0, "executions": 1}


@pytest.mark.unit
def test_tick_advances_attempt_end_without_progress_file(tmp_path):
    from neural_pbf.uq.pipeline import Status

    s = Status(tmp_path, "smoke")
    s.begin_attempt("S0")
    s.data["attempts"][-1]["last"] = 0.0
    s.tick("S0")
    assert s.data["attempts"][-1]["last"] > 0.0


def _with_run(cfg, **kw):
    return cfg.model_copy(update={"run": cfg.run.model_copy(update=kw)})


def _markers(root):
    for stage in ("S3", "S4"):
        write_marker(root, stage, "i", "o", {})


@pytest.mark.unit
def test_draw_batch_does_not_invalidate_finished_stages(tmp_path):
    from neural_pbf.uq.pipeline import input_hash

    _markers(tmp_path)
    cfg = UQConfig.smoke(tmp_path)
    lowered = _with_run(cfg, draw_batch=1)
    assert input_hash(cfg, "S0", tmp_path) == input_hash(lowered, "S0", tmp_path)
    assert input_hash(cfg, "S5", tmp_path) == input_hash(lowered, "S5", tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize("field,value", [("chunk_size", 99), ("legacy_sample_limit", 99)])
def test_settings_that_change_sampling_output_invalidate_s5(tmp_path, field, value):
    from neural_pbf.uq.pipeline import input_hash

    _markers(tmp_path)
    cfg = UQConfig.smoke(tmp_path)
    assert input_hash(cfg, "S5", tmp_path) != input_hash(_with_run(cfg, **{field: value}), "S5", tmp_path)


@pytest.mark.unit
def test_ordinary_error_mentioning_driver_is_not_gpu_failure():
    assert classify_failure(1, "ValueError: unsupported database driver") == "fatal"
    assert classify_failure(1, "Triton Error: Invalid argument in kernel") == "fatal"
    assert classify_failure(1, "Failed to initialize NVML: Driver/library version mismatch") == "gpu"


@pytest.mark.unit
def test_smoke_mlflow_is_kept_out_of_the_main_database(tmp_path):
    assert UQConfig.smoke(tmp_path).run.mlflow_uri == f"sqlite:///{tmp_path}/mlflow.db"
    assert UQConfig.full().run.mlflow_uri == "sqlite:///mlflow.db"


@pytest.mark.unit
def test_interrupted_rerun_is_not_skipped_as_done(tmp_path):
    from neural_pbf.uq.pipeline import mark_started

    write_marker(tmp_path, "S5", "h", "o", {})
    mark_started(tmp_path, "S5", "h")  # the forced rerun began, its output is gone, then it was stopped
    assert read_marker(tmp_path, "S5") is None
    assert decide_stage(tmp_path, "S5", "h", force=False, upstream_reran=False) == StageState.RUN


@pytest.mark.unit
def test_s1_code_hash_excludes_physics_loss_files() -> None:
    """Editing the training loss must not mark the 17 h dataset stale; the solver files must stay hashed."""
    from neural_pbf.uq.stages import STAGE_DEFS

    code = STAGE_DEFS["S1"]["code"]
    assert "src/neural_pbf/physics" not in code
    for loss_file in ("fm_physics.py", "triton_pde_loss.py"):
        assert not any(p.endswith(loss_file) for p in code)
    for solver_file in ("material.py", "ops.py", "triton_ops.py"):
        assert f"src/neural_pbf/physics/{solver_file}" in code
    assert "src/neural_pbf/integrator/stepper.py" in code


@pytest.mark.unit
def test_dirty_code_paths_keeps_first_path_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """_git() strips its output, which removes the leading space of the first ' M path' line."""
    from neural_pbf.uq import stages

    monkeypatch.setattr(stages, "_git", lambda *a: " M src/a.py\n M src/b.py\n?? experiments/c.py".strip())
    assert stages.dirty_code_paths() == ["src/a.py", "src/b.py", "experiments/c.py"]
