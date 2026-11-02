"""Pausing the driver at a stage boundary or a wall-clock time, and honoring --stop between stages.

The driver runs for real; only the stage children are replaced by small scripts.
"""

from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from neural_pbf.uq import pipeline, stages
from neural_pbf.uq.config import UQConfig

NOW = datetime.datetime(2026, 11, 2, 21, 0)


LATER = "2099-01-01T00:00"
EXPIRE = "(root / 'late').write_text('x')"


def _in_seconds(seconds: float) -> str:
    return (datetime.datetime.now() + datetime.timedelta(seconds=seconds)).isoformat(timespec="seconds")


# --- argument handling ------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("22:30", datetime.datetime(2026, 11, 2, 22, 30)),
        ("2026-11-03T10:30", datetime.datetime(2026, 11, 3, 10, 30)),
        ("2026-11-03 10:30", datetime.datetime(2026, 11, 3, 10, 30)),
        ("2026-11-02T21:00:05", datetime.datetime(2026, 11, 2, 21, 0, 5)),
    ],
)
def test_stop_time_parses_local_times(text, expected):
    assert pipeline.parse_stop_time(text, NOW) == expected


@pytest.mark.unit
@pytest.mark.parametrize("text", ["10:30", "21:00", "2026-11-02T10:30", "2020-01-01T00:00"])
def test_stop_time_in_the_past_is_refused_not_moved_to_tomorrow(text):
    with pytest.raises(pipeline.PipelineError, match="past"):
        pipeline.parse_stop_time(text, NOW)


@pytest.mark.unit
@pytest.mark.parametrize("text", ["25:00", "10:61", "tomorrow", "2026-11-03", "1030", "2026-13-01T10:30", ""])
def test_stop_time_garbage_is_refused(text):
    with pytest.raises(pipeline.PipelineError, match="--stop-at-time"):
        pipeline.parse_stop_time(text, NOW)


@pytest.mark.unit
@pytest.mark.parametrize(
    "argv",
    [
        ["--stop-after-stage", "S2", "--only-stage", "S2"],
        ["--stop-after-stage", "S1", "--from-stage", "S3"],
        ["--stop-after-stage", "S2", "--stop"],
        ["--stop-at-time", "23:59", "--stop"],
        ["--stop-after-stage", "S2", "--render-figures"],
        ["--stop-at-time", "23:59", "--render-figures"],
        ["--stop-at-time", "2020-01-01T00:00"],
        ["--stop-at-time", "2020-01-01T00:00", "--dry-run"],
    ],
)
def test_bad_pause_arguments_are_refused(tmp_path, monkeypatch, capsys, argv):
    monkeypatch.setenv("UQ_SMOKE_ROOT", str(tmp_path))
    assert pipeline.main(["--smoke", *argv]) == pipeline.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err
    assert not (tmp_path / "status.json").exists()


@pytest.mark.unit
def test_unknown_stage_fails_clearly(capsys):
    with pytest.raises(SystemExit) as exc:
        pipeline.main(["--stop-after-stage", "S9"])
    assert exc.value.code == pipeline.EXIT_REFUSED
    assert "S9" in capsys.readouterr().err


@pytest.mark.unit
def test_status_ignores_pause_arguments(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("UQ_SMOKE_ROOT", str(tmp_path))
    (tmp_path / "status.json").write_text(json.dumps({"stage": "S2", "state": "paused: --stop-after-stage S2"}))
    assert pipeline.main(["--smoke", "--status", "--stop-at-time", "2020-01-01T00:00"]) == pipeline.EXIT_OK
    assert "paused: --stop-after-stage S2" in capsys.readouterr().out


@pytest.mark.unit
def test_smoke_root_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("UQ_SMOKE_ROOT", raising=False)
    assert UQConfig.smoke().root == "artifacts/uq_smoke"
    monkeypatch.setenv("UQ_SMOKE_ROOT", str(tmp_path))
    assert UQConfig.smoke().root == str(tmp_path)
    assert UQConfig.full().root == "artifacts/uq"


@pytest.mark.unit
def test_deadline_reaches_the_child_and_is_absent_by_default():
    base = {"smoke": True, "allow_reeval": False}
    assert "--stop-deadline" not in pipeline._child_cmd(SimpleNamespace(**base), "S1")
    cmd = pipeline._child_cmd(SimpleNamespace(**base), "S1", 12.5)
    assert cmd[cmd.index("--stop-deadline") + 1] == "12.5"


@pytest.mark.unit
def test_status_text_names_pause_and_next_stage():
    text = pipeline.format_status(
        {"stage": "S2", "state": "paused: --stop-after-stage S2", "next_stage": "S3", "resume_command": "go"}
    )
    assert "paused: --stop-after-stage S2" in text and "Next stage: S3" in text and "go" in text
    planned = pipeline.format_status({"stage": "S1", "state": "running", "stop_after_stage": "S2"})
    assert "--stop-after-stage S2" in planned


# --- driver with fake stage children -----------------------------------------------------


class _Fake:
    def __init__(self, root) -> None:
        self.root = root

    def ran(self) -> list[str]:
        path = self.root / "ran.log"
        return path.read_text().split() if path.exists() else []

    def status(self) -> dict:
        return json.loads((self.root / "status.json").read_text())

    def done(self, stage: str) -> bool:
        return (self.root / "markers" / f"{stage}.done.json").exists()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Real driver, scripted children. ``use({"S4": "code"})`` runs that code before the stage reports success."""
    monkeypatch.setenv("UQ_SMOKE_ROOT", str(tmp_path))
    monkeypatch.setattr(pipeline, "dirty_code_paths", lambda: [])
    handlers = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    script = tmp_path / "child.py"

    def use(behaviour: dict[str, str] | None = None) -> None:
        script.write_text(
            "import json, os, pathlib, signal, sys, time\n"
            f"behaviour = {behaviour or {}!r}\n"
            f"root = pathlib.Path({str(tmp_path)!r}); stage = sys.argv[1]\n"
            "with (root / 'ran.log').open('a') as f: f.write(stage + '\\n')\n"
            "exec(behaviour.get(stage, ''))\n"
            "(root / 'outputs').mkdir(exist_ok=True)\n"
            "(root / 'outputs' / f'{stage}.json').write_text(json.dumps({'output_hash': stage}))\n"
        )

    use()
    monkeypatch.setattr(pipeline, "_child_cmd", lambda args, stage, deadline=None: [sys.executable, str(script), stage])
    fake_env = _Fake(tmp_path)
    fake_env.use = use  # type: ignore[attr-defined]
    # The deadline "passes" when a child says so, so these tests do not depend on machine load.
    late = tmp_path / "late"
    monkeypatch.setattr(pipeline, "_deadline_passed", lambda plan: plan.deadline is not None and late.exists())
    yield fake_env
    pipeline.STOP_REQUESTED.clear()
    signal.signal(signal.SIGTERM, handlers[0])
    signal.signal(signal.SIGINT, handlers[1])


@pytest.mark.integration
def test_stop_after_stage_pauses_then_normal_start_continues(fake, capsys):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S2"]) == pipeline.EXIT_OK
    assert fake.ran() == ["S0", "S1", "S2"]
    status = fake.status()
    assert status["state"] == "paused: --stop-after-stage S2" and status["next_stage"] == "S3"
    assert fake.done("S2") and not fake.done("S3")
    out = capsys.readouterr().out
    assert "scripts/run_uq_pipeline.sh --smoke" in out and "S3" in out
    assert "--smoke" in (fake.root / "logs" / "driver.log").read_text()

    assert pipeline.main(["--smoke"]) == pipeline.EXIT_OK
    assert fake.ran() == list(stages.STAGE_IDS), "finished stages must not run twice"
    status = fake.status()
    assert status["state"] == "finished" and status["next_stage"] is None


@pytest.mark.integration
def test_from_stage_continues_after_pause(fake):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S2"]) == pipeline.EXIT_OK
    assert pipeline.main(["--smoke", "--from-stage", "S3"]) == pipeline.EXIT_OK
    assert fake.ran() == list(stages.STAGE_IDS)


@pytest.mark.integration
def test_stop_after_a_finished_stage_pauses_without_work(fake):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S2"]) == pipeline.EXIT_OK
    assert pipeline.main(["--smoke", "--stop-after-stage", "S1"]) == pipeline.EXIT_OK
    assert fake.ran() == ["S0", "S1", "S2"]
    assert fake.status()["state"] == "paused: --stop-after-stage S1"


@pytest.mark.integration
def test_stop_after_last_stage_is_a_normal_finish(fake):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S7"]) == pipeline.EXIT_OK
    assert fake.status()["state"] == "finished"


@pytest.mark.integration
def test_child_pause_is_not_retried_and_resumes_the_same_stage(fake):
    fake.use({"S4": f"sys.exit({stages.EXIT_PAUSED})"})
    code = pipeline.main(["--smoke", "--stop-at-time", _in_seconds(3600)])
    assert code == pipeline.EXIT_OK
    assert fake.ran().count("S4") == 1
    status = fake.status()
    assert status["state"].startswith("paused: --stop-at-time") and status["next_stage"] == "S4"
    assert status["stages"]["S4"]["state"] == "paused"
    assert not fake.done("S4") and (fake.root / "markers" / "S4.started.json").exists()

    fake.use()
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_OK
    assert fake.ran() == [*stages.STAGE_IDS[:5], "S4", *stages.STAGE_IDS[5:]]
    assert fake.status()["state"] == "finished"


@pytest.mark.integration
def test_deadline_stops_before_the_next_stage(fake):
    fake.use({"S1": EXPIRE})
    assert pipeline.main(["--smoke", "--stop-at-time", LATER]) == pipeline.EXIT_OK
    assert fake.ran() == ["S0", "S1"]
    status = fake.status()
    assert status["state"].startswith("paused: --stop-at-time") and status["next_stage"] == "S2"
    assert fake.done("S1")


@pytest.mark.integration
def test_deadline_during_the_last_stage_is_a_finish(fake):
    fake.use({"S7": EXPIRE})
    assert pipeline.main(["--smoke", "--stop-at-time", LATER]) == pipeline.EXIT_OK
    assert fake.status()["state"] == "finished"


@pytest.mark.integration
def test_deadline_does_not_pause_when_only_skipped_stages_remain(fake):
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_OK
    (fake.root / "late").write_text("x")
    # nothing left to run: everything is skipped, so the deadline is irrelevant
    assert pipeline.main(["--smoke", "--stop-at-time", LATER]) == pipeline.EXIT_OK
    assert fake.status()["state"] == "finished"
    assert fake.ran() == list(stages.STAGE_IDS)


@pytest.mark.integration
def test_stage_boundary_first_wins_and_status_says_so(fake):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S1", "--stop-at-time", _in_seconds(3600)]) == 0
    assert fake.status()["state"] == "paused: --stop-after-stage S1"


@pytest.mark.integration
def test_time_first_wins_and_status_says_so(fake):
    fake.use({"S1": EXPIRE})
    assert pipeline.main(["--smoke", "--stop-after-stage", "S5", "--stop-at-time", LATER]) == 0
    assert fake.status()["state"].startswith("paused: --stop-at-time")
    assert fake.ran() == ["S0", "S1"]


@pytest.mark.integration
def test_new_start_clears_the_previous_pause(fake):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S0"]) == 0
    assert pipeline.main(["--smoke", "--stop-after-stage", "S1"]) == 0
    status = fake.status()
    assert status["state"] == "paused: --stop-after-stage S1" and status["next_stage"] == "S2"
    assert status["stop_at_time"] is None


@pytest.mark.integration
def test_force_stage_is_not_repeated_in_the_resume_command(fake):
    fake.use({"S4": f"sys.exit({stages.EXIT_PAUSED})"})
    assert pipeline.main(["--smoke", "--force-stage", "S4", "--stop-at-time", _in_seconds(3600)]) == 0
    assert "--force-stage" not in fake.status()["resume_command"]


@pytest.mark.integration
def test_default_run_has_no_pause(fake):
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_OK
    assert fake.ran() == list(stages.STAGE_IDS)
    assert fake.status()["state"] == "finished"


# --- --stop between stages ---------------------------------------------------------------


_TERM_THEN_SUCCEED = (
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); os.kill(os.getppid(), signal.SIGTERM); time.sleep(1)"
)


@pytest.mark.integration
def test_sigterm_between_stages_is_honored(fake, capsys):
    fake.use({"S1": _TERM_THEN_SUCCEED})
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_STOPPED
    assert fake.ran() == ["S0", "S1"], "the next stage must not start"
    assert fake.done("S1"), "the stage that finished keeps its marker"
    assert fake.status()["state"] == "stopped"
    assert not (fake.root / "driver.pid").exists()


@pytest.mark.integration
def test_stop_beats_a_pending_stage_boundary_pause(fake):
    fake.use({"S1": _TERM_THEN_SUCCEED})
    assert pipeline.main(["--smoke", "--stop-after-stage", "S1"]) == pipeline.EXIT_STOPPED
    assert fake.status()["state"] == "stopped"


@pytest.mark.integration
def test_sigterm_mid_stage_is_still_exit_130(fake):
    fake.use({"S1": "os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"})
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_STOPPED
    assert fake.ran() == ["S0", "S1"] and not fake.done("S1")


# --- dry run -----------------------------------------------------------------------------


@pytest.mark.integration
def test_dry_run_prints_the_pause_point_and_runs_nothing(fake, capsys):
    assert pipeline.main(["--smoke", "--dry-run", "--stop-after-stage", "S2"]) == 0
    out = capsys.readouterr().out
    assert "PAUSE after S2" in out and "S3: not reached" in out
    assert fake.ran() == []
    assert pipeline.main(["--smoke", "--dry-run", "--stop-at-time", "2099-01-01T10:30"]) == 0
    assert "2099-01-01T10:30" in capsys.readouterr().out


@pytest.mark.integration
def test_dry_run_names_what_made_a_stage_stale(fake, capsys, monkeypatch):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S2"]) == 0
    capsys.readouterr()
    marker_path = fake.root / "markers" / "S1.done.json"
    marker = json.loads(marker_path.read_text())
    marker["input_hash"] = "other"
    marker["inputs"]["config"]["gen"]["nx"] = -1
    marker_path.write_text(json.dumps(marker))
    assert pipeline.main(["--smoke", "--dry-run"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("S1:"))
    assert "BLOCKED" in line and "config gen.nx" in line and "S1: BLOCKED" in line


@pytest.mark.integration
def test_dry_run_flags_partial_output_from_other_inputs(fake, capsys):
    fake.use({"S4": f"sys.exit({stages.EXIT_PAUSED})"})
    assert pipeline.main(["--smoke", "--stop-at-time", _in_seconds(3600)]) == 0
    capsys.readouterr()
    started = fake.root / "markers" / "S4.started.json"
    data = json.loads(started.read_text())
    data["input_hash"] = "other"
    data["inputs"]["config"]["train"]["lr"] = 9.0
    started.write_text(json.dumps(data))
    assert pipeline.main(["--smoke", "--dry-run"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("S4:"))
    assert "BLOCKED" in line and "partial" in line and "config train.lr" in line


@pytest.mark.unit
def test_dry_run_reports_markers_without_recorded_inputs():
    reasons = pipeline.stale_reasons(None, {"code": {}, "config": {}, "run": {}, "upstream": {}})
    assert "not recorded" in reasons[0]


@pytest.mark.unit
def test_stale_reasons_cover_config_run_and_upstream():
    old = {"code": {}, "config": {"gen": {"nx": 1, "ny": 2}}, "run": {"chunk_size": 1}, "upstream": {"S3": "a"}}
    new = {"code": {}, "config": {"gen": {"nx": 9, "ny": 2}}, "run": {"chunk_size": 2}, "upstream": {"S3": "b"}}
    assert pipeline.stale_reasons(old, new) == ["config gen.nx", "run chunk_size", "upstream S3 output"]


@pytest.mark.integration
def test_stale_reasons_name_the_changed_file_inside_a_directory(tmp_path, monkeypatch):
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("1")
    (tmp_path / "pkg" / "b.py").write_text("1")
    (tmp_path / "top.txt").write_text("1")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one")
    old = {"pkg": git("rev-parse", "HEAD:pkg"), "top.txt": git("rev-parse", "HEAD:top.txt")}
    (tmp_path / "pkg" / "b.py").write_text("2")
    (tmp_path / "top.txt").write_text("2")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "two")
    new = {"pkg": git("rev-parse", "HEAD:pkg"), "top.txt": git("rev-parse", "HEAD:top.txt")}
    monkeypatch.chdir(tmp_path)
    empty = {"config": {}, "run": {}, "upstream": {}}
    assert pipeline.stale_reasons({"code": old, **empty}, {"code": new, **empty}) == ["pkg/b.py", "top.txt"]


@pytest.mark.unit
def test_marker_inputs_do_not_change_the_input_hash(tmp_path):
    cfg = UQConfig.smoke(tmp_path)
    payload = pipeline.input_payload(cfg, "S0", tmp_path)
    assert pipeline.input_hash(cfg, "S0", tmp_path) == pipeline.hash_payload(payload)


@pytest.mark.unit
def test_docs_list_every_hashed_path():
    from pathlib import Path

    text = Path("docs/uq.md").read_text()
    for stage, definition in stages.STAGE_DEFS.items():
        for path in definition["code"]:
            assert f"`{path}`" in text, f"{stage}: {path} missing from docs/uq.md"


# --- stage-level pause points ------------------------------------------------------------


def _ctx(tmp_path, deadline=None):
    cfg = UQConfig.smoke(tmp_path)
    return stages.Context("S1", cfg, False, stop_deadline=deadline)


@pytest.mark.unit
def test_check_pause_only_after_the_deadline(tmp_path):
    _ctx(tmp_path).check_pause()
    _ctx(tmp_path, deadline=4e9).check_pause()
    with pytest.raises(stages.StagePaused):
        _ctx(tmp_path, deadline=1.0).check_pause()


@pytest.mark.unit
def test_run_stage_turns_a_pause_into_its_exit_code(tmp_path, monkeypatch):
    def pauses(ctx):
        ctx.check_pause()

    monkeypatch.setitem(stages.STAGE_DEFS["S1"], "fn", pauses)
    cfg = UQConfig.smoke(tmp_path)
    assert stages.run_stage("S1", cfg, stop_deadline=1.0) == stages.EXIT_PAUSED
    assert not (tmp_path / "outputs" / "S1.json").exists(), "a paused stage reports no output"


def _fake_generation(monkeypatch, on_run):
    made = []

    def generate(cfg, run, path, log_dir, check_pause=None):
        made.append(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        on_run(run)

    monkeypatch.setattr(stages, "_generate_one", generate)
    monkeypatch.setattr(stages, "_is_valid_run", lambda *a, **k: True)
    monkeypatch.setattr(stages, "_data_hash", lambda cfg: "h")
    return made


@pytest.mark.unit
def test_s1_finishes_the_current_run_then_pauses_and_resumes_only_the_rest(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, deadline=4e9)

    def expire(run):
        ctx.stop_deadline = 1.0

    made = _fake_generation(monkeypatch, expire)
    with pytest.raises(stages.StagePaused):
        stages.s1_generate(ctx)
    assert made == [0]

    made_again = _fake_generation(monkeypatch, lambda run: None)
    assert stages.s1_generate(_ctx(tmp_path))["output_hash"] == "h"
    assert made_again == [1, 2, 3]


@pytest.mark.unit
def test_s1_deadline_during_the_last_run_completes_the_stage(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, deadline=4e9)

    def expire(run):
        if run == 3:
            ctx.stop_deadline = 1.0

    _fake_generation(monkeypatch, expire)
    assert stages.s1_generate(ctx)["output_hash"] == "h"


@pytest.mark.unit
def test_s5_pauses_at_a_chunk_boundary_but_not_after_the_final_chunk(tmp_path):
    ctx = _ctx(tmp_path, deadline=1.0)
    with pytest.raises(stages.StagePaused):
        stages._s5_progress(ctx, 0, 7)(1, 4)
    with pytest.raises(stages.StagePaused):
        stages._s5_progress(ctx, 0, 7)(4, 4)  # one job done, six more to go
    stages._s5_progress(ctx, 6, 7)(4, 4)  # everything is done


# --- review fixes -------------------------------------------------------------------------


@pytest.mark.unit
def test_driver_deadline_uses_the_clock():
    assert not pipeline._deadline_passed(pipeline.PausePlan())
    assert not pipeline._deadline_passed(pipeline.PausePlan(deadline=4e9))
    assert pipeline._deadline_passed(pipeline.PausePlan(deadline=1.0))


@pytest.mark.integration
def test_force_of_a_stage_not_yet_started_survives_a_pause(fake, capsys):
    assert pipeline.main(["--smoke"]) == 0
    assert pipeline.main(["--smoke", "--force-stage", "S5", "--stop-after-stage", "S3"]) == 0
    assert "--force-stage S5" in fake.status()["resume_command"]
    assert pipeline.main(["--smoke", "--force-stage", "S5"]) == 0
    assert fake.ran().count("S5") == 2


@pytest.mark.integration
def test_force_of_a_finished_stage_is_dropped_from_the_resume_command(fake):
    assert pipeline.main(["--smoke"]) == 0
    assert pipeline.main(["--smoke", "--force-stage", "S4", "--force-stage", "S6", "--stop-after-stage", "S4"]) == 0
    command = fake.status()["resume_command"]
    assert "--force-stage S6" in command and "--force-stage S4" not in command


@pytest.mark.integration
def test_sigterm_just_before_the_child_starts_still_kills_it(fake, monkeypatch):
    real = pipeline.mark_started

    def term_at_s1(root, stage, *args, **kwargs):
        if stage == "S1":
            os.kill(os.getpid(), signal.SIGTERM)
        return real(root, stage, *args, **kwargs)

    monkeypatch.setattr(pipeline, "mark_started", term_at_s1)
    fake.use({"S1": "time.sleep(3); (root / 'finished').write_text('y')"})
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_STOPPED
    assert not (fake.root / "finished").exists() and not fake.done("S1")


def _make_stale(fake, stage):
    path = fake.root / "markers" / f"{stage}.done.json"
    marker = json.loads(path.read_text())
    marker["input_hash"] = "other"
    path.write_text(json.dumps(marker))


@pytest.mark.integration
def test_refused_start_keeps_the_paused_status(fake, capsys):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S2"]) == 0
    _make_stale(fake, "S1")
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_REFUSED
    status = fake.status()
    assert status["state"] == "paused: --stop-after-stage S2" and status["next_stage"] == "S3"
    assert status["resume_command"]


@pytest.mark.integration
def test_refused_from_stage_keeps_the_paused_status(fake, capsys):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S0"]) == 0
    assert pipeline.main(["--smoke", "--from-stage", "S3"]) == pipeline.EXIT_REFUSED
    assert fake.status()["state"] == "paused: --stop-after-stage S0"


@pytest.mark.integration
def test_refusal_keeps_a_finished_state(fake, capsys):
    assert pipeline.main(["--smoke"]) == 0
    _make_stale(fake, "S3")
    assert pipeline.main(["--smoke", "--only-stage", "S3"]) == pipeline.EXIT_REFUSED
    assert fake.status()["state"] == "finished", "nothing ran, so the earlier state still holds"
    assert fake.ran().count("S3") == 1


@pytest.mark.integration
def test_refusal_on_a_fresh_root_is_visible(fake, capsys):
    assert pipeline.main(["--smoke", "--from-stage", "S3"]) == pipeline.EXIT_REFUSED
    assert fake.status()["state"] == "refused"


def _change_output(fake, stage):
    path = fake.root / "markers" / f"{stage}.done.json"
    marker = json.loads(path.read_text())
    marker["output_hash"] = "new"
    path.write_text(json.dumps(marker))


@pytest.mark.integration
def test_paused_rerun_continues_like_an_uninterrupted_one(fake):
    assert pipeline.main(["--smoke"]) == 0
    assert pipeline.main(["--smoke", "--force-stage", "S4", "--stop-after-stage", "S4"]) == 0
    assert fake.status()["rerun_from"] == "S5"
    _change_output(fake, "S4")
    assert pipeline.main(["--smoke"]) == 0
    assert fake.ran()[-3:] == ["S5", "S6", "S7"]
    assert fake.status()["rerun_from"] is None


@pytest.mark.integration
def test_stages_before_the_pause_point_stay_guarded_during_a_rerun(fake, capsys):
    assert pipeline.main(["--smoke"]) == 0
    assert pipeline.main(["--smoke", "--force-stage", "S4", "--stop-after-stage", "S4"]) == 0
    _make_stale(fake, "S1")
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_REFUSED
    assert "S1 finished with different inputs" in capsys.readouterr().err
    assert fake.status()["rerun_from"] == "S5", "the refusal must not lose the rerun"


@pytest.mark.integration
def test_fresh_chain_keeps_the_stale_guard_after_a_pause(fake, capsys):
    assert pipeline.main(["--smoke", "--stop-after-stage", "S2"]) == 0
    assert fake.status()["rerun_from"] == "S3"
    _make_stale(fake, "S1")
    assert pipeline.main(["--smoke"]) == pipeline.EXIT_REFUSED


@pytest.mark.unit
def test_stale_reasons_name_a_path_that_is_no_longer_hashed():
    empty = {"config": {}, "run": {}, "upstream": {}}
    reasons = pipeline.stale_reasons({"code": {"gone.py": "a"}, **empty}, {"code": {}, **empty})
    assert reasons == ["gone.py (no longer hashed)"]


@pytest.mark.unit
def test_render_figures_still_wins_over_status(monkeypatch):
    monkeypatch.setattr(pipeline, "_render_figures", lambda cfg: 42)
    assert pipeline.main(["--smoke", "--status", "--render-figures"]) == 42


@pytest.mark.unit
@pytest.mark.parametrize("text", ["2026-10-25T02:30", "2026-03-29T02:30"])
def test_stop_time_inside_a_clock_change_is_refused(monkeypatch, text):
    import time

    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    try:
        with pytest.raises(pipeline.PipelineError, match="clock change"):
            pipeline.parse_stop_time(text, datetime.datetime(2026, 1, 1))
        assert pipeline.parse_stop_time("2026-10-25T04:30", datetime.datetime(2026, 1, 1))
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


@pytest.mark.unit
def test_failed_run_is_not_retried_after_the_deadline(tmp_path, monkeypatch):
    calls = []

    def fail(cfg, run, out, log_dir):
        calls.append(run)
        out.write_text("broken")
        return 1

    monkeypatch.setattr(stages, "_run_generator", fail)
    monkeypatch.setattr(stages, "gpu_alive", lambda: True)
    monkeypatch.setattr(stages.time, "sleep", lambda s: None)
    ctx = _ctx(tmp_path, deadline=1.0)
    path = tmp_path / "run_000.h5"
    with pytest.raises(stages.StagePaused):
        stages._generate_one(ctx.cfg, 0, path, tmp_path, ctx.check_pause)
    assert calls == [0] and not path.with_suffix(".partial.h5").exists()
