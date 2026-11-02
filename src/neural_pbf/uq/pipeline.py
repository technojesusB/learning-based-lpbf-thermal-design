"""UQ pipeline driver. Start it through scripts/run_uq_pipeline.sh.

Each stage runs in a child process so a GPU failure cannot take the driver down.
Exit codes: 0 ok, 1 failed, 2 refused (validation or guard), 3 GPU lost, 130 stopped.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import enum
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from neural_pbf.uq.config import UQConfig  # noqa: E402
from neural_pbf.uq.guard import atomic_write_text  # noqa: E402
from neural_pbf.uq.stages import (  # noqa: E402
    STAGE_DEFS,
    STAGE_IDS,
    PipelineError,
    code_tree_hash,
    dirty_code_paths,
    disk_guard,  # noqa: F401  (re-exported)
    git_commit,
    gpu_alive,
    package_versions,
    run_stage,
)

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_GPU_LOST, EXIT_STOPPED = 0, 1, 2, 3, 130
_GPU_RE = re.compile(
    r"CUDA error|cuda runtime error|device-side assert|GPU is lost|driver|nvidia-smi has failed|"
    r"Unable to determine the device handle|no longer available|Triton Error",
    re.IGNORECASE,
)
_OOM_RE = re.compile(r"out of memory", re.IGNORECASE)
_GPU_MESSAGE = (
    "GPU/driver lost and not recoverable from here (WSL2 TDR). Finished work is safe on disk.\n"
    "What to do: 1) in Windows PowerShell run `wsl --shutdown`, restart WSL. "
    "2) Check `nvidia-smi`. 3) Rerun the same command; it resumes where it stopped."
)


class StageState(enum.Enum):
    RUN = "run"
    SKIP = "skip"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_marker(root: Path, stage: str, input_hash: str, output_hash: str, provenance: dict[str, Any]) -> None:
    marker = {"input_hash": input_hash, "output_hash": output_hash, "finished": _now(), "provenance": provenance}
    atomic_write_text(Path(root) / "markers" / f"{stage}.done.json", json.dumps(marker, indent=1))
    (Path(root) / "markers" / f"{stage}.started.json").unlink(missing_ok=True)


def read_marker(root: Path, stage: str) -> dict[str, Any] | None:
    path = Path(root) / "markers" / f"{stage}.done.json"
    return json.loads(path.read_text()) if path.exists() else None


def mark_started(root: Path, stage: str, input_hash: str) -> None:
    atomic_write_text(Path(root) / "markers" / f"{stage}.started.json", json.dumps({"input_hash": input_hash}))


def decide_stage(root: Path, stage: str, input_hash: str, force: bool, upstream_reran: bool) -> StageState:
    """Skip finished work; refuse stale work unless a rerun was asked for or is implied."""
    if force or upstream_reran:
        return StageState.RUN
    marker = read_marker(root, stage)
    if marker is not None:
        if marker["input_hash"] == input_hash:
            return StageState.SKIP
        raise PipelineError(
            f"{stage} finished with different inputs (config, code, or upstream data changed). "
            f"Rerun it deliberately with --force-stage {stage}, or restore the old inputs."
        )
    started = Path(root) / "markers" / f"{stage}.started.json"
    if started.exists() and json.loads(started.read_text())["input_hash"] != input_hash:
        raise PipelineError(
            f"{stage} has partial output from different inputs. Mixing them would corrupt results. "
            f"Use --force-stage {stage}, or restore the old inputs."
        )
    return StageState.RUN


def select_stages(from_stage: str | None, only_stage: str | None) -> list[str]:
    if from_stage and only_stage:
        raise PipelineError("--from-stage and --only-stage are mutually exclusive")
    for stage in filter(None, (from_stage, only_stage)):
        if stage not in STAGE_IDS:
            raise PipelineError(f"unknown stage {stage}; expected one of {STAGE_IDS}")
    if only_stage:
        return [only_stage]
    return list(STAGE_IDS[STAGE_IDS.index(from_stage) :]) if from_stage else list(STAGE_IDS)


def classify_failure(returncode: int, tail: str) -> str:
    if returncode == EXIT_GPU_LOST or returncode < 0:
        return "gpu"
    if returncode == EXIT_REFUSED:
        return "fatal"
    if _OOM_RE.search(tail):
        return "oom"
    return "gpu" if _GPU_RE.search(tail) else "fatal"


def _tail(path: Path, n: int = 6000) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - n))
        return f.read().decode(errors="replace")


_current_child: subprocess.Popen[bytes] | None = None


def run_child_with_retries(
    cmd: list[str],
    log_path: Path,
    max_retries: int,
    backoff_s: int,
    gpu_alive: Callable[[], bool],
    on_tick: Callable[[], None] | None = None,
) -> int:
    """Run a stage child with bounded retries on GPU or memory failures. Returns an exit code."""
    global _current_child
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_retries + 1):
        start = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open("ab") as log:
            log.write(f"\n=== {_now()} attempt {attempt + 1}/{max_retries + 1}: {' '.join(cmd)}\n".encode())
            log.flush()
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            _current_child = child
            while True:
                try:
                    code = child.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    if on_tick is not None:
                        on_tick()
            _current_child = None
        if code == EXIT_OK:
            return EXIT_OK
        if code == EXIT_STOPPED:
            return EXIT_STOPPED
        tail = _tail(log_path, min(6000, log_path.stat().st_size - start))
        kind = classify_failure(code, tail)
        print(f"[{_now()}] stage child failed: exit {code}, class {kind}, see {log_path}", flush=True)
        if kind == "fatal":
            return EXIT_REFUSED if code == EXIT_REFUSED else EXIT_FAILED
        if kind == "gpu" and not gpu_alive():
            print(_GPU_MESSAGE, flush=True)
            return EXIT_GPU_LOST
        if attempt < max_retries:
            wait = backoff_s * (attempt + 1)
            print(f"[{_now()}] retrying in {wait}s", flush=True)
            time.sleep(wait)
    if kind == "oom":
        print("Out of GPU memory on every attempt. Lower run.draw_batch in the config and rerun.", flush=True)
        return EXIT_FAILED
    print(_GPU_MESSAGE, flush=True)
    return EXIT_GPU_LOST


def load_status(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise PipelineError(f"no status file at {path}; the pipeline has not started")
    return json.loads(Path(path).read_text())


def format_status(status: dict[str, Any]) -> str:
    lines = [f"Stage: {status.get('stage')}  state: {status.get('state')}"]
    progress = status.get("progress") or {}
    if progress:
        lines.append(f"Progress: {progress.get('done')}/{progress.get('total')}")
    if status.get("eta_s") is not None:
        lines.append(f"ETA for this stage: {status['eta_s'] / 3600:.1f} h")
    heartbeat = status.get("heartbeat")
    if heartbeat:
        age = (datetime.datetime.now() - datetime.datetime.fromisoformat(heartbeat)).total_seconds()
        lines.append(f"Last heartbeat: {heartbeat} ({age:.0f}s ago)")
        if status.get("state") == "running" and age > 120:
            lines.append("WARNING: no heartbeat for over 2 minutes. The driver may be dead; check tmux and logs.")
    if status.get("timeline"):
        lines.append(f"Timeline: {json.dumps(status['timeline'])}")
    for stage, info in (status.get("stages") or {}).items():
        lines.append(f"  {stage}: {info.get('state')}  {info.get('duration_s', '')}")
    return "\n".join(lines)


class Status:
    """status.json is written only here; children report through progress files."""

    def __init__(self, root: Path, mode: str) -> None:
        self.path = root / "status.json"
        self.root = root
        self.lock = threading.Lock()
        self.data: dict[str, Any] = (
            load_status(self.path) if self.path.exists() else {"mode": mode, "stages": {}, "timeline": {}}
        )

    def update(self, **fields: Any) -> None:
        with self.lock:
            self.data.update(fields)
            self.data["heartbeat"] = _now()
            atomic_write_text(self.path, json.dumps(self.data, indent=1))

    def stage(self, stage: str, **fields: Any) -> None:
        with self.lock:
            self.data["stages"].setdefault(stage, {}).update(fields)
        self.update()

    def tick(self, stage: str) -> None:
        path = self.root / "progress" / f"{stage}.json"
        if path.exists():
            try:
                progress = json.loads(path.read_text())
            except json.JSONDecodeError:
                return
            timeline = {**self.data.get("timeline", {}), **progress.get("timeline", {})}
            self.update(
                progress={"done": progress["done"], "total": progress["total"]},
                eta_s=progress.get("eta_s"),
                timeline=timeline,
            )
        else:
            self.update()


def input_hash(cfg: UQConfig, stage: str, root: Path) -> str:
    definition = STAGE_DEFS[stage]
    upstream: dict[str, str] = {}
    for dep in definition["upstream"]:
        marker = read_marker(root, dep)
        if marker is None:
            raise PipelineError(f"{stage} needs {dep}, which has not finished")
        upstream[dep] = marker["output_hash"]
    payload = {
        "stage": stage,
        "config": {section: getattr(cfg, section).model_dump(mode="json") for section in definition["config"]},
        "code": code_tree_hash(definition["code"]),
        "upstream": upstream,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _child_cmd(args: argparse.Namespace, stage: str) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "neural_pbf.uq.pipeline", "--child", stage]
    if args.smoke:
        cmd.append("--smoke")
    if args.allow_reeval:
        cmd.append("--allow-reeval")
    return cmd


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _claim_lock(root: Path) -> Path:
    lock = root / "driver.pid"
    if lock.exists():
        pid = int(lock.read_text().strip() or 0)
        if _pid_alive(pid):
            raise PipelineError(f"another driver is already running (pid {pid}); use --status or --stop")
    atomic_write_text(lock, str(os.getpid()))
    return lock


def _handle_stop(signum: int, frame: Any) -> None:
    if _current_child is not None:
        with contextlib.suppress(OSError):
            os.killpg(_current_child.pid, signal.SIGTERM)


def drive(cfg: UQConfig, args: argparse.Namespace) -> int:
    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    dirty = dirty_code_paths()
    if dirty and not args.dry_run:
        raise PipelineError(f"uncommitted changes in pipeline code: {dirty}. Commit them so results stay traceable.")
    selected = select_stages(args.from_stage, args.only_stage)
    forced = set(args.force_stage or [])
    stop_requested = threading.Event()

    if args.dry_run:
        return _dry_run(cfg, root, selected, forced)

    lock = _claim_lock(root)
    signal.signal(signal.SIGTERM, lambda s, f: (stop_requested.set(), _handle_stop(s, f)))
    signal.signal(signal.SIGINT, lambda s, f: (stop_requested.set(), _handle_stop(s, f)))
    status = Status(root, cfg.mode)
    status.update(state="running", started=_now(), git_commit=git_commit())
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    upstream_reran = False
    try:
        for stage in STAGE_IDS:
            if stage not in selected:
                continue
            hash_in = input_hash(cfg, stage, root)
            force = stage in forced
            decision = decide_stage(root, stage, hash_in, force, upstream_reran)
            if decision is StageState.SKIP:
                print(f"[{_now()}] {stage}: done, skipping", flush=True)
                status.stage(stage, state="done (skipped)")
                continue
            if force:
                _wipe(cfg, stage)
            mark_started(root, stage, hash_in)
            print(f"[{_now()}] {stage}: {STAGE_DEFS[stage]['title']}", flush=True)
            (root / "progress" / f"{stage}.json").unlink(missing_ok=True)
            status.update(stage=stage, state="running", progress=None, eta_s=None)
            status.stage(stage, state="running", started=_now())
            t0 = time.time()
            code = run_child_with_retries(
                _child_cmd(args, stage),
                Path(cfg.log_dir) / f"{stamp}_{stage}.log",
                cfg.run.max_stage_retries,
                cfg.run.retry_backoff_s,
                gpu_alive,
                on_tick=lambda s=stage: status.tick(s),
            )
            duration = round(time.time() - t0, 1)
            if code != EXIT_OK:
                state = "stopped" if stop_requested.is_set() else "failed"
                status.stage(stage, state=state, duration_s=duration, exit_code=code)
                status.update(state=state)
                print(
                    f"[{_now()}] {stage}: {state} with exit {code}. Log: {cfg.log_dir}/{stamp}_{stage}.log", flush=True
                )
                return EXIT_STOPPED if stop_requested.is_set() else code
            status.tick(stage)
            output = json.loads((root / "outputs" / f"{stage}.json").read_text())
            provenance = {
                "git_commit": git_commit(),
                "seeds": {
                    "gen": cfg.gen.base_seed, "split": cfg.split.seed, "train": cfg.train.seed,
                    "sampling": cfg.prereg.sampling_seed,
                },
                "config": cfg.model_dump(mode="json"),
                "packages": package_versions(),
                "data_hash": output.get("data_hash"),
                "duration_s": duration,
                **output.get("provenance", {}),
            }  # fmt: skip
            write_marker(root, stage, hash_in, output["output_hash"], provenance)
            upstream_reran = True  # anything downstream of an executed stage may legitimately be stale
            status.stage(stage, state="done", duration_s=duration)
            print(f"[{_now()}] {stage}: done in {duration}s", flush=True)
        status.update(state="finished", progress=None, eta_s=None)
        print(f"[{_now()}] Pipeline finished. Report: {cfg.report_path}", flush=True)
        return EXIT_OK
    finally:
        lock.unlink(missing_ok=True)


def _wipe(cfg: UQConfig, stage: str) -> None:
    for path in STAGE_DEFS[stage].get("wipe", lambda c: [])(cfg):
        print(f"[{_now()}] --force-stage {stage}: removing {path}", flush=True)
        shutil.rmtree(path, ignore_errors=True)


def _dry_run(cfg: UQConfig, root: Path, selected: list[str], forced: set[str]) -> int:
    for stage in STAGE_IDS:
        if stage not in selected:
            print(f"{stage}: not selected")
            continue
        marker = read_marker(root, stage)
        if stage in forced:
            action = "RUN (forced)"
        elif marker is None:
            action = "RUN"
        else:
            try:
                same = marker["input_hash"] == input_hash(cfg, stage, root)
                action = "SKIP (done)" if same else "BLOCKED (stale; needs --force-stage)"
            except PipelineError as exc:
                action = f"BLOCKED ({exc})"
        print(f"{stage}: {action}  {STAGE_DEFS[stage]['title']}")
    return EXIT_OK


def _stop(cfg: UQConfig) -> int:
    lock = Path(cfg.root) / "driver.pid"
    if not lock.exists():
        print("No running driver.")
        return EXIT_OK
    pid = int(lock.read_text())
    os.kill(pid, signal.SIGTERM)
    print(f"Sent stop to driver {pid}. Finished work stays on disk; rerun the start command to resume.")
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="UQ pipeline driver")
    p.add_argument("--resume", action="store_true", help="default behavior; finished stages are skipped")
    p.add_argument("--from-stage", choices=STAGE_IDS)
    p.add_argument("--only-stage", choices=STAGE_IDS)
    p.add_argument("--force-stage", action="append", choices=STAGE_IDS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--allow-reeval", action="store_true", help="permit a second test evaluation; recorded in the report"
    )
    p.add_argument("--child", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = UQConfig.smoke() if args.smoke else UQConfig.full()
    try:
        if args.child:
            return run_stage(args.child, cfg, allow_reeval=args.allow_reeval)
        if args.status:
            print(format_status(load_status(Path(cfg.root) / "status.json")))
            return EXIT_OK
        if args.stop:
            return _stop(cfg)
        return drive(cfg, args)
    except PipelineError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr, flush=True)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
