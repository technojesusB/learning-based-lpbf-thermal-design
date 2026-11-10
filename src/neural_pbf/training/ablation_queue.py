"""The ablation queue runner: train the queued ablation experiments one after another, pausable and resumable.

Part of the ablation layer (``training/ablation.py``). Each experiment runs as a child process that imports its
experiment script and calls ``main(argv, hooks=...)`` with the argv built from the pinned recipe; the runner itself
takes infrastructure options only (paths, MLflow URI, UQ root, workers), so the recipe cannot be bypassed through it.
A finished experiment (its progress.json) is skipped, a paused or killed one resumes from its latest.pt.

``--stop`` asks the running queue to pause: the training child finishes its current epoch, saves latest.pt and
exits 75, and the runner marks the experiment paused and exits 0; the same command continues later. The child
processes, retries on GPU failures and the stall watchdog are the UQ pipeline's (``uq/pipeline.py``).

Frozen code: the queue refuses to start with uncommitted code (``dirty_code_paths``) or with ``neural_pbf`` imported
from another checkout than the current directory's (a worktree run with the main repo's virtualenv). A pinned resume
refuses another git commit (``RunTracker.resume``). Test mode (``--test-overrides``, hidden) replaces recipe values
for tiny runs, skips the dirty check and marks every run (provenance ``test_overrides``, MLflow lpbf_debug).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import neural_pbf
from neural_pbf.data.splits import _hash_manifest, load_manifest, save_manifest
from neural_pbf.training.ablation import EXPERIMENTS, QUEUE, ROOT, argv_for, hooks_for, load_script
from neural_pbf.training.tracking import DEBUG_EXPERIMENT, RunHooks
from neural_pbf.uq import pipeline
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.guard import atomic_write_text
from neural_pbf.uq.pipeline import EXIT_OK, EXIT_REFUSED, EXIT_STOPPED, run_child_with_retries
from neural_pbf.uq.stages import EXIT_PAUSED, dirty_code_paths, gpu_alive

RUNNER = ROOT / "experiments" / "run_ablation.py"
SLOW_EPOCH_MIN = 15.0  # above this, pausing at an epoch boundary is too slow for a meeting


@dataclass(frozen=True)
class Paths:
    """Everything the queue writes or reads besides the code; all of it can point at the main repo from a worktree."""

    root: Path  # queue state, per-experiment logs, preflight.json
    checkpoint_root: Path  # one directory per experiment: latest.pt, best.pt, progress.json, STOP
    mlflow_uri: str
    uq_root: Path  # the UQ pipeline root with the S1 dataset marker
    manifest: Path
    num_workers: int

    def ckpt(self, name: str) -> Path:
        return self.checkpoint_root / name

    def infra_args(self) -> list[str]:
        return ["--root", str(self.root), "--checkpoint_root", str(self.checkpoint_root), "--mlflow_uri",
                self.mlflow_uri, "--uq_root", str(self.uq_root), "--manifest", str(self.manifest), "--num_workers",
                str(self.num_workers)]  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation queue runner (docs/ablation.md)", allow_abbrev=False)
    p.add_argument("--only", metavar="NAME", help="run this one ablation experiment instead of the queue")
    p.add_argument("--status", action="store_true", help="current experiment, epoch, ETA, finished ones")
    p.add_argument("--stop", action="store_true", help="pause the running queue after the current epoch")
    p.add_argument("--dry-run", action="store_true", help="state of every experiment; validate every argv")
    p.add_argument("--preflight", action="store_true", help="a short GPU run per experiment: VRAM and time per epoch")
    p.add_argument("--root", default="logs/ablation")
    p.add_argument("--checkpoint_root", default="checkpoints")
    p.add_argument("--mlflow_uri", default="sqlite:///mlflow.db")
    p.add_argument("--uq_root", default=None, help="default: the UQ configuration's root")
    p.add_argument("--manifest", default="artifacts/uq/manifest.json")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test-overrides", help=argparse.SUPPRESS)
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--preflight-child", help=argparse.SUPPRESS)
    p.add_argument("--preflight-dir", help=argparse.SUPPRESS)
    return p


def paths_from(args: argparse.Namespace) -> Paths:
    return Paths(Path(args.root), Path(args.checkpoint_root), args.mlflow_uri,
                 Path(args.uq_root or UQConfig().root), Path(args.manifest), args.num_workers)  # fmt: skip


def queue_for(only: str | None, overrides: Mapping[str, Any] | None) -> list[str]:
    """The experiments to run, in queue order; test overrides restrict the queue to their experiments."""
    if only is not None:
        if only not in QUEUE:
            raise SystemExit(f"--only {only!r}: not a queued ablation experiment; queued: {list(QUEUE)}")
        return [only]
    return [n for n in QUEUE if overrides is None or n in overrides]


def experiment_state(paths: Paths, name: str) -> tuple[str, int, int | None]:
    """(pending | paused | finished, next epoch, epochs); a run killed while running resumes like a paused one."""
    path = paths.ckpt(name) / "progress.json"
    if not path.exists():
        return "pending", 0, None
    progress = json.loads(path.read_text())
    state = "finished" if progress["state"] == "finished" else "paused"
    return state, int(progress["next_epoch"]), int(progress["epochs"])


def _checkout_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def _frozen_code_problem(test_mode: bool) -> str | None:
    if not test_mode:
        dirty = dirty_code_paths()
        if dirty:
            return f"uncommitted code: {dirty}; commit it, or run the queue from a worktree at a fixed commit"
    package, checkout = Path(neural_pbf.__file__).resolve(), _checkout_root().resolve()
    if not package.is_relative_to(checkout):
        return (f"neural_pbf is imported from {package.parent}, not from this checkout ({checkout}); "
                "in a worktree, run `uv sync` there first")  # fmt: skip
    return None


def _ensure_mlflow_experiment(uri: str, experiment: str) -> None:
    """A sqlite store fixes an experiment's artifact location from the current directory at creation; create it next
    to the database instead, so a queue in a worktree never writes artifacts into the worktree."""
    if not uri.startswith("sqlite:///"):
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    if client.get_experiment_by_name(experiment) is None:
        db = Path(uri.removeprefix("sqlite:///")).resolve()
        client.create_experiment(experiment, artifact_location=str(db.parent / "mlruns" / experiment))


# --- child processes ------------------------------------------------------------------------------------------


def run_child(paths: Paths, name: str, overrides: Mapping[str, Any] | None) -> int:
    """One pinned training run, in this process; a pause ends it with SystemExit(75)."""
    module = load_script(EXPERIMENTS[name].script)
    parser = module.build_parser()
    hooks = hooks_for(name, overrides)
    _ensure_mlflow_experiment(paths.mlflow_uri, hooks.mlflow_experiment)
    argv = [*argv_for(name, parser, overrides), "--manifest", str(paths.manifest), "--mlflow_uri",
            paths.mlflow_uri, "--checkpoint_dir", str(paths.ckpt(name)), "--uq_root", str(paths.uq_root),
            "--num_workers", str(paths.num_workers)]  # fmt: skip
    module.main(argv, hooks=hooks)
    return EXIT_OK


def preflight_store(work: Path, name: str) -> str:
    """The throwaway MLflow store of one preflight run; absolute, since MLflow refuses a relative file:// URI."""
    return (work / "mlruns" / name).resolve().as_uri()


def run_preflight_child(paths: Paths, name: str, overrides: Mapping[str, Any] | None, work: Path) -> int:
    """One epoch of the pinned recipe on a small manifest, unpinned, into a throwaway store and checkpoint dir."""
    module = load_script(EXPERIMENTS[name].script)
    parser = module.build_parser()
    hooks = RunHooks(name=f"preflight_{name}", mlflow_experiment=DEBUG_EXPERIMENT)
    argv = [*argv_for(name, parser, overrides), "--epochs", "1", "--snapshot_every", "1000000", "--manifest",
            str(work / "manifest.json"), "--mlflow_uri", preflight_store(work, name), "--checkpoint_dir",
            str(work / "ckpt" / name), "--uq_root", str(paths.uq_root), "--num_workers",
            str(paths.num_workers)]  # fmt: skip
    module.main(argv, hooks=hooks)
    return EXIT_OK


def _child_cmd(paths: Paths, flag: str, name: str, overrides_file: Path | None, *extra: str) -> list[str]:
    cmd = [sys.executable, str(RUNNER), flag, name, *paths.infra_args(), *extra]
    return [*cmd, "--test-overrides", str(overrides_file)] if overrides_file is not None else cmd


# --- the queue ------------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class _Status:
    def __init__(self, paths: Paths) -> None:
        self.path = paths.root / "status.json"
        self.data: dict[str, Any] = json.loads(self.path.read_text()) if self.path.exists() else {"experiments": {}}

    def update(self, **fields: Any) -> None:
        self.data.update(fields, heartbeat=_now())
        atomic_write_text(self.path, json.dumps(self.data, indent=1))

    def experiment(self, name: str, **fields: Any) -> None:
        self.data["experiments"].setdefault(name, {}).update(fields)
        self.update()


def _lock_pid(lock: Path) -> int | None:
    """The PID in the lock file, only if that process really is a queue runner (not a child)."""
    try:
        pid = int(lock.read_text().strip())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    return pid if b"run_ablation.py" in cmdline and b"-child" not in cmdline else None


def _write_overrides(paths: Paths, overrides: Mapping[str, Any] | None) -> Path | None:
    if overrides is None:
        return None
    path = paths.root / "test_overrides.json"
    atomic_write_text(path, json.dumps(dict(overrides)))
    return path


def drive(paths: Paths, names: list[str], overrides: Mapping[str, Any] | None) -> int:
    """Run the experiments in order; 0 when all finished or the queue paused, else the failing child's code."""
    problem = _frozen_code_problem(test_mode=overrides is not None)
    if problem is not None:
        print(f"REFUSED: {problem}", file=sys.stderr, flush=True)
        return EXIT_REFUSED
    paths.root.mkdir(parents=True, exist_ok=True)
    lock = paths.root / "driver.pid"
    pid = _lock_pid(lock)
    if pid is not None:
        print(f"REFUSED: a queue is already running (pid {pid}); use --status or --stop", file=sys.stderr)
        return EXIT_REFUSED
    atomic_write_text(lock, str(os.getpid()))
    signal.signal(signal.SIGTERM, pipeline._handle_stop)  # forwards SIGTERM to the child's process group
    signal.signal(signal.SIGINT, pipeline._handle_stop)
    cfg = UQConfig().run
    status = _Status(paths)
    overrides_file = _write_overrides(paths, overrides)
    try:
        for name in names:
            state, next_epoch, epochs = experiment_state(paths, name)
            if state == "finished":
                print(f"[{_now()}] {name}: skip (finished)", flush=True)
                continue
            if pipeline.STOP_REQUESTED.is_set():
                break
            where = f"resuming at epoch {next_epoch}/{epochs}" if state == "paused" else "starting"
            print(f"[{_now()}] {name}: {where}", flush=True)
            status.update(current=name, state="running", pid=os.getpid())
            status.experiment(name, state="running", started=_now())
            log = paths.root / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{name}.log"
            code = run_child_with_retries(
                _child_cmd(paths, "--child", name, overrides_file), log, cfg.max_stage_retries, cfg.retry_backoff_s,
                gpu_alive, on_tick=lambda: status.update(), progress_path=paths.ckpt(name) / "progress.json",
                stall_timeout_s=cfg.stall_timeout_s,
            )  # fmt: skip
            state, next_epoch, epochs = experiment_state(paths, name)
            if state == "finished" and (code == EXIT_OK or pipeline.STOP_REQUESTED.is_set()):
                # a stop that reached the child after its last save (or before training began) loses nothing
                status.experiment(name, state="finished", finished=_now())
                print(f"[{_now()}] {name}: finished", flush=True)
                continue
            if code in (EXIT_PAUSED, EXIT_STOPPED) and (state == "paused" or pipeline.STOP_REQUESTED.is_set()):
                status.experiment(name, state="paused", next_epoch=next_epoch)
                status.update(state="paused")
                print(f"[{_now()}] {name}: paused at epoch {next_epoch}/{epochs}; the same command continues",
                      flush=True)  # fmt: skip
                return EXIT_OK
            if code != EXIT_OK:
                status.experiment(name, state="failed", exit_code=code)
                status.update(state="failed")
                print(f"[{_now()}] {name}: failed with exit {code}; see {log}", file=sys.stderr, flush=True)
                return code
        status.update(current=None, state="paused" if pipeline.STOP_REQUESTED.is_set() else "done")
        return EXIT_OK
    finally:
        lock.unlink(missing_ok=True)


def stop(paths: Paths) -> int:
    pid = _lock_pid(paths.root / "driver.pid")
    if pid is None:
        print("No running queue.")
        return EXIT_OK
    os.kill(pid, signal.SIGTERM)
    print(f"Asked queue {pid} to pause after the current epoch; the same start command continues.")
    return EXIT_OK


def _estimates(paths: Paths) -> dict[str, float]:
    path = paths.root / "preflight.json"
    if not path.exists():
        return {}
    return {r["name"]: r["est_total_h"] for r in json.loads(path.read_text())["experiments"] if r["ok"]}


def status_report(paths: Paths, names: list[str]) -> int:
    pid = _lock_pid(paths.root / "driver.pid")
    print(f"Queue: {'running (pid ' + str(pid) + ')' if pid else 'not running'}")
    estimates, total_h = _estimates(paths), 0.0
    for name in names:
        state, next_epoch, epochs = experiment_state(paths, name)
        progress = paths.ckpt(name) / "progress.json"
        eta_h = json.loads(progress.read_text())["eta_s"] / 3600 if progress.exists() else estimates.get(name)
        if state == "paused" and pid is not None and json.loads(progress.read_text())["state"] == "running":
            state = "running"
        epoch = f"epoch {next_epoch}/{epochs}" if epochs else ""
        eta = f"ETA {eta_h:.1f} h" if eta_h is not None and state != "finished" else ""
        total_h += (eta_h or 0.0) if state != "finished" else 0.0
        print(f"  {name:<32} {state:<9} {epoch:<14} {eta}")
    print(f"Remaining (known estimates): {total_h:.1f} h")
    return EXIT_OK


def dry_run(paths: Paths, names: list[str], overrides: Mapping[str, Any] | None = None) -> int:
    """Print each experiment's state and validate its argv against the pinned recipe; 2 if one is invalid."""
    problem = _frozen_code_problem(test_mode=overrides is not None)
    if problem is not None:
        print(f"BLOCKED: {problem}")
    code = EXIT_OK
    for name in names:
        state, next_epoch, epochs = experiment_state(paths, name)
        where = f"paused at epoch {next_epoch}/{epochs}" if state == "paused" else state
        module = load_script(EXPERIMENTS[name].script)
        parser = module.build_parser()
        try:
            args = parser.parse_args([*argv_for(name, parser, overrides), "--manifest", str(paths.manifest)])
            hooks = hooks_for(name, overrides)
            assert hooks.validate is not None
            hooks.validate(args, module.spec(args).recipe_fixed())
            check = "argv OK"
        except (ValueError, SystemExit) as exc:
            check, code = f"argv INVALID ({exc})", EXIT_REFUSED
        print(f"{name}: {where}, {check}")
    return code


# --- preflight ------------------------------------------------------------------------------------------------


def _small_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    small = {**manifest, "sets": {"train": manifest["sets"]["train"][:2], "val": manifest["sets"]["val"][:1],
                                  "cal": [], "test": []}}  # fmt: skip
    small["hash"] = _hash_manifest(small)
    return small


def _preflight_result(work: Path, name: str, full: dict[str, Any], small: dict[str, Any], epochs: int) -> dict:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=preflight_store(work, name))
    experiment = client.get_experiment_by_name(DEBUG_EXPERIMENT)
    assert experiment is not None, f"the preflight of {name} logged no MLflow run"
    run = client.search_runs([experiment.experiment_id])[0]
    m = run.data.metrics
    count = lambda mf, s: sum(len(r["sample_keys"]) for r in mf["sets"][s])  # noqa: E731
    n_small = int(run.data.params["data.n_train"])
    train_s = n_small / m["throughput_samples_s"]
    val_s = max(m["epoch_time_s"] - train_s, 0.0)
    epoch_s = count(full, "train") / m["throughput_samples_s"] + val_s * count(full, "val") / count(small, "val")
    return {"name": name, "ok": True, "peak_vram_gb": m["peak_gpu_mem_mb"] / 1024, "est_epoch_min": epoch_s / 60,
            "epochs": epochs, "est_total_h": epoch_s * epochs / 3600}  # fmt: skip


def preflight(paths: Paths, names: list[str], overrides: Mapping[str, Any] | None) -> int:
    """One epoch per experiment on two train runs and one val run, scaled to the full manifest."""
    full = load_manifest(paths.manifest)
    small = _small_manifest(full)
    work = paths.root / "preflight_work"
    shutil.rmtree(work, ignore_errors=True)
    save_manifest(small, work / "manifest.json")
    overrides_file = _write_overrides(paths, overrides)
    results = []
    print(f"{'experiment':<32} {'':<6} {'peak VRAM':>10} {'min/epoch':>10} {'epochs':>7} {'est. total':>11}")
    for name in names:
        log = paths.root / f"preflight_{name}.log"
        with log.open("w") as f:
            code = subprocess.run(_child_cmd(paths, "--preflight-child", name, overrides_file, "--preflight-dir",
                                             str(work)), stdout=f, stderr=subprocess.STDOUT).returncode  # fmt: skip
        epochs = int({**EXPERIMENTS[name].recipe, **(overrides or {}).get(name, {})}["epochs"])
        if code == EXIT_OK:
            r = _preflight_result(work, name, full, small, epochs)
            print(f"{name:<32} {'OK':<6} {r['peak_vram_gb']:>7.2f} GB {r['est_epoch_min']:>10.1f} {epochs:>7} "
                  f"{r['est_total_h']:>9.1f} h", flush=True)  # fmt: skip
        else:
            r = {"name": name, "ok": False, "exit_code": code, "log": str(log)}
            print(f"{name:<32} {'FAILED':<6} exit {code}, see {log}", flush=True)
        results.append(r)
    ok = [r for r in results if r["ok"]]
    total = sum(r["est_total_h"] for r in ok)
    print(f"{'total':<32} {'':<6} {'':>10} {'':>10} {'':>7} {total:>9.1f} h  ({total / 24:.1f} days)")
    slow = [r["name"] for r in ok if r["est_epoch_min"] > SLOW_EPOCH_MIN]
    if slow:
        print(f"WARNING: above {SLOW_EPOCH_MIN:.0f} min per epoch, too slow to pause for a meeting: {slow}")
    atomic_write_text(paths.root / "preflight.json", json.dumps({
        "created": _now(), "num_workers": paths.num_workers, "experiments": results, "total_h": total,
    }, indent=1))  # fmt: skip
    shutil.rmtree(work, ignore_errors=True)
    return EXIT_OK if len(ok) == len(results) else EXIT_REFUSED


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = paths_from(args)
    overrides = json.loads(Path(args.test_overrides).read_text()) if args.test_overrides else None
    if args.child:
        return run_child(paths, args.child, (overrides or {}).get(args.child) if overrides else None)
    if args.preflight_child:
        name = args.preflight_child
        return run_preflight_child(paths, name, (overrides or {}).get(name) if overrides else None,
                                   Path(args.preflight_dir))  # fmt: skip
    names = queue_for(args.only, overrides)
    if args.status:
        return status_report(paths, names)
    if args.stop:
        return stop(paths)
    per_name = {n: overrides[n] for n in names} if overrides else None
    if args.dry_run:
        return dry_run(paths, names, None) if per_name is None else _dry_run_test(paths, names, per_name)
    if args.preflight:
        paths.root.mkdir(parents=True, exist_ok=True)
        return preflight(paths, names, per_name)
    return drive(paths, names, per_name)


def _dry_run_test(paths: Paths, names: list[str], per_name: Mapping[str, Any]) -> int:
    return max((dry_run(paths, [n], per_name[n]) for n in names), default=EXIT_OK)
