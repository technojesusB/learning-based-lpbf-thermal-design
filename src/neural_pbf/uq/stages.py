"""The eight pipeline stages. Each runs in its own process through run_stage()."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.guard import GuardError, atomic_write_text

logger = logging.getLogger(__name__)

STAGE_IDS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
_GENERATOR = "experiments/generate_offline_dataset.py"
_REPO_CODE = ["src/neural_pbf", _GENERATOR, "pyproject.toml", "uv.lock"]
_PHYSICS_CODE = [
    _GENERATOR,
    "src/neural_pbf/physics",
    "src/neural_pbf/integrator/stepper.py",
    "src/neural_pbf/scan",
    "src/neural_pbf/pipelines",
    "src/neural_pbf/core",
    "src/neural_pbf/data/sampling.py",
]
_MODEL_CODE = [
    "src/neural_pbf/models/generative/fm",
    "src/neural_pbf/data/fm_dataset.py",
    "src/neural_pbf/data/patch_dataset.py",
    "src/neural_pbf/uq/train.py",
    "src/neural_pbf/training/factories.py",
]
_SAMPLING_CODE = [
    *_MODEL_CODE,
    "src/neural_pbf/uq/harness.py",
    "src/neural_pbf/uq/sampling.py",
    "src/neural_pbf/eval/benchmark",
    "src/neural_pbf/eval/metrics/geometry.py",
    "src/neural_pbf/integrator/fm_stepper.py",
]
_EVAL_CODE = ["src/neural_pbf/uq/analysis.py", "src/neural_pbf/uq/conformal.py", "src/neural_pbf/uq/report.py"]

# Planning figures for the timeline, from the README and legacy generation logs. Replaced by measurements as they exist.
_LEGACY_TRAIN_H_PER_SAMPLE_EPOCH = 1.39 / (135 * 100)
_LEGACY_ROLLOUT_S = 1.92


class PipelineError(RuntimeError):
    """Refuse to continue; never retried."""


class GpuLost(RuntimeError):
    """The GPU or driver is gone."""


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


def code_tree_hash(paths: list[str]) -> dict[str, str]:
    return {p: _git("rev-parse", f"HEAD:{p}") for p in paths}


def dirty_code_paths() -> list[str]:
    tracked = ["src", "experiments", "scripts/run_uq_pipeline.sh", "pyproject.toml", "uv.lock"]
    out = _git("status", "--porcelain", "--", *tracked)
    return [line[3:] for line in out.splitlines()]


def package_versions() -> dict[str, str]:
    names = ("torch", "numpy", "h5py", "mlflow", "triton", "pydantic", "pandas")
    return {n: importlib.metadata.version(n) for n in names} | {"python": sys.version.split()[0]}


def gpu_alive() -> bool:
    """Fresh-process check, because a process that saw the failure cannot be trusted."""
    code = "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda').sum().item()"
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=60)
        probe = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return smi.returncode == 0 and probe.returncode == 0


def disk_guard(path: str | Path, min_free_gb: float, purpose: str) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 1e9
    if free < min_free_gb:
        raise PipelineError(f"{purpose}: {free:.1f} GB free on {path}, needs at least {min_free_gb:.1f} GB")
    logger.info("%s: %.1f GB free on %s", purpose, free, path)


class Context:
    def __init__(self, stage: str, cfg: UQConfig, allow_reeval: bool) -> None:
        self.stage, self.cfg, self.allow_reeval = stage, cfg, allow_reeval
        self.root = Path(cfg.root)
        self._t0 = time.time()
        self._d0: int | None = None

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def prereg_path(self) -> Path:
        return self.root / "prereg.json"

    @property
    def ckpt_dir(self) -> Path:
        return self.root / "ckpt"

    @property
    def draws_dir(self) -> Path:
        return self.root / "draws"

    def progress(self, done: int, total: int, timeline: dict[str, Any] | None = None) -> None:
        """Report progress; the ETA counts only items finished in this attempt."""
        if self._d0 is None:
            self._d0, self._t0 = done, time.time()
        rate_items = done - self._d0
        eta = (time.time() - self._t0) / rate_items * (total - done) if rate_items > 0 else None
        payload = {"done": done, "total": total, "eta_s": eta, "timeline": timeline or {}}
        atomic_write_text(self.root / "progress" / f"{self.stage}.json", json.dumps(payload))
        logger.info("progress %d/%d eta=%s", done, total, f"{eta / 3600:.2f} h" if eta is not None else "unknown")


def _run_paths(cfg: UQConfig) -> list[Path]:
    return [Path(cfg.data_dir) / f"run_{i:03d}.h5" for i in range(cfg.gen.n_runs)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_hash(cfg: UQConfig) -> str:
    """Names, sizes, and run identities; hashing tens of GB on every resume is not worth it."""
    from neural_pbf.data.splits import discover_runs

    paths = _run_paths(cfg)
    runs, _ = discover_runs([str(p) for p in paths])
    body = [[p.name, p.stat().st_size] for p in paths] + sorted(r.run_id for r in runs)
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


# --- S0 --------------------------------------------------------------------------------


def s0_preflight(ctx: Context) -> dict[str, Any]:
    cfg = ctx.cfg
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "feat/uq-conformal":
        raise PipelineError(f"must run on feat/uq-conformal, currently on {branch}")
    dirty = _git("status", "--porcelain")
    if dirty:
        raise PipelineError(f"working tree is not clean:\n{dirty}")
    if not gpu_alive():
        raise PipelineError("GPU not visible (nvidia-smi or torch.cuda failed)")
    disk_guard(cfg.data_dir, cfg.run.min_free_gb_generate, "preflight")
    subprocess.run(["uv", "sync", "--locked", "--inexact"], check=True)
    tests = [
        "tests/uq/test_conformal.py", "tests/uq/test_guard.py", "tests/uq/test_sampling.py",
        "tests/data/test_splits.py", "tests/data/test_mask_source.py", "tests/scripts/test_generator_causality.py",
    ]  # fmt: skip
    subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", "-o", "addopts="], check=True)
    return {"output_hash": "preflight-ok"}


# --- S1 --------------------------------------------------------------------------------


def _generator_cmd(cfg: UQConfig, run: int, out: Path) -> list[str]:
    g = cfg.gen
    cmd = [
        sys.executable, _GENERATOR, "--out", str(out), "--materials", g.materials, "--patterns", g.patterns,
        "--nx", str(g.nx), "--ny", str(g.ny), "--nz", str(g.nz), "--device", cfg.run.device, "--seed", str(g.base_seed),
        "--samples-per-run", str(g.samples_per_run), "--run-index", str(run), "--no-diag-csv",
        "--orchestration-id", f"uq_{cfg.mode}",
    ]  # fmt: skip
    return [*cmd, "--wsl-safe"] if g.wsl_safe else cmd


def _timeline(cfg: UQConfig, seconds_per_run: float) -> dict[str, Any]:
    sample_epochs = cfg.gen.samples_per_run * cfg.gen.n_runs * cfg.train.max_epochs * 0.55  # train + val share
    rollouts = cfg.gen.samples_per_run * cfg.gen.n_runs * 0.45 * cfg.prereg.K
    generation = seconds_per_run * cfg.gen.n_runs / 3600
    training = _LEGACY_TRAIN_H_PER_SAMPLE_EPOCH * sample_epochs
    sampling = rollouts * _LEGACY_ROLLOUT_S / 3600 * 2.1  # new Baseline, legacy Baseline, second seed
    return {
        "measured_s_per_run": round(seconds_per_run, 1),
        "projected_generation_h": round(generation, 1),
        "planning_training_h_upper_bound": round(training, 1),
        "planning_sampling_h_unbatched_upper_bound": round(sampling, 1),
        "projected_total_h": round(generation + training + sampling, 1),
    }


def s1_generate(ctx: Context) -> dict[str, Any]:
    from neural_pbf.uq.validate import ValidationError, validate_run_file

    cfg = ctx.cfg
    data = Path(cfg.data_dir)
    disk_guard(data, cfg.run.min_free_gb_generate if not any(data.glob("run_*.h5")) else 5.0, "generation")
    log_dir = Path(cfg.log_dir) / "generation"
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = _run_paths(cfg)

    def valid(path: Path) -> bool:
        try:
            validate_run_file(str(path), cfg.gen.samples_per_run, deep=False)
            return True
        except ValidationError as exc:
            logger.warning("%s", exc)
            return False

    for path in paths:
        if path.exists() and not valid(path):
            quarantine = path.with_suffix(".invalid")
            logger.warning("moving invalid run aside: %s -> %s", path, quarantine)
            os.replace(path, quarantine)

    done = sum(p.exists() for p in paths)
    ctx.progress(done, len(paths))
    measured: list[float] = []
    for run, path in enumerate(paths):
        if path.exists():
            continue
        partial = path.with_suffix(".partial.h5")
        t0 = time.time()
        for attempt in range(cfg.gen.max_retries_per_run + 1):
            partial.unlink(missing_ok=True)
            with (log_dir / f"run_{run:03d}.log").open("a") as log:
                result = subprocess.run(_generator_cmd(cfg, run, partial), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0 and valid(partial):
                os.replace(partial, path)
                break
            logger.error("run %d attempt %d failed (exit %d)", run, attempt + 1, result.returncode)
            if not gpu_alive():
                raise GpuLost(f"GPU lost while generating run {run}")
            time.sleep(cfg.gen.backoff_s * (attempt + 1))
        else:
            partial.unlink(missing_ok=True)
            raise PipelineError(f"run {run} failed {cfg.gen.max_retries_per_run + 1} times; see {log_dir}")
        for leftover in data.glob(f"run_{run:03d}.partial.run*.metrics.json"):
            leftover.unlink()
        measured.append(time.time() - t0)
        done += 1
        timeline = _timeline(cfg, sum(measured) / len(measured))
        logger.info("measured time per run and projected timeline: %s", timeline)
        ctx.progress(done, len(paths), timeline)
    hash_out = _data_hash(cfg)
    return {"output_hash": hash_out, "data_hash": hash_out}


# --- S2, S3 ----------------------------------------------------------------------------


def s2_validate(ctx: Context) -> dict[str, Any]:
    from neural_pbf.uq.validate import validate_dataset

    cfg = ctx.cfg
    report = validate_dataset(
        [str(p) for p in _run_paths(cfg)], cfg.gen.samples_per_run, cfg.gen.n_runs, cfg.gen.min_per_pattern,
        progress=ctx.progress,
    )  # fmt: skip
    logger.info("dataset summary: %s", report)
    return {"output_hash": _data_hash(cfg), "data_hash": _data_hash(cfg), "provenance": {"dataset": report}}


def s3_split(ctx: Context) -> dict[str, Any]:
    from neural_pbf.data.fm_dataset import _DEFAULT_CONDITIONING_KEYS
    from neural_pbf.data.splits import make_manifest, save_manifest
    from neural_pbf.uq.guard import check_no_shared_conditioning, freeze_prereg

    cfg = ctx.cfg
    manifest = make_manifest([str(p) for p in _run_paths(cfg)], cfg.split.seed, cfg.split.fractions, cfg.split.min_runs)
    check_no_shared_conditioning(manifest, _DEFAULT_CONDITIONING_KEYS)
    save_manifest(manifest, ctx.manifest_path)
    prereg = freeze_prereg(cfg, ctx.prereg_path)
    logger.info("manifest %s, runs %s", manifest["hash"], {k: len(v) for k, v in manifest["sets"].items()})
    return {
        "output_hash": manifest["hash"],
        "provenance": {"manifest_hash": manifest["hash"], "prereg_hash": prereg},
    }


# --- S4 --------------------------------------------------------------------------------


def s4_train(ctx: Context) -> dict[str, Any]:
    from neural_pbf.uq.train import train_baseline

    result = train_baseline(ctx.cfg, ctx.manifest_path, ctx.ckpt_dir, ctx.progress, git_commit())
    atomic_write_text(ctx.root / "train_result.json", json.dumps(result, indent=1))
    return {
        "output_hash": _sha256(ctx.ckpt_dir / "best.pt"),
        "provenance": {k: result[k] for k in ("best_epoch", "best_val", "epochs_run", "stopped_early")},
    }


# --- S5 --------------------------------------------------------------------------------


def _jobs(ctx: Context) -> list[dict[str, Any]]:
    """Every sampling job, calibration before test."""
    import torch

    from neural_pbf.data.splits import load_manifest
    from neural_pbf.uq.harness import ModelSpec, legacy_own_split

    cfg = ctx.cfg
    p = cfg.prereg
    manifest = load_manifest(ctx.manifest_path)
    new = ModelSpec("baseline_pre", str(ctx.ckpt_dir / "best.pt"), "net")
    legacy = ModelSpec("legacy_baseline", cfg.run.legacy_checkpoint, "net")
    if not Path(legacy.checkpoint).exists():
        raise PipelineError(f"legacy checkpoint missing: {legacy.checkpoint}")
    legacy_paths = torch.load(legacy.checkpoint, map_location="cpu", weights_only=True)["dataset_cfg"]["h5_paths"]
    own_cal, own_test = legacy_own_split(list(legacy_paths))
    every = max(1, round(1 / p.second_seed_fraction))
    limit = cfg.run.legacy_sample_limit
    return [
        {"spec": new, "name": "cal", "runs": manifest["sets"]["cal"], "seed": p.sampling_seed},
        {"spec": new, "name": "test", "runs": manifest["sets"]["test"], "seed": p.sampling_seed},
        {"spec": new, "name": "test_seed2", "runs": manifest["sets"]["test"], "seed": p.second_sampling_seed,
         "every": every},
        {"spec": legacy, "name": "own_cal", "runs": own_cal, "seed": p.sampling_seed, "limit": limit},
        {"spec": legacy, "name": "own_test", "runs": own_test, "seed": p.sampling_seed, "limit": limit},
        {"spec": legacy, "name": "new_cal", "runs": manifest["sets"]["cal"], "seed": p.sampling_seed, "limit": limit},
        {"spec": legacy, "name": "new_test", "runs": manifest["sets"]["test"], "seed": p.sampling_seed, "limit": limit},
    ]  # fmt: skip


def _hash_dir(directory: Path) -> str:
    body = [[str(p.relative_to(directory)), _sha256(p)] for p in sorted(directory.rglob("chunk_*.npz"))]
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()


def bind_draws_to_checkpoint(draws_dir: Path, ckpt_hash: str, manifest_hash: str) -> None:
    """Refuse to add draws to a directory that holds draws from another checkpoint."""
    binding = draws_dir / "binding.json"
    current = {"checkpoint": ckpt_hash, "manifest": manifest_hash}
    if binding.exists() and json.loads(binding.read_text()) != current:
        raise PipelineError(f"{draws_dir} holds draws from another checkpoint or manifest; use --force-stage S5")
    atomic_write_text(binding, json.dumps(current))


def s5_sampling(ctx: Context) -> dict[str, Any]:
    import torch

    from neural_pbf.data.splits import load_manifest
    from neural_pbf.uq.guard import TestAccessGuard, verify_prereg
    from neural_pbf.uq.harness import sample_set

    cfg = ctx.cfg
    prereg = verify_prereg(cfg, ctx.prereg_path)
    disk_guard(ctx.root, cfg.run.min_free_gb_sampling, "sampling")
    manifest = load_manifest(ctx.manifest_path)
    ckpt = torch.load(ctx.ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
    if ckpt.get("manifest_hash") != manifest["hash"]:
        raise PipelineError(f"checkpoint manifest {ckpt.get('manifest_hash')} differs from {manifest['hash']}")
    bind_draws_to_checkpoint(ctx.draws_dir, _sha256(ctx.ckpt_dir / "best.pt"), manifest["hash"])
    guard = TestAccessGuard(ctx.root / "test_access.jsonl", prereg)
    jobs = _jobs(ctx)
    n = len(jobs)
    for i, job in enumerate(jobs):
        if "test" in job["name"]:
            guard.log_access("sampling", model=job["spec"].name, set=job["name"])
        logger.info("job %d/%d: %s on %s", i + 1, n, job["spec"].name, job["name"])
        sample_set(
            job["spec"], job["runs"], ctx.draws_dir / job["spec"].name / job["name"], cfg, job["seed"],
            progress=lambda d, t, i=i: ctx.progress(i * 1000 + round(1000 * d / max(t, 1)), n * 1000),
            every=job.get("every", 1), limit=job.get("limit"),
        )  # fmt: skip
    hash_out = _hash_dir(ctx.draws_dir)
    return {"output_hash": hash_out, "data_hash": hash_out, "provenance": {"prereg_hash": prereg}}


# --- S6 --------------------------------------------------------------------------------


def s6_evaluate(ctx: Context) -> dict[str, Any]:
    import mlflow
    import torch

    from neural_pbf.data.splits import load_manifest
    from neural_pbf.uq.guard import TestAccessGuard, leakage_label, verify_prereg
    from neural_pbf.uq.harness import legacy_own_split, load_draws
    from neural_pbf.uq.report import build_report

    cfg = ctx.cfg
    prereg = verify_prereg(cfg, ctx.prereg_path)
    manifest = load_manifest(ctx.manifest_path)
    ckpt = torch.load(ctx.ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
    legacy = torch.load(cfg.run.legacy_checkpoint, map_location="cpu", weights_only=True)
    _, own_test = legacy_own_split(list(legacy["dataset_cfg"]["h5_paths"]))
    own_manifest = {"hash": None, "sets": {"test": own_test}}

    new_dir, legacy_dir = ctx.draws_dir / "baseline_pre", ctx.draws_dir / "legacy_baseline"
    guard = TestAccessGuard(ctx.root / "test_access.jsonl", prereg)
    results = [
        {"name": "Baseline (pre-step mask)", "label": leakage_label(ckpt, manifest, "test"), "role": "primary",
         "cal": load_draws(new_dir / "cal"), "test": load_draws(new_dir / "test"),
         "second": load_draws(new_dir / "test_seed2")},
        {"name": "Legacy Baseline on its own random-split held-out samples",
         "label": leakage_label(legacy, own_manifest, "test"), "role": "leakage_demo",
         "cal": load_draws(legacy_dir / "own_cal"), "test": load_draws(legacy_dir / "own_test"), "second": None},
        {"name": "Legacy Baseline on the new test runs", "label": leakage_label(legacy, manifest, "test"),
         "role": "leakage_demo", "cal": load_draws(legacy_dir / "new_cal"), "test": load_draws(legacy_dir / "new_test"),
         "second": None},
    ]  # fmt: skip
    guard.begin_evaluation(ctx.allow_reeval)

    train_result = json.loads((ctx.root / "train_result.json").read_text())
    meta = {
        "split_type": "run-level (grouped by scan family, stratified by material)",
        "runs_per_set": {k: len(v) for k, v in manifest["sets"].items()},
        "manifest_hash": manifest["hash"],
        "prereg_hash": prereg,
        "access_events": guard.events(),
        "training": {k: train_result[k] for k in ("best_epoch", "epochs_run", "stopped_early")},
    }
    rows = build_report(results, meta, cfg, cfg.report_path, cfg.figure_dir)
    rows.to_csv(ctx.root / "coverage_rows.csv", index=False)

    mlflow.set_tracking_uri(cfg.run.mlflow_uri)
    mlflow.set_experiment(cfg.run.mlflow_experiment)
    with mlflow.start_run(run_name=f"uq_eval_{cfg.mode}"):
        mlflow.log_params({"manifest_hash": manifest["hash"], "prereg_hash": prereg, "K": cfg.prereg.K})
        mlflow.set_tag("leakage_label", results[0]["label"])
        primary = rows[(rows.model == results[0]["name"]) & (rows.stratum_type == "all")]
        row: Any
        for row in primary.itertuples():
            prefix = f"{row.quantity}.{row.method}.a{row.alpha}"
            mlflow.log_metric(f"{prefix}.coverage", float(row.coverage))
            width = float(row.mean_width)
            if width != float("inf"):
                mlflow.log_metric(f"{prefix}.mean_width", width)
        mlflow.log_artifact(cfg.report_path)
        mlflow.log_artifact(str(ctx.root / "coverage_rows.csv"))
    return {"output_hash": _sha256(ctx.root / "coverage_rows.csv")}


# --- S7 --------------------------------------------------------------------------------


def s7_summary(ctx: Context) -> dict[str, Any]:
    status = json.loads((ctx.root / "status.json").read_text())
    lines = ["\n## Timeline actually used\n", "| stage | seconds (last execution) |", "| --- | --- |"]
    total = 0.0
    for stage in STAGE_IDS[:-1]:
        marker = ctx.root / "markers" / f"{stage}.done.json"
        duration = json.loads(marker.read_text())["provenance"]["duration_s"] if marker.exists() else 0.0
        total += duration
        lines.append(f"| {stage} {STAGE_DEFS[stage]['title']} | {duration:.0f} |")
    lines.append(f"| total | {total:.0f} ({total / 3600:.1f} h) |")
    section = "\n".join(lines) + "\n"
    report = Path(ctx.cfg.report_path)
    body = report.read_text().split("\n## Timeline actually used\n")[0].rstrip() + "\n"
    atomic_write_text(report, body + section)
    atomic_write_text(ctx.root / "summary.md", section + f"\nFinal status: {json.dumps(status.get('stages'))}\n")
    print(section, flush=True)
    return {"output_hash": hashlib.sha256(section.encode()).hexdigest()}


STAGE_DEFS: dict[str, dict[str, Any]] = {
    "S0": {"title": "preflight", "fn": s0_preflight, "config": ("run",), "upstream": (), "code": ["uv.lock"]},
    "S1": {"title": "generate runs", "fn": s1_generate, "config": ("gen",), "upstream": ("S0",), "code": _PHYSICS_CODE},
    "S2": {
        "title": "validate dataset", "fn": s2_validate, "config": ("gen",), "upstream": ("S1",),
        "code": ["src/neural_pbf/uq/validate.py"],
    },
    "S3": {
        "title": "split manifest and pre-registration", "fn": s3_split, "config": ("split", "prereg"),
        "upstream": ("S2",), "code": ["src/neural_pbf/data/splits.py", "src/neural_pbf/uq/guard.py"],
    },
    "S4": {
        "title": "train Baseline", "fn": s4_train, "config": ("train",), "upstream": ("S3",), "code": _MODEL_CODE,
        "wipe": lambda c: [Path(c.root) / "ckpt"],
    },
    "S5": {
        "title": "K-draw sampling", "fn": s5_sampling, "config": ("prereg",), "upstream": ("S3", "S4"),
        "code": _SAMPLING_CODE, "wipe": lambda c: [Path(c.root) / "draws"],
    },
    "S6": {
        "title": "conformal, evaluation, report", "fn": s6_evaluate, "config": ("prereg",), "upstream": ("S3", "S5"),
        "code": _EVAL_CODE,
    },
    "S7": {"title": "final summary", "fn": s7_summary, "config": (), "upstream": ("S6",), "code": []},
}  # fmt: skip


def run_stage(stage: str, cfg: UQConfig, allow_reeval: bool = False) -> int:
    """Child-process entry. Exit 2 = refused, 3 = GPU lost; other exceptions propagate as a traceback."""
    from neural_pbf.data.splits import SplitError
    from neural_pbf.uq.train import TrainingError
    from neural_pbf.uq.validate import ValidationError

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)
    ctx = Context(stage, cfg, allow_reeval)
    try:
        fn: Callable[[Context], dict[str, Any]] = STAGE_DEFS[stage]["fn"]
        output = fn(ctx)
    except GpuLost as exc:
        logger.error("%s", exc)
        return 3
    except (PipelineError, GuardError, ValidationError, SplitError, TrainingError, FloatingPointError) as exc:
        logger.error("REFUSED: %s", exc)
        return 2
    atomic_write_text(ctx.root / "outputs" / f"{stage}.json", json.dumps(output))
    return 0
