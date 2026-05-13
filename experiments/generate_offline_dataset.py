from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

# Must be set before h5py is imported so the library skips POSIX file locks.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.data.diagnostics import (
    DiagnosticsCsvWriter,
    StepProfiler,
    format_diag_line,
)
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.pipelines.dataset import (
    TrajectoryPlan,
    prepare_trajectory,
    save_path_preview,
)
from neural_pbf.scan.sources import GaussianBeam
from neural_pbf.tracking import RunContext
from neural_pbf.utils.units import LengthUnit

logger = logging.getLogger(__name__)

LF_FACTOR = 8
Lx_mm, Ly_mm, Lz_mm = 1.0, 0.5, 0.125

MATERIAL_ZOO: dict[str, MaterialConfig] = {}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_material_zoo() -> dict[str, MaterialConfig]:
    if not MATERIAL_ZOO:
        MATERIAL_ZOO.update({
            "SS316L": MaterialConfig.ss316l_preset(),
            "Ti64": MaterialConfig.ti64_preset(),
            "IN718": MaterialConfig.in718_preset(),
        })
    return MATERIAL_ZOO


def _open_hdf5(path: Path, mode: str, retries: int = 3, delay: float = 0.5) -> h5py.File:
    for attempt in range(retries):
        try:
            return h5py.File(path, mode)
        except OSError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")


def _is_wsl2(_path: Path = Path("/proc/version")) -> bool:
    try:
        return "microsoft" in _path.read_text().lower()
    except OSError:
        return False


def _resolve_wsl_safe(args: argparse.Namespace) -> bool:
    if args.no_wsl_safe:
        return False
    if args.wsl_safe:
        return True
    return _is_wsl2()


def _chunked_step(
    stepper: TimeStepper,
    state: SimulationState,
    exposure_time: float,
    Q_vol: torch.Tensor,
    max_chunk: float = 5e-6,
    use_triton: bool = True,
) -> tuple[SimulationState, int]:
    """Step through exposure_time in ≤max_chunk slices to avoid Windows GPU TDR."""
    t_remaining = exposure_time
    total_n_sub = 0
    while t_remaining > 0.0:
        dt = min(t_remaining, max_chunk)
        state = stepper.step_adaptive(state, dt_target=dt, Q_ext=Q_vol, use_triton=use_triton)
        total_n_sub += state.last_n_sub or 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_remaining -= dt
    return state, total_n_sub


def _write_sample(
    samples_grp: h5py.Group,
    global_idx: int,
    arrays: dict[str, np.ndarray],
    attrs: dict,
    lut_arrays: dict[str, np.ndarray] | None,
) -> None:
    grp = samples_grp.create_group(f"sample_{global_idx:06d}")
    for name, data in arrays.items():
        grp.create_dataset(name, data=data, compression="gzip")
    grp.attrs.update(attrs)
    if lut_arrays is not None:
        for name, data in lut_arrays.items():
            grp.create_dataset(name, data=data)


def plot_path_preview(path_points, sample_indices, Lx, Ly, out_path):
    """Legacy pyplot-API path preview retained for backward compatibility.

    New code should use ``pipelines.dataset.save_path_preview`` instead.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(path_points[:, 0] * 1e3, path_points[:, 1] * 1e3, "r-", lw=0.5, alpha=0.3)
    plt.scatter(path_points[:, 0] * 1e3, path_points[:, 1] * 1e3, c="black", s=2)
    sample_list = np.array(sorted(sample_indices))
    plt.scatter(
        path_points[sample_list, 0] * 1e3,
        path_points[sample_list, 1] * 1e3,
        c="blue", s=8, zorder=5,
    )
    plt.xlim(0, Lx * 1e3)
    plt.ylim(0, Ly * 1e3)
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Scan Path Preview (red=path, black=all, blue=samples)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Offline LPBF Dataset")
    parser.add_argument("--out", type=str, default="data/offline_dataset.h5")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--materials", type=str, default="SS316L,IN718")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-run", type=int, default=50)
    parser.add_argument(
        "--wsl-safe", action="store_true",
        help="Chunk each exposure into ≤5µs slices to prevent Windows GPU TDR timeouts.",
    )
    parser.add_argument(
        "--no-wsl-safe", action="store_true", default=False,
        help="Disable WSL2 auto-detection and force-disable chunked stepping.",
    )
    parser.add_argument(
        "--post-failure-delay", type=int, default=30, metavar="SECS",
    )
    parser.add_argument(
        "--max-retries", type=int, default=2, metavar="N",
    )
    parser.add_argument(
        "--mlflow", action="store_true", default=False,
        help="Enable per-step MLflow metric logging (off by default).",
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, default="lpbf-dataset-gen-diagnostics",
    )
    parser.add_argument(
        "--no-diag-csv", action="store_true", default=False,
    )
    # Hidden internal flags
    parser.add_argument("--run-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--orchestration-id", type=str, default="manual", help=argparse.SUPPRESS)
    parser.add_argument("--no-triton", action="store_true", default=False, help=argparse.SUPPRESS)
    return parser


def _init_hdf5(out_path: Path, args: argparse.Namespace) -> None:
    sim_cfg = SimulationConfig(
        Lx=Lx_mm, Ly=Ly_mm, Lz=Lz_mm,
        Nx=args.nx, Ny=args.ny, Nz=args.nz,
        length_unit=LengthUnit.MILLIMETERS,
    )
    with h5py.File(out_path, "w") as f:
        f.attrs.update({
            "Nx": args.nx, "Ny": args.ny, "Nz": args.nz,
            "Lx_m": sim_cfg.Lx_m, "Ly_m": sim_cfg.Ly_m, "Lz_m": sim_cfg.Lz_m,
            "T_ambient": sim_cfg.T_ambient,
            "lf_factor": LF_FACTOR,
            "schema_version": 2,
            "generator_script": "generate_offline_dataset.py",
        })
        f.create_group("samples")


def _count_samples(out_path: Path) -> int:
    if not out_path.exists():
        return 0
    with _open_hdf5(out_path, "r") as f:
        return len(f.get("samples", {}).keys())  # type: ignore[arg-type]


def run_worker(
    args: argparse.Namespace,
    plan: TrajectoryPlan | None = None,
) -> None:
    """Simulate exactly one trajectory and append samples to HDF5.

    Accepts an optional pre-computed *plan* (from ``prepare_trajectory``).
    When *plan* is None the trajectory is generated deterministically from
    *args* and *args.run_index*, matching notebook pre-preview behaviour.
    """
    run_idx: int = args.run_index  # type: ignore[assignment]

    # If a plan was pre-computed (e.g. notebook pre-rendered the path), reuse
    # it.  Otherwise generate it now — identical results since seeds match.
    if plan is None:
        zoo = _get_material_zoo()
        plan = prepare_trajectory(args, run_idx, zoo)

    # Torch seed: set after prepare_trajectory so numpy/random state is stable.
    torch.manual_seed(plan.run_seed)

    DEVICE = torch.device(args.device)

    sim_cfg = SimulationConfig(
        Lx=Lx_mm, Ly=Ly_mm, Lz=Lz_mm,
        Nx=args.nx, Ny=args.ny, Nz=args.nz,
        length_unit=LengthUnit.MILLIMETERS,
    )
    Lx_m, Ly_m, Lz_m = sim_cfg.Lx_m, sim_cfg.Ly_m, sim_cfg.Lz_m

    xs = torch.linspace(0.0, Lx_m, args.nx, device=DEVICE).view(1, 1, 1, 1, -1)
    ys = torch.linspace(0.0, Ly_m, args.ny, device=DEVICE).view(1, 1, 1, -1, 1)
    zs = torch.linspace(0.0, Lz_m, args.nz, device=DEVICE).view(1, 1, -1, 1, 1)

    logger.info("--- Run %d | Mat: %s ---", run_idx + 1, plan.mat_key)

    out_path = Path(args.out)
    path_points = plan.path_points
    sample_indices = plan.sample_indices
    mat_cfg = plan.mat_cfg

    state = SimulationState.zeros(sim_cfg, device=DEVICE)
    stepper = TimeStepper(sim_cfg, mat_cfg)
    beam = GaussianBeam(plan.beam_cfg)

    use_triton = not getattr(args, "no_triton", False)
    mlflow_enabled = getattr(args, "mlflow", False)
    no_diag_csv = getattr(args, "no_diag_csv", False)
    orchestration_id = getattr(args, "orchestration_id", "manual")
    mlflow_experiment = getattr(args, "mlflow_experiment", "lpbf-dataset-gen-diagnostics")

    if args.viz:
        save_path_preview(
            plan, out_path.parent / f"path_run_{run_idx:03d}.png", Lx_m, Ly_m
        )

    global_sample_idx = _count_samples(out_path)
    profiler = StepProfiler(sync_cuda=torch.cuda.is_available())

    diag_csv: DiagnosticsCsvWriter | None = None
    if not no_diag_csv:
        diag_csv = DiagnosticsCsvWriter(
            out_path.with_suffix(f".run{run_idx:03d}.diag.csv")
        )

    run_start_t = time.time()
    n_written = 0

    tracking_ctx = None
    if mlflow_enabled:
        tracking_ctx = RunContext.with_full_tracking(
            mlflow_uri=f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}",
            experiment_name=mlflow_experiment,
            run_name=f"run_{run_idx:03d}_{plan.mat_key}",
            tags={
                "run_index": str(run_idx),
                "orchestration_id": orchestration_id,
                "mat_key": plan.mat_key,
                "nx": str(args.nx), "ny": str(args.ny), "nz": str(args.nz),
            },
            out_dir=out_path.parent / f"run_{run_idx:03d}_artifacts",
            n_steps=len(path_points),
            log_every_n_steps=1,
        )

    def _do_run(ctx: RunContext | None) -> None:
        nonlocal global_sample_idx, n_written, state

        if ctx is not None and ctx._flight_recorder is not None:
            ctx._flight_recorder.set_run_context(
                mat_params={
                    "mat_key": plan.mat_key,
                    "k_solid": mat_cfg.k_solid,
                    "k_liquid": mat_cfg.k_liquid,
                    "k_powder": mat_cfg.k_powder,
                    "cp_base": mat_cfg.cp_base,
                    "rho": mat_cfg.rho,
                    "T_solidus": mat_cfg.T_solidus,
                    "T_liquidus": mat_cfg.T_liquidus,
                },
                run_info={
                    "run_idx": run_idx,
                    "orchestration_id": orchestration_id,
                    "power": plan.power,
                    "sigma": plan.sigma,
                    "pattern": plan.pattern,
                    "exposure_time": plan.exposure_time,
                },
            )

        Q_vol: torch.Tensor | None = None
        with _open_hdf5(out_path, "a") as f:
            samples_grp = f["samples"]
            for step_idx, (x0, y0) in enumerate(path_points):
                is_sample = step_idx in sample_indices
                T_in = state.T.clone() if is_sample else None

                Q_vol = beam.intensity(xs, ys, zs, x0, y0, z0=Lz_m)

                with profiler.phase("sim"):
                    if args.wsl_safe:
                        new_state, total_substeps = _chunked_step(
                            stepper, state, plan.exposure_time, Q_vol,
                            use_triton=use_triton,
                        )
                    else:
                        new_state = stepper.step_adaptive(
                            state,
                            dt_target=plan.exposure_time,
                            Q_ext=Q_vol,
                            use_triton=use_triton,
                        )
                        total_substeps = new_state.last_n_sub or 0

                # Rebind via new name to keep the outer `state` available for
                # rollback if needed; reassign at end of step.
                state_next = new_state  # type: ignore[possibly-undefined]
                peak_temperature = float(state_next.T.amax().item())

                T_in_np = Q_np = T_target_np = T_lf_np = mask_np = None
                if is_sample:
                    with profiler.phase("transfer"):
                        assert T_in is not None
                        T_coarse = F.interpolate(
                            T_in, scale_factor=1.0 / LF_FACTOR,
                            mode="trilinear", align_corners=False,
                        )
                        T_lf = F.interpolate(
                            T_coarse, size=(args.nz, args.ny, args.nx),
                            mode="trilinear", align_corners=False,
                        )
                        T_in_np = T_in.half().cpu().numpy()
                        Q_np = Q_vol.half().cpu().numpy()
                        T_target_np = state_next.T.half().cpu().numpy()
                        T_lf_np = T_lf.half().cpu().numpy()
                        mask_np = state_next.material_mask.cpu().numpy()  # type: ignore[union-attr]

                if is_sample:
                    assert T_in_np is not None and Q_np is not None
                    assert T_target_np is not None and T_lf_np is not None
                    assert mask_np is not None
                    try:
                        with profiler.phase("io"):
                            _write_sample(
                                samples_grp,
                                global_sample_idx,
                                arrays={
                                    "T_in": T_in_np, "Q": Q_np,
                                    "T_target": T_target_np, "T_lf": T_lf_np,
                                    "mask": mask_np,
                                },
                                attrs={
                                    "mat_name": plan.mat_key,
                                    "k_s": mat_cfg.k_solid,
                                    "k_l": mat_cfg.k_liquid,
                                    "k_p": mat_cfg.k_powder,
                                    "cp": mat_cfg.cp_base,
                                    "rho": mat_cfg.rho,
                                    "L": mat_cfg.latent_heat_L,
                                    "T_s": mat_cfg.T_solidus,
                                    "T_l": mat_cfg.T_liquidus,
                                    "power": plan.power,
                                    "sigma": beam.config.sigma,
                                    "exposure_time": plan.exposure_time,
                                    "point_distance": plan.point_distance,
                                    "hatch_spacing": plan.hatch_spacing,
                                    "pattern": plan.pattern,
                                    "angle_deg": plan.path_kwargs["angle_deg"],
                                    "x": x0,
                                    "y": y0,
                                },
                                lut_arrays=(
                                    {
                                        "T_lut": np.array(mat_cfg.T_lut, dtype=np.float32),
                                        "k_lut": np.array(mat_cfg.k_lut, dtype=np.float32),
                                        "cp_lut": np.array(mat_cfg.cp_lut, dtype=np.float32),
                                    }
                                    if mat_cfg.use_lut and mat_cfg.T_lut is not None
                                    else None
                                ),
                            )
                            global_sample_idx += 1
                            n_written += 1
                    except (RuntimeError, OSError) as exc:
                        logger.error(
                            "Run %d: sample write failed at step %d — aborting: %s",
                            run_idx, step_idx, exc,
                        )
                        raise

                # Diagnostics
                timing = profiler.build(
                    step_idx=step_idx,
                    substeps=total_substeps,
                    peak_temperature=peak_temperature,
                    is_sample=is_sample,
                )
                diag_line = format_diag_line(timing)
                logger.info(diag_line)
                if diag_csv is not None:
                    diag_csv.write(timing)
                if ctx is not None:
                    ctx.record_step(step_idx, timing=timing)

                # Advance state after all diagnostics are recorded
                state = state_next  # type: ignore[assignment]

    try:
        if tracking_ctx is not None:
            with tracking_ctx as ctx:
                _do_run(ctx)
        else:
            _do_run(None)

        duration = time.time() - run_start_t
        metrics = {
            "run": run_idx, "mat": plan.mat_key, "duration": duration,
            "steps": len(path_points), "seed": plan.run_seed, "samples": n_written,
        }
        metrics_path = out_path.with_suffix(f".run{run_idx:03d}.metrics.json")
        with open(metrics_path, "w") as f_met:
            json.dump(metrics, f_met, indent=2)
        logger.info("Run %d done — %d samples in %.1fs", run_idx + 1, n_written, duration)

    finally:
        del state, stepper, beam
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _run_worker(args: argparse.Namespace) -> None:
    run_worker(args)


def _orchestrate(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not (args.append and out_path.exists()):
        _init_hdf5(out_path, args)

    orch_id = uuid.uuid4().hex[:8]

    script = str(Path(__file__).resolve())
    base_cmd = [
        sys.executable, script,
        "--out", args.out,
        "--materials", args.materials,
        "--nx", str(args.nx), "--ny", str(args.ny), "--nz", str(args.nz),
        "--device", args.device,
        "--seed", str(args.seed),
        "--samples-per-run", str(args.samples_per_run),
        "--orchestration-id", orch_id,
        "--append",
    ]
    if args.viz:
        base_cmd.append("--viz")
    if args.wsl_safe:
        base_cmd.append("--wsl-safe")
    if args.mlflow:
        base_cmd.extend(["--mlflow", "--mlflow-experiment", args.mlflow_experiment])
    if args.no_diag_csv:
        base_cmd.append("--no-diag-csv")

    failed: list[int] = []
    for run_idx in range(args.runs):
        cmd = base_cmd + ["--run-index", str(run_idx)]
        logger.info("=== Spawning run %d/%d ===", run_idx + 1, args.runs)

        for attempt in range(1 + args.max_retries):
            if attempt > 0:
                logger.info("=== Retry %d/%d for run %d ===", attempt, args.max_retries, run_idx + 1)
            result = subprocess.run(cmd)
            if result.returncode == 0:
                break
            logger.error(
                "Run %d/%d FAILED (exit %d, attempt %d/%d)",
                run_idx + 1, args.runs, result.returncode,
                attempt + 1, 1 + args.max_retries,
            )
            sleep_s = args.post_failure_delay * (attempt + 1)
            if sleep_s > 0:
                logger.info("GPU cooldown — sleeping %ds", sleep_s)
                time.sleep(sleep_s)
        else:
            failed.append(run_idx)

    all_metrics = []
    for run_idx in range(args.runs):
        p = out_path.with_suffix(f".run{run_idx:03d}.metrics.json")
        if p.exists():
            with open(p) as f:
                all_metrics.append(json.load(f))
            p.unlink()
    with open(out_path.with_suffix(".metrics.json"), "w") as f_met:
        json.dump(all_metrics, f_met, indent=2)

    total = _count_samples(out_path)
    logger.info("Done. Total snapshots: %d. Failed runs: %s", total, failed if failed else "none")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _build_parser().parse_args()

    if args.run_index is None:
        resolved = _resolve_wsl_safe(args)
        if resolved and not args.wsl_safe:
            logger.info(
                "WSL2 detected — enabling --wsl-safe automatically "
                "(pass --no-wsl-safe to opt out)"
            )
        args.wsl_safe = resolved
        _orchestrate(args)
    else:
        _run_worker(args)


if __name__ == "__main__":
    main()
