from __future__ import annotations

import datetime
import logging
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from ..diagnostics.recorder import DiagnosticsRecorder
from ..schemas.artifacts import ArtifactConfig
from ..schemas.diagnostics import DiagnosticsConfig
from ..schemas.instrumentation import InstrumentationConfig
from ..schemas.run_meta import RunMeta
from ..schemas.tracking import TrackingConfig
from ..viz.artifacts_base import ArtifactBuilder
from .factory import build_tracker

logger = logging.getLogger(__name__)


class RunContext:
    """Integration context for simulation tracking, diagnostics, and artifacts.

    Can be used as a plain helper (call start/end manually) or as a context
    manager (``with ctx:``).  For the dataset-generation pipeline, prefer the
    ``with_full_tracking(...)`` classmethod which wires up system monitoring,
    the flight recorder, and the progress bar in one call.
    """

    def __init__(
        self,
        tracking_cfg: TrackingConfig,
        artifact_cfg: ArtifactConfig,
        diagnostics_cfg: DiagnosticsConfig,
        run_meta: RunMeta,
        out_dir: Path,
        artifact_builder: ArtifactBuilder | None = None,
        instrumentation_cfg: InstrumentationConfig | None = None,
        _n_steps: int | None = None,
    ):
        self.tracking_cfg = tracking_cfg
        self.artifact_cfg = artifact_cfg
        self.diagnostics_cfg = diagnostics_cfg
        self.run_meta = run_meta
        self.out_dir = out_dir

        self.tracker = build_tracker(tracking_cfg)
        self.diagnostics = DiagnosticsRecorder(diagnostics_cfg)
        self.artifact_builder = artifact_builder

        self._tracker_ctx = None
        self._prev_state_T = None

        # ── Instrumentation ──────────────────────────────────────────────────
        self._inst_cfg = instrumentation_cfg
        self._n_steps = _n_steps
        self._system_monitor: Any | None = None
        self._flight_recorder: Any | None = None
        self._progress_reporter: Any | None = None
        self._nvml_step_counter = 0

        if instrumentation_cfg is not None:
            self._setup_instrumentation(instrumentation_cfg, _n_steps)

        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Instrumentation setup ────────────────────────────────────────────────

    def _setup_instrumentation(
        self, cfg: InstrumentationConfig, n_steps: int | None
    ) -> None:
        from .instrumentation import FlightRecorder, ProgressReporter, SystemMonitor

        if cfg.system_metrics:
            self._system_monitor = SystemMonitor()
        if cfg.flight_recorder:
            self._flight_recorder = FlightRecorder(capacity=cfg.flight_recorder_history)
        if cfg.progress_reporter and n_steps is not None:
            self._progress_reporter = ProgressReporter(total=n_steps)

    # ── Context-manager protocol ─────────────────────────────────────────────

    def __enter__(self) -> RunContext:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        status = "FINISHED" if exc_type is None else "FAILED"
        if exc_type is not None and self._flight_recorder is not None:
            dump_path = self.out_dir / "flight_recorder.json"
            self._flight_recorder.dump(dump_path, exc=exc_val)
            with suppress(Exception):
                self.tracker.log_artifact(str(dump_path), artifact_path="diagnostics")
        if self._system_monitor is not None:
            self._system_monitor.stop()
        if self._progress_reporter is not None:
            self._progress_reporter.close()
        self.end(final_state=None, status=status)
        return False  # never suppress the exception

    # ── One-liner factory ────────────────────────────────────────────────────

    @classmethod
    @contextmanager
    def with_full_tracking(
        cls,
        *,
        mlflow_uri: str,
        experiment_name: str,
        run_name: str,
        tags: dict[str, str] | None = None,
        out_dir: Path,
        n_steps: int | None = None,
        log_every_n_steps: int = 1,
        instrumentation_cfg: InstrumentationConfig | None = None,
        run_meta: RunMeta | None = None,
    ) -> Generator[RunContext, None, None]:
        """Context manager that starts a fully instrumented tracking run.

        Usage::

            with RunContext.with_full_tracking(
                mlflow_uri="sqlite:///mlflow.db",
                experiment_name="lpbf-dataset-gen-diagnostics",
                run_name="run_000_SS316L",
                tags={"mat": "SS316L"},
                out_dir=Path("artifacts/run_000"),
                n_steps=740,
            ) as ctx:
                for step_idx, (x0, y0) in enumerate(path_points):
                    ...
                    ctx.record_step(step_idx, timing=timing)
        """
        inst_cfg = instrumentation_cfg or InstrumentationConfig(
            system_metrics=True,
            flight_recorder=True,
            progress_reporter=True,
        )
        tracking_cfg = TrackingConfig(
            enabled=True,
            backend="mlflow",
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags or {},
            log_every_n_steps=log_every_n_steps,
            mlflow_tracking_uri=mlflow_uri,
        )
        if run_meta is None:
            logger.warning(
                "with_full_tracking: no run_meta provided; using placeholder values. "
                "Pass run_meta=RunMeta(...) to record accurate grid/device metadata."
            )
            run_meta = RunMeta(
                seed=0,
                device="unknown",
                dtype="float32",
                started_at=datetime.datetime.now().isoformat(),
                dx=0.0,
                dy=0.0,
                dz=0.0,
                dt=0.0,
                grid_shape=[0, 0, 0],
            )
        ctx = cls(
            tracking_cfg=tracking_cfg,
            artifact_cfg=ArtifactConfig(enabled=False),
            diagnostics_cfg=DiagnosticsConfig(enabled=False),
            run_meta=run_meta,
            out_dir=out_dir,
            instrumentation_cfg=inst_cfg,
            _n_steps=n_steps,
        )
        with ctx:
            yield ctx

    # ── Per-step recording ───────────────────────────────────────────────────

    def record_step(
        self,
        step_idx: int,
        timing: Any | None = None,
        extra_metrics: dict[str, float] | None = None,
    ) -> dict[str, float | None]:
        """Fan out per-step data to all active instruments.

        *timing* may be a ``StepTiming`` from ``data.diagnostics`` or any
        object with ``t_sim``, ``t_transfer``, ``t_io``, ``substeps``,
        ``peak_temperature`` attributes. Pass ``extra_metrics`` for any
        additional float values to log to MLflow.

        Returns the merged system-metrics sample dict (empty when system
        monitoring is disabled) so callers can inspect GPU temperature etc.
        """
        sys_sample: dict[str, float | None] = {}
        inst_cfg = self._inst_cfg

        # ── System metrics ───────────────────────────────────────────────────
        if (
            self._system_monitor is not None
            and inst_cfg is not None
            and self._nvml_step_counter % inst_cfg.nvml_sample_interval_steps == 0
        ):
            sys_sample = self._system_monitor.sample()
        self._nvml_step_counter += 1

        # ── MLflow logging ───────────────────────────────────────────────────
        if step_idx % self.tracking_cfg.log_every_n_steps == 0:
            mlflow_metrics: dict[str, float] = {}
            if timing is not None:
                mlflow_metrics.update(
                    {
                        "t_sim": getattr(timing, "t_sim", 0.0),
                        "t_transfer": getattr(timing, "t_transfer", 0.0),
                        "t_io": getattr(timing, "t_io", 0.0),
                        "substeps": float(getattr(timing, "substeps", 0)),
                        "peak_temperature": getattr(timing, "peak_temperature", 0.0),
                    }
                )
            if extra_metrics:
                mlflow_metrics.update(extra_metrics)
            # System metrics: skip None values
            mlflow_metrics.update(
                {k: v for k, v in sys_sample.items() if v is not None}
            )
            if mlflow_metrics:
                self.tracker.log_metrics(mlflow_metrics, step=step_idx)

        # ── Flight recorder ──────────────────────────────────────────────────
        if self._flight_recorder is not None:
            snapshot: dict[str, Any] = {"step_idx": step_idx}
            if timing is not None:
                snapshot["timing"] = {
                    "t_sim": getattr(timing, "t_sim", None),
                    "t_transfer": getattr(timing, "t_transfer", None),
                    "t_io": getattr(timing, "t_io", None),
                    "substeps": getattr(timing, "substeps", None),
                    "peak_temperature": getattr(timing, "peak_temperature", None),
                }
            snapshot["system"] = {k: v for k, v in sys_sample.items() if v is not None}
            self._flight_recorder.record_step(snapshot)

        # ── Progress bar ─────────────────────────────────────────────────────
        if self._progress_reporter is not None:
            pr_metrics: dict[str, float | None] = {
                "Tmax": getattr(timing, "peak_temperature", None) if timing else None,
                "t_sim": getattr(timing, "t_sim", None) if timing else None,
                "gpu_temp": sys_sample.get("sys/gpu_temperature"),
            }
            self._progress_reporter.update(pr_metrics)

        return sys_sample

    def log_artifact(self, path: Path | str, artifact_subdir: str = "") -> None:
        """Log a local file to the active tracker run."""
        self.tracker.log_artifact(str(path), artifact_path=artifact_subdir or None)

    # ── Existing API (unchanged) ──────────────────────────────────────────────

    def start(self) -> None:
        self._tracker_ctx = self.tracker.start_run(
            run_name=self.tracking_cfg.run_name,
            config=self.tracking_cfg.model_dump(),
            tags=self.tracking_cfg.tags,
        )
        self._tracker_ctx.__enter__()
        if self.artifact_builder:
            self.artifact_builder.on_run_start(self.run_meta, self.out_dir)
        if self._system_monitor is not None:
            self._system_monitor.start()
        logger.info("RunContext started. Artifacts: %s", self.out_dir)

    def on_step_start(self, step: int, state: Any) -> None:
        meta = {"dt": self.run_meta.dt}
        self.diagnostics.on_step_start(state, meta)

    def log_step(
        self, step: int, state: Any, meta: dict[str, Any] | None = None
    ) -> dict[str, float]:
        if meta is None:
            meta = {}
        if isinstance(state, dict):
            T = state.get("T", state.get("temperature"))
        else:
            T = state
        metrics = self.diagnostics.on_step_end(step, state, self._prev_state_T, meta)
        if step % self.tracking_cfg.log_every_n_steps == 0:
            self.tracker.log_metrics(metrics, step=step)
        if T is not None:
            if hasattr(T, "detach"):
                self._prev_state_T = T.detach().clone()
            elif hasattr(T, "copy"):
                self._prev_state_T = T.copy()
            else:
                self._prev_state_T = T
        return metrics

    def maybe_snapshot(
        self, step: int, state: Any, meta: dict[str, Any] | None = None
    ) -> None:
        if meta is None:
            meta = {}
        if not self.artifact_builder:
            return
        paths = self.artifact_builder.on_snapshot(step, state, meta)
        if paths:
            for p in paths:
                rel_path = p.relative_to(self.out_dir)
                self.tracker.log_artifact(str(p), artifact_path=str(rel_path.parent))

    def end(
        self,
        final_state: Any,
        meta: dict[str, Any] | None = None,
        status: str = "FINISHED",
    ) -> None:
        if meta is None:
            meta = {}
        if self.artifact_builder:
            paths = self.artifact_builder.on_run_end(final_state, meta)
            for p in paths:
                rel_path = p.relative_to(self.out_dir)
                self.tracker.log_artifact(str(p), artifact_path=str(rel_path.parent))
        if self._tracker_ctx:
            self._tracker_ctx.__exit__(None, None, None)
            self._tracker_ctx = None
        logger.info("RunContext ended with status %s", status)

    @property
    def dirs(self) -> dict[str, Path]:
        if self.artifact_builder:
            return self.artifact_builder.dirs
        return {}
