import logging
from pathlib import Path
from typing import Any

from ..diagnostics.recorder import DiagnosticsRecorder
from ..schemas.artifacts import ArtifactConfig
from ..schemas.diagnostics import DiagnosticsConfig
from ..schemas.run_meta import RunMeta
from ..schemas.tracking import TrackingConfig
from ..viz.artifacts_base import ArtifactBuilder
from .factory import build_tracker

logger = logging.getLogger(__name__)


class RunContext:
    """Integration context for simulation tracking, diagnostics, and artifacts."""

    def __init__(
        self,
        tracking_cfg: TrackingConfig,
        artifact_cfg: ArtifactConfig,
        diagnostics_cfg: DiagnosticsConfig,
        run_meta: RunMeta,
        out_dir: Path,
        artifact_builder: ArtifactBuilder | None = None,
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
        self._prev_state_T = None  # Cache for diffs

        # Ensure out dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start the tracking run."""
        self._tracker_ctx = self.tracker.start_run(
            run_name=self.tracking_cfg.run_name,
            config=self.tracking_cfg.model_dump(),
            tags=self.tracking_cfg.tags,
        )
        self._tracker_ctx.__enter__()

        # Initialize artifact builder
        if self.artifact_builder:
            self.artifact_builder.on_run_start(self.run_meta, self.out_dir)

        logger.info(f"RunContext started. Artifacts: {self.out_dir}")

    def on_step_start(self, step: int, state: Any):
        """Hook at start of step."""
        meta = {"dt": self.run_meta.dt}
        self.diagnostics.on_step_start(state, meta)

    def log_step(
        self, step: int, state: Any, meta: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Process step completion: Diagnostics -> Metrics -> Logging."""
        if meta is None:
            meta = {}

        # Diagnostics
        # Need to handle extracting T for prev_state comparison
        # Assuming state is tensor or dict.
        if isinstance(state, dict):
            T = state.get("T", state.get("temperature"))
        else:
            T = state

        metrics = self.diagnostics.on_step_end(step, state, self._prev_state_T, meta)

        # Logging cadence
        if step % self.tracking_cfg.log_every_n_steps == 0:
            self.tracker.log_metrics(metrics, step=step)

        # Update prev state (clone to avoid reference issues)
        if T is not None:
            if hasattr(T, "detach"):
                self._prev_state_T = T.detach().clone()
            elif hasattr(T, "copy"):
                self._prev_state_T = T.copy()
            else:
                self._prev_state_T = T

        return metrics

    def maybe_snapshot(self, step: int, state: Any, meta: dict[str, Any] | None = None):
        """Check artifact cadence and produce snapshots."""
        if meta is None:
            meta = {}
        if not self.artifact_builder:
            return

        # Let builder handle internal cadence logic or check here?
        # The prompt says builder gets `on_snapshot(step, ...)`.
        # But `log_artifact` is on tracker.
        # RunContext coordinates.

        # We delegate to builder. Builder logic should check cadence or we check here?
        # The prompt schemas has `artifact_every_n_steps` in TrackingConfig AND
        # ArtifactConfig.
        # Let's assume ArtifactBuilder handles its own internal checks for types.
        # However, for simplicity and robustness, passing every step (or
        # reasonably frequent)
        # to builder allow specific logic there.
        # But to save perf, we might pre-check if we are in general artifact cadence.

        # Actually, ArtifactConfig has specific `png_every_n_steps` etc.
        # So we should call builder and let it decide, or we only call if ANY match.
        # Let's call builder - it's designed to be smart.

        # Wait, if `artifact_builder.on_snapshot` returns paths, we should log
        # them with tracker.

        # Optimization: Only call if step matches GCD or simple modulo of min(cadences)
        # to builder allow specific logic there.
        # But to save perf, we might pre-check if we are in general artifact cadence.
        # For safety, let's just calling it.

        paths = self.artifact_builder.on_snapshot(step, state, meta)

        # Log generated artifacts to tracker (e.g. MLflow)
        if paths:
            for p in paths:
                # We can batch or specific path?
                # MLflow log_artifact takes local path.
                rel_path = p.relative_to(self.out_dir)
                # We might want to keep folder structure in MLflow artifact store
                # parent of p is usually subdirectory.
                # artifact_path arg in log_artifact allows specifying dest dir.
                self.tracker.log_artifact(str(p), artifact_path=str(rel_path.parent))

    def end(
        self, final_state: Any, meta: dict[str, Any] | None = None, status="FINISHED"
    ):
        if meta is None:
            meta = {}
        """End run, generate final reports."""
        if self.artifact_builder:
            # paths = self.artifact_builder.on_run_end(final_state, meta)
            # for p in paths:
            #     rel_path = p.relative_to(self.out_dir)
            #     self.tracker.log_artifact(str(p), artifact_path=str(rel_path.parent))
            paths = self.artifact_builder.on_run_end(final_state, meta)
            for p in paths:
                rel_path = p.relative_to(self.out_dir)
                self.tracker.log_artifact(str(p), artifact_path=str(rel_path.parent))

        if self._tracker_ctx:
            self._tracker_ctx.__exit__(None, None, None)

        logger.info(f"RunContext ended with status {status}")

    @property
    def dirs(self) -> dict[str, Path]:
        """Access artifact directories if available."""
        if self.artifact_builder:
            return self.artifact_builder.dirs
        return {}
