import abc
import logging
from pathlib import Path
from typing import Any

from ..schemas.run_meta import RunMeta

logger = logging.getLogger(__name__)


class ArtifactBuilder(abc.ABC):
    """Abstract base class for artifact builders."""

    def __init__(self):
        self.out_dir: Path | None = None
        self.meta: RunMeta | None = None
        self.dirs: dict[str, Path] = {}

    def on_run_start(self, meta: RunMeta, out_dir: Path) -> None:
        """Called when run starts. Sets up directory structure."""
        self.meta = meta
        self.out_dir = out_dir

        # Canonical structure
        self.dirs = {
            "config": out_dir / "config",
            "diagnostics": out_dir / "diagnostics",
            "plots_png": out_dir / "plots" / "png",
            "plots_interactive": out_dir / "plots" / "interactive",
            "report": out_dir / "report",
            "video": out_dir / "video",
            "states": out_dir / "states",
            "checkpoints": out_dir / "checkpoints",
        }

        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def on_snapshot(
        self, step_idx: int, state: Any, meta: dict[str, Any]
    ) -> list[Path]:
        """Called periodically to generate artifacts from state."""
        pass

    @abc.abstractmethod
    def on_run_end(self, final_state: Any, meta: dict[str, Any]) -> list[Path]:
        """Called at end of run for final artifacts/reports."""
        pass
