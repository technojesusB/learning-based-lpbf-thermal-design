from typing import Protocol, Dict, Optional, Any, Literal
from contextlib import contextmanager

class ExperimentTracker(Protocol):
    """Protocol for experiment trackers."""

    def start_run(self, run_name: Optional[str], config: Dict[str, Any], tags: Dict[str, str]) -> Any:
        """Start a new run. Returns a context manager or self."""
        ...

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters."""
        ...

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        ...

    def log_text(self, text: str, artifact_path: str) -> None:
        """Log text as an artifact."""
        ...

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact."""
        ...

    def flush(self) -> None:
        """Flush logging queue."""
        ...

    def end_run(self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED") -> None:
        """End the current run."""
        ...

class NullTracker:
    """No-op tracker implementation."""

    @contextmanager
    def start_run(self, run_name: Optional[str], config: Dict[str, Any], tags: Dict[str, str]):
        yield self

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_text(self, text: str, artifact_path: str) -> None:
        pass

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass

    def flush(self) -> None:
        pass

    def end_run(self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED") -> None:
        pass
