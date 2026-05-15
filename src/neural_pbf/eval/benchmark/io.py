"""I/O helpers: persist raw metrics and log artifacts to MLflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from neural_pbf.tracking.run_context import RunContext

logger = logging.getLogger(__name__)


def save_metrics_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Write DataFrame to CSV, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.debug("Saved metrics CSV → %s", path)


def save_npz(path: Path | str, **arrays: np.ndarray) -> None:
    """Save named arrays to a .npz file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrays)  # type: ignore[call-overload]
    logger.debug("Saved NPZ → %s", path)


def mlflow_log_paths(
    ctx: RunContext,
    paths: list[Path | str],
    artifact_subdir: str = "",
) -> None:
    """Log each existing file as an MLflow artifact via RunContext."""
    for p in paths:
        p = Path(p)
        if not p.exists():
            logger.warning("mlflow_log_paths: skipping missing file %s", p)
            continue
        ctx.log_artifact(p, artifact_subdir=artifact_subdir)
