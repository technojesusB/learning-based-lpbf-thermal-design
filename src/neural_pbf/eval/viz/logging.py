"""Unified figure-logging helper for experiment scripts."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt


def log_figure(
    tracker: Any,
    fig: Any,
    path: str,
    *,
    dpi: int = 150,
    artifact_path: str = "eval",
) -> None:
    """Save *fig* to *path*, log to *tracker*, then close fig and delete temp."""
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        tracker.log_artifact(path, artifact_path=artifact_path)
    finally:
        plt.close(fig)
        if os.path.exists(path):
            os.remove(path)
