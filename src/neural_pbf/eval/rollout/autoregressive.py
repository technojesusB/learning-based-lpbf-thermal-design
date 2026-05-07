"""Autoregressive rollout analytics: cumulative drift and divergence detection."""
from __future__ import annotations

from .engine import RolloutResult


def cumulative_mae(result: RolloutResult) -> list[float]:
    """Running average of global MAE across steps.

    Returns a list of length ``len(result.per_step_metrics)`` where entry *i*
    is the mean MAE over steps 0..i.
    """
    cumulative: list[float] = []
    total = 0.0
    for i, m in enumerate(result.per_step_metrics):
        total += m["mae_global"]
        cumulative.append(total / (i + 1))
    return cumulative


def has_diverged(result: RolloutResult) -> bool:
    """Return ``True`` if the rollout was aborted due to divergence."""
    return result.diverged_at_step is not None


def ar_summary(result: RolloutResult) -> dict[str, float]:
    """Aggregate AR rollout statistics into a flat dict.

    Keys: ``mean_mae``, ``final_mae``, ``peak_mae``, ``cumulative_drift``,
    ``diverged`` (0.0 or 1.0).
    Returns ``nan`` for MAE values when no steps were evaluated.
    """
    if not result.per_step_metrics:
        return {
            "mean_mae": float("nan"),
            "final_mae": float("nan"),
            "peak_mae": float("nan"),
            "cumulative_drift": float("nan"),
            "diverged": float(has_diverged(result)),
        }
    maes = [m["mae_global"] for m in result.per_step_metrics]
    return {
        "mean_mae": sum(maes) / len(maes),
        "final_mae": maes[-1],
        "peak_mae": max(maes),
        "cumulative_drift": sum(maes),
        "diverged": float(has_diverged(result)),
    }
