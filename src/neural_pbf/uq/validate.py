"""Dataset validation: fails loudly on anything that would invalidate the evaluation."""

from __future__ import annotations

from collections import Counter
from typing import Any

import h5py
import numpy as np

from neural_pbf.data.fm_dataset import _DEFAULT_CONDITIONING_KEYS
from neural_pbf.data.splits import discover_runs

_REQUIRED = ("T_in", "T_target", "Q", "mask", "mask_in")
_REQUIRED_ATTRS = (
    "mat_name", "pattern", "hatch_spacing", "angle_deg", "exposure_time", *_DEFAULT_CONDITIONING_KEYS,
)  # fmt: skip


class ValidationError(ValueError):
    pass


def validate_run_file(path: str, n_samples: int, deep: bool) -> None:
    """Check one run file. ``deep`` also reads every array (NaN/inf and mask_in subset of mask)."""
    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        raise ValidationError(f"{path}: unreadable: {exc}") from exc
    with f:
        samples: Any = f.get("samples")
        if samples is None or len(samples) != n_samples:
            found = 0 if samples is None else len(samples)
            raise ValidationError(f"{path}: expected {n_samples} samples, found {found}")
        for key in samples:
            sample: Any = samples[key]
            missing = [d for d in _REQUIRED if d not in sample] + [a for a in _REQUIRED_ATTRS if a not in sample.attrs]
            if missing:
                raise ValidationError(f"{path}:{key}: missing {missing}")
            if not deep:
                continue
            for name in ("T_in", "T_target", "Q"):
                if not np.isfinite(sample[name][()]).all():
                    raise ValidationError(f"{path}:{key}: non-finite values in {name}")
            if np.any((sample["mask_in"][()] > 0) & (sample["mask"][()] == 0)):
                raise ValidationError(f"{path}:{key}: mask_in is not a subset of the post-step mask")


def validate_dataset(
    paths: list[str],
    n_samples: int,
    expected_runs: int,
    min_per_pattern: dict[str, int],
    progress: Any = None,
) -> dict[str, Any]:
    """Deep-validate all run files and the dataset-level invariants; returns a summary."""
    if len(paths) != expected_runs:
        raise ValidationError(f"expected {expected_runs} run files, found {len(paths)}")
    for i, path in enumerate(paths):
        validate_run_file(path, n_samples, deep=True)
        if progress is not None:
            progress(i + 1, len(paths))
    runs, dropped = discover_runs(paths)
    if dropped:
        raise ValidationError(f"duplicate parameter tuple in {[d.h5_path for d in dropped]}")
    if len(runs) != len(paths) or {r.h5_path for r in runs} != set(paths):
        raise ValidationError("run groups cannot be reconstructed: expected exactly one run per file")
    families = Counter(r.family_id for r in runs)
    if max(families.values()) > 1:
        raise ValidationError("duplicate parameter tuple: two runs share a scan")
    patterns = dict(Counter(r.pattern for r in runs))
    for pattern, minimum in min_per_pattern.items():
        if patterns.get(pattern, 0) < minimum:
            raise ValidationError(f"pattern {pattern}: {patterns.get(pattern, 0)} runs, need {minimum}")
    return {"n_runs": len(runs), "materials": dict(Counter(r.material for r in runs)), "patterns": patterns}
