"""Deterministic run-level train/val/cal/test split, persisted as a hashed manifest.

Samples of one simulated run are strongly correlated, so a random sample-level split
leaks. Runs that share a scan (same process parameters, different material) are kept
together as one *family*. Cross-file duplicates are dropped and recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

SET_NAMES = ("train", "val", "cal", "test")
_PROCESS_KEYS = ("power", "sigma", "pattern", "hatch_spacing", "angle_deg", "exposure_time")


class SplitError(ValueError):
    """The requested split is impossible or a manifest violates its invariants."""


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    family_id: str
    material: str
    pattern: str
    h5_path: str
    sample_keys: tuple[str, ...]


def _digest(values: tuple[Any, ...]) -> str:
    return hashlib.sha256(json.dumps(values, default=str).encode()).hexdigest()[:16]


def _attr(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def discover_runs(h5_paths: list[str]) -> tuple[list[RunInfo], list[RunInfo]]:
    """Reconstruct runs from sample attributes; returns ``(runs, dropped_duplicates)``.

    A run is a maximal block of consecutive samples with the same material and process
    parameters. A block whose identity appeared before (same or earlier file) is a duplicate.
    """
    runs: dict[str, RunInfo] = {}
    dropped: list[RunInfo] = []
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            if "samples" not in f:
                raise SplitError(f"{path}: no 'samples' group")
            blocks: list[tuple[tuple[Any, ...], list[str]]] = []
            samples = f["samples"]
            for key in samples:  # type: ignore[union-attr]
                attrs = samples[key].attrs  # type: ignore[index]
                process = tuple(_attr(attrs[k]) for k in _PROCESS_KEYS)
                identity = (_attr(attrs["mat_name"]), *process)
                if blocks and blocks[-1][0] == identity:
                    blocks[-1][1].append(key)
                else:
                    blocks.append((identity, [key]))
        for identity, keys in blocks:
            info = RunInfo(
                run_id=_digest(identity),
                family_id=_digest(identity[1:]),
                material=str(identity[0]),
                pattern=str(identity[1 + _PROCESS_KEYS.index("pattern")]),
                h5_path=path,
                sample_keys=tuple(keys),
            )
            if info.run_id in runs:
                dropped.append(info)
            else:
                runs[info.run_id] = info
    return list(runs.values()), dropped


def _allocate(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of *n* items; train gets at least one when n > 0."""
    exact = {k: fractions[k] * n for k in SET_NAMES}
    counts = {k: int(exact[k] + 1e-9) for k in SET_NAMES}
    by_remainder = sorted(SET_NAMES, key=lambda k: (-(exact[k] - counts[k]), SET_NAMES.index(k)))
    for k in by_remainder[: n - sum(counts.values())]:
        counts[k] += 1
    if n and counts["train"] == 0:
        donor = max(SET_NAMES, key=lambda k: counts[k])
        counts[donor] -= 1
        counts["train"] = 1
    return counts


def _evenly_spaced_labels(counts: dict[str, int]) -> list[str]:
    """Interleave labels so that items sorted by pattern reach every set."""
    n = sum(counts.values())
    assigned = dict.fromkeys(counts, 0)
    labels: list[str] = []
    for i in range(1, n + 1):
        label = max(counts, key=lambda k: counts[k] * i / n - assigned[k])
        assigned[label] += 1
        labels.append(label)
    return labels


def _hash_manifest(manifest: dict[str, Any]) -> str:
    body = {k: v for k, v in manifest.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _check_disjoint(sets: dict[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for name, runs in sets.items():
        for run in runs:
            for field in ("run_id", "family_id"):
                token = f"{field}:{run[field]}"
                if seen.setdefault(token, name) != name:
                    raise SplitError(f"{token} appears in more than one set ({seen[token]}, {name})")


def make_manifest(
    h5_paths: list[str],
    seed: int,
    fractions: dict[str, float],
    min_runs: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the manifest; raises :class:`SplitError` instead of degrading silently."""
    if set(fractions) != set(SET_NAMES) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SplitError(f"fractions must cover {SET_NAMES} and sum to 1, got {fractions}")
    runs, dropped = discover_runs(h5_paths)

    families: dict[str, list[RunInfo]] = {}
    for run in runs:
        families.setdefault(run.family_id, []).append(run)

    # A mixed-material family is stratified under its alphabetically first material.
    by_material: dict[str, list[str]] = {}
    for family_id, members in sorted(families.items()):
        by_material.setdefault(min(m.material for m in members), []).append(family_id)

    rng = random.Random(seed)
    sets: dict[str, list[dict[str, Any]]] = {name: [] for name in SET_NAMES}
    for _material, family_ids in sorted(by_material.items()):
        rng.shuffle(family_ids)
        family_ids.sort(key=lambda fid: min(m.pattern for m in families[fid]))  # stable: random within a pattern
        for family_id, label in zip(
            family_ids, _evenly_spaced_labels(_allocate(len(family_ids), fractions)), strict=True
        ):
            sets[label].extend(
                {
                    "run_id": m.run_id,
                    "family_id": m.family_id,
                    "material": m.material,
                    "pattern": m.pattern,
                    "h5_path": m.h5_path,
                    "sample_keys": [[m.h5_path, k] for k in m.sample_keys],
                }
                for m in families[family_id]
            )

    train_materials = {r["material"] for r in sets["train"]}
    for name in SET_NAMES:
        required = (min_runs or {}).get(name, 1)
        if len(sets[name]) < required:
            raise SplitError(f"set '{name}' has {len(sets[name])} runs, needs at least {required}")
        missing = {r["material"] for r in sets[name]} - train_materials
        if missing:
            raise SplitError(f"set '{name}' contains materials absent from train: {sorted(missing)}")

    _check_disjoint(sets)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "fractions": fractions,
        "sets": sets,
        "dropped_duplicates": [{"run_id": d.run_id, "h5_path": d.h5_path} for d in dropped],
    }
    manifest["hash"] = _hash_manifest(manifest)
    return manifest


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write the manifest atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest, verifying its hash and that its sets are disjoint."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("hash") != _hash_manifest(manifest):
        raise SplitError(f"{path}: manifest hash mismatch")
    _check_disjoint(manifest["sets"])
    return manifest


def manifest_sample_keys(manifest: dict[str, Any], set_name: str) -> list[tuple[str, str]]:
    """All ``(h5_path, sample_key)`` pairs of one set, in manifest order."""
    return [(p, k) for run in manifest["sets"][set_name] for p, k in run["sample_keys"]]
