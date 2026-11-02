"""Methodological guards: frozen pre-registration, test-access log, leakage labels."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py

from neural_pbf.data.splits import discover_runs
from neural_pbf.uq.config import UQConfig


class GuardError(RuntimeError):
    """A methodological rule would be violated."""


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def prereg_hash(cfg: UQConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.prereg.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def freeze_prereg(cfg: UQConfig, path: str | Path) -> str:
    """Write the pre-registered config once; an existing freeze must match exactly."""
    path = Path(path)
    if path.exists():
        return verify_prereg(cfg, path)
    atomic_write_text(path, json.dumps({**cfg.prereg.model_dump(mode="json"), "hash": prereg_hash(cfg)}, indent=1))
    return prereg_hash(cfg)


def verify_prereg(cfg: UQConfig, path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise GuardError(f"pre-registration has not been frozen: {path} is missing")
    frozen = json.loads(path.read_text())["hash"]
    if frozen != prereg_hash(cfg):
        raise GuardError(f"config differs from the pre-registered one in {path} ({frozen[:12]})")
    return frozen


class TestAccessGuard:
    """Append-only log of test-set access. Allows one evaluation unless re-evaluation is explicit."""

    __test__ = False  # not a pytest class

    def __init__(self, path: str | Path, prereg_hash: str) -> None:
        self.path = Path(path)
        self.prereg_hash = prereg_hash

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def n_evaluations(self) -> int:
        return sum(e["event"] == "evaluation" for e in self.events())

    def _append(self, event: str, **fields: Any) -> None:
        previous = {e["prereg_hash"] for e in self.events()}
        if previous - {self.prereg_hash}:
            raise GuardError(f"pre-registration changed after test access; see {self.path}")
        entry = {
            "event": event,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "prereg_hash": self.prereg_hash,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log_access(self, purpose: str, **fields: Any) -> None:
        self._append(purpose, **fields)

    def begin_evaluation(self, allow_reeval: bool) -> None:
        if self.n_evaluations() and not allow_reeval:
            raise GuardError("test set was already evaluated; a second evaluation needs --allow-reeval")
        self._append("evaluation", allow_reeval=allow_reeval)


def leakage_label(ckpt: dict[str, Any], manifest: dict[str, Any], set_name: str) -> str:
    """Label a result ``run-disjoint: yes/no/unknown`` from the checkpoint's recorded training data."""
    stored = ckpt.get("manifest_hash")
    if stored is not None:
        if stored == manifest["hash"]:
            return "run-disjoint: yes (checkpoint manifest matches)"
        return "run-disjoint: no (checkpoint trained under a different manifest)"
    paths = (ckpt.get("dataset_cfg") or {}).get("h5_paths")
    if not paths or not all(Path(p).exists() for p in paths):
        return "run-disjoint: unknown (checkpoint has no manifest or readable training data)"
    runs, dropped = discover_runs(list(paths))
    trained = {r.family_id for r in [*runs, *dropped]}
    shared = trained & {r["family_id"] for r in manifest["sets"][set_name]}
    if shared:
        return f"run-disjoint: no ({len(shared)} scan families shared with training data, from dataset_cfg)"
    return "run-disjoint: yes (no shared scan family, from dataset_cfg)"


def _manifest_samples(manifest: dict[str, Any]) -> Iterator[tuple[str, str, str]]:
    for name, runs in manifest["sets"].items():
        for run in runs:
            for path, key in run["sample_keys"]:
                yield name, path, key


def check_no_shared_conditioning(manifest: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Raise if any sample's conditioning vector appears in two sets."""
    seen: dict[tuple[Any, ...], str] = {}
    files: dict[str, h5py.File] = {}
    try:
        for name, path, key in _manifest_samples(manifest):
            f = files.setdefault(path, h5py.File(path, "r"))
            attrs = f["samples"][key].attrs  # type: ignore[index]
            vector = tuple(float(attrs[k]) for k in keys)  # type: ignore[arg-type]
            if seen.setdefault(vector, name) != name:
                raise GuardError(f"identical conditioning in sets {seen[vector]} and {name}: {path}:{key}")
    finally:
        for f in files.values():
            f.close()
