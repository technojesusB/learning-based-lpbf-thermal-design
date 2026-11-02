"""Chunked, resumable K-draw sampling for any registered model; scalars only reach disk."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import random_split

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.data.splits import discover_runs
from neural_pbf.eval.benchmark.dataset import build_test_dataset
from neural_pbf.eval.benchmark.model_loader import load_model_adaptive
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.sampling import sample_k, sample_seed, scalar_functionals, volume_limited


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: str
    model_type: str


def legacy_own_split(h5_paths: list[str], seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The legacy Baseline's own random val and test samples, as manifest-style run lists.

    Same arithmetic as build_patch_data's default. Used only for the leakage demonstration.
    """
    keys = FMThermalDataset(FMDatasetConfig(h5_paths=h5_paths))._keys
    n = len(keys)
    n_train, n_val = int(n * 0.7), int(n * 0.2)
    _, val, test = random_split(
        list(range(n)),  # type: ignore[arg-type]
        [n_train, n_val, n - n_train - n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    runs, _ = discover_runs(h5_paths)
    owner = {(r.h5_path, k): r for r in runs for k in r.sample_keys}

    def as_runs(indices: Any) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for i in indices:
            run = owner[keys[i]]
            entry = grouped.setdefault(
                run.run_id,
                {"run_id": run.run_id, "family_id": run.family_id, "material": run.material, "pattern": run.pattern,
                 "h5_path": run.h5_path, "sample_keys": []},
            )  # fmt: skip
            entry["sample_keys"].append(list(keys[i]))
        return list(grouped.values())

    return as_runs(val.indices), as_runs(test.indices)


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        np.savez(f, **arrays)  # type: ignore[arg-type]
    os.replace(tmp, path)


def load_draws(directory: str | Path) -> dict[str, np.ndarray]:
    """Concatenate all chunks; raises on a missing or duplicated sample."""
    chunks = sorted(Path(directory).glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no sampling chunks in {directory}")
    loaded = [np.load(c, allow_pickle=False) for c in chunks]
    merged = {k: np.concatenate([c[k] for c in loaded]) for k in loaded[0].files}
    if np.unique(merged["sample_index"]).size != merged["sample_index"].size:
        raise ValueError(f"duplicate samples in {directory}")
    return merged


_CHUNK_FIELDS = (
    "draws", "truth", "run_id", "material", "pattern", "nz", "sample_index", "h5_path", "sample_key",
    "volume_limited", "draw_batch",
)  # fmt: skip


def _chunk_arrays(
    items: list[tuple[int, str, str, dict[str, Any]]],
    ds: Any,
    index: dict[tuple[str, str], int],
    model: Any,
    encoder: Any,
    spec: ModelSpec,
    grid_attrs: Any,
    ds_cfg: FMDatasetConfig,
    cfg: UQConfig,
    base_seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[Any]] = {k: [] for k in _CHUNK_FIELDS}
    for sample_index, path, key, run in items:
        with h5py.File(path, "r") as f:
            T_l = float(f["samples"][key].attrs["T_l"])  # type: ignore[index, arg-type]
        sample = ds[index[(path, key)]]
        batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k, v in sample.items()}
        draws = sample_k(
            model, encoder, batch, spec.model_type, cfg.prereg.K, sample_seed(base_seed, sample_index), device,
            n_steps=cfg.prereg.n_euler_steps, grid_attrs=grid_attrs, draw_batch=cfg.run.draw_batch,
        )  # fmt: skip
        to_phys = lambda t: t * ds_cfg.T_ref + ds_cfg.T_ambient  # noqa: E731
        truth = to_phys(sample["T_target"].squeeze().float())
        rows["draws"].append([scalar_functionals(to_phys(d), T_l) for d in draws])
        rows["truth"].append(scalar_functionals(truth, T_l))
        rows["run_id"].append(run["run_id"])
        rows["material"].append(run["material"])
        rows["pattern"].append(run["pattern"])
        rows["nz"].append(truth.shape[0])
        rows["sample_index"].append(sample_index)
        rows["h5_path"].append(path)
        rows["sample_key"].append(key)
        rows["volume_limited"].append(volume_limited(truth, T_l))
        rows["draw_batch"].append(cfg.run.draw_batch)  # affects the noise stream, so it is recorded per input
    return {k: np.asarray(v, dtype=float if k in ("draws", "truth") else None) for k, v in rows.items()}


def sample_set(
    spec: ModelSpec,
    runs: list[dict[str, Any]],
    out_dir: str | Path,
    cfg: UQConfig,
    base_seed: int,
    progress: Callable[[int, int], None] | None = None,
    every: int = 1,
    limit: int | None = None,
) -> None:
    """Sample every input of *runs* into atomic chunks; finished chunks are skipped.

    ``every`` keeps each n-th input (sampling-seed sensitivity), ``limit`` caps the total.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [(p, k, r) for r in runs for p, k in r["sample_keys"]]
    if not items:
        raise ValueError(f"no samples to draw for {spec.name}")
    selected = [(i, p, k, r) for i, (p, k, r) in enumerate(items) if i % every == 0][:limit]
    chunks = [selected[i : i + cfg.run.chunk_size] for i in range(0, len(selected), cfg.run.chunk_size)]
    todo = [(i, c) for i, c in enumerate(chunks) if not (out / f"chunk_{i:05d}.npz").exists()]

    def report() -> None:
        if progress is not None:
            progress(len(list(out.glob("chunk_*.npz"))), len(chunks))

    report()
    if not todo:
        return
    device = torch.device(cfg.run.device)
    model, encoder, _, grid_attrs = load_model_adaptive(spec.name, spec.checkpoint, spec.model_type, device)
    ckpt = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    ds_cfg = FMDatasetConfig(**{**ckpt["dataset_cfg"], "h5_paths": sorted({p for p, _, _ in items})})
    base = FMThermalDataset(ds_cfg)
    index = {key: i for i, key in enumerate(base._keys)}
    ds = build_test_dataset(base, patch_size=cfg.train.patch_size, model_type=spec.model_type)
    try:
        for i, chunk in todo:
            arrays = _chunk_arrays(chunk, ds, index, model, encoder, spec, grid_attrs, ds_cfg, cfg, base_seed, device)
            _save_npz_atomic(out / f"chunk_{i:05d}.npz", arrays)
            report()
    finally:
        del model, encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
