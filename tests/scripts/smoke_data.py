"""Tiny manifest-backed dataset, a subprocess runner and the standalone smoke check of the training scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from mlflow.tracking import MlflowClient

from neural_pbf.data.splits import make_manifest, save_manifest
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.training.tracking import REQUIRED_EPOCH, REQUIRED_STEP, STANDALONE_EXPERIMENT

ROOT = Path(__file__).resolve().parents[2]
FRACS = {"train": 0.25, "val": 0.25, "cal": 0.25, "test": 0.25}


def make_smoke_data(tmp_path: Path, n_runs: int = 4, per_run: int = 2, depth: int = 8) -> Path:
    """One file, n_runs runs (told apart by power), heating under a source, SS316L tables; returns the manifest.

    ``depth`` is Nz; DiT v1/v2 need 64 (their position embedding is sized for 64³ patches).
    """
    mat = MaterialConfig.ss316l_preset()
    SHAPE = (1, 1, depth, 64, 64)
    rng = np.random.default_rng(0)
    path = tmp_path / "runs.h5"
    z, y, x = np.meshgrid(*(np.arange(n) for n in SHAPE[2:]), indexing="ij")
    hot = np.exp(-((x - 32) ** 2 + (y - 32) ** 2 + (z - depth + 1) ** 2) / 40.0)[None, None]
    with h5py.File(path, "w") as f:
        f.attrs.update(Lx_m=1.26e-4, Ly_m=1.26e-4, Lz_m=2e-6 * (depth - 1), Nx=64, Ny=64, Nz=depth)
        for i in range(n_runs * per_run):
            g = f.require_group("samples").create_group(f"sample_{i:06d}")
            T_in = 400.0 + 300.0 * hot + rng.random(SHAPE)
            g.create_dataset("T_in", data=T_in.astype(np.float16))
            g.create_dataset("T_target", data=(T_in + 1500.0 * hot).astype(np.float16))
            g.create_dataset("Q", data=(2e15 * hot).astype(np.float32))
            mask = np.zeros(SHAPE, dtype=np.uint8)
            mask[..., : depth // 2, :, :] = 1
            g.create_dataset("mask", data=mask)
            g.create_dataset("mask_in", data=mask)
            for name in ("T_lut", "k_lut", "cp_lut"):
                g.create_dataset(name, data=np.asarray(getattr(mat, name), dtype=np.float32))
            g.attrs.update(
                mat_name="SS316L", pattern="raster", power=100.0 + i // per_run, sigma=2e-5, rho=mat.rho,
                cp=mat.cp_base, k_s=mat.k_solid, k_l=mat.k_liquid, k_p=mat.k_powder, L=mat.latent_heat_L,
                T_s=mat.T_solidus, T_l=mat.T_liquidus, x=6.3e-5, y=6.3e-5, hatch_spacing=5e-5, angle_deg=1.0,
                exposure_time=5e-5, point_distance=4e-5,
            )  # fmt: skip
    manifest = tmp_path / "manifest.json"
    save_manifest(make_manifest([str(path)], seed=1, fractions=FRACS), manifest)
    return manifest


def run_script(
    script: str, argv: list[str], timeout: int = 600, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """``env``: extra variables on top of the current environment (e.g. ``CUDA_VISIBLE_DEVICES=""`` for CPU)."""
    return subprocess.run(
        [sys.executable, str(ROOT / "experiments" / script), *argv], cwd=ROOT, capture_output=True, text=True,
        timeout=timeout, env={**os.environ, **(env or {})},
    )  # fmt: skip


def only_run(mlruns: Path, experiment: str):
    client = MlflowClient(tracking_uri=f"file://{mlruns}")
    exp = client.get_experiment_by_name(experiment)
    assert exp is not None, f"no MLflow experiment {experiment}"
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1, f"{len(runs)} runs in {experiment}"
    return client, runs[0]


def smoke(tmp_path: Path, script: str, args: list[str], tiny: list[str], depth: int = 8) -> tuple:
    """Run ``script`` standalone and check what every run logs: lpbf_experiments, pinned=false, no ablation key."""
    manifest = make_smoke_data(tmp_path, depth=depth)
    paths = ["--mlflow_uri", f"file://{tmp_path / 'mlruns'}", "--checkpoint_dir", str(tmp_path / "ckpt")]
    out = run_script(script, ["--manifest", str(manifest), *args, *tiny, *paths])
    assert out.returncode == 0, out.stderr[-3000:]
    client, run = only_run(tmp_path / "mlruns", STANDALONE_EXPERIMENT)
    name = Path(script).stem
    assert run.info.run_name == name and run.data.tags["pinned"] == "false"
    assert "ablation_experiment" not in run.data.tags
    assert set(REQUIRED_STEP) | set(REQUIRED_EPOCH) | {"best_val_loss"} <= set(run.data.metrics)
    assert "num_workers" in run.data.params  # infrastructure: logged, never pinned
    assert "resolved_config.json" in {a.path for a in client.list_artifacts(run.info.run_id)}
    figures = {a.path for a in client.list_artifacts(run.info.run_id, "eval")}
    assert {"eval/loss_panel.png", "eval/gallery_evolution.png"} <= figures, figures
    progress = json.loads((tmp_path / "ckpt" / "progress.json").read_text())
    assert progress["name"] == name and progress["state"] == "finished" and progress["next_epoch"] == progress["epochs"]
    assert progress["global_step"] > 0 and progress["sec_per_epoch"] > 0 and progress["eta_s"] == 0
    for ckpt in ("best.pt", "latest.pt"):
        prov = torch.load(tmp_path / "ckpt" / ckpt, weights_only=False)["provenance"]
        assert prov["name"] == name and prov["pinned"] is False and "ablation_experiment" not in prov
        assert prov["train_runs"] and prov["val_runs"]
    return client, run
