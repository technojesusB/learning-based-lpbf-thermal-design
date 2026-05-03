"""compare_methods.py — Compare physics solver, ThermalSurrogate3D, and FMStepper.

Usage::

    uv run python scripts/compare_methods.py \\
        --h5 data/offline_dataset.h5 \\
        --sample-key sample_000000 \\
        --fm-ckpt checkpoints/fm/best.pt \\
        --surrogate-ckpt checkpoints/surrogate.pt \\   # optional
        --output-dir artifacts/compare \\
        --device cpu

The ThermalSurrogate3D arm is skipped gracefully if --surrogate-ckpt is not
provided or the file does not exist.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred.float() - target.float()
    mae = diff.abs().mean().item()
    rmse = (diff ** 2).mean().sqrt().item()
    max_err = diff.abs().max().item()
    return {"MAE [K]": mae, "RMSE [K]": rmse, "MaxErr [K]": max_err}


def _print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    keys = list(rows[0][1].keys())
    header = f"{'Method':<30}" + "".join(f"{k:>15}" for k in keys)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, m in rows:
        row = f"{name:<30}" + "".join(f"{m[k]:>15.4f}" for k in keys)
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_sample(h5_path: str, sample_key: str):
    """Return (T_in, T_target, Q, mask, attrs) as float32 CPU tensors + dict."""
    with h5py.File(h5_path, "r") as f:
        grp = f["samples"][sample_key]
        T_in = torch.from_numpy(grp["T_in"][:].astype(np.float32))
        T_target = torch.from_numpy(grp["T_target"][:].astype(np.float32))
        Q = torch.from_numpy(grp["Q"][:].astype(np.float32))
        mask = torch.from_numpy(grp["mask"][:].astype(np.uint8))
        attrs = dict(grp.attrs)
        # Read grid shape from HDF5 root attrs if available
        root_attrs = dict(f.attrs)
    return T_in, T_target, Q, mask, attrs, root_attrs


def _reconstruct_configs(attrs: dict, root_attrs: dict, device: torch.device):
    """Build MaterialConfig and SimulationConfig from per-sample attributes."""
    from neural_pbf.core.config import SimulationConfig
    from neural_pbf.physics.material import MaterialConfig
    from neural_pbf.utils.units import LengthUnit

    mat_cfg = MaterialConfig(
        k_powder=float(attrs["k_p"]),
        k_solid=float(attrs["k_s"]),
        k_liquid=float(attrs["k_l"]),
        cp_base=float(attrs["cp"]),
        rho=float(attrs["rho"]),
        T_solidus=float(attrs["T_s"]),
        T_liquidus=float(attrs["T_l"]),
        latent_heat_L=float(attrs["L"]),
    )

    Nx = int(root_attrs.get("Nx", 64))
    Ny = int(root_attrs.get("Ny", 32))
    Nz = int(root_attrs.get("Nz", 8))
    Lx_m = float(root_attrs.get("Lx_m", 1e-3))
    Ly_m = float(root_attrs.get("Ly_m", 0.5e-3))
    Lz_m = float(root_attrs.get("Lz_m", 0.125e-3))

    # SimulationConfig expects user-unit lengths; pass metres directly
    sim_cfg = SimulationConfig(
        Lx=Lx_m * 1000, Ly=Ly_m * 1000, Lz=Lz_m * 1000,
        Nx=Nx, Ny=Ny, Nz=Nz,
        length_unit=LengthUnit.MILLIMETERS,
    )
    return sim_cfg, mat_cfg


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def _run_physics_solver(T_in, Q, mask, sim_cfg, mat_cfg, device):
    from neural_pbf.core.state import SimulationState
    from neural_pbf.integrator.stepper import TimeStepper

    state = SimulationState(
        T=T_in.to(device),
        material_mask=mask.to(device),
    )
    stepper = TimeStepper(sim_cfg, mat_cfg)
    dt_target = float(sim_cfg.dt_base)
    state = stepper.step_adaptive(state, dt_target=dt_target, Q_ext=Q.to(device))
    return state.T.cpu()


def _run_surrogate(T_in, Q, surrogate_ckpt: str, device):
    import os

    from neural_pbf.models.config import SurrogateConfig
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    if not os.path.isfile(surrogate_ckpt):
        logger.warning("Surrogate checkpoint not found: %s — skipping", surrogate_ckpt)
        return None

    ckpt = torch.load(surrogate_ckpt, map_location=device)
    sur_cfg_dict = ckpt.get("surrogate_cfg", {})
    sur_cfg = SurrogateConfig(**sur_cfg_dict) if sur_cfg_dict else SurrogateConfig()
    model = ThermalSurrogate3D(sur_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Squeeze stored (1,1,Nz,Ny,Nx) to correct channel layout
    t = T_in.to(device)
    q = Q.to(device)

    with torch.no_grad():
        pred_dT = model(t, q)

    return (t + pred_dT).cpu()


def _run_fm_stepper(T_in, Q, mask, conditioning_vec, fm_ckpt: str, sim_cfg, device):
    from neural_pbf.core.state import SimulationState
    from neural_pbf.integrator.fm_stepper import FMStepper
    from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
    from neural_pbf.models.generative.fm.config import FMConfig
    from neural_pbf.models.generative.fm.velocity_net import VelocityNet

    ckpt = torch.load(fm_ckpt, map_location=device)
    fm_cfg = FMConfig(**ckpt["fm_cfg"])
    model = VelocityNet(fm_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    cond_encoder = ConditioningEncoder(fm_cfg.cond_dim, fm_cfg.cond_embed_dim).to(device)
    cond_encoder.load_state_dict(ckpt["cond_encoder_state"])

    state = SimulationState(
        T=T_in.to(device),
        material_mask=mask.to(device),
    )

    stepper = FMStepper(model, cond_encoder, sim_cfg, fm_cfg, device)
    out_state = stepper.step(
        state,
        conditioning=conditioning_vec.to(device),
        n_steps=fm_cfg.n_inference_steps,
    )
    return out_state.T.cpu()


def _build_conditioning(attrs: dict, dataset_cfg) -> torch.Tensor:
    """Build a z-score normalised conditioning vector from sample attrs."""
    values = []
    for key in dataset_cfg.conditioning_keys:
        values.append(float(attrs[key]))
    cond = torch.tensor(values, dtype=torch.float32)
    means = torch.tensor([dataset_cfg.cond_means[k] for k in dataset_cfg.conditioning_keys])
    stds = torch.tensor([dataset_cfg.cond_stds[k] for k in dataset_cfg.conditioning_keys])
    return (cond - means) / stds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Compare LPBF thermal prediction methods")
    parser.add_argument("--h5", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--sample-key", default="sample_000000", help="Sample group key")
    parser.add_argument("--surrogate-ckpt", default=None, help="ThermalSurrogate3D checkpoint (optional)")
    parser.add_argument("--fm-ckpt", default=None, help="FM surrogate checkpoint")
    parser.add_argument("--output-dir", default="artifacts/compare")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-png", action="store_true", help="Save side-by-side mid-slice PNGs")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load sample ----------------------------------------------------------
    logger.info("Loading sample '%s' from %s", args.sample_key, args.h5)
    T_in, T_target, Q, mask, attrs, root_attrs = _load_sample(args.h5, args.sample_key)
    sim_cfg, mat_cfg = _reconstruct_configs(attrs, root_attrs, device)

    logger.info("Grid: %dx%dx%d  T_in range: %.0f–%.0f K",
                sim_cfg.Nx, sim_cfg.Ny, sim_cfg.Nz,
                T_in.min().item(), T_in.max().item())

    rows: list[tuple[str, dict]] = []

    # ---- Physics solver -------------------------------------------------------
    logger.info("Running physics solver...")
    try:
        pred_solver = _run_physics_solver(T_in, Q, mask, sim_cfg, mat_cfg, device)
        rows.append(("Physics Solver", _metrics(pred_solver, T_target)))
    except Exception as exc:
        logger.warning("Physics solver failed: %s", exc)

    # ---- ThermalSurrogate3D (optional) ----------------------------------------
    if args.surrogate_ckpt:
        logger.info("Running ThermalSurrogate3D...")
        try:
            pred_sur = _run_surrogate(T_in, Q, args.surrogate_ckpt, device)
            if pred_sur is not None:
                rows.append(("ThermalSurrogate3D", _metrics(pred_sur, T_target)))
        except Exception as exc:
            logger.warning("Surrogate failed: %s", exc)
    else:
        logger.info("--surrogate-ckpt not provided; skipping ThermalSurrogate3D arm")

    # ---- FM Stepper -----------------------------------------------------------
    if args.fm_ckpt:
        logger.info("Running FM Stepper...")
        try:
            from neural_pbf.data.fm_dataset import FMDatasetConfig

            ds_cfg = FMDatasetConfig(h5_paths=[args.h5])
            conditioning = _build_conditioning(attrs, ds_cfg)
            pred_fm = _run_fm_stepper(T_in, Q, mask, conditioning, args.fm_ckpt, sim_cfg, device)
            rows.append(("FM Surrogate", _metrics(pred_fm, T_target)))
        except Exception as exc:
            logger.warning("FM stepper failed: %s", exc)
    else:
        logger.info("--fm-ckpt not provided; skipping FM Surrogate arm")

    # ---- Results table --------------------------------------------------------
    if rows:
        print("\n=== Method Comparison: %s ===\n" % args.sample_key)
        _print_table(rows)
    else:
        logger.warning("No methods produced results.")

    # ---- Optional PNG output --------------------------------------------------
    if args.save_png and rows:
        _save_comparison_png(T_in, T_target, rows, out_dir, args.sample_key)


def _save_comparison_png(T_in, T_target, rows, out_dir, sample_key):
    """Save mid-slice PNG comparisons using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # T_in/T_target shape: (1, 1, Nz, Ny, Nx) or (1, Nz, Ny, Nx)
        t = T_in.squeeze()
        mid_z = t.shape[-3] // 2 if t.ndim == 3 else 0
        t_in_slice = t[mid_z] if t.ndim == 3 else t
        t_tgt_slice = T_target.squeeze()
        if t_tgt_slice.ndim == 3:
            t_tgt_slice = t_tgt_slice[mid_z]

        n_cols = 2 + len(rows)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        vmin = T_in.min().item()
        vmax = max(T_in.max().item(), T_target.max().item())

        axes[0].imshow(t_in_slice.numpy(), vmin=vmin, vmax=vmax, cmap="inferno")
        axes[0].set_title("T_in")
        axes[1].imshow(t_tgt_slice.numpy(), vmin=vmin, vmax=vmax, cmap="inferno")
        axes[1].set_title("T_target")

        for ax, (name, _metrics_dict) in zip(axes[2:], rows):
            ax.set_title(name)
            ax.set_xlabel(f"MAE={_metrics_dict['MAE [K]']:.2f}K")

        plt.tight_layout()
        out_path = out_dir / f"compare_{sample_key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info("Saved comparison PNG: %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save PNG: %s", exc)


if __name__ == "__main__":
    main()
