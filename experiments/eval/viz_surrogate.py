"""viz_surrogate.py — Post-training visualisation for the ThermalSurrogate3D.

Usage::

    uv run python scripts/viz_surrogate.py \\
        --checkpoint ./checkpoints/checkpoint_final.pt \\
        --n_steps 20 \\
        --output_dir ./viz_output \\
        --save_gif

Loads a saved surrogate checkpoint, runs autoregressive inference over a
synthetic scan trajectory, and saves cross-section PNGs (and optionally a GIF)
using the existing visualisation utilities.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")  # non-interactive backend for script mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise surrogate temperature predictions."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file produced by train_surrogate.py.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=20,
        help="Number of autoregressive prediction steps to visualise.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./viz_output",
        help="Directory to save output PNG frames.",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        default=False,
        help="If set, stitch all frames into an animated GIF.",
    )
    parser.add_argument(
        "--domain_mm",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        metavar=("Lx", "Ly", "Lz"),
        help="Domain size in millimetres for generating the scan trajectory.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=3,
        default=[32, 32, 32],
        metavar=("Nx", "Ny", "Nz"),
        help="Grid resolution for the visualisation domain.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_surrogate(ckpt_path: str, device: torch.device):
    """Load checkpoint and reconstruct ThermalSurrogate3D."""
    from neural_pbf.models.config import SurrogateConfig
    from neural_pbf.models.surrogate import ThermalSurrogate3D

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = SurrogateConfig(**ckpt["cfg"])
    model = ThermalSurrogate3D(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded surrogate: strategy=%s depth=%d (sim_step=%s train_step=%s)",
        cfg.strategy,
        cfg.depth,
        ckpt.get("sim_step", "?"),
        ckpt.get("train_step", "?"),
    )
    return model, cfg


def _build_q_sequence(
    n_steps: int,
    Lx_m: float,
    Ly_m: float,
    Lz_m: float,
    Nx: int,
    Ny: int,
    Nz: int,
    dt: float = 1e-5,
) -> list[torch.Tensor]:
    """Generate a sequence of Gaussian beam Q fields along a linear scan path."""
    from neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig

    beam_cfg = GaussianSourceConfig(power=200.0, eta=0.35, sigma=40e-6, depth=30e-6)
    beam = GaussianBeam(beam_cfg)

    xs = torch.linspace(0, Lx_m, Nx)
    ys = torch.linspace(0, Ly_m, Ny)
    zs = torch.linspace(0, Lz_m, Nz)
    Z_g, Y_g, X_g = torch.meshgrid(zs, ys, xs, indexing="ij")
    X_g = X_g.unsqueeze(0).unsqueeze(0)
    Y_g = Y_g.unsqueeze(0).unsqueeze(0)
    Z_g = Z_g.unsqueeze(0).unsqueeze(0)

    scan_speed = 0.8  # m/s
    z0 = Lz_m  # surface

    Q_sequence: list[torch.Tensor] = []
    for step in range(n_steps):
        t_sim = step * dt
        x0 = (scan_speed * t_sim) % Lx_m
        y0 = Ly_m / 2.0
        Q = beam.intensity(X_g, Y_g, Z_g, x0=x0, y0=y0, z0=z0)
        Q_sequence.append(Q)

    return Q_sequence


def _save_frame(
    T_np: np.ndarray,
    step: int,
    dx: float,
    dy: float,
    dz: float,
    output_dir: Path,
    vmin: float,
    vmax: float,
) -> Path:
    """Save cross-section plot of one temperature frame."""
    from neural_pbf.viz.plots import plot_cross_sections

    fig = plt.figure(figsize=(15, 4))
    plot_cross_sections(fig, T_np, dx, dy, dz, unit="mm", vmin=vmin, vmax=vmax)
    fig.suptitle(f"Surrogate Prediction — Step {step}", fontsize=12)

    frame_path = output_dir / f"frame_{step:04d}.png"
    fig.savefig(frame_path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return frame_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load model ---------------------------------------------------------
    model, cfg = _load_surrogate(args.checkpoint, device)

    # ---- Domain parameters --------------------------------------------------
    Lx_mm, Ly_mm, Lz_mm = args.domain_mm
    Nx, Ny, Nz = args.grid_size
    Lx_m, Ly_m, Lz_m = Lx_mm * 1e-3, Ly_mm * 1e-3, Lz_mm * 1e-3

    dx = Lx_m / max(Nx - 1, 1)
    dy = Ly_m / max(Ny - 1, 1)
    dz = Lz_m / max(Nz - 1, 1)

    # ---- Initial temperature field ------------------------------------------
    T_ambient = 293.15
    T_init = torch.full(
        (1, 1, Nz, Ny, Nx), T_ambient, dtype=torch.float32, device=device
    )

    # ---- Heat source sequence -----------------------------------------------
    Q_sequence = _build_q_sequence(args.n_steps, Lx_m, Ly_m, Lz_m, Nx, Ny, Nz)
    Q_sequence = [Q.to(device) for Q in Q_sequence]

    # ---- Autoregressive inference -------------------------------------------
    logger.info("Running autoregressive inference for %d steps…", args.n_steps)
    with torch.no_grad():
        predictions = model.predict_autoregressive(T_init, Q_sequence, device=device)

    # ---- Visualise each frame -----------------------------------------------
    # Determine global T range for consistent colour scale
    all_T = torch.stack([p.squeeze() for p in predictions])
    vmin = float(all_T.min().item())
    vmax = float(all_T.max().item())
    logger.info("Temperature range: %.1f K – %.1f K", vmin, vmax)

    frame_paths: list[Path] = []
    for step_idx, T_pred in enumerate(predictions):
        # T_pred shape: (1, 1, Nz, Ny, Nx) → numpy (Nx, Ny, Nz) for viz
        T_np = T_pred.squeeze().cpu().numpy()
        # Reorder from (Nz, Ny, Nx) to (Nx, Ny, Nz)
        T_np = T_np.transpose(2, 1, 0)

        frame_path = _save_frame(T_np, step_idx, dx, dy, dz, output_dir, vmin, vmax)
        frame_paths.append(frame_path)

        if step_idx % 5 == 0:
            logger.info("Saved frame %d/%d → %s", step_idx + 1, args.n_steps, frame_path)

    logger.info("Saved %d frames to %s", len(frame_paths), output_dir)

    # ---- Optional GIF -------------------------------------------------------
    if args.save_gif:
        try:
            import imageio.v2 as imageio

            gif_path = output_dir / "surrogate_prediction.gif"
            frames = [np.array(plt.imread(str(p))) for p in frame_paths]
            imageio.mimsave(str(gif_path), frames, fps=5, loop=0)
            logger.info("GIF saved to %s", gif_path)
        except ImportError:
            logger.warning("imageio not available — GIF not saved. Install with: pip install imageio")


if __name__ == "__main__":
    main()
