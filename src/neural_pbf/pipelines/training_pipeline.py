"""Device-safe surrogate training pipeline for LPBF thermal simulation.

All functions accept an explicit ``device`` argument and ensure tensors are
moved to the correct device before computation.  Buffer storage always remains
on CPU to avoid GPU VRAM pressure.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.optim as optim

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.models.config import SurrogateConfig
from neural_pbf.models.loss import PhysicsInformedLoss
from neural_pbf.models.replay_buffer import ExperienceReplayBuffer
from neural_pbf.models.surrogate import ThermalSurrogate3D
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.pipelines.grids import make_coordinate_grids
from neural_pbf.scan.sources import GaussianBeam, GaussianSourceConfig


def generate_hf_dataset(
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
    scan_positions: list[tuple[float, float, float]],
    dt_macro: float,
    buffer_capacity: int,
    patch_size: int,
    device: torch.device,
    beam_cfg: GaussianSourceConfig,
    dtype: torch.dtype = torch.float32,
) -> ExperienceReplayBuffer:
    """Generate a high-fidelity dataset by running the physics solver.

    Loops over ``scan_positions``, evaluates the Gaussian beam heat source,
    advances the simulation by one macro-step, and stores each
    ``(T_in, Q, T_target)`` triple in the replay buffer.  All simulation
    tensors are allocated on ``device``; buffer storage is always CPU.

    Args:
        sim_cfg:          Simulation domain configuration.
        mat_cfg:          Material properties.
        scan_positions:   List of ``(x0, y0, z0)`` beam centre positions [m].
        dt_macro:         Macro time step for each scan position [s].
        buffer_capacity:  Maximum buffer capacity.
        patch_size:       Spatial patch size for the replay buffer.
        device:           Device for simulation tensors.
        beam_cfg:         Gaussian beam source configuration.
        dtype:            Floating-point dtype (default float32).

    Returns:
        :class:`ExperienceReplayBuffer` populated with one experience per
        scan position.
    """
    beam = GaussianBeam(beam_cfg)
    stepper = TimeStepper(sim_cfg, mat_cfg)
    buffer = ExperienceReplayBuffer(capacity=buffer_capacity, patch_size=patch_size)

    # Build simulation state on device using the new factory classmethod
    state = SimulationState.zeros(sim_cfg, device=device, dtype=dtype)

    # Build coordinate grids on device (once, reuse each step)
    X3, Y3, Z3 = make_coordinate_grids(sim_cfg, device=device, dtype=dtype)

    for x0, y0, z0 in scan_positions:
        # Compute volumetric heat source on device
        Q_field = beam.intensity(X3, Y3, Z3, x0=x0, y0=y0, z0=z0)

        # Reshape Q to (1, 1, Nx, Ny, Nz) to match state.T
        Q_ext = Q_field.unsqueeze(0).unsqueeze(0)

        # Snapshot input temperature
        T_in = state.T.clone()

        # Advance physics solver
        state = stepper.step_adaptive(state, dt_target=dt_macro, Q_ext=Q_ext)

        # Store experience — push CPU tensors to the buffer
        buffer.push(
            T_in.cpu(),
            Q_ext.cpu(),
            state.T.cpu(),
        )

    return buffer


def train_surrogates(
    buffer: ExperienceReplayBuffer,
    surrogate_direct: ThermalSurrogate3D,
    surrogate_residual: ThermalSurrogate3D | None,
    sim_cfg: SimulationConfig,
    mat_cfg: MaterialConfig,
    surrogate_cfg: SurrogateConfig,
    device: torch.device,
    num_epochs: int,
    dt: float,
    tracker: Any | None = None,
) -> dict[str, list[float]]:
    """Train one or two surrogate models using experience replay.

    Args:
        buffer:             Populated :class:`ExperienceReplayBuffer`.
        surrogate_direct:   Primary surrogate (direct strategy).
        surrogate_residual: Optional second surrogate (residual strategy).
        sim_cfg:            Simulation domain configuration.
        mat_cfg:            Material properties (for physics loss).
        surrogate_cfg:      Surrogate hyper-parameters.
        device:             Training device.
        num_epochs:         Number of full passes through the buffer.
        dt:                 Time step used during data generation [s].
        tracker:            Optional experiment tracker (MLflow etc.).

    Returns:
        Dictionary with keys:
        - ``"direct_loss"``:   List of per-epoch total loss (floats).
        - ``"residual_loss"``: List of per-epoch total loss, or ``[]`` when
                               ``surrogate_residual`` is None.
    """
    # Move models to device
    surrogate_direct.to(device)
    if surrogate_residual is not None:
        surrogate_residual.to(device)

    # Build optimisers using AdamW
    opt_direct = optim.AdamW(surrogate_direct.parameters(), lr=surrogate_cfg.lr)
    opt_residual: optim.Optimizer | None = None
    if surrogate_residual is not None:
        opt_residual = optim.AdamW(surrogate_residual.parameters(), lr=surrogate_cfg.lr)

    # Build physics-informed loss from config
    criterion = PhysicsInformedLoss.from_config(surrogate_cfg, sim_cfg, mat_cfg)

    direct_losses: list[float] = []
    residual_losses: list[float] = []

    batch_size = surrogate_cfg.batch_size

    for epoch in range(num_epochs):
        # Sample a batch and move to device via sample_to
        batch = buffer.sample_to(batch_size, device=device)

        T_in = batch["T_in"]
        Q = batch["Q"]
        T_target = batch["T_target"]

        # ---- Direct surrogate training --------------------------------
        surrogate_direct.train()
        opt_direct.zero_grad()

        T_pred = surrogate_direct(T_in, Q)
        loss_dict = criterion(T_pred, T_target, T_in, Q, dt)
        loss = loss_dict["loss"]
        loss.backward()
        opt_direct.step()

        direct_losses.append(loss.item())

        # ---- Residual surrogate training (optional) ------------------
        if surrogate_residual is not None and opt_residual is not None:
            surrogate_residual.train()
            opt_residual.zero_grad()

            T_lf = batch.get("T_lf", T_in)  # fallback to T_in when no T_lf
            T_pred_res = surrogate_residual(T_in, Q, T_lf=T_lf)
            loss_dict_res = criterion(T_pred_res, T_target, T_in, Q, dt)
            loss_res = loss_dict_res["loss"]
            loss_res.backward()
            opt_residual.step()

            residual_losses.append(loss_res.item())

        # Optional: log to tracker
        if tracker is not None and hasattr(tracker, "log_metrics"):
            metrics: dict[str, float] = {"direct_loss": direct_losses[-1]}
            if residual_losses:
                metrics["residual_loss"] = residual_losses[-1]
            tracker.log_metrics(metrics, step=epoch)

    return {
        "direct_loss": direct_losses,
        "residual_loss": residual_losses,
    }


def evaluate_autoregressive(
    surrogate: ThermalSurrogate3D,
    T_init: torch.Tensor,
    Q_sequence: list[torch.Tensor],
    gt_sequence: list[torch.Tensor],
    device: torch.device,
    T_lf_sequence: list[torch.Tensor] | None = None,
) -> dict[str, float | list[float]]:
    """Evaluate a surrogate model autoregressively and compute MAE vs ground truth.

    All input tensors are moved to ``device`` at function entry.

    Args:
        surrogate:      Trained surrogate model (will be set to eval mode).
        T_init:         Initial temperature field. Shape: (1, 1, D, H, W).
        Q_sequence:     List of heat-source fields (one per step).
        gt_sequence:    List of ground-truth temperature fields (one per step).
        device:         Device for inference.
        T_lf_sequence:  Optional list of low-fidelity fields (for residual strategy).

    Returns:
        Dictionary with:
        - ``"mae_per_step"``:  List of per-step MAE values (Python floats).
        - ``"mean_mae"``:      Scalar mean MAE over all steps (Python float).
    """
    surrogate.to(device)
    surrogate.eval()

    # Move all inputs to device once
    T_init_d = T_init.to(device)
    Q_seq_d = [q.to(device) for q in Q_sequence]
    gt_seq_d = [g.to(device) for g in gt_sequence]
    lf_seq_d: list[torch.Tensor] | None = None
    if T_lf_sequence is not None:
        lf_seq_d = [t.to(device) for t in T_lf_sequence]

    # Run autoregressive prediction
    predictions = surrogate.predict_autoregressive(
        T_init_d,
        Q_seq_d,
        T_lf_sequence=lf_seq_d,
        device=device,
    )

    # Compute per-step MAE on device, convert to Python floats
    mae_per_step: list[float] = []
    for pred, gt in zip(predictions, gt_seq_d, strict=False):
        T_pred = pred[0] if isinstance(pred, tuple) else pred
        mae = torch.mean(torch.abs(T_pred - gt)).item()
        mae_per_step.append(float(mae))

    mean_mae = float(sum(mae_per_step) / len(mae_per_step)) if mae_per_step else 0.0

    return {
        "mae_per_step": mae_per_step,
        "mean_mae": mean_mae,
    }
