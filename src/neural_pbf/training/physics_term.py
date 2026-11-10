"""The physics term of the training scripts: loss, ReLoBRaLo combination, val residual, loaders and pre-flight.

One place turns the run arguments (``script_args.add_run_args``) into the physics part of a training step, so the
U-Net and DiT scripts differ only in the model, never in how the physics loss is computed, weighted or logged.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler

from neural_pbf.data.material_dataset import MaterialDataset
from neural_pbf.physics.fm_physics import PHYSICS_FORMS, physics_loss, physics_residual_sums
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.physics.material_vector import SCALARS, encode
from neural_pbf.physics.triton_pde_loss import _TRITON_AVAILABLE
from neural_pbf.training.relobralo import ReLoBRaLoWeighter
from neural_pbf.uq.config import UQConfig

Reduce = Callable[[Tensor], Tensor]
PREFLIGHT_TOL = 1e-8


def _identity(t: Tensor) -> Tensor:
    return t


def _grid(grid_attrs: Mapping[str, float]) -> tuple[float, float, float]:
    return float(grid_attrs["dx_m"]), float(grid_attrs["dy_m"]), float(grid_attrs["dz_m"])


@dataclass
class PhysicsTerm:
    """The physics loss of one run and its ReLoBRaLo weight. Batch fields are (B, 1, D, H, W) on the model's device:
    ``T_in``, ``T_target``, ``Q``, ``mask`` (mask_in) and ``material`` (B, MATERIAL_DIM)."""

    form: str
    backend: str | None
    weighter: ReLoBRaLoWeighter
    T_ref: float
    T_ambient: float
    Q_ref: float
    grid: tuple[float, float, float]

    @classmethod
    def from_args(cls, args: argparse.Namespace, ds_cfg: Any, grid_attrs: Mapping[str, float]) -> PhysicsTerm:
        if args.physics_form != "none" and args.physics_backend is None:
            raise ValueError(f"physics form {args.physics_form!r} needs a physics backend")
        weighter = ReLoBRaLoWeighter(
            alpha=args.relobralo_alpha, beta=args.relobralo_beta, temperature=args.relobralo_temperature,
            eps=args.relobralo_epsilon, max_lambda=args.max_lambda,
        )  # fmt: skip
        return cls(args.physics_form, args.physics_backend, weighter, float(ds_cfg.T_ref), float(ds_cfg.T_ambient),
                   float(ds_cfg.Q_ref), _grid(grid_attrs))  # fmt: skip

    @property
    def active(self) -> bool:
        return self.form != "none"

    def loss(self, v_pred: Tensor, x_tau: Tensor, tau: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        """The normalized physics loss; zero without a physics form, which never touches the kernels."""
        if not self.active:
            return v_pred.new_zeros(())
        return physics_loss(
            v_pred, x_tau, tau, batch["T_in"], batch["T_target"], batch["Q"], batch["mask"], batch["material"],
            self.form, self.T_ref, self.T_ambient, self.Q_ref, *self.grid, backend=str(self.backend),
        )  # fmt: skip

    def combine(
        self, loss_fm: Tensor, loss_phys: Tensor, reduce: Reduce = _identity
    ) -> tuple[Tensor, dict[str, float]]:
        """Total loss ``loss_fm + λ_phys·loss_phys`` and the step metrics.

        ``reduce`` gathers a detached loss across processes (e.g. a mean over Accelerate processes), so every
        process steps ReLoBRaLo with the same values and keeps the same weight.
        """
        fm = float(reduce(loss_fm.detach()))
        if not self.active:
            zeros = dict.fromkeys(("loss_phys", "lambda_raw", "lambda_phys", "cap_hit", "skipped"), 0.0)
            return loss_fm, {"loss_fm": fm, "loss_total": fm, **zeros}
        phys = float(reduce(loss_phys.detach()))
        lam = self.weighter.step(fm, phys)
        w = self.weighter
        metrics = {
            "loss_fm": fm, "loss_phys": phys, "loss_total": fm + lam * phys, "lambda_raw": w.lambda_raw,
            "lambda_phys": lam, "cap_hit": float(w.cap_hit), "skipped": float(w.skipped),
        }  # fmt: skip
        return loss_fm + lam * loss_phys, metrics


class ValResidual:
    """The val physics residual of every training run, in both forms, with the loss's normalization.

    T̂ = x_τ + (1 − τ)·v is the same one-step estimate the training loss uses. Σ R² and Σ S² are pooled over the
    whole val set; the melt value pools only the samples whose true T_max lies in [T_solidus, T_boil), the regimes
    of ``uq.regimes``. A set without melt samples gives NaN.
    """

    _KEYS = tuple(f"{f}_{part}" for f in PHYSICS_FORMS for part in ("num", "den", "num_melt", "den_melt"))

    def __init__(self, ds_cfg: Any, grid_attrs: Mapping[str, float]) -> None:
        self.T_ref, self.T_ambient, self.Q_ref = float(ds_cfg.T_ref), float(ds_cfg.T_ambient), float(ds_cfg.Q_ref)
        self.grid = _grid(grid_attrs)
        self._sums = torch.zeros(len(self._KEYS), dtype=torch.float64)

    @torch.no_grad()
    def update(self, v_pred: Tensor, x_tau: Tensor, tau: Tensor, batch: Mapping[str, Tensor]) -> None:
        T_pred = x_tau + (1.0 - tau.view(-1, 1, 1, 1, 1)) * v_pred
        backend = "triton" if T_pred.is_cuda and _TRITON_AVAILABLE else "reference"  # same values, checked by tests
        T_max = batch["T_target"].flatten(1).amax(1).double() * self.T_ref + self.T_ambient
        T_s = batch["material"][:, SCALARS.index("T_s")].double()
        melt = ((T_s <= T_max) & (T_max < batch["T_boil"].double())).cpu()
        row = []
        for form in PHYSICS_FORMS:
            num, den = physics_residual_sums(
                T_pred, batch["T_in"], batch["T_target"], batch["Q"], batch["mask"], batch["material"], form,
                self.T_ref, self.T_ambient, self.Q_ref, *self.grid, backend=backend,
            )  # fmt: skip
            num, den = num.double().cpu(), den.double().cpu()
            row += [num.sum(), den.sum(), num[melt].sum(), den[melt].sum()]
        self._sums += torch.stack(row)

    def compute(self, reduce: Reduce = _identity) -> dict[str, float]:
        """``reduce`` sums the accumulators across processes before the ratios are taken."""
        s = dict(zip(self._KEYS, reduce(self._sums).tolist(), strict=True))
        out = {}
        for form in PHYSICS_FORMS:
            out[f"val_phys_{form}"] = s[f"{form}_num"] / s[f"{form}_den"] if s[f"{form}_den"] > 0 else math.nan
            melt_den = s[f"{form}_den_melt"]
            out[f"val_phys_{form}_melt"] = s[f"{form}_num_melt"] / melt_den if melt_den > 0 else math.nan
        return out


def with_material(loader: DataLoader, t_boil: Mapping[str, float] | None = None) -> DataLoader:
    """The same loader over ``MaterialDataset``: batches gain ``material`` and ``T_boil``.

    ``t_boil`` defaults to the UQ configuration's boiling points. Wrap before ``accelerator.prepare``.
    """
    if t_boil is None:
        t_boil = UQConfig().exploratory.T_boil_K
    return DataLoader(
        MaterialDataset(loader.dataset, t_boil=t_boil),
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, RandomSampler),
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        persistent_workers=loader.persistent_workers,
        generator=loader.generator,
    )


def preflight(backend: str | None, device: torch.device) -> dict[str, Any]:
    """Check the Triton kernels on this GPU before training: one float64 batch, both forms, value and gradient
    against the PyTorch reference. Raises RuntimeError on a mismatch; returns the result for the run's tags."""
    if backend != "triton":
        return {"preflight": "not needed"}
    g = torch.Generator().manual_seed(0)
    mat = MaterialConfig.ss316l_preset()
    scalars = {"rho": mat.rho, "L": mat.latent_heat_L, "T_s": mat.T_solidus, "T_l": mat.T_liquidus,
               "k_p": mat.k_powder, "k_s": mat.k_solid, "k_l": mat.k_liquid, "cp": mat.cp_base,
               "exposure_time": 50e-6}  # fmt: skip
    vec = encode(scalars, {name: list(getattr(mat, name) or []) for name in ("T_lut", "k_lut", "cp_lut")})
    shape = (2, 1, 6, 6, 6)

    def rand(scale: float = 1.0, offset: float = 0.0) -> Tensor:
        return (offset + scale * torch.rand(shape, generator=g, dtype=torch.float64)).to(device)

    T_in = rand(0.6, 0.05)
    case = {"T_in": T_in, "T_target": T_in + rand(0.4), "Q": rand(1e3), "mask": (rand() > 0.5).double(),
            "material": torch.stack([vec, vec]).to(device)}  # fmt: skip
    x_tau, v0 = rand(), rand(2.0, -1.0)
    tau = torch.rand(2, generator=g, dtype=torch.float64).to(device)
    worst = 0.0
    for form in PHYSICS_FORMS:
        results = []
        for backend_ in ("triton", "reference"):
            v = v0.clone().requires_grad_(True)
            fields = (case["T_in"], case["T_target"], case["Q"], case["mask"], case["material"])
            loss = physics_loss(v, x_tau, tau, *fields, form, 2000.0, 300.0, 1e12, 2e-6, 2e-6, 2e-6, backend=backend_)
            (grad,) = torch.autograd.grad(loss, v)
            results.append((loss.detach(), grad))
        (lt, gt), (lr, gr) = results
        worst = max(worst, float((lt - lr).abs() / lr.abs()), float((gt - gr).norm() / gr.norm()))
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    if not worst < PREFLIGHT_TOL:
        raise RuntimeError(f"physics pre-flight failed on {name}: Triton differs from the reference by {worst:.2e}")
    return {"preflight": "passed", "preflight_max_rel_err": worst, "preflight_device": name}
