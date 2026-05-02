"""PhysicsInformedLoss — combined MSE + PDE residual loss for surrogate training."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neural_pbf.core.config import SimulationConfig
from neural_pbf.models.config import SurrogateConfig
from neural_pbf.physics.material import MaterialConfig, cp_eff, k_eff
from neural_pbf.physics.ops import div_k_grad


class PhysicsInformedLoss(nn.Module):
    """Combined MSE + physics (PDE residual) loss for thermal surrogate training.

    The total loss is::

        L = MSE(T_pred, T_target)
          + pde_weight  * L_pde
          + mask_weight * L_mask   (only when mask_pred / mask_target are given)

    where ``L_pde`` is the mean-squared PDE residual of the heat equation::

        rho * cp(T) * dT/dt  =  div(k(T) * grad(T))  +  Q

    and ``L_mask`` is binary cross-entropy for consolidation-mask prediction.

    Material properties ``k`` and ``cp`` may be supplied as pre-computed
    per-voxel tensors (from a physics-context buffer) or derived from
    ``mat_config`` at runtime.  When pre-computed tensors are provided they
    take precedence, enabling multi-material batches where ``mat_config`` is
    used only as a fallback.

    Args:
        sim_config:  Simulation domain parameters (grid spacing, 3D flag).
        mat_config:  Material physical properties (fallback when k/cp not given).
        pde_weight:  Scalar multiplier for the PDE residual term.
        mask_weight: Scalar multiplier for the mask BCE term (default 0 = disabled).
        T_ref:       Temperature normalisation reference [K].
        Q_ref:       Heat-source normalisation reference [W/m³].
    """

    def __init__(
        self,
        sim_config: SimulationConfig,
        mat_config: MaterialConfig,
        pde_weight: float = 0.1,
        mask_weight: float = 0.0,
        T_ref: float = 2000.0,
        Q_ref: float = 1e12,
    ) -> None:
        super().__init__()
        self.sim_config = sim_config
        self.mat_config = mat_config
        self.pde_weight = pde_weight
        self.mask_weight = mask_weight
        self.T_ref = T_ref
        self.Q_ref = Q_ref

    @classmethod
    def from_config(
        cls,
        surrogate_cfg: SurrogateConfig,
        sim_config: SimulationConfig,
        mat_config: MaterialConfig,
    ) -> PhysicsInformedLoss:
        """Construct from a :class:`SurrogateConfig`, forwarding all matching fields.

        Prefer this over direct construction when training with a SurrogateConfig to
        avoid the ``mask_weight`` default mismatch (``PhysicsInformedLoss`` defaults to
        0.0, ``SurrogateConfig`` defaults to 1.0).
        """
        return cls(
            sim_config=sim_config,
            mat_config=mat_config,
            pde_weight=surrogate_cfg.pde_weight,
            mask_weight=surrogate_cfg.mask_weight,
            T_ref=surrogate_cfg.T_ref,
            Q_ref=surrogate_cfg.Q_ref,
        )

    def forward(
        self,
        T_pred: Tensor,
        T_target: Tensor,
        T_in: Tensor,
        Q: Tensor,
        dt: float,
        k_field: Tensor | None = None,
        cp_field: Tensor | None = None,
        rho_field: Tensor | None = None,
        mask_pred: Tensor | None = None,
        mask_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute loss terms.

        Args:
            T_pred:     Predicted temperature field.  Shape: (B, 1, D, H, W).
            T_target:   Ground-truth temperature.     Shape: (B, 1, D, H, W).
            T_in:       Input temperature (t step).   Shape: (B, 1, D, H, W).
            Q:          Volumetric heat source [W/m³]. Shape: (B, 1, D, H, W).
            dt:         Time step duration [s].
            k_field:    Pre-computed thermal conductivity [W/(m·K)].
                        Shape: (B, 1, D, H, W). When provided, overrides the
                        ``mat_config``-based k_eff computation.
            cp_field:   Pre-computed specific heat [J/(kg·K)].
                        Shape: (B, 1, D, H, W). When provided, overrides cp_eff.
            rho_field:  Pre-computed density [kg/m³].
                        Shape: (B, 1, D, H, W). When provided, overrides scalar rho.
            mask_pred:  Raw consolidation-mask logits (pre-sigmoid).
                        Shape: (B, 1, D, H, W). Required to compute mask loss.
            mask_target: Binary ground-truth mask {0, 1}.
                        Shape: (B, 1, D, H, W). Required to compute mask loss.

        Returns:
            Dictionary with scalar tensors:
            - ``"loss"``:     Weighted total loss.
            - ``"mse"``:      MSE between T_pred and T_target.
            - ``"pde"``:      Mean-squared PDE residual.
            - ``"mask_bce"``: Mask BCE (0.0 when mask args not provided).
        """
        # ---- Data loss (normalised by T_ref² → O(1) scale) -----------------
        mse = F.mse_loss(T_pred, T_target) / (self.T_ref**2)

        # ---- PDE residual ---------------------------------------------------
        T_det = T_pred.detach()

        k = (
            k_field.detach()
            if k_field is not None
            else k_eff(T_det, self.mat_config, mask=None)
        )

        if cp_field is not None:
            cp: Tensor = cp_field.detach()
        else:
            cp = cp_eff(T_det, self.mat_config)

        if rho_field is not None:
            rho: Tensor | float = rho_field
        else:
            rho = self.mat_config.rho

        dx = self.sim_config.dx
        dy = self.sim_config.dy
        dz: float | None = self.sim_config.dz if self.sim_config.is_3d else None

        div_term = div_k_grad(T_pred, k, dx, dy, dz)

        dT_dt = (T_pred - T_in) / dt

        # PDE: rho * cp * dT/dt - div(k grad T) - Q = 0
        residual = rho * cp * dT_dt - div_term - Q
        pde_loss = (residual / self.Q_ref).pow(2).mean()

        # ---- Mask BCE (consolidation state) ---------------------------------
        mask_bce: Tensor
        if mask_pred is not None and mask_target is not None:
            mask_bce = F.binary_cross_entropy_with_logits(
                mask_pred, mask_target.float()
            )
        else:
            mask_bce = T_pred.new_zeros(())

        # ---- Total ----------------------------------------------------------
        total = mse + self.pde_weight * pde_loss + self.mask_weight * mask_bce

        return {"loss": total, "mse": mse, "pde": pde_loss, "mask_bce": mask_bce}
