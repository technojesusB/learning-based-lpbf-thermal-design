"""
Sampling utilities for the offline LPBF dataset pipeline.

Provides:
- heuristic_sample_indices: physics-informed index selection across a scan path
- randomize_material: stochastic material property perturbation for data augmentation
"""
from __future__ import annotations

import math
import random

import numpy as np

from neural_pbf.physics.material import MaterialConfig


def heuristic_sample_indices(n_steps: int, n_samples: int = 50) -> set[int]:
    """
    Return a set of step indices distributed across three physically meaningful
    windows of a laser scan path.

    The distribution is:
    - Early (melt pool formation):    first 5% of steps  -> 20% of n_samples
    - Mid   (steady-state):           5% - 90% of steps  -> 60% of n_samples
    - Late  (solidification/cooling): last 10% of steps  -> remaining n_samples

    Args:
        n_steps:   Total number of steps in the scan path. Must be >= n_samples.
        n_samples: Total number of sample indices to return. Default 50.

    Returns:
        A Python set of int indices, all in [0, n_steps). The set may contain
        fewer than n_samples elements only when windows overlap and duplicates
        are deduplicated (only possible for very small n_steps relative to
        n_samples).

    Raises:
        AssertionError: If n_steps < n_samples.
    """
    if n_steps <= n_samples:
        return set(range(n_steps))

    n_early = round(n_samples * 0.20)
    n_mid = round(n_samples * 0.60)
    n_late = n_samples - n_early - n_mid

    # Window boundaries (index arithmetic; upper bounds are exclusive)
    early_end = math.ceil(0.05 * n_steps)  # first 5%
    late_start = math.floor(0.90 * n_steps)  # last 10%

    # Clamp to valid range so windows stay in-bounds for small n_steps
    early_end = min(early_end, n_steps)
    late_start = max(late_start, 0)

    early_indices = np.linspace(0, early_end - 1, n_early).astype(int)
    mid_indices = np.linspace(early_end, late_start - 1, n_mid).astype(int)
    late_indices = np.linspace(late_start, n_steps - 1, n_late).astype(int)

    result: set[int] = set(
        int(i) for i in np.concatenate([early_indices, mid_indices, late_indices])
    )
    return result


def _topup_sample_indices(
    sample_indices: set[int],
    n_steps: int,
    n_samples: int,
    rng: random.Random | None = None,
) -> set[int]:
    """
    Top up a set of sample indices to exactly n_samples by drawing randomly
    from remaining available indices.

    Returns a new set (does not mutate the input).
    If len(available) < need, returns the original set unchanged.

    Args:
        sample_indices: Existing set of sampled indices.
        n_steps:        Total number of steps in the path.
        n_samples:      Target number of samples.
        rng:            Optional ``random.Random`` instance for deterministic
                        testing. Defaults to the module-level ``random`` functions.

    Returns:
        A new set of int indices of length
        ``min(n_samples, len(available) + len(sample_indices))``.
    """
    need = n_samples - len(sample_indices)
    if need <= 0:
        return set(sample_indices)

    available = sorted(set(range(n_steps)) - sample_indices)
    if len(available) < need:
        return set(sample_indices)

    if rng is not None:
        extras = rng.sample(available, need)
    else:
        extras = random.sample(available, need)

    return set(sample_indices) | set(extras)


def randomize_material(cfg: MaterialConfig, scale: float = 0.1) -> MaterialConfig:
    """
    Return a new MaterialConfig with random perturbations applied to key properties.

    Perturbation strategy:
    - Scalar conductivities (k_powder, k_solid, k_liquid), heat capacity
      (cp_base), and density (rho) are each multiplied by
      ``(1 + uniform(-scale, scale))``.
    - T_solidus and T_liquidus are both shifted by the **same** random offset
      in [-20, 20] K, preserving the solidus-liquidus gap.
    - LUT value arrays (k_lut, cp_lut) are perturbed element-wise in the
      same multiplicative fashion.  The temperature axis (T_lut) is **never
      modified**.

    Args:
        cfg:   Source MaterialConfig (Pydantic v2 frozen model).
        scale: Fractional perturbation magnitude. Default 0.1 (+-10%).

    Returns:
        A new MaterialConfig with perturbed properties.
    """

    def perturb(v: float) -> float:
        return v * (1.0 + random.uniform(-scale, scale))

    k_powder = perturb(cfg.k_powder)
    k_solid = perturb(cfg.k_solid)
    k_liquid = perturb(cfg.k_liquid)
    cp_base = perturb(cfg.cp_base)
    rho = perturb(cfg.rho)
    latent_heat_L = perturb(cfg.latent_heat_L)

    # Shift both phase-change temperatures by the same random offset so that
    # T_liquidus - T_solidus (and thus latent_width) is preserved exactly.
    shift = random.uniform(-20.0, 20.0)
    T_solidus = cfg.T_solidus + shift
    T_liquidus = cfg.T_liquidus + shift

    # LUT perturbation: perturb VALUE arrays only, never T_lut
    k_lut: list[float] | None = None
    cp_lut: list[float] | None = None
    if cfg.k_lut is not None:
        k_lut = [perturb(v) for v in cfg.k_lut]
    if cfg.cp_lut is not None:
        cp_lut = [perturb(v) for v in cfg.cp_lut]

    return cfg.model_copy(
        update={
            "k_powder": k_powder,
            "k_solid": k_solid,
            "k_liquid": k_liquid,
            "cp_base": cp_base,
            "rho": rho,
            "latent_heat_L": latent_heat_L,
            "T_solidus": T_solidus,
            "T_liquidus": T_liquidus,
            "k_lut": k_lut,
            "cp_lut": cp_lut,
        }
    )
