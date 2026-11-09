"""Fixed-size vector for one run's material and exposure_time, so a batch can carry one per sample.

Layout: the scalars in ``SCALARS``, the LUT length n, then ``T_lut``, ``k_lut`` and ``cp_lut``, each padded to
``MAX_LUT``. ``data.material_dataset`` builds it from the stored values; the physics loss decodes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from neural_pbf.physics.material import MaterialConfig

SCALARS = ("rho", "L", "T_s", "T_l", "k_p", "k_s", "k_l", "cp", "exposure_time")
LUTS = ("T_lut", "k_lut", "cp_lut")
MAX_LUT = 8
_N_LUT = len(SCALARS)
_LUT0 = _N_LUT + 1
MATERIAL_DIM = _LUT0 + len(LUTS) * MAX_LUT


def encode(scalars: Mapping[str, float], luts: Mapping[str, Sequence[float]]) -> Tensor:
    """(MATERIAL_DIM,) float64 vector. The three tables must have one common length in [2, MAX_LUT]."""
    lengths = {name: len(luts[name]) for name in LUTS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"LUT length differs between tables: {lengths}")
    n = lengths["T_lut"]
    if not 2 <= n <= MAX_LUT:
        raise ValueError(f"LUT length {n} outside [2, {MAX_LUT}]")
    vec = torch.zeros(MATERIAL_DIM, dtype=torch.float64)
    vec[:_N_LUT] = torch.tensor([float(scalars[k]) for k in SCALARS], dtype=torch.float64)
    vec[_N_LUT] = n
    for i, name in enumerate(LUTS):
        vec[_LUT0 + i * MAX_LUT : _LUT0 + i * MAX_LUT + n] = torch.tensor(list(luts[name]), dtype=torch.float64)
    return vec


def exposure_time_of(vec: Tensor) -> float:
    return float(vec[SCALARS.index("exposure_time")])


def material_from_vector(vec: Tensor) -> MaterialConfig:
    """The generator's MaterialConfig for this run (presets set only these fields; the rest keep defaults)."""
    s = dict(zip(SCALARS, (float(v) for v in vec[:_N_LUT]), strict=True))
    n = int(vec[_N_LUT])
    T_lut, k_lut, cp_lut = ([float(v) for v in vec[_LUT0 + i * MAX_LUT : _LUT0 + i * MAX_LUT + n]] for i in range(3))
    return MaterialConfig(
        k_powder=s["k_p"], k_solid=s["k_s"], k_liquid=s["k_l"], cp_base=s["cp"], rho=s["rho"],
        T_solidus=s["T_s"], T_liquidus=s["T_l"], latent_heat_L=s["L"], use_T_dep=True, use_lut=True,
        T_lut=T_lut, k_lut=k_lut, cp_lut=cp_lut,
    )  # fmt: skip
