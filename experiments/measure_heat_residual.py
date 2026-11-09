"""Measure the large-step heat residual on solver data, as pre-registered in docs/physics_residual_check.md.

Reads only the train and val runs of the UQ manifest. Runs the one-substep bug gate on every sample first and stops
if it fails; only then computes the full-exposure residuals and the per-regime table.

    uv run python experiments/measure_heat_residual.py [--device cuda]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.physics import residual_check
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.uq.config import UQConfig
from neural_pbf.uq.regimes import assign_regimes
from neural_pbf.utils.units import LengthUnit

PRESETS = {
    "SS316L": MaterialConfig.ss316l_preset,
    "IN718": MaterialConfig.in718_preset,
    "Ti64": MaterialConfig.ti64_preset,
}
# Relative positions in each run; for a 50-sample run these are indices 0, 12, 25, 37, 49.
POSITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
MEASURED_SETS = ("train", "val")
FLOAT32_KEY = "float32 solver form"


class GateFailed(RuntimeError):
    """The solver form does not reproduce one substep: an implementation bug, not the large-step error."""


def position_indices(n: int, positions: tuple[float, ...]) -> list[int]:
    """Sample index at each relative position of a run with n samples: round half up of p·(n − 1)."""
    if any(not 0.0 <= p <= 1.0 for p in positions):
        raise ValueError(f"every position must lie in [0, 1], got {positions}")
    return [int(p * (n - 1) + 0.5) for p in positions]


def sample_keys(
    manifest: dict[str, Any], positions: tuple[float, ...] = POSITIONS, sets: tuple[str, ...] = MEASURED_SETS
) -> list[tuple[str, str]]:
    """(h5_path, sample_key) at the given relative positions of every run in `sets`. Other sets are never read."""
    forbidden = sorted(set(sets) - set(MEASURED_SETS))
    if forbidden:
        raise ValueError(f"{forbidden}: cal and test belong to the UQ pipeline and its logged test access")
    keys = []
    for name in sets:
        for run in manifest["sets"][name]:
            for i in position_indices(len(run["sample_keys"]), positions):
                path, key = run["sample_keys"][i]
                keys.append((path, key))
    return keys


_STORED_SCALARS = {
    "k_solid": "k_s", "k_liquid": "k_l", "k_powder": "k_p", "cp_base": "cp", "rho": "rho",
    "latent_heat_L": "L", "T_solidus": "T_s", "T_liquidus": "T_l",
}  # fmt: skip


def material_for(name: str, attrs: Any, luts: dict[str, np.ndarray]) -> MaterialConfig:
    """The preset with the run's randomized values applied, as the generator built it (data.sampling).

    Every field randomize_material changes is stored with the sample. Its T_lut axis is never perturbed, so a
    stored axis that differs from the preset means the sample came from another material definition.
    """
    mat = PRESETS[name]()
    if mat.T_lut is None or not np.allclose(np.asarray(mat.T_lut, dtype=np.float32), luts["T_lut"]):
        raise ValueError(f"{name}: stored T_lut differs from the preset axis")
    update: dict[str, Any] = {field: float(attrs[key]) for field, key in _STORED_SCALARS.items()}
    update |= {"k_lut": [float(v) for v in luts["k_lut"]], "cp_lut": [float(v) for v in luts["cp_lut"]]}
    return mat.model_copy(update=update)


def load_sample(path: str, key: str, device: torch.device) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        a, g = f.attrs, f["samples"][key]
        sim = SimulationConfig(
            Lx=float(a["Lx_m"]), Ly=float(a["Ly_m"]), Lz=float(a["Lz_m"]),
            Nx=int(a["Nx"]), Ny=int(a["Ny"]), Nz=int(a["Nz"]), length_unit=LengthUnit.METERS,
        )  # fmt: skip
        fields = {n: torch.from_numpy(g[n][()]).to(device) for n in ("T_in", "T_target", "Q", "mask_in", "mask")}
        luts = {n: g[n][()] for n in ("T_lut", "k_lut", "cp_lut")}
        name = str(g.attrs["mat_name"])
        return {
            **{n: fields[n].float() for n in ("T_in", "T_target", "Q")},
            "mask_in": fields["mask_in"], "mask": fields["mask"], "sim": sim, "mat_name": name,
            "mat": material_for(name, g.attrs, luts), "exposure_time": float(g.attrs["exposure_time"]),
            "T_s": float(g.attrs["T_s"]),
        }  # fmt: skip


def measure(
    keys: list[tuple[str, str]], t_boil: dict[str, float], device: torch.device
) -> tuple[list[dict], list[dict]]:
    """Bug gate (float64) on every sample, then the full-exposure residuals. Raises GateFailed before any exposure."""
    gate = []
    for path, key in keys:
        s = load_sample(path, key, device)
        args = (s["T_in"], s["mask_in"], s["Q"], s["mat"], s["sim"], s["exposure_time"])
        exact = residual_check.substep_check(*args, dtype=torch.float64)
        f32 = residual_check.substep_check(*args, dtype=torch.float32)[residual_check.SOLVER_FORM]
        gate.append({**exact, FLOAT32_KEY: f32})
        print(f"gate {path}:{key} float64 {exact[residual_check.SOLVER_FORM]:.2e} float32 {f32:.2e}", flush=True)
    tol, form = residual_check.SUBSTEP_TOL, residual_check.SOLVER_FORM
    failed = [k for k, g in zip(keys, gate, strict=True) if not g[form] < tol]
    if failed:
        raise GateFailed(f"{len(failed)} samples fail the one-substep gate, e.g. {failed[:3]}")

    rows = []
    for path, key in keys:
        s = load_sample(path, key, device)
        t_max = float(s["T_target"].max())
        regime = assign_regimes(np.array([t_max]), np.array([s["mat_name"]]), np.array([s["T_s"]]), t_boil)[0]
        res = residual_check.exposure_residuals(
            s["T_in"], s["T_target"], s["Q"], s["mask_in"], s["mask"], s["mat"], s["sim"], s["exposure_time"]
        )
        rows.append({"path": path, "key": key, "material": s["mat_name"], "T_max": t_max, "regime": str(regime), **res})
        print(f"exposure {path}:{key} {regime}", flush=True)
    return gate, rows


def decision_band(value: float) -> str:
    return "< 0.1" if value < 0.1 else ("0.1 to 0.5" if value <= 0.5 else "> 0.5")


def _name(v: residual_check.Variant) -> str:
    return f"{v[0]}, {v[1]}"


def _fmt(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.3g}"


def render(gate: list[dict], rows: list[dict]) -> str:
    lines = ["### One-substep check (all samples, float64 unless marked)", ""]
    lines += ["| Variant | median | max |", "| --- | --- | --- |"]
    for v in residual_check.VARIANTS:
        vals = np.array([g[v] for g in gate])
        lines.append(f"| {_name(v)} | {_fmt(float(np.median(vals)))} | {_fmt(float(vals.max()))} |")
    f32 = np.array([g[FLOAT32_KEY] for g in gate])
    f32_name = f"{_name(residual_check.SOLVER_FORM)}, float32"
    lines.append(f"| {f32_name} | {_fmt(float(np.median(f32)))} | {_fmt(float(f32.max()))} |")
    lines += ["", "### Full exposure, |ΔT| > 50 K", "", "| Regime | Variant | n | empty | median | p90 |"]
    lines.append("| --- | --- | --- | --- | --- | --- |")
    table = residual_check.summarize(rows, residual_check.VARIANTS)
    for (regime, v), t in table.items():
        lines.append(f"| {regime} | {_name(v)} | {t['n']} | {t['n_empty']} | {_fmt(t['median'])} | {_fmt(t['p90'])} |")
    chosen = residual_check.select_variant(table, set(residual_check.VARIANTS))
    lines.append("")
    if chosen is None:
        lines.append("**Decision variant:** none, no melt-regime sample. **Band:** undecided.")
    else:
        median = table[("melt", chosen)]["median"]
        band = decision_band(median)
        lines.append(f"**Decision variant:** {_name(chosen)}, melt median {_fmt(median)}. **Band:** {band}.")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"{path}: no rows to write")
    flat = [{(_name(k) if isinstance(k, tuple) else k): v for k, v in r.items()} for r in rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="artifacts/uq/manifest.json")
    p.add_argument("--positions", default=",".join(map(str, POSITIONS)), help="relative sample positions per run")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="artifacts/physics_residual")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    positions = tuple(float(x) for x in args.positions.split(","))
    keys = sample_keys(json.loads(Path(args.manifest).read_text()), positions)
    print(f"{len(keys)} samples from {MEASURED_SETS}", flush=True)
    try:
        gate, rows = measure(keys, UQConfig().exploratory.T_boil_K, torch.device(args.device))
    except GateFailed as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    _write_csv(out / "gate.csv", [{"path": p, "key": k, **g} for (p, k), g in zip(keys, gate, strict=True)])
    _write_csv(out / "exposures.csv", rows)
    text = render(gate, rows)
    (out / "result.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
