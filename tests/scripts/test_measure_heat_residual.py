"""The measurement script for docs/physics_residual_check.md: train runs only, bug gate first, then the table."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from neural_pbf.core.config import SimulationConfig
from neural_pbf.core.state import SimulationState
from neural_pbf.integrator.stepper import TimeStepper
from neural_pbf.physics import residual_check
from neural_pbf.physics.material import MaterialConfig
from neural_pbf.utils.units import LengthUnit

_SCRIPT = Path(__file__).parent.parent.parent / "experiments" / "measure_heat_residual.py"
_spec = importlib.util.spec_from_file_location("measure_heat_residual", _SCRIPT)
mhr = importlib.util.module_from_spec(_spec)
sys.modules["measure_heat_residual"] = mhr
_spec.loader.exec_module(mhr)  # type: ignore[union-attr]

MAT = MaterialConfig.ss316l_preset()
SIM = SimulationConfig(Lx=0.032, Ly=0.024, Lz=0.016, Nx=17, Ny=13, Nz=9, length_unit=LengthUnit.MILLIMETERS)
T_BOIL = {"SS316L": 3090.0}


def _mat_attrs(mat: MaterialConfig) -> dict[str, float]:
    return {
        "k_s": mat.k_solid, "k_l": mat.k_liquid, "k_p": mat.k_powder, "cp": mat.cp_base, "rho": mat.rho,
        "L": mat.latent_heat_L, "T_s": mat.T_solidus, "T_l": mat.T_liquidus,
    }  # fmt: skip


def _write_run(path: Path, peaks: list[float], exposure: float = 2e-6) -> None:
    """A run file in the generator's layout, with targets from the real solver."""
    shape = (1, 1, SIM.Nz, SIM.Ny, SIM.Nx)
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in shape[2:]), indexing="ij")
    r2 = ((x - SIM.Nx // 2) ** 2 + (y - SIM.Ny // 2) ** 2 + (z - SIM.Nz + 1) ** 2).view(shape)
    with h5py.File(path, "w") as f:
        f.attrs.update({"Lx_m": SIM.Lx_m, "Ly_m": SIM.Ly_m, "Lz_m": SIM.Lz_m, "Nx": SIM.Nx, "Ny": SIM.Ny, "Nz": SIM.Nz})
        for i, peak in enumerate(peaks):
            T = 300.0 + (peak - 300.0) * torch.exp(-r2 / 8.0)
            mask = (z < SIM.Nz // 2).to(torch.uint8).view(shape)
            Q = 2e15 * torch.exp(-r2 / 4.0)
            state = SimulationState.zeros(SIM, device=torch.device("cpu"))
            state.T, state.material_mask = T.clone(), mask.clone()
            out = TimeStepper(SIM, MAT).step_adaptive(state, exposure, Q)
            g = f.create_group(f"samples/sample_{i:06d}")
            for name, arr in {
                "T_in": T.half(), "T_target": out.T.half(), "Q": Q, "mask_in": mask, "mask": out.material_mask,
                "T_lut": torch.tensor(MAT.T_lut), "k_lut": torch.tensor(MAT.k_lut), "cp_lut": torch.tensor(MAT.cp_lut),
            }.items():  # fmt: skip
                g.create_dataset(name, data=arr.numpy())
            g.attrs.update({"mat_name": "SS316L", "exposure_time": exposure, **_mat_attrs(MAT)})


def _manifest(tmp_path: Path) -> Path:
    train = tmp_path / "run_000.h5"
    _write_run(train, [300.0, 1900.0, 3500.0])  # Q adds about 1000 K: solid, melt, keyhole
    sets = {
        "train": [{"h5_path": str(train), "material": "SS316L", "sample_keys": [
            [str(train), f"sample_{i:06d}"] for i in range(3)]}],
        "val": [{"h5_path": str(train), "material": "SS316L", "sample_keys": [[str(train), "sample_000001"]]}],
        **{name: [{"h5_path": str(tmp_path / f"missing_{name}.h5"), "material": "SS316L", "sample_keys": [
            [str(tmp_path / f"missing_{name}.h5"), "sample_000000"]]}] for name in ("cal", "test")},
    }  # fmt: skip
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"sets": sets}))
    return path


@pytest.mark.unit
def test_sample_keys_reads_train_and_val_only(tmp_path: Path) -> None:
    manifest = json.loads(_manifest(tmp_path).read_text())
    keys = mhr.sample_keys(manifest, positions=(0.0,))
    assert [k[1] for k in keys] == ["sample_000000", "sample_000001"]  # one train run, one val run
    assert all("missing" not in k[0] for k in keys)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["cal", "test"])
def test_sample_keys_refuses_cal_and_test(tmp_path: Path, name: str) -> None:
    manifest = json.loads(_manifest(tmp_path).read_text())
    with pytest.raises(ValueError, match="UQ pipeline"):
        mhr.sample_keys(manifest, positions=(0.0,), sets=("train", name))


@pytest.mark.unit
@pytest.mark.parametrize(("n", "expected"), [(50, [0, 12, 25, 37, 49]), (44, [0, 11, 22, 32, 43]), (1, [0] * 5)])
def test_positions_map_to_indices_within_each_run(n: int, expected: list[int]) -> None:
    """Runs have 44 to 50 samples; positions keep the pre-registered indices for full runs and stay in range."""
    assert mhr.position_indices(n, mhr.POSITIONS) == expected


@pytest.mark.unit
def test_positions_outside_the_run_are_refused() -> None:
    with pytest.raises(ValueError, match="position"):
        mhr.position_indices(50, (0.0, 1.5))


@pytest.mark.unit
def test_material_is_rebuilt_from_the_stored_perturbed_values() -> None:
    """The generator randomizes every run's material; the stored values, not the preset, are what the solver used."""
    from neural_pbf.data.sampling import randomize_material

    perturbed = randomize_material(MAT, scale=0.1)
    luts = {n: np.asarray(getattr(perturbed, n), dtype=np.float32) for n in ("T_lut", "k_lut", "cp_lut")}
    rebuilt = mhr.material_for("SS316L", _mat_attrs(perturbed), luts)
    for field in ("k_solid", "k_liquid", "k_powder", "cp_base", "rho", "latent_heat_L", "T_solidus", "T_liquidus"):
        assert getattr(rebuilt, field) == pytest.approx(getattr(perturbed, field))
    for field in ("k_lut", "cp_lut"):
        assert getattr(rebuilt, field) == pytest.approx(getattr(perturbed, field), rel=1e-6)
    assert rebuilt.transition_sharpness == MAT.transition_sharpness and rebuilt.use_lut


@pytest.mark.unit
def test_material_refuses_a_temperature_axis_the_generator_never_writes() -> None:
    luts = {n: np.asarray(getattr(MAT, n), dtype=np.float32) for n in ("T_lut", "k_lut", "cp_lut")}
    luts["T_lut"] = luts["T_lut"] + 10.0
    with pytest.raises(ValueError, match="T_lut"):
        mhr.material_for("SS316L", _mat_attrs(MAT), luts)


@pytest.mark.unit
def test_load_sample_rebuilds_grid_and_fields(tmp_path: Path) -> None:
    _manifest(tmp_path)
    s = mhr.load_sample(str(tmp_path / "run_000.h5"), "sample_000001", torch.device("cpu"))
    assert s["sim"].dx == pytest.approx(SIM.dx) and s["sim"].dz == pytest.approx(SIM.dz)
    assert s["T_in"].dtype == torch.float32 and s["T_in"].shape == (1, 1, SIM.Nz, SIM.Ny, SIM.Nx)
    assert s["mat"].k_lut == pytest.approx(MAT.k_lut) and s["mat"].T_solidus == pytest.approx(MAT.T_solidus)
    assert s["exposure_time"] == pytest.approx(2e-6)


@pytest.mark.unit
def test_measure_runs_gate_then_exposures_with_regimes(tmp_path: Path) -> None:
    manifest = json.loads(_manifest(tmp_path).read_text())
    keys = mhr.sample_keys(manifest, positions=(0.0, 0.5, 1.0), sets=("train",))
    gate, rows = mhr.measure(keys, T_BOIL, torch.device("cpu"))
    assert len(gate) == len(rows) == 3
    assert all(g[residual_check.SOLVER_FORM] < residual_check.SUBSTEP_TOL for g in gate)
    assert all(g[residual_check.SOLVER_FORM] < 1e-10 for g in gate)  # the gate runs in float64
    assert all(0.0 < g["float32 solver form"] < residual_check.SUBSTEP_TOL for g in gate)
    assert [r["regime"] for r in rows] == ["solid", "melt", "keyhole"]
    assert all(set(residual_check.VARIANTS) <= set(r) for r in rows)


@pytest.mark.unit
def test_measure_stops_before_any_exposure_when_the_gate_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = json.loads(_manifest(tmp_path).read_text())
    keys = mhr.sample_keys(manifest, positions=(0.0, 0.5), sets=("train",))
    monkeypatch.setattr(mhr.residual_check, "substep_check", lambda *a, **k: {residual_check.SOLVER_FORM: 1.0})
    called = []
    monkeypatch.setattr(mhr.residual_check, "exposure_residuals", lambda *a, **k: called.append(1))
    with pytest.raises(mhr.GateFailed):
        mhr.measure(keys, T_BOIL, torch.device("cpu"))
    assert not called


@pytest.mark.unit
@pytest.mark.parametrize(("value", "band"), [(0.05, "< 0.1"), (0.1, "0.1 to 0.5"), (0.5, "0.1 to 0.5"), (0.7, "> 0.5")])
def test_decision_band(value: float, band: str) -> None:
    assert mhr.decision_band(value) == band


@pytest.mark.unit
def test_render_names_the_chosen_variant_and_the_band(tmp_path: Path) -> None:
    manifest = json.loads(_manifest(tmp_path).read_text())
    keys = mhr.sample_keys(manifest, positions=(0.0, 0.5, 1.0), sets=("train",))
    gate, rows = mhr.measure(keys, T_BOIL, torch.device("cpu"))
    text = mhr.render(gate, rows)
    assert "Decision variant" in text and "Band" in text
    for regime in ("solid", "melt", "keyhole"):
        assert f"| {regime} |" in text
