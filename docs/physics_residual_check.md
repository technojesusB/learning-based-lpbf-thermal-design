# Physics residual check

Does the heat-equation residual over one whole exposure vanish on the solver's own data? Brainstorming 2.2 raises
the doubt: the model predicts the state after 600 to 1700 explicit substeps, while the PDE residual holds for one
small step. This note fixes the measurement before any number exists, then records the result.

Reference implementation: `heat_residual_si` and `enthalpy` in `src/neural_pbf/physics/fm_physics.py` (commit
`e09d484`).

## Pre-registration (fixed 2026-11-09, before measuring)

### What the result decides

The physics-loss variants are trained in the ablation regardless of the outcome. The residual size decides how the
loss is weighted and what the report says about it.

| Relative residual | Weighting | Report |
| --- | --- | --- |
| < 0.1 | ReLoBRaLo as planned | the residual size |
| 0.1 to 0.5 | ReLoBRaLo with the physics weight capped at the data-loss weight, so a residual that cannot reach zero is not up-weighted for stagnating | the residual size |
| > 0.5 | the planned variants, with the same cap | that the residual is inconsistent with the one-exposure target |

### Definition

- `R = heat_residual_si(T_in, T_target, Q, mask_in, mat, dt = exposure_time, ...)` in W/m³.
- Relative residual of one sample: `‖R‖₂ / ‖ρ·(H(T_target) − H(T_in))/dt‖₂`, a ratio of norms, not a per-voxel
  ratio. `H` is `enthalpy`: the solver's cp(T) plus latent heat. The same denominator is used for every variant, so
  the variants are compared on equal terms. `cp_eff(T_in)` is not used: a voxel that melts during the exposure starts
  below solidus, so it would leave out the latent heat exactly in the melt regime.
- Both norms run over the interior voxels where `|T_target − T_in| > 50 K`. That is well above the float16 storage
  rounding (about 1 to 8 K) and well below melt-pool changes. A sample with no such voxel is reported as a count and
  left out of the statistics.
- The relative residual is reported per regime (`solid`, `melt`, `keyhole`), assigned by `uq.regimes.assign_regimes`
  from the **true** T_max of `T_target`, with the same boiling points as the UQ pre-registration. For each regime the
  report gives n, the median, and p90.
- All six variants are reported: storage `enthalpy` or `cp_in`, crossed with diffusion `explicit`, `implicit`, or
  `trapezoid`.
- **Decision variant:** among the variants that pass the sanity check below, the one with the lowest median in the
  melt regime. That variant is ported to Triton in step 3.3.
- **Samples:** the relative positions 0, ¼, ½, ¾ and 1 of every **train** and **val** run of the UQ manifest
  (`artifacts/uq/manifest.json`), index ⌊p·(n − 1) + ½⌋, so 120 samples from 24 runs. For a 50-sample run these are
  indices 0, 12, 25, 37 and 49. Four train runs are shorter (run_001: 49, run_002: 44, run_009: 48, run_024: 49);
  the first version named fixed indices, and the first launch stopped on run_002 before computing anything, so the
  selection was switched to positions (2026-11-09). Cal and test runs belong to the UQ pipeline and its
  logged test access; the script refuses them and never opens them.
  - train (21 runs; 7 per material, 7 per pattern): run_000, 001, 002, 009, 012, 014, 022, 023, 024, 025, 026, 028,
    032, 033, 035, 037, 038, 039, 040, 042, 043.
  - val (3 runs; one per material and pattern): run_003, 007, 011.
- **Material:** each run's material is the preset with the generator's per-run randomization applied
  (`data.sampling.randomize_material`, up to ±10 % on k, cp, ρ, L and the LUT values, ±20 K on T_s and T_l). The
  measurement rebuilds it from the values stored with every sample, not from the preset.

### Sanity check (an implementation bug, not the large-step error, if it fails)

- For each measured sample, run **one** solver substep live, in **float64**, with the solver's own substep dt,
  starting from the stored `T_in`, `mask_in`, and `Q`. The stored data is float16 and too coarse for this. The same
  substep in float32 is reported alongside (see below why it is not the gate).
- The mask is promoted above T_solidus before the substep, which is what the Triton kernel does inside it.
- Every variant is evaluated on that substep. The relative residual uses the same ratio of norms and the same
  denominator as above, over the whole interior, **without** the 50 K filter.
- **Bug gate:** `cp_in` with diffusion at `T_in` (the solver's own update rule) must stay below 1e-3 for every
  sample. If it does not, it is an implementation bug, and no measurement result is reported until it is fixed.
- **Eligibility (decided 2026-11-09, see below):** once the bug gate passes, all six variants are eligible. The
  decision rule above is unchanged.
- The one-substep value of every variant is reported alongside, as the error each form has over one substep.

### Why the 1e-3 bound is not applied to the other variants

The stop rule of the first version of this note triggered on synthetic test fields, before any dataset sample was
opened. Relative residual over one substep (`tests/physics/test_residual_check.py`, warm spot under a source):

| Peak T | enthalpy, explicit | enthalpy, implicit | enthalpy, trapezoid | cp_in, explicit | cp_in, implicit | cp_in, trapezoid |
| --- | --- | --- | --- | --- | --- | --- |
| 1500 K (no melting) | 1.8e-3 | 2.5e-2 | 1.2e-2 | 5.1e-6 | 2.5e-2 | 1.2e-2 |
| 2500 K | 2.5e-4 | 1.9e-1 | 9.6e-2 | 1.9e-6 | 1.9e-1 | 9.6e-2 |

- `implicit` and `trapezoid` cannot reach 1e-3 for reasons other than a bug. The solver takes each substep close to
  its stability limit, where the fastest modes of ∇·k∇T change by a large fraction within the substep, so evaluating
  it at the end instead of the start differs by O(1) for those modes.
- `enthalpy` with explicit diffusion misses it even without melting: with a sloped cp table, ΔH and cp(T_in)·ΔT
  differ by ½·cp′·ΔT².
- The original rule would have left only `cp_in` with explicit diffusion, which leaves out the latent heat in the
  melt regime that decides the outcome.
- The variants share everything the gate tests (k, mask, stencil, Q, units). They differ only in the storage form,
  which the `dH/dT = cp_eff` unit test checks, and in where the same operator is evaluated.

### Why the gate runs in float64 (decided 2026-11-09)

The first version said float32. On one real train sample (run_014, sample 25; gate only, no full-exposure number
computed), the solver form gave 5.55e-4 in float32 and 2.02e-12 in float64, with the same substep dt (8.46e-8 s). The
reference reproduces the substep to machine precision; the float32 value is rounding of `T_out` summed over about
8 million voxels that barely change in one substep. That is a factor of 2 below the bound, so a hotter sample could
fail the gate on rounding alone. Float32 was only meant to avoid the stored float16, and float64 serves that better.
The 1e-3 bound stays.

### Known solver limitation

The solver uses the apparent-heat-capacity method: each explicit substep multiplies `ΔT` by `cp_eff` at the start
temperature. Across the narrow latent-heat bump this is not energy-exact, so the solver gains or loses a little
latent heat on every substep in which a voxel crosses the melting range. The `enthalpy` residual over one substep
measures that error.

## Result (2026-11-09)

Measured with `experiments/measure_heat_residual.py --device cuda` at commit `9900766`, on 120 samples from the 24
train and val runs. The raw rows are in `artifacts/physics_residual/` (`gate.csv`, `exposures.csv`, `result.md`).
Regimes: 15 solid (14 SS316L, 1 IN718), 59 melt (25 IN718, 24 Ti64, 10 SS316L), 46 keyhole (16 SS316L, 16 Ti64,
14 IN718). No sample had an empty 50 K selection.

**Bug gate: passed.** The solver form in float64 stays at or below 4.5e-12 on every sample. In float32 the same
substep reaches up to 1.84e-3, and 7 of the 120 samples are above 1e-3: a float32 gate would have stopped the run on
rounding alone.

#### One-substep check (all samples, float64 unless marked)

| Variant | median | max |
| --- | --- | --- |
| enthalpy, explicit | 0.0614 | 0.863 |
| enthalpy, implicit | 0.291 | 0.883 |
| enthalpy, trapezoid | 0.18 | 0.873 |
| cp_in, explicit | 1.86e-12 | 4.52e-12 |
| cp_in, implicit | 0.156 | 0.691 |
| cp_in, trapezoid | 0.078 | 0.345 |
| cp_in, explicit, float32 | 6.5e-05 | 0.00184 |

#### Full exposure, |ΔT| > 50 K

| Regime | Variant | n | empty | median | p90 |
| --- | --- | --- | --- | --- | --- |
| keyhole | enthalpy, explicit | 46 | 0 | 2.81 | 4.91 |
| keyhole | enthalpy, implicit | 46 | 0 | 3.03 | 5.06 |
| keyhole | enthalpy, trapezoid | 46 | 0 | 2.06 | 3.56 |
| keyhole | cp_in, explicit | 46 | 0 | 3.03 | 4.89 |
| keyhole | cp_in, implicit | 46 | 0 | 3.16 | 5.05 |
| keyhole | cp_in, trapezoid | 46 | 0 | 2.13 | 3.54 |
| melt | enthalpy, explicit | 59 | 0 | 0.121 | 1.84 |
| melt | enthalpy, implicit | 59 | 0 | 0.627 | 1.84 |
| melt | enthalpy, trapezoid | 59 | 0 | 0.308 | 1.28 |
| melt | cp_in, explicit | 59 | 0 | 0.398 | 2.06 |
| melt | cp_in, implicit | 59 | 0 | 0.682 | 2.16 |
| melt | cp_in, trapezoid | 59 | 0 | 0.52 | 1.56 |
| solid | enthalpy, explicit | 15 | 0 | 0.0015 | 0.00205 |
| solid | enthalpy, implicit | 15 | 0 | 0.0027 | 0.00343 |
| solid | enthalpy, trapezoid | 15 | 0 | 0.00149 | 0.00192 |
| solid | cp_in, explicit | 15 | 0 | 0.0537 | 0.0721 |
| solid | cp_in, implicit | 15 | 0 | 0.0522 | 0.0706 |
| solid | cp_in, trapezoid | 15 | 0 | 0.0529 | 0.0713 |

**Decision variant:** enthalpy, explicit, melt median 0.121. **Band:** 0.1 to 0.5.

### Decision (pre-registered rule)

- **Decision variant:** storage `enthalpy`, diffusion `explicit`. Melt median 0.121, p90 1.84.
- **Band: 0.1 to 0.5.** The physics-loss variants are trained with ReLoBRaLo and the physics weight capped at the
  data-loss weight. The report states the residual size.
- **What the picked form constrains.** With diffusion evaluated at `T_in`, the prediction enters the residual only
  through `H(T_pred)` in each voxel: `R = ρ·(H(T_pred) − H(T_in))/dt − ∇·(k∇T_in) − Q`. The loss constrains the
  per-voxel energy balance of the prediction and has no spatial coupling, so it cannot smooth the prediction.
- **Two forms are kept for the ablation (decided 2026-11-09).** The measurement picked a different form than the old
  loss used: `physics_heat_residual` in `fm_physics.py` and `triton_pde_loss.py` evaluate ∇·(k∇T) at the prediction.
  Both forms are trained, behind one argument `--physics_form`:
  - `coupled`: enthalpy storage, diffusion at the prediction. The corrected successor of the old v4/v5 loss.
  - `tin`: enthalpy storage, diffusion at `T_in`. The form the measurement picked.

### Observations (exploratory, not part of the pre-registered decision)

- **Solid:** the enthalpy forms stay near 0.0015. Without melting, the one-exposure target is nearly consistent with
  the PDE. `cp_in` sits near 0.05 there, from the sloped cp table over a large ΔT.
- **Melt:** the median is near the lower band edge, but the tail is heavy (p90 1.84). The residual is small on some
  melt samples and larger than the stored energy on others.
- **Keyhole:** the median is 2 to 3 for every variant. The residual exceeds the stored energy, so the one-exposure
  target is inconsistent with the PDE there. The solver has no evaporation, so these samples are already outside its
  validity.
- **The solver's own latent-heat error is large.** Over one substep, `enthalpy` with explicit diffusion differs from
  the solver's update by a median of 0.06 and up to 0.86 of the enthalpy change. Voxels that heat through the narrow
  latent-heat bump within one substep do not absorb its latent heat correctly. So part of the melt-regime residual
  is the solver's own energy error, not the large-step error. Whether the solver gains or loses energy, and how much
  that changes melt depth and T_max, is not measured here. It belongs to the Phase B solver work (brainstorming 2.9).
