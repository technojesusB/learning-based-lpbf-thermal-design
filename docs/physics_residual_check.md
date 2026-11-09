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
- **Samples:** indices 0, 12, 25, 37 and 49 of every **train** and **val** run of the UQ manifest
  (`artifacts/uq/manifest.json`), so 120 samples from 24 runs. Cal and test runs belong to the UQ pipeline and its
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

## Result

Not measured yet.
