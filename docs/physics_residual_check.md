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
- Samples come from the **train** runs of the manifest only. Cal and test runs are not opened.

### Sanity check (an implementation bug, not the large-step error, if it fails)

- For each measured sample, run **one** solver substep live, in float32, with the solver's own substep dt, starting
  from the stored `T_in`, `mask_in`, and `Q`. The stored data is float16 and too coarse for this.
- The mask is promoted above T_solidus before the substep, which is what the Triton kernel does inside it.
- Every variant is evaluated on that substep. The relative residual uses the same ratio of norms and the same
  denominator as above, over the whole interior, **without** the 50 K filter.
- **Pass/fail:** `cp_in` with diffusion at `T_in` (the solver's own update rule) must stay below 1e-3 for every
  sample. If it does not, it is an implementation bug, and no measurement result is reported until it is fixed.
- The other variants get the same 1e-3 bound for eligibility. The `enthalpy` value over one substep is reported
  alongside; it measures the solver's own latent-heat error per substep (see below).
- **Stop rule:** if a variant fails only for a reason that is not a bug (the solver's per-substep latent-heat error
  for `enthalpy`, or the O(dt) difference of `implicit` and `trapezoid` over one substep), the measurement stops and
  the eligibility rule is decided again before any full-exposure number is looked at.

### Known solver limitation

The solver uses the apparent-heat-capacity method: each explicit substep multiplies `ΔT` by `cp_eff` at the start
temperature. Across the narrow latent-heat bump this is not energy-exact, so the solver gains or loses a little
latent heat on every substep in which a voxel crosses the melting range. The `enthalpy` residual over one substep
measures that error.

## Result

Not measured yet.
