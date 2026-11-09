# Uncertainty quantification

Prediction intervals for scalar melt-pool quantities (depth, T_max, volume) from the
flow-matching surrogate, calibrated with split conformal prediction and evaluated on
held-out simulation runs. It is developed and evaluated on the Baseline (`VelocityNet`).

**Measured results are only in the generated [report](uq_report.md).** This page holds
no result numbers. If the report is missing, the full pipeline has not been run yet.

## Input causality audit

A surrogate input must exist before the step it predicts. Checked against
`_do_run` in `experiments/generate_offline_dataset.py`. Tests:
`tests/scripts/test_generator_causality.py`.

| Input | Source in the generator | Available before the predicted step |
| --- | --- | --- |
| Noise | Sampled at inference, independent of the data | Yes |
| `mask_in` | `state.material_mask.clone()`, taken before the step runs | Yes |
| `mask` (legacy only) | `state_next.material_mask`, the state after the step | **No** |
| `T_in` | `state.T.clone()`, taken before the step runs | Yes |
| `Q` | `beam.intensity(...)` at the planned laser position, before the step | Yes |
| Conditioning: power, sigma, exposure_time, material properties | Fixed per run by `prepare_trajectory` | Yes |
| Conditioning: x, y | `plan.path_points`, computed before the time loop | Yes |

`T_target` and the legacy `mask` are the only stored arrays derived from `state_next`.
Since run 2 the model gets `T_in` as a fourth input channel and `exposure_time` as conditioning: run 1
lacked both, so the model saw neither the heat of earlier tracks nor the deposited energy
(power x exposure time). Run 1 stays in the report appendix.

The tests check that `mask_in` is a subset of the post-step mask, that `mask_in` is empty at
a cold start, and that nothing except targets and diagnostics derives from `state_next`.

**Every legacy result uses the post-step mask**, so part of the answer was in the input.
Old checkpoints cannot be corrected. They need retraining on data with `mask_in`.

## Method

- **Data:** new runs in a material x pattern factorial design, with a seed range that
  does not collide with legacy data. Legacy data has no `mask_in` and is used only for the
  leakage demonstration.
- **Split:** by run, stratified by material, into train, val, calibration, and test. Runs
  sharing a scan stay together. The manifest is hashed and stored in the checkpoint.
- **Ensemble:** K noise draws per input through `run_euler_rollout`, with a `torch.Generator`
  on both noise paths. Default behavior without a generator is unchanged.
- **Scalars:** computed per draw, using each sample's own T_l rather than the fixed 1600 K
  of the legacy benchmark.

| Interval method | Definition |
| --- | --- |
| `absolute` (primary) | Conformal quantile of the ensemble-mean residual; constant width |
| `normalized` | Conformal quantile of residual / (ensemble sigma + eps) |
| `constant_single_draw` | Constant-width baseline: conformal on one draw, no ensemble |
| `raw_ensemble` | Uncalibrated mean +- z sigma, reference only |

`absolute` is itself constant-width, so the separate baseline uses a single draw. That
shows what the ensemble adds.

**Coverage intervals:** Wilson (assumes independent samples) and a run-level bootstrap.
Samples within a run are correlated, so prefer the bootstrap. Conformal's guarantee needs
exchangeability, which holds only loosely within runs.

**Run-level support:** a finite run-level quantile at level alpha needs
`ceil((n+1)(1-alpha)) <= n` calibration runs: 4 for 0.2, 9 for 0.1, 19 for 0.05. The
planned 12 calibration runs do not support alpha = 0.05, so the report marks it
`not supported at run level`.

## Deviations from the legacy Baseline training

| Item | Legacy (`train_fm_patches.py`) | This pipeline | Reason |
| --- | --- | --- | --- |
| Mask input | Post-step `mask` | `mask_in` | Remove the leak |
| Split | Random by sample, 3 runs | By run, about 45 runs | Honest held-out data |
| Stopping | Fixed 100 epochs | At most 100, with patience in `TrainConfig` | Requested |
| Validation noise | Fresh each epoch | Fixed seed | Epochs compare on equal terms |
| Stored `Q` | Float32 after a repair | Float32 at write time | float16 overflows to inf |
| Inputs | `[x_tau, mask, Q]`, 12 conditioning values | `[x_tau, mask_in, Q, T_in]`, plus `exposure_time` (run 2) | Run 1 showed both were missing |

Architecture, learning rate, batch size, optimizer, and loss are otherwise unchanged.

## Exploratory: coverage by thermal regime

Added after the first test evaluation, so it is reported as exploratory. Each sample falls in a regime by
T_max: `solid` (below its solidus), `melt` (up to the boiling point), or `keyhole` (above it; the solver has no
evaporation, so these temperatures are unphysical). Every regime gets its own conformal calibration.
Regimes from the predicted T_max are usable at prediction time; regimes from the true T_max are a diagnostic.
Boiling points (`ExploratoryConfig.T_boil_K`, model values from LPBF simulation literature): SS316L 3090 K,
IN718 3188 K, Ti64 3315 K. The usefulness criterion was fixed before run 2: in the melt regime at the primary
alpha, coverage of at least 0.85, depth width below 25 voxels, and T_max width below 1000 K.

## Legacy Baseline provenance gap

`checkpoints/fm/best.pt` was trained by `experiments/train_fm_patches.py` on
`offline_dataset_notebook.h5` (3 runs). MLflow names commit `f7add59`, but the script is
not in that commit, so the exact training code cannot be recovered. The stored data also
has a `Q_saturated` dataset that no committed generator version writes.

## Leakage controls

- `LEAKAGE GUARD`: every result is labelled `run-disjoint: yes/no/unknown`, from the
  checkpoint's manifest hash or, for old checkpoints, from actual scan overlap with
  `dataset_cfg`. Models with different labels are never ranked together.
- Pre-registered K, alpha, primary method, and seeds are frozen and hashed before test access.
- Test access after the split is logged: the conditioning check in S3 (attributes only),
  sampling in S5, and evaluation in S6. S2 reads every run before any test set exists, to
  check data integrity. A second evaluation needs `--allow-reeval` and is stated in the report.
- Checkpoint selection uses val only. In manifest mode, training cannot load cal or test runs.
- Normalization constants are fixed defaults, not fitted on any data.
- The old benchmark test reconstruction (hard-coded seed 42) is not used.
- Leakage demonstration: the old checkpoint on its own random-split held-out samples and
  on the new test runs. Its small held-out n is an illustration, not a measurement.

## Running it

```bash
uv run python experiments/uq_conformal.py --smoke   # same entry point; the shell script wraps it
scripts/run_uq_pipeline.sh --smoke      # whole chain on a tiny grid; writes to artifacts/uq_smoke
scripts/run_uq_pipeline.sh              # full run, or resume; finished stages are skipped
scripts/run_uq_pipeline.sh --status     # progress, ETA, heartbeat
scripts/run_uq_pipeline.sh --stop       # stop now; the stage in progress is cut off, finished work is kept
scripts/run_uq_pipeline.sh --dry-run    # what would run
```

Other options: `--from-stage`, `--only-stage`, `--force-stage`, `--allow-reeval`, `--render-figures`,
`--stop-after-stage`, `--stop-at-time`.

### Pausing

```bash
scripts/run_uq_pipeline.sh --stop-after-stage S2              # run S0 to S2, then pause
scripts/run_uq_pipeline.sh --stop-at-time 10:30               # today; a past time is refused
scripts/run_uq_pipeline.sh --stop-at-time 2026-11-03T10:30    # local time with a date
scripts/run_uq_pipeline.sh                                    # continue; --from-stage S3 also works
```

A pause exits 0. `--status` and `status.json` show `paused: --stop-after-stage S2` or
`paused: --stop-at-time ...`, the next stage, and the command to continue. That command is also in
`logs/uq/driver.log`. Done-markers stay valid, so finished stages are not rerun.

`--stop-at-time` never kills work. It takes effect at the first safe boundary after the given time,
so the actual stop can be later by up to the length of the unit in progress:

| Stage | Stops | Continues with |
| --- | --- | --- |
| S1 | After the current run is finished and validated | The missing runs |
| S4 | After the current epoch, once `latest.pt` is saved | The next epoch, with the same shuffle order |
| S5 | After the current chunk | The missing chunks |
| S0, S2, S3, S6, S7 | Run to the end, then before the next stage | The next stage |

If both options are given, whichever comes first wins and the status names it. A deadline reached
during the last stage, or with only finished stages left, is a normal finish. Neither option can be
combined with `--stop` or `--render-figures`, and `--stop-after-stage` cannot be combined with
`--only-stage` or name a stage before `--from-stage`. `--dry-run` prints the pause point. A time
inside a daylight-saving change is refused.

Use the printed continue command. It keeps `--force-stage` only for stages that have not started,
because repeating it for a stage in progress wipes its saved S4 or S5 progress. After a pause, stages
from the pause point on may rerun because of a rerun before it, as in an uninterrupted chain; earlier
stages are still refused if stale. After `--stop` or a crash, such a rerun is refused instead.

`--stop` is separate. It kills the running stage child and exits 130, and it is also honored when
the signal arrives between stages.

### What a stage hashes

A stage is skipped only if its inputs still hash the same. Inputs are the committed state of these
paths, the named config sections, and the output hash of its upstream stages.

| Stage | Code paths | Config | Upstream |
| --- | --- | --- | --- |
| S0 | `uv.lock` | None | None |
| S1 | `experiments/generate_offline_dataset.py`, `src/neural_pbf/physics/material.py`, `src/neural_pbf/physics/ops.py`, `src/neural_pbf/physics/triton_ops.py` (solver only; the loss files are not hashed), `src/neural_pbf/integrator/stepper.py`, `src/neural_pbf/scan`, `src/neural_pbf/pipelines`, `src/neural_pbf/core`, `src/neural_pbf/data/sampling.py` | gen | S0 |
| S2 | `src/neural_pbf/uq/validate.py` | gen | S1 |
| S3 | `src/neural_pbf/data/splits.py`, `src/neural_pbf/uq/guard.py` | split, prereg | S2 |
| S4 | `src/neural_pbf/models/generative/fm`, `src/neural_pbf/data/fm_dataset.py`, `src/neural_pbf/data/patch_dataset.py`, `src/neural_pbf/uq/train.py`, `src/neural_pbf/training/factories.py` | train | S3 |
| S5 | `src/neural_pbf/models/generative/fm`, `src/neural_pbf/data/fm_dataset.py`, `src/neural_pbf/data/patch_dataset.py`, `src/neural_pbf/uq/train.py`, `src/neural_pbf/training/factories.py`, `src/neural_pbf/uq/harness.py`, `src/neural_pbf/uq/sampling.py`, `src/neural_pbf/eval/benchmark`, `src/neural_pbf/eval/metrics/geometry.py`, `src/neural_pbf/integrator/fm_stepper.py` | prereg, run: chunk_size, legacy_sample_limit | S3, S4 |
| S6 | `src/neural_pbf/uq/analysis.py`, `src/neural_pbf/uq/conformal.py`, `src/neural_pbf/uq/report.py`, `src/neural_pbf/uq/regimes.py` | prereg, exploratory | S3, S5 |
| S7 | None | None | S6 |

Figure code, `uq/pipeline.py`, `uq/stages.py`, and `uq/config.py` are not hashed; config values are.
`--dry-run` lists stale stages as `BLOCKED` and names what changed: each changed file, config field,
or upstream output.

**What can change during a pause.** Commit first: a start with uncommitted changes under `src`,
`experiments`, `scripts/run_uq_pipeline.sh`, `pyproject.toml`, or `uv.lock` is refused.

- Safe: `docs/`, `tests/`, and figure code. Paths in the table for stages that have not started are
  also accepted, and those stages then use the new code.
- Refused as stale: anything in the table for a finished or partly finished stage. Changing
  `uv.lock` makes S0 stale, and physics code makes S1 stale, which would mean regenerating data.
- Do not change: `uq/stages.py`, `uq/pipeline.py`, and `uq/config.py` defaults that are not hashed.
  Stage logic such as the sampling job list lives in `uq/stages.py`, and a change there goes
  **undetected**, so results could mix two versions.

`--render-figures` redraws all figures from the tables S6 saved (`coverage_rows.csv`,
`sample_points.csv`, `sensitivity_rows.csv`, `point_errors.csv`). It loads no model, uses no GPU,
and does not count as a test evaluation, so it needs no `--allow-reeval`. Figure code is outside
the S6 code hash, so restyling a figure does not make S6 stale, but figure file names must not
change after a run because the report links to them. It is refused while a driver is running and
warns on uncommitted pipeline code. Figure specs are in [plot_specs.md](plot_specs.md).

| Stage | Purpose |
| --- | --- |
| S0 | Preflight: branch, clean tree, GPU, disk, dependencies, quick tests |
| S1 | Generate runs: one process per run, atomic, deep-checked as each finishes |
| S2 | Validate: schema, NaN/inf, counts, unique parameters, `mask_in` subset of `mask` |
| S3 | Split manifest and pre-registration freeze |
| S4 | Train the Baseline: resumable, best on val |
| S5 | K-draw sampling: atomic chunks, resumable |
| S6 | Conformal, evaluation, saved tables, figures, report, MLflow |
| S7 | Summary with the timeline actually used |

A figure that fails to render does not fail S6. The report and the S7 summary name it and give
the `--render-figures` command to run. An MLflow failure does not fail S6 either; the S7 summary
states it.

Exit codes: 0 ok or paused, 1 failed, 2 refused, 3 GPU lost, 130 stopped. Logs are under `logs/uq/`
and status is in `artifacts/uq/status.json`.

**Reproducibility.** Two identical smoke runs produce byte-identical run data, S0 to S3 outputs,
manifest, pre-registration, and legacy-model draws and table rows. GPU training (S4) is not bitwise
reproducible because no deterministic mode is enabled, so `best.pt`, the new model's draws, and its
report numbers differ slightly between runs. A paused and continued chain is therefore compared with
a normal one only on the reproducible parts. Timeline seconds also differ by nature.

A stage whose inputs changed is refused, never rerun silently. `--force-stage S1` never
deletes generated runs, because that data is expensive.

## Limitations

- Results are on simulated data only, with no real-world validation.
- There is a small number of runs, so run-level intervals are wide.
- Conformal validity under within-run correlation is approximate.
- Depth is capped by the domain height, and zero-depth samples make coverage easy. The
  report shows these strata separately.
- Volume is capped by the 64-voxel patch, so a wider pool is truncated and its volume is a
  lower bound. The report shows zero-volume and patch-limited samples separately.
- The simulator has no vaporization limit, so T_max can reach unphysical values.
- Results apply to the Baseline only. Other models need retraining on `mask_in` data.
- `load_model_adaptive` currently cannot load `checkpoints/dit`, `dit_physics`, or
  `dit_accelerate`.
