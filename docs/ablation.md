# Ablation queue

The experiment scripts (`experiments/train_fm_*.py`) are standalone training entry points: run on their own they use
their defaults, take every value from the command line and log to the MLflow experiment `lpbf_experiments`, unpinned.
The ablation is a layer on top: `training/ablation.py` defines each ablation experiment as a script plus a pinned
recipe, physics form and backend and a chapter, and the queue runner calls the scripts with exactly those arguments.
Dependencies run one way only (ablation layer → scripts → shared training run); a test enforces it.

## The queue

`QUEUE` in `training/ablation.py`, in this order: `unet_none`, `unet_physics_coupled`, `unet_physics_tin`,
`dit_v5_none_triton`, `dit_v5_coupled_triton`, `dit_v5_tin_triton`, `dit_v6_overlap_coupled_triton`, `dit_v3_rope`,
`dit_v2_sincos`, `dit_v4_coupled_reference`, `dit_v1_ape`. `unet_baseline` is the UQ pipeline's S4 model (run 2) and
is not retrained. Each step of the chain changes one thing, except the named steps `scale_up` (v3 → v5) and `recipe`
(UQ baseline → `unet_none`); the module docstring has the chain and why. Every experiment trains 100 epochs: on 1040
train samples that is about twice the sample-steps of the old 500-epoch runs on 105 samples. The historical recipes are
recorded in each experiment's `note` as "historical recipe, not used".

```bash
scripts/run_ablation.sh --dry-run      # state of every experiment; every argv validated against its recipe
scripts/run_ablation.sh --preflight    # one short GPU run per experiment: OK/failed, peak VRAM, min/epoch, total
scripts/run_ablation.sh                # run the queue, or continue it
scripts/run_ablation.sh --only dit_v3_rope
scripts/run_ablation.sh --status       # current experiment, epoch, ETA, finished ones
scripts/run_ablation.sh --stop         # pause after the current epoch; the same start command continues
```

The runner takes infrastructure options only (`--root`, `--checkpoint_root`, `--mlflow_uri`, `--uq_root`,
`--manifest`, `--num_workers`), never a recipe value, so the pinned recipe cannot be bypassed through it. Each run is
pinned: MLflow experiment `lpbf_ablation_rerun`, the experiment name as run name, provenance key and tag
`ablation_experiment`. The UQ ablation evaluation refuses checkpoints with `pinned == false`, so a standalone run never
slips into the study. Logs: one file per experiment and attempt under `logs/ablation/`, plus `status.json`.

## Pausing

- `--stop`, a single Ctrl-C in the runner's terminal, or a `STOP` file in the experiment's checkpoint directory: the
  run finishes the current epoch (about 3 min with 4 workers), writes `latest.pt` and exits 75; the runner marks the
  experiment paused and exits 0. Starting the same command again resumes it into the same MLflow run.
- A second Ctrl-C kills at once; at most the running epoch is lost. `latest.pt` and `best.pt` are written to a
  temporary file and renamed, so a kill never leaves a half-written checkpoint.
- A paused run resumes bitwise on the CPU in fp32. On the GPU with bf16 and the Triton atomics, the same seed means the
  same configuration and the same data order, not bitwise-identical weights; that holds for two uninterrupted runs too.
- Step metrics a killed run logged after its last saved epoch are logged again on resume, so MLflow shows two values
  at those steps. Epoch metrics are logged once per epoch.

## Running from a worktree (frozen code)

The queue runs for days while development continues in the main checkout. The checkpoints record the git commit, and a
pinned resume refuses another commit, so the queue runs from a dedicated worktree at a fixed commit:

```bash
git worktree add ../lpbf-ablation-wt <commit>
cd ../lpbf-ablation-wt
uv sync                                   # its own virtualenv: the main repo's would import the main checkout's code
MAIN=$HOME/Projekte/learning-based-lpbf-thermal-design
mkdir -p "$MAIN/logs/ablation"
nohup scripts/run_ablation.sh \
    --root "$MAIN/logs/ablation" --checkpoint_root "$MAIN/checkpoints" \
    --mlflow_uri "sqlite:///$MAIN/mlflow.db" --uq_root "$MAIN/artifacts/uq" \
    --manifest "$MAIN/artifacts/uq/manifest.json" > "$MAIN/logs/ablation/queue.out" 2>&1 &
```

Pass the same options to `--status` and `--stop`. Every path the runner and the training runs write (checkpoints,
`progress.json`, logs, queue state, the MLflow store) then lives in the main repository; the S1 dataset marker is read
from `--uq_root`. The runner creates the MLflow experiment with its artifact location next to the database, since a
sqlite store would otherwise fix it relative to the worktree. The runner refuses to start with uncommitted code, or
with `neural_pbf` imported from another checkout than the current directory's.
