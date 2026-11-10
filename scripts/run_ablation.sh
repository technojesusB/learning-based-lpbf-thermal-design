#!/usr/bin/env bash
# Ablation queue entry point. See docs/ablation.md. Options are passed to the Python runner:
#   (none: run or continue the queue)  --only NAME  --status  --stop  --dry-run  --preflight
#   --root  --checkpoint_root  --mlflow_uri  --uq_root  --manifest  --num_workers
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# experiments/run_ablation.py sets HDF5_USE_FILE_LOCKING and PYTHONUNBUFFERED itself.
exec uv run python experiments/run_ablation.py "$@"
