#!/usr/bin/env bash
# UQ pipeline entry point. See docs/uq.md. Options are passed to the Python driver:
#   --resume (default)  --smoke  --dry-run  --status  --stop  --allow-reeval
#   --from-stage Sx  --only-stage Sx  --force-stage Sx
#   --stop-after-stage Sx  --stop-at-time [YYYY-MM-DDT]HH:MM
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# experiments/uq_conformal.py sets HDF5_USE_FILE_LOCKING and PYTHONUNBUFFERED itself.
exec uv run python experiments/uq_conformal.py "$@"
