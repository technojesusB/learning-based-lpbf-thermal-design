"""Ablation queue runner: train the queued ablation experiments one after another, pausable and resumable.

Options and the worktree setup are in docs/ablation.md and neural_pbf.training.ablation_queue.

    uv run python experiments/run_ablation.py --dry-run
    uv run python experiments/run_ablation.py --preflight
    uv run python experiments/run_ablation.py               # the queue, or continue it
    uv run python experiments/run_ablation.py --only dit_v3_rope
    uv run python experiments/run_ablation.py --status
    uv run python experiments/run_ablation.py --stop
"""

from __future__ import annotations

import os
import sys

# Before h5py loads: HDF5 file locks raise BlockingIOError (errno 11) here, see docs/offline_dataset_generation.md.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # the children inherit it, so their logs stream

from neural_pbf.training.ablation_queue import main  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    sys.exit(main(sys.argv[1:]))
