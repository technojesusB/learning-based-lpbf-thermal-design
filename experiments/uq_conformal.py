"""Conformal UQ experiment: data generation, run-level split, training, K-draw sampling, report.

The stages, their markers, and every flag live in neural_pbf.uq.pipeline; docs/uq.md explains them.

    uv run python experiments/uq_conformal.py --smoke     # whole chain on a tiny grid
    uv run python experiments/uq_conformal.py             # full run, or resume
    uv run python experiments/uq_conformal.py --status
    uv run python experiments/uq_conformal.py --stop-after-stage S5 --stop-at-time 08:00
"""

from __future__ import annotations

import os
import sys

# Before h5py loads: HDF5 file locks raise BlockingIOError (errno 11) here, see docs/offline_dataset_generation.md.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")  # stage children inherit it, so their logs stream

from neural_pbf.uq.pipeline import main  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    sys.exit(main(sys.argv[1:]))
