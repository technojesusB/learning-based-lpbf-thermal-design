"""Pausing and resuming a training run at an epoch boundary (the pattern of ``uq/train.py``, which is S4-hashed).

``latest.pt`` holds everything an uninterrupted run would still have in memory after an epoch: weights, optimizer,
scheduler, grad scaler, ReLoBRaLo weights, every RNG, the step counter and the history. The train order of epoch
``e`` comes from ``epoch_seed(seed, e)`` alone, so a resumed run sees the same batches as an uninterrupted one.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import random
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch

STOP_FILE = "STOP"
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_PR_SET_PDEATHSIG = 1


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def remove_stale_tmp(ckpt_dir: Path) -> list[str]:
    """Temporary files a SIGKILL left during a save; the checkpoints themselves are always whole."""
    stale = sorted(ckpt_dir.glob("*.tmp"))
    for path in stale:
        path.unlink(missing_ok=True)
    return [p.name for p in stale]


def epoch_seed(base_seed: int, epoch: int) -> int:
    return int(np.random.SeedSequence([base_seed, epoch]).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    """Python's random (ReLoBRaLo's lookback), NumPy and torch on CPU and all CUDA devices; same on every process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds every CUDA device too


def rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),  # ReLoBRaLo draws its lookback from here
    }


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def ignore_stop_signals(_worker_id: int, parent: int | None = None) -> None:
    """DataLoader ``worker_init_fn`` (bind ``parent=os.getpid()`` before the fork).

    Ctrl-C sends SIGINT to the whole process group, and the queue runner's stop (shared with the UQ pipeline) sends
    SIGTERM to it; only the main process acts on them. So a worker leads its own process group, which no group
    signal reaches, and ignores SIGINT as well. SIGTERM keeps its default action: torch's shutdown ends a worker that
    does not exit by itself with SIGTERM to its pid and then waits for it, so an ignored SIGTERM hangs a run's end.

    A worker that ignores SIGINT must still die with its main process (killed by a second signal): on Linux the
    kernel sends it SIGKILL then. The main process can die between the fork and this call, before the kernel knows;
    then the parent is no longer ``parent`` and the worker exits at once. Without both, orphaned workers were seen
    to outlive the main process for good.
    """
    os.setpgid(0, 0)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sys.platform.startswith("linux"):
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    if parent is not None and os.getppid() != parent:
        os._exit(0)


class PauseRequest:
    """A stop file in the run's checkpoint directory, or SIGINT/SIGTERM, asks the run to pause after the epoch.

    Under ``watch_signals`` the first signal is a pause request; a second one kills the process at once, with the
    signal's default action, which loses at most the running epoch. Raising from the handler instead is not safe: the
    exception lands anywhere, e.g. in logging's at-fork hook, where it is swallowed and leaves a lock held.
    """

    def __init__(self, ckpt_dir: str | Path) -> None:
        self.path = Path(ckpt_dir) / STOP_FILE
        self._signalled = False

    def requested(self) -> bool:
        return self._signalled or self.path.exists()

    def clear(self) -> None:
        self._signalled = False
        self.path.unlink(missing_ok=True)

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        name = signal.Signals(signum).name
        if self._signalled:
            os.write(2, f"\n{name} again: stopping now; latest.pt holds the last finished epoch\n".encode())
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        self._signalled = True
        os.write(2, f"\n{name}: pausing after the current epoch (send it again to stop at once)\n".encode())

    @contextlib.contextmanager
    def watch_signals(self) -> Iterator[None]:
        previous = {sig: signal.getsignal(sig) for sig in STOP_SIGNALS}
        for sig in STOP_SIGNALS:
            signal.signal(sig, self._handle)
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
