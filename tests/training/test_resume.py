"""Resumable training runs: per-epoch data order, RNG state, the stop file and the atomic latest.pt."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_pbf.training.checkpointing import atomic_torch_save
from neural_pbf.training.resume import PauseRequest, epoch_seed, rng_state, seed_everything, set_rng_state


@pytest.mark.unit
def test_the_epoch_seed_depends_only_on_seed_and_epoch() -> None:
    assert epoch_seed(42, 3) == epoch_seed(42, 3)
    assert len({epoch_seed(42, e) for e in range(50)}) == 50
    assert epoch_seed(42, 3) != epoch_seed(43, 3)


@pytest.mark.unit
def test_the_rng_state_restores_every_generator() -> None:
    state = rng_state()
    first = (torch.rand(3), np.random.rand(3), random.random())
    set_rng_state(state)
    second = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(first[0], second[0]) and np.array_equal(first[1], second[1]) and first[2] == second[2]


@pytest.mark.unit
def test_seed_everything_fixes_python_numpy_and_torch() -> None:
    draws = []
    for _ in range(2):
        seed_everything(42)
        draws.append((torch.rand(3), np.random.rand(3), random.random()))
    (t1, n1, p1), (t2, n2, p2) = draws
    assert torch.equal(t1, t2) and np.array_equal(n1, n2) and p1 == p2
    seed_everything(7)
    assert random.random() != p1


@pytest.mark.unit
def test_the_stop_file_is_a_pause_request_until_cleared(tmp_path: Path) -> None:
    pause = PauseRequest(tmp_path)
    assert not pause.requested()
    (tmp_path / "STOP").touch()
    assert pause.requested()
    pause.clear()
    assert not pause.requested() and not (tmp_path / "STOP").exists()


@pytest.mark.unit
def test_atomic_save_replaces_the_file_and_leaves_no_temp(tmp_path: Path) -> None:
    path = tmp_path / "latest.pt"
    atomic_torch_save({"epoch": 0}, path)
    atomic_torch_save({"epoch": 1}, path)
    assert torch.load(path)["epoch"] == 1
    assert [p.name for p in tmp_path.iterdir()] == ["latest.pt"]



@pytest.mark.unit
def test_a_signal_is_a_pause_request(tmp_path: Path) -> None:
    import os
    import signal

    pause = PauseRequest(tmp_path)
    before = signal.getsignal(signal.SIGINT)
    with pause.watch_signals():
        os.kill(os.getpid(), signal.SIGTERM)
        assert pause.requested()
    assert signal.getsignal(signal.SIGINT) is before  # handlers are restored
    pause.clear()
    assert not pause.requested()


@pytest.mark.unit
@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_a_worker_dies_with_its_main_process() -> None:
    import ctypes
    import multiprocessing as mp
    import os
    import signal

    from neural_pbf.training.resume import ignore_stop_signals

    def _child(q) -> None:
        ignore_stop_signals(0)
        sig = ctypes.c_int()
        ctypes.CDLL(None).prctl(2, ctypes.byref(sig))  # PR_GET_PDEATHSIG
        q.put((sig.value, signal.getsignal(signal.SIGINT) is signal.SIG_IGN,
           signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, os.getpgid(0) == os.getpid()))

    q = mp.get_context("fork").Queue()
    p = mp.get_context("fork").Process(target=_child, args=(q,))
    p.start()
    # SIGTERM stays (torch's shutdown terminates with it); the worker leads its own process group, so a signal to the
    # main process's group (Ctrl-C, the queue's stop) never reaches it
    assert q.get(timeout=10) == (signal.SIGKILL, True, True, True)
    p.join(timeout=10)


@pytest.mark.unit
@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_a_worker_whose_main_process_died_before_its_init_exits() -> None:
    """The main process can die between the fork and the worker's prctl; the worker must not live on."""
    import multiprocessing as mp

    from neural_pbf.training.resume import ignore_stop_signals

    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _child(q) -> None:
        ignore_stop_signals(0, parent=1)  # a parent pid that is not ours: the main process is gone
        q.put("still alive")

    p = ctx.Process(target=_child, args=(q,))
    p.start()
    p.join(timeout=10)
    assert p.exitcode == 0 and q.empty()


@pytest.mark.unit
@pytest.mark.parametrize("where", ["before_replace", "mid_write"])
def test_a_save_that_dies_leaves_the_previous_latest_intact(tmp_path: Path, monkeypatch, where: str) -> None:
    """A hard kill during a save: the old latest.pt still loads, so the run resumes from the last finished epoch."""
    import os

    path = tmp_path / "latest.pt"
    atomic_torch_save({"epoch": 3, "w": torch.arange(4.0)}, path)

    def _die(*args, **kwargs):
        raise KeyboardInterrupt("killed during the save")

    if where == "before_replace":
        monkeypatch.setattr(os, "replace", _die)
    else:
        real_save = torch.save

        def _half(obj, f, *a, **kw):
            real_save({"partial": True}, f)
            with open(f, "r+b") as fh:
                fh.truncate(10)  # a torn write
            raise KeyboardInterrupt("killed during the save")

        monkeypatch.setattr(torch, "save", _half)
    with pytest.raises(KeyboardInterrupt):
        atomic_torch_save({"epoch": 4, "w": torch.zeros(4)}, path)
    monkeypatch.undo()
    state = torch.load(path)
    assert state["epoch"] == 3 and torch.equal(state["w"], torch.arange(4.0))
    assert [p.name for p in tmp_path.iterdir()] == ["latest.pt"]
