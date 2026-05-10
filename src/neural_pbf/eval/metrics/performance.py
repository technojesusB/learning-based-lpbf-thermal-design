"""Performance profiling helpers: latency, VRAM, speedup."""
from __future__ import annotations

import time


class LatencyTimer:
    """Context manager that measures wall-clock latency of a code block.

    Synchronises CUDA before/after measurement when a GPU is available so that
    async kernel launches are fully counted.

    Usage::

        with LatencyTimer() as t:
            model(x)
        logging.info("elapsed: %s s", t.elapsed_s)
    """

    elapsed_s: float = 0.0

    def __enter__(self) -> LatencyTimer:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        self.elapsed_s = time.perf_counter() - self._t0


def vram_peak_bytes() -> int:
    """Return peak CUDA memory allocated since last reset [bytes]."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    return 0


def reset_vram_peak() -> None:
    """Reset the CUDA peak memory counter."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def speedup_ratio(baseline_s: float, candidate_s: float) -> float:
    """Return ``baseline_s / candidate_s``.

    Returns ``inf`` when *candidate_s* is zero to avoid division by zero.
    """
    if candidate_s <= 0.0:
        return float("inf")
    return baseline_s / candidate_s
