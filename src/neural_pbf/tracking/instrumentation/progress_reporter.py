"""tqdm progress bar with current-vs-previous metric postfix."""

from __future__ import annotations

from tqdm import tqdm


class ProgressReporter:
    """Wraps tqdm and renders a current+prev postfix for key metrics."""

    _TRACKED = ("Tmax", "t_sim", "gpu_temp")

    def __init__(self, total: int, desc: str = "Simulating") -> None:
        self._pbar = tqdm(total=total, desc=desc, unit="step")
        self._prev: dict[str, float] = {}

    def update(self, metrics: dict[str, float | None]) -> None:
        """Advance bar by one step and refresh the postfix."""
        postfix: dict[str, str] = {}
        for key in self._TRACKED:
            cur = metrics.get(key)
            if cur is None:
                continue
            prev = self._prev.get(key)
            if prev is not None:
                postfix[key] = f"{cur:.0f} ({prev:.0f})"
            else:
                postfix[key] = f"{cur:.0f}"
        if postfix:
            self._pbar.set_postfix(postfix)
        self._pbar.update(1)
        for key in self._TRACKED:
            val = metrics.get(key)
            if val is not None:
                self._prev[key] = val

    def close(self) -> None:
        self._pbar.close()

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
