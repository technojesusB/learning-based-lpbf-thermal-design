"""Crash-dump flight recorder: rolling step buffer + env snapshot → JSON."""

from __future__ import annotations

import json
import logging
import platform
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _collect_env() -> dict[str, Any]:
    """Capture environment metadata that is safe to call at run start."""
    env: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "uname": list(platform.uname()),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        t_v = getattr(torch, "version", None)
        env["torch_cuda_version"] = getattr(t_v, "cuda", None) if t_v else None
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_total_memory_mb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            )
    except Exception:
        pass
    try:
        import mlflow

        env["mlflow_version"] = mlflow.__version__
    except Exception:
        pass
    try:
        import triton

        env["triton_version"] = triton.__version__
    except Exception:
        pass

    import os
    from urllib.parse import urlparse, urlunparse

    for var in ("CUDA_VISIBLE_DEVICES", "HDF5_USE_FILE_LOCKING"):
        val = os.environ.get(var)
        if val is not None:
            env[var] = val

    # Strip Basic Auth credentials from tracking URI before storing
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is not None:
        try:
            parsed = urlparse(tracking_uri)
            safe = parsed._replace(netloc=parsed.hostname or "")
            env["MLFLOW_TRACKING_URI"] = urlunparse(safe)
        except Exception:
            env["MLFLOW_TRACKING_URI"] = "<redacted>"

    return env


def _json_default(obj: Any) -> Any:
    """Fallback serialiser: coerce numpy/torch scalars and unknown types to str."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


class FlightRecorder:
    """Maintains a rolling window of per-step snapshots for crash forensics.

    Call ``record_step`` before each simulation step so the buffer contains
    pre-step data. On exception, call ``dump`` — it never touches torch/CUDA.
    """

    def __init__(self, capacity: int = 5) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._run_meta: dict[str, Any] = {}
        self._env: dict[str, Any] = _collect_env()

    def set_run_context(
        self,
        mat_params: dict[str, Any] | None = None,
        run_info: dict[str, Any] | None = None,
    ) -> None:
        """Store immutable run metadata captured before the simulation loop."""
        self._run_meta = {
            "material_params": mat_params or {},
            "run_info": run_info or {},
        }

    def record_step(self, snapshot: dict[str, Any]) -> None:
        """Append a snapshot dict to the ring buffer (replaces oldest if full)."""
        self._buf.append(snapshot)

    def dump(
        self,
        path: Path,
        exc: BaseException | None = None,
    ) -> Path:
        """Serialise the blackbox to *path* and return it.

        Safe to call after a CUDA exception — no torch/CUDA APIs are invoked.
        """
        tb_str = None
        if exc is not None:
            try:
                tb_str = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            except Exception:
                tb_str = repr(exc)

        payload: dict[str, Any] = {
            "env": self._env,
            "run_meta": self._run_meta,
            "last_steps": list(self._buf),
            "exception_type": type(exc).__name__ if exc is not None else None,
            "exception_repr": repr(exc) if exc is not None else None,
            "traceback": tb_str,
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2, default=_json_default)
            logger.info("FlightRecorder: blackbox written to %s", path)
        except Exception as write_exc:
            logger.error("FlightRecorder: failed to write blackbox: %s", write_exc)

        return path
