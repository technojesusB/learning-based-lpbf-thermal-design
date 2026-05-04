"""Lazy import guards for optional hardware-monitoring dependencies."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_NVML_WARNED = False
_PSUTIL_WARNED = False


def try_import_nvml() -> tuple[Any | None, str | None]:
    """Return (pynvml, None) or (None, error_message)."""
    global _NVML_WARNED
    try:
        import pynvml  # type: ignore[import-untyped]

        return pynvml, None
    except ImportError as exc:
        msg = f"pynvml not available ({exc}). GPU metrics disabled."
        if not _NVML_WARNED:
            logger.debug(msg)
            _NVML_WARNED = True
        return None, msg


def try_import_psutil() -> tuple[Any | None, str | None]:
    """Return (psutil, None) or (None, error_message)."""
    global _PSUTIL_WARNED
    try:
        import psutil

        return psutil, None
    except ImportError as exc:
        msg = f"psutil not available ({exc}). CPU/RAM metrics disabled."
        if not _PSUTIL_WARNED:
            logger.debug(msg)
            _PSUTIL_WARNED = True
        return None, msg
