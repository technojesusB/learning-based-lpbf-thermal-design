from __future__ import annotations

from pathlib import Path
from typing import Type, TypeVar
import torch
import os
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def _ensure_parent_dir(path: Path) -> None:
    parent = path.parent.resolve()

    # If parent exists but is a file -> hard error with clear message
    if parent.exists() and not parent.is_dir():
        raise RuntimeError(
            f"Cannot save to {path}: parent path {parent} exists but is not a directory "
            f"(it might be a file named '{parent.name}')."
        )

    parent.mkdir(parents=True, exist_ok=True)

    # Extra sanity: ensure we can write into that directory
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"No write permission for directory: {parent}")
    
def save_state(state: BaseModel, path: str | Path) -> Path:
    """
    Save a Pydantic state object via torch.save (zipfile format).

    Returns the resolved path actually written.
    """
    path = Path(str(path)).expanduser()

    # If someone passed a directory by mistake, create default filename
    if path.exists() and path.is_dir():
        path = path / "states.pt"

    _ensure_parent_dir(path)

    try:
        torch.save(state, path)
        return path.resolve()
    except RuntimeError as e:
        # Fallback: try a simpler legacy format (rarely helps, but sometimes does)
        # and try saving to current directory as ultimate fallback.
        fallback = Path("states_run_fallback.pt").resolve()
        try:
            torch.save(state, fallback, _use_new_zipfile_serialization=False)
            return fallback
        except Exception:
            # Re-raise original error with extra context
            raise RuntimeError(
                f"Failed to save state to {path.resolve()}.\n"
                f"Also failed to save fallback to {fallback}.\n"
                f"Original error: {e}"
            ) from e


def load_state(path: str | Path, model_type: Type[T], map_location: str | torch.device | None = "cpu") -> T:
    path = Path(str(path)).expanduser()
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(obj, model_type):
        raise TypeError(f"Loaded object type {type(obj)} does not match expected {model_type}.")
    return obj