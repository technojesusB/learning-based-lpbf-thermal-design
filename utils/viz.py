from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import torch


def _to_numpy_2d(field: torch.Tensor):
    """
    field: [1,1,H,W] or [H,W]
    """
    if field.ndim == 4:
        field = field[0, 0]
    return field.detach().float().cpu().numpy()


def save_field_png(
    field: torch.Tensor,
    out_path: str | Path,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = _to_numpy_2d(field)

    plt.figure()
    plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_process_summary(
    T_final: torch.Tensor,
    T_peak: torch.Tensor,
    E_acc: torch.Tensor,
    t_since: torch.Tensor,
    out_dir: str | Path,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_field_png(T_final, out_dir / "T_final.png", "T final")
    save_field_png(T_peak, out_dir / "T_peak_global.png", "T peak (global)")
    save_field_png(E_acc, out_dir / "E_acc.png", "Accumulated energy proxy (E_acc)")
    save_field_png(t_since, out_dir / "t_since.png", "Time since last hit (t_since)")
