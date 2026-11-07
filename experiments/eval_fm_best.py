"""eval_fm_best.py — Point accuracy of one trained FM checkpoint on held-out runs.

Reads the checkpoint's own dataset config (mask source, conditioning keys) and the runs of one manifest
set, so the model sees exactly the inputs it was trained on, and no run it was trained on.

    uv run python experiments/eval_fm_best.py --checkpoint checkpoints/baseline_unet/<run>/best.pt
    uv run python experiments/eval_fm_best.py --checkpoint ... --model-type dit --split cal

The UQ test runs stay reserved for the UQ pipeline; ``--split test`` also needs ``--allow-test``.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import torch

from neural_pbf.eval.benchmark.dataset import build_test_dataset
from neural_pbf.eval.benchmark.model_loader import load_model_adaptive
from neural_pbf.eval.benchmark.rollout import run_euler_rollout
from neural_pbf.eval.heldout import heldout_dataset
from neural_pbf.eval.metrics.geometry import iou_melt_volumes, melt_pool_extent
from neural_pbf.training.script_args import DEFAULT_MANIFEST

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", default="net", choices=("net", "dit", "rope", "triton"))
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="val", choices=("val", "cal", "test"))
    parser.add_argument("--allow-test", action="store_true", help="Read the UQ test runs outside the pipeline.")
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="eval_fm_best.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, cond_enc, patch_size, grid_attrs = load_model_adaptive("eval", args.checkpoint, args.model_type, device)
    model.eval()
    cond_enc.eval()
    heldout = heldout_dataset(ckpt, args.manifest, args.split, allow_test=args.allow_test)
    base = heldout.dataset
    patches = build_test_dataset(base, patch_size=64, model_type=args.model_type)  # type: ignore[arg-type]
    cfg = base.cfg  # type: ignore[attr-defined]
    to_phys = lambda t: t * cfg.T_ref + cfg.T_ambient  # noqa: E731
    gen = torch.Generator(device=device).manual_seed(args.seed)

    rows = []
    for i in heldout.indices:
        path, key = base._keys[i]  # type: ignore[attr-defined]
        with h5py.File(path, "r") as f:
            T_l = float(f["samples"][key].attrs["T_l"])  # type: ignore[index, arg-type]
        sample = patches[i]
        batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]) for k, v in sample.items()}
        with torch.no_grad():
            pred = run_euler_rollout(model, cond_enc, batch, args.model_type, device, args.n_steps, grid_attrs, gen)
        T_pred, T_true = to_phys(pred.cpu()), to_phys(batch["T_target"].squeeze(1).cpu())
        rows.append({
            "h5_path": path, "sample_key": key,
            "rmse_K": float(torch.sqrt(torch.mean((T_pred - T_true) ** 2))),
            "T_max_pred_K": float(T_pred.max()), "T_max_true_K": float(T_true.max()),
            "depth_pred_vox": melt_pool_extent(T_pred, T_l)["D"], "depth_true_vox": melt_pool_extent(T_true, T_l)["D"],
            "melt_iou": iou_melt_volumes(T_pred, T_true, T_l),
        })  # fmt: skip

    with Path(args.out).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("rmse_K", "melt_iou")}
    logger.info("%d %s samples (%s): mean RMSE %.1f K, mean melt IoU %.3f -> %s",
                len(rows), args.split, args.checkpoint, mean["rmse_K"], mean["melt_iou"], args.out)  # fmt: skip


if __name__ == "__main__":
    main()
