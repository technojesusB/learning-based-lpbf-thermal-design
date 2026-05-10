import argparse
import logging
import torch
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.models.generative.fm.velocity_net import VelocityNet
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.config import FMConfig
from neural_pbf.models.generative.fm.flow import fm_loss, sample_noise, interpolate
from neural_pbf.eval.metrics.geometry import iou_melt_volumes, melt_pool_extent

# Reuse the same dataset wrapper
from experiments.train_fm_patches import PatchFMThermalDataset, _log_validation_image

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fm/best.pt")
    parser.add_argument("--mlflow_experiment", type=str, default="fm_eval_detailed")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. SETUP DATASET (Exactly as in training)
    ds_cfg = FMDatasetConfig(h5_paths=[args.h5], Q_ref=1.35e15)
    full_ds = FMThermalDataset(ds_cfg)
    patch_ds = PatchFMThermalDataset(full_ds, patch_size=64)
    
    n_train = int(len(patch_ds) * 0.7)
    n_val = int(len(patch_ds) * 0.2)
    n_test = len(patch_ds) - n_train - n_val
    _, _, test_ds = random_split(patch_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    logger.info(f"Loaded {len(test_ds)} test samples for evaluation.")

    # 2. LOAD MODEL
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    # weights_only=False required: checkpoint includes fm_cfg dict (not pure tensors)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    fm_cfg = FMConfig(**checkpoint["fm_cfg"])
    
    model = VelocityNet(fm_cfg).to(device)
    cond_encoder = ConditioningEncoder(fm_cfg.cond_dim, 128).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    cond_encoder.load_state_dict(checkpoint["cond_encoder_state"])
    model.eval()
    cond_encoder.eval()
    logger.info(f"Model loaded from {args.checkpoint} (Epoch {checkpoint.get('epoch', 'N/A')})")

    # 3. EVALUATION
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"eval_detailed_{checkpoint.get('epoch', 'final')}"):
        mlflow.log_params({
            "checkpoint": args.checkpoint,
            "test_samples": len(test_ds),
            "orig_val_loss": checkpoint.get("val_loss", 0.0)
        })

        test_mse = 0.0
        results_log = []

        # We want to trace back to original indices
        # test_ds is a Subset, so test_ds.indices contains original indices in patch_ds
        test_indices = test_ds.indices

        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating Samples")):
            # Get original H5 sample info
            orig_idx = test_indices[i]
            h5_path, sample_key = patch_ds.base_ds._keys[orig_idx]
            
            T_tgt = batch["T_target"].to(device).squeeze(1)
            mask = batch["mask"].to(device).squeeze(1)
            Q = batch["Q"].to(device).squeeze(1)
            cond = batch["conditioning"].to(device)
            
            with torch.no_grad():
                cond_emb = cond_encoder(cond)
                # For IoU/Extent, we should do a simple 1-step or 10-step inference
                # to get a "final" T field. Let's do 5-step Euler for speed/accuracy balance.
                dt = 0.2
                xt = sample_noise(T_tgt)
                for step_idx in range(5):
                    t_val = torch.ones(T_tgt.shape[0], device=device) * (step_idx * dt)
                    vt = model(torch.cat([xt, mask, Q], dim=1), t_val, cond_emb)
                    xt = xt + vt * dt
                
                mse = torch.mean((xt - T_tgt)**2).item()
                iou = iou_melt_volumes(xt, T_tgt, T_liquidus=0.6)
                ext_pred = melt_pool_extent(xt, T_liquidus=0.6)
                ext_gt = melt_pool_extent(T_tgt, T_liquidus=0.6)
                
            test_mse += mse
            results_log.append(f"TestIdx {i} -> H5: {os.path.basename(h5_path)} | Key: {sample_key} | MSE: {mse:.6f} | IoU: {iou:.4f} | D_pred: {ext_pred['D']:.1f} | D_gt: {ext_gt['D']:.1f}")

            # Visualization for every 10th sample or if MSE is high
            if i % 10 == 0 or i < 5:
                # Custom label for the image
                _log_validation_image(model, mlflow, cond_encoder, batch, i, device)
                # Note: I'll manually rename/tag it in MLflow or just accept the index i

        avg_mse = test_mse / len(test_ds)
        mlflow.log_metric("avg_test_mse", avg_mse)
        
        # Save the mapping log
        log_path = "test_sample_mapping.txt"
        with open(log_path, "w") as f:
            f.write("\n".join(results_log))
        mlflow.log_artifact(log_path)
        
        logger.info(f"Evaluation finished. Avg MSE: {avg_mse:.6f}")
        logger.info(f"Detailed mapping saved to {log_path} and MLflow.")

if __name__ == "__main__":
    main()
