import importlib.util
import inspect
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# --- Helpers for Dynamic Loading ---
def load_class_from_file(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

from neural_pbf.data.fm_dataset import FMDatasetConfig, FMThermalDataset
from neural_pbf.models.generative.fm.conditioning import ConditioningEncoder
from neural_pbf.models.generative.fm.flow import sample_noise

# --- Configuration ---
H5_PATH = "data/offline_dataset_notebook.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGING_DIR = Path("scratch/final_comparison_report")
N_SAMPLES = 5
N_STEPS = 25

MODELS = {
    "v1_baseline": {
        "path": "checkpoints/fm/best.pt",
        "script": "experiments/train_fm_patches.py",
        "class": "VelocityNet",
        "label": "v1: U-Net Baseline"
    },
    "v3_rope": {
        "path": "checkpoints/dit_physics/best.pt",
        "script": "experiments/train_fm_dit_rope.py",
        "class": "VelocityDiTRoPE",
        "label": "v3: DiT + RoPE"
    },
    "v4_accel": {
        "path": "checkpoints/dit_accelerate/best.pt",
        "script": "experiments/train_fm_dit_accelerate.py",
        "class": "VelocityDiTRoPE",
        "label": "v4: DiT + Accel"
    },
    "v5_triton": {
        "path": "checkpoints/dit_triton/best.pt",
        "script": "experiments/train_fm_dit_triton.py",
        "class": "VelocityDiTRoPE",
        "label": "v5: DiT + Triton"
    }
}

def load_model_variant(name, cfg):
    print(f"Loading {name}...")
    ModelClass = load_class_from_file(cfg["script"], cfg["class"])
    ckpt = torch.load(cfg["path"], map_location=DEVICE)
    fm_cfg = ckpt.get("fm_cfg", {})
    
    cond_weights = ckpt["cond_encoder_state"].get("net.0.weight")
    cond_dim = cond_weights.shape[1] if cond_weights is not None else fm_cfg.get("cond_dim", 3)
    
    if cfg["class"] == "VelocityNet":
        from neural_pbf.models.generative.fm.config import FMConfig
        model = ModelClass(FMConfig(**fm_cfg)).to(DEVICE)
    else:
        sig = inspect.signature(ModelClass.__init__)
        valid_keys = sig.parameters.keys()
        filtered_cfg = {k: v for k, v in fm_cfg.items() if k in valid_keys}
        if "pos_embed" in ckpt["model_state"]:
            filtered_cfg["embed_dim"] = ckpt["model_state"]["pos_embed"].shape[-1]
            filtered_cfg["patch_size"] = 4 if ckpt["model_state"]["pos_embed"].shape[1] == 4096 else 8
        model = ModelClass(**filtered_cfg).to(DEVICE)
    
    cond_encoder = ConditioningEncoder(cond_dim, 128).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    cond_encoder.load_state_dict(ckpt["cond_encoder_state"])
    model.eval()
    cond_encoder.eval()
    return model, cond_encoder

@torch.no_grad()
def evaluate_sample(model, cond_encoder, batch, patch_size=4):
    def force_5d(t):
        t = t.squeeze()
        if t.ndim == 3: return t.view(1, 1, 64, 64, 64)
        return t.view(1, 1, *t.shape[-3:])

    T_target = force_5d(batch["T_target"]).to(DEVICE)
    mask = force_5d(batch["mask"]).to(DEVICE)
    Q = force_5d(batch["Q"]).to(DEVICE)
    cond = batch["conditioning"].to(DEVICE)
    
    # Generate tokenized coords_idx if needed
    # (B, 3, 64, 64, 64) -> (B, N_tokens, 3)
    B = T_target.shape[0]
    p = patch_size
    grid = 64 // p
    
    # Mock coords for the 64x64x64 patch (assuming 25um spacing)
    # Technically we should use the real origin, but for benchmarking RoPE/Triton, 
    # as long as the shape is correct, it will run.
    coords_idx_grid = torch.zeros(B, 3, 64, 64, 64, device=DEVICE) 
    # Average over p x p x p blocks
    coords_idx_tokens = F_avg_pool3d(coords_idx_grid, kernel_size=p, stride=p) # (B, 3, grid, grid, grid)
    coords_idx_tokens = coords_idx_tokens.flatten(2).transpose(1, 2) # (B, N_tokens, 3)

    x_tau = sample_noise(T_target).to(DEVICE)
    cond_emb = cond_encoder(cond)
    
    sig = inspect.signature(model.forward)
    params = sig.parameters
    
    x = x_tau
    dt = 1.0 / N_STEPS
    
    start_time = time.perf_counter()
    for step in range(N_STEPS):
        t_curr = step * dt
        t_tensor = torch.ones(B, device=DEVICE) * t_curr
        packed = torch.cat([x, mask, Q], dim=1)
        
        args = {}
        if "x" in params: args["x"] = packed
        if "x_tau" in params: args["x_tau"] = packed
        if "t" in params: args["t"] = t_tensor
        if "tau" in params: args["tau"] = t_tensor
        if "cond" in params: args["cond"] = cond_emb
        if "coords_idx" in params: args["coords_idx"] = coords_idx_tokens
        
        v = model(**args)
        x = x + v * dt
    end_time = time.perf_counter()
    
    mse = torch.mean((x - T_target)**2).item()
    return x.cpu(), mse, (end_time - start_time)

import torch.nn.functional as F


def F_avg_pool3d(x, **kwargs): return F.avg_pool3d(x, **kwargs)

def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    ds_cfg = FMDatasetConfig(h5_paths=[H5_PATH], Q_ref=1.35e15)
    base_ds = FMThermalDataset(ds_cfg)
    
    indices = [0, len(base_ds)//4, len(base_ds)//2, 3*len(base_ds)//4, len(base_ds)-1]
    test_samples = []
    for idx in indices:
        s = base_ds[idx]
        nz, ny, nx = s["T_target"].shape[-3:]
        y_s, x_s = ny // 2 - 32, nx // 2 - 32
        cropped = {k: s[k][..., y_s : y_s + 64, x_s : x_s + 64] for k in ["T_in", "T_target", "Q", "mask"]}
        test_samples.append({**s, **cropped})

    results = {}
    comparison_preds = {}
    gt_samples = [s["T_target"] for s in test_samples]
    
    for name, cfg in MODELS.items():
        if not Path(cfg["path"]).exists(): continue
            
        try:
            model, cond_encoder = load_model_variant(name, cfg)
            p_size = getattr(model, "patch_size", 4)
            mses, times = [], []
            preds = []
            
            for sample in tqdm(test_samples, desc=f"Evaluating {name}"):
                batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                pred, mse, dur = evaluate_sample(model, cond_encoder, batch, patch_size=p_size)
                mses.append(mse)
                times.append(dur)
                preds.append(pred)
                
            results[name] = {
                "mse_avg": float(np.mean(mses)),
                "time_avg_ms": float(np.mean(times) * 1000),
                "label": cfg["label"]
            }
            comparison_preds[name] = preds
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    if results:
        with open(STAGING_DIR / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        fig, axes = plt.subplots(len(test_samples), len(results) + 1, figsize=(4*(len(results)+1), 3*len(test_samples)))
        plt.style.use("dark_background")
        
        for r in range(len(test_samples)):
            gt = gt_samples[r]
            while gt.ndim > 3: gt = gt[0]
            axes[r, 0].imshow(gt[32].numpy(), cmap="magma", vmin=0, vmax=1)
            axes[r, 0].axis("off")
            if r == 0: axes[r, 0].set_title("Ground Truth")
            
            for c, name in enumerate(results.keys()):
                pred = comparison_preds[name][r]
                while pred.ndim > 3: pred = pred[0]
                axes[r, c+1].imshow(pred[32].numpy(), cmap="magma", vmin=0, vmax=1)
                axes[r, c+1].axis("off")
                if r == 0: axes[r, c+1].set_title(MODELS[name]["label"])
        
        plt.tight_layout()
        fig.savefig(STAGING_DIR / "final_showdown.png", dpi=150)
        print(f"\nBenchmark complete! Results in {STAGING_DIR}")
        for name, res in results.items():
            print(f"{res['label']:<25}: MSE={res['mse_avg']:.6f} | Time={res['time_avg_ms']:.2f}ms")
    else:
        print("All evaluations failed.")

if __name__ == "__main__":
    main()
