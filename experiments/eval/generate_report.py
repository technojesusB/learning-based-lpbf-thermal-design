import mlflow
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import argparse

def moving_average(a, n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def extract_parts(img_path):
    img = Image.open(img_path)
    w, h = img.size
    return {
        'gt_surf': img.crop((0, 0, w//2, h//2)),
        'pred_surf': img.crop((w//2, 0, w, h//2)),
        'gt_depth': img.crop((0, h//2, w//2, h)),
        'pred_depth': img.crop((w//2, h//2, w, h))
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    run_id = args.run_id
    out_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("scratch/cache", exist_ok=True)

    # 1. LOSSES
    h_train = client.get_metric_history(run_id, "train_loss")
    h_val = client.get_metric_history(run_id, "val_loss")
    if h_train:
        t_steps = [m.step for m in h_train]
        t_vals = [m.value for m in h_train]
        plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
        ax = plt.gca()
        ax.set_facecolor('#2d2d2d')
        plt.plot(t_steps, t_vals, color='#2ecc71', alpha=0.3, label='Train Loss')
        if h_val:
            v_steps = [m.step for m in h_val]
            v_vals = [m.value for m in h_val]
            plt.plot(v_steps, v_vals, color='#e74c3c', alpha=0.3, label='Val Loss')
            if len(v_vals) > 5:
                plt.plot(v_steps[4:], moving_average(v_vals, 5), color='#e74c3c', lw=2, label='Val (MA-5)')
        if len(t_vals) > 5:
            plt.plot(t_steps[4:], moving_average(t_vals, 5), color='#2ecc71', lw=2, label='Train (MA-5)')
        plt.yscale('log')
        plt.title(f"Loss Evolution (Run: {run_id[:8]})", color='white', fontsize=14)
        plt.tick_params(colors='white')
        plt.legend()
        plt.savefig(os.path.join(out_dir, "losses.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 2. TRAINING PROGRESS
    epochs = range(0, 100, 10)
    img_paths = []
    for ep in epochs:
        p = f"plots/val_epoch_{ep:03d}.png"
        try:
            local = client.download_artifacts(run_id, p, "scratch/cache")
            img_paths.append((ep, local))
        except: pass
    
    if img_paths:
        n_cols = len(img_paths) + 1
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols*2.5, 6), facecolor='#1e1e1e')
        p_0 = extract_parts(img_paths[0][1])
        axes[0, 0].imshow(p_0['gt_surf'])
        axes[1, 0].imshow(p_0['gt_depth'])
        for i, (ep, path) in enumerate(img_paths):
            p = extract_parts(path)
            axes[0, i+1].imshow(p['pred_surf'])
            axes[0, i+1].set_title(f"Ep {ep}", color='white', fontsize=9)
            axes[1, i+1].imshow(p['pred_depth'])
        for ax in axes.flatten(): ax.axis('off')
        plt.savefig(os.path.join(out_dir, "training_progress.png"), dpi=120, bbox_inches='tight')
        plt.close()

    # 3. TEST EVAL
    test_img_paths = []
    for i in range(4):
        p = f"plots/val_epoch_{990+i:03d}.png"
        try:
            local = client.download_artifacts(run_id, p, "scratch/cache")
            test_img_paths.append(local)
        except: pass
    if test_img_paths:
        fig, axes = plt.subplots(2, 8, figsize=(20, 6), facecolor='#1e1e1e')
        for i, path in enumerate(test_img_paths):
            p = extract_parts(path)
            axes[0, i*2].imshow(p['gt_surf'])
            axes[0, i*2+1].imshow(p['pred_surf'])
            axes[1, i*2].imshow(p['gt_depth'])
            axes[1, i*2+1].imshow(p['pred_depth'])
        for ax in axes.flatten(): ax.axis('off')
        plt.savefig(os.path.join(out_dir, "test_evaluation.png"), dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
