from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

# --- Configuration ---
MLFLOW_URI = "http://localhost:5000"
STAGING_DIR = Path("scratch/final_comparison_report")
EXPERIMENT_NAME = "lpbf_harmonization_final_verification" # From hero_smoke_test

# Map of run names to labels
RUN_MAP = {
    "v1_baseline": "v1: U-Net Baseline",
    "v3_rope": "v3: DiT + RoPE",
    "v4_accel": "v4: DiT + Accel",
    "v5_triton": "v5: DiT + Triton"
}

def get_run_losses(experiment_id, run_name):
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    if runs.empty: return None
    
    run_id = runs.iloc[0].run_id
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_metric_history(run_id, "loss")
    return pd.DataFrame([(m.step, m.value) for m in metrics], columns=["epoch", "loss"])

def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"Experiment {EXPERIMENT_NAME} not found. Ensure hero_smoke_test ran.")
        return
        
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")
    
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#D0021B"]
    
    for i, (run_name, label) in enumerate(RUN_MAP.items()):
        df = get_run_losses(exp.experiment_id, run_name)
        if df is not None:
            ax.plot(df["epoch"], df["loss"], label=label, color=colors[i], linewidth=2, alpha=0.8)
    
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss (log)", fontsize=12, fontweight="bold")
    ax.set_title("Training Loss Comparison (v1 - v5)", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    
    plt.tight_layout()
    fig.savefig(STAGING_DIR / "loss_overlay_comparison.png", dpi=200)
    print(f"Loss overlay saved to {STAGING_DIR / 'loss_overlay_comparison.png'}")

if __name__ == "__main__":
    main()
