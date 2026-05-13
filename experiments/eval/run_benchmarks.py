import os
import subprocess

import mlflow

# Config
H5_150 = "data/offline_dataset_notebook.h5"
H5_500 = "data/offline_dataset_test.h5"
BASELINE_RUN_ID = "30ba4110bfd048d3a269da05ce338f8d"
EPOCHS = 100
MLFLOW_URI = "sqlite:///mlflow.db"

def run_cmd(cmd, env=None):
    print(f"Executing: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode

def get_latest_run_id(experiment_name):
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if not exp: return None
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs.empty:
        return runs.iloc[0]["run_id"]
    return None

def main():
    os.makedirs("reports", exist_ok=True)
    
    # 0. REPORT FOR EXISTING BASELINE
    print("\n" + "="*60)
    print(f"STEP 0: Generating report for existing Baseline ({BASELINE_RUN_ID[:8]})")
    print("="*60)
    run_cmd(["uv", "run", "python", "scripts/generate_report.py", "--run_id", BASELINE_RUN_ID, "--output_dir", "reports/baseline"])

    # 1. DiT (150 Samples) - For direct comparison
    exp_dit_150 = "benchmark_dit_v1_150samples"
    cmd_dit_150 = [
        "uv", "run", "python", "experiments/train_fm_dit.py",
        "--h5", H5_150,
        "--epochs", str(EPOCHS),
        "--batch_size", "2",
        "--mlflow_experiment", exp_dit_150,
        "--mlflow_uri", MLFLOW_URI
    ]
    print("\n" + "="*60)
    print("STEP 1: Training VelocityDiT (150 Samples - Comparison)")
    print("="*60)
    run_cmd(cmd_dit_150)
    
    run_id_150 = get_latest_run_id(exp_dit_150)
    if run_id_150:
        run_cmd(["uv", "run", "python", "scripts/generate_report.py", "--run_id", run_id_150, "--output_dir", "reports/dit_150"])

    # 2. DiT (500 Samples) - Full dataset scaling
    exp_dit_500 = "benchmark_dit_v1_500samples"
    cmd_dit_500 = [
        "uv", "run", "python", "experiments/train_fm_dit.py",
        "--h5", H5_500,
        "--epochs", str(EPOCHS),
        "--batch_size", "2",
        "--mlflow_experiment", exp_dit_500,
        "--mlflow_uri", MLFLOW_URI
    ]
    print("\n" + "="*60)
    print("STEP 2: Training VelocityDiT (500 Samples - Full Scale)")
    print("="*60)
    run_cmd(cmd_dit_500)
    
    run_id_500 = get_latest_run_id(exp_dit_500)
    if run_id_500:
        run_cmd(["uv", "run", "python", "scripts/generate_report.py", "--run_id", run_id_500, "--output_dir", "reports/dit_500"])

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print(f"Reports available in reports/baseline/{run_id_patches} and reports/dit/{run_id_dit}")
    print("="*60)

if __name__ == "__main__":
    main()
