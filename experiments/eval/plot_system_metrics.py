import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
MLRUNS_DIR = Path("mlruns/21")
STAGING_DIR = Path("scratch/final_comparison_report")

def parse_trace_durations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    events = data.get("traceEvents", [])
    triton_time = 0
    standard_gpu_time = 0
    cpu_time = 0
    
    for ev in events:
        dur = ev.get("dur", 0)
        cat = ev.get("cat", "")
        name = ev.get("name", "").lower()
        
        if cat == "kernel":
            # Identify Triton kernels by name pattern
            if "triton" in name or "compiled_kernel" in name:
                triton_time += dur
            else:
                standard_gpu_time += dur
        elif cat == "cpu_op":
            cpu_time += dur
            
    return triton_time / 1000.0, standard_gpu_time / 1000.0, cpu_time / 1000.0

def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    traces = glob.glob(str(MLRUNS_DIR / "**" / "artifacts" / "profiler" / "*" / "*.json"), recursive=True)
    
    results = {}
    for t in traces:
        model_name = Path(t).parent.name
        triton_ms, standard_ms, cpu_ms = parse_trace_durations(t)
        results[model_name] = {"triton": triton_ms, "standard": standard_ms, "cpu": cpu_ms}
        
    if not results:
        print("No profiler traces found yet.")
        return

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")
    
    models = sorted(results.keys())
    triton_vals = [results[m]["triton"] for m in models]
    standard_vals = [results[m]["standard"] for m in models]
    cpu_vals = [results[m]["cpu"] for m in models]
    
    x = np.arange(len(models))
    width = 0.6
    
    # Stacked bar chart for GPU components
    ax.bar(x, standard_vals, width, label="Standard GPU Kernels", color="#4A90E2", alpha=0.8)
    ax.bar(x, triton_vals, width, bottom=standard_vals, label="Custom Triton Kernels", color="#F5A623", alpha=0.9)
    
    # Side bar for CPU overhead
    ax.bar(x + 0.35, cpu_vals, 0.2, label="CPU Dispatch Overhead", color="#50E3C2", alpha=0.6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Duration per Step [ms]", fontweight="bold")
    ax.set_title("Hardware Performance: Triton vs. ATen Kernels", fontsize=16, fontweight="bold", pad=20)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.15)
    
    # Annotate Triton share for v5
    for i, m in enumerate(models):
        total_gpu = triton_vals[i] + standard_vals[i]
        if total_gpu > 0 and triton_vals[i] > 0:
            share = (triton_vals[i] / total_gpu) * 100
            ax.text(i, total_gpu + 1, f"{share:.1f}% Triton", ha="center", color="#F5A623", fontweight="bold")

    plt.tight_layout()
    fig.savefig(STAGING_DIR / "system_metrics_profiling.png", dpi=200)
    print(f"Detailed system metrics plot saved to {STAGING_DIR / 'system_metrics_profiling.png'}")

if __name__ == "__main__":
    main()
