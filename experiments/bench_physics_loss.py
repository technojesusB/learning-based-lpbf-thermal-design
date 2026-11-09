"""Benchmark: physics loss forward+backward, Triton vs PyTorch reference, B=8 patches of 64^3, float32, CUDA.

    uv run python experiments/bench_physics_loss.py   # numbers on record in docs/physics_residual_check.md
"""
import sys, time
sys.path.insert(0, "tests/physics")
import torch, test_triton_pde_loss as t

def make(B=8, n=64):
    c = t._case((n, n, n), torch.float32, seed=0)  # 2 samples (SS316L, Ti64)
    reps = B // c["v"].shape[0]
    return {k: (v.repeat(reps, *([1] * (v.ndim - 1))) if v.ndim > 0 else v) for k, v in c.items()}

def timed(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        torch.cuda.synchronize(); s = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
    ts.sort(); return ts[len(ts) // 2]

c = make()
print(f"GPU {torch.cuda.get_device_name(0)}, B={c['v'].shape[0]}, patch {tuple(c['v'].shape[2:])}, float32")
for form in ("tin", "coupled"):
    def tri():
        v = c["v"].clone().requires_grad_(True); t._triton_loss(c, form, v).backward()
    def ref():
        v = c["v"].clone().requires_grad_(True)
        from neural_pbf.physics.fm_physics import physics_loss_reference
        si = lambda x: x * t.T_REF + t.T_AMB
        T_pred = si(c["x_tau"] + (1 - c["tau"].view(-1, 1, 1, 1, 1)) * v)
        physics_loss_reference(T_pred, si(c["T_in"]), si(c["T_target"]), c["Q"] * t.Q_REF, c["mask_in"],
                               c["material"], form, t.DX, t.DY, t.DZ).backward()
    torch.cuda.reset_peak_memory_stats(); a = timed(tri); m_tri = torch.cuda.max_memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats(); b = timed(ref); m_ref = torch.cuda.max_memory_allocated() / 2**20
    print(f"{form:8s} triton {a*1e3:7.2f} ms  ({m_tri:6.0f} MiB peak)   reference {b*1e3:7.2f} ms  ({m_ref:6.0f} MiB peak)   speedup {b/a:5.1f}x")
