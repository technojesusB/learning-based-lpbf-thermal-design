from __future__ import annotations
import torch
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProfileResult:
    name: str
    elapsed_ms: float
    max_vram_mb: float
    success: bool = True

class PerformanceTracker:
    """
    A simple context manager for benchmarking PyTorch operations.
    Measures CUDA execution time and peak VRAM consumption.
    """
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        self.end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        self.start_time = 0.0
        self.result: Optional[ProfileResult] = None

    def __enter__(self) -> PerformanceTracker:
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device == "cuda":
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(self.end_event)
            max_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            max_vram = 0.0 # RAM tracking not implemented

        self.result = ProfileResult(
            name=self.name,
            elapsed_ms=elapsed_ms,
            max_vram_mb=max_vram,
            success=(exc_type is None)
        )

def get_torch_profiler(log_dir: str = "./artifacts/profiler_logs"):
    """
    Returns a torch.profiler.profile instance configured for deeper trace analysis.
    """
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
