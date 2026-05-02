"""Tests for ExperienceReplayBuffer — written BEFORE implementation (TDD RED phase)."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture()
def small_buffer():
    """Tiny capacity buffer for fast tests."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    return ExperienceReplayBuffer(capacity=16, patch_size=8)


def _make_patch(size: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper: create (T_in, Q, T_target) of shape (1,1,D,H,W)."""
    shape = (1, 1, size, size, size)
    T_in = torch.rand(shape, dtype=torch.float32) * 500 + 300
    Q = torch.rand(shape, dtype=torch.float32) * 1e9
    T_target = T_in + torch.rand(shape, dtype=torch.float32) * 10
    return T_in, Q, T_target


def test_push_and_sample_basic(small_buffer):
    """Push 10 items, sample a batch of 4, check shapes."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf: ExperienceReplayBuffer = small_buffer
    for _ in range(10):
        T_in, Q, T_target = _make_patch(size=8)
        buf.push(T_in, Q, T_target)

    assert len(buf) == 10

    batch = buf.sample(4)
    assert "T_in" in batch
    assert "Q" in batch
    assert "T_target" in batch

    # Batch dimension must be 4
    assert batch["T_in"].shape[0] == 4
    assert batch["Q"].shape[0] == 4
    assert batch["T_target"].shape[0] == 4

    # Spatial dims must equal patch_size
    patch_size = 8
    assert batch["T_in"].shape[2:] == (patch_size, patch_size, patch_size)


def test_ring_buffer_eviction(small_buffer):
    """Push capacity + 10 items; buffer length stays at capacity."""
    capacity = 16
    for _i in range(capacity + 10):
        T_in, Q, T_target = _make_patch(size=8)
        small_buffer.push(T_in, Q, T_target)

    assert len(small_buffer) == capacity


def test_sample_raises_when_empty():
    """Sampling from an empty buffer must raise an error."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=32, patch_size=8)
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        buf.sample(1)


def test_is_ready(small_buffer):
    """is_ready(n) returns False when fewer than n samples are stored."""
    assert not small_buffer.is_ready(5)

    for _ in range(4):
        T_in, Q, T_target = _make_patch(size=8)
        small_buffer.push(T_in, Q, T_target)

    assert not small_buffer.is_ready(5)

    T_in, Q, T_target = _make_patch(size=8)
    small_buffer.push(T_in, Q, T_target)
    assert small_buffer.is_ready(5)


def test_estimate_memory_mb(small_buffer):
    """estimate_memory_mb returns a positive float after pushing data."""
    T_in, Q, T_target = _make_patch(size=8)
    small_buffer.push(T_in, Q, T_target)

    mem = small_buffer.estimate_memory_mb()
    assert isinstance(mem, float)
    assert mem > 0.0


def test_push_full_field_crops_to_patch():
    """Pushing a large field larger than patch_size must be cropped internally."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    patch_size = 8
    buf = ExperienceReplayBuffer(capacity=4, patch_size=patch_size)

    # Large field — bigger than patch_size
    large_size = 32
    T_in, Q, T_target = _make_patch(size=large_size)
    buf.push(T_in, Q, T_target)

    batch = buf.sample(1)
    assert batch["T_in"].shape[2:] == (patch_size, patch_size, patch_size)
    assert batch["Q"].shape[2:] == (patch_size, patch_size, patch_size)
    assert batch["T_target"].shape[2:] == (patch_size, patch_size, patch_size)


def test_t_lf_optional(small_buffer):
    """Pushing without T_lf results in a batch without the 'T_lf' key."""
    T_in, Q, T_target = _make_patch(size=8)
    small_buffer.push(T_in, Q, T_target)  # No T_lf

    batch = small_buffer.sample(1)
    assert "T_lf" not in batch


def test_t_lf_stored_when_provided(small_buffer):
    """Pushing with T_lf makes the batch include the 'T_lf' key."""
    T_in, Q, T_target = _make_patch(size=8)
    T_lf = torch.rand_like(T_in)
    small_buffer.push(T_in, Q, T_target, T_lf=T_lf)

    batch = small_buffer.sample(1)
    assert "T_lf" in batch
    assert batch["T_lf"].shape[2:] == (8, 8, 8)


def test_sample_returns_float32(small_buffer):
    """All tensors in sampled batch must be float32."""
    T_in, Q, T_target = _make_patch(size=8)
    small_buffer.push(T_in, Q, T_target)

    batch = small_buffer.sample(1)
    for key, tensor in batch.items():
        assert tensor.dtype == torch.float32, f"Key '{key}' has dtype {tensor.dtype}"


def test_invalid_capacity_raises():
    """Constructing a buffer with capacity <= 0 must raise ValueError."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    with pytest.raises(ValueError, match="capacity"):
        ExperienceReplayBuffer(capacity=0, patch_size=8)


def test_invalid_patch_size_raises():
    """Constructing a buffer with patch_size <= 0 must raise ValueError."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    with pytest.raises(ValueError, match="patch_size"):
        ExperienceReplayBuffer(capacity=16, patch_size=0)


def test_sample_insufficient_entries_raises():
    """Requesting more samples than available entries must raise RuntimeError."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=8)
    T_in, Q, T_target = _make_patch(size=8)
    buf.push(T_in, Q, T_target)

    with pytest.raises(RuntimeError, match="Buffer has only"):
        buf.sample(5)  # only 1 entry


def test_estimate_memory_mb_empty_returns_zero():
    """estimate_memory_mb on an empty buffer must return 0.0."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=8)
    assert buf.estimate_memory_mb() == 0.0


def test_sample_reproducible_with_torch_seed():
    """sample() must return the same items when torch seed is reset to the same value.

    RED  (before fix, using random.sample): Python's stdlib random is not governed by
         torch.manual_seed, so re-seeding torch does NOT reproduce the same sample.
    GREEN (after fix, using torch.randperm): seeding torch to 42 both times produces
         identical batches.
    """
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    capacity = 20
    buf = ExperienceReplayBuffer(capacity=capacity, patch_size=4)

    # Fill buffer with distinguishable entries (unique T value per entry)
    for i in range(capacity):
        shape = (1, 1, 4, 4, 4)
        T_in = torch.full(shape, float(i), dtype=torch.float32)
        Q = torch.zeros(shape)
        T_target = torch.full(shape, float(i + 0.5), dtype=torch.float32)
        buf.push(T_in, Q, T_target)

    batch_size = 5

    # First sample: seed torch, sample
    torch.manual_seed(42)
    batch_a = buf.sample(batch_size)

    # Second sample: re-seed torch to the same value, sample again
    torch.manual_seed(42)
    batch_b = buf.sample(batch_size)

    assert torch.equal(batch_a["T_in"], batch_b["T_in"]), (
        "sample() is not reproducible under torch.manual_seed. "
        "Replace random.sample(...) with torch.randperm(n)[:batch_size].tolist() "
        "so sampling is governed by PyTorch's RNG."
    )


# ---------------------------------------------------------------------------
# Tests for sample_to method (TDD RED phase)
# ---------------------------------------------------------------------------


def test_sample_to_method_exists():
    """ExperienceReplayBuffer must expose a sample_to(batch_size, device) method."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=8)
    assert hasattr(buf, "sample_to"), (
        "ExperienceReplayBuffer.sample_to does not exist. "
        "Add method sample_to(self, batch_size, device) -> dict[str, Tensor]."
    )
    assert callable(buf.sample_to)


def test_sample_to_returns_tensors_on_requested_device():
    """sample_to(n, device) must return all tensors on the specified device (CPU test)."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=4)
    for _ in range(5):
        shape = (1, 1, 4, 4, 4)
        T_in = torch.rand(shape)
        Q = torch.rand(shape)
        T_target = torch.rand(shape)
        buf.push(T_in, Q, T_target)

    cpu = torch.device("cpu")
    batch = buf.sample_to(3, device=cpu)

    for key, tensor in batch.items():
        assert tensor.device.type == "cpu", (
            f"sample_to(device=cpu): key '{key}' is on {tensor.device}, expected cpu"
        )


def test_sample_to_returns_correct_batch_size():
    """sample_to must return batches with the requested batch dimension."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=4)
    for _ in range(8):
        shape = (1, 1, 4, 4, 4)
        buf.push(torch.rand(shape), torch.rand(shape), torch.rand(shape))

    batch = buf.sample_to(4, device=torch.device("cpu"))
    assert batch["T_in"].shape[0] == 4
    assert batch["Q"].shape[0] == 4
    assert batch["T_target"].shape[0] == 4


def test_sample_to_includes_mandatory_keys():
    """sample_to must include T_in, Q, T_target in the returned dict."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=4)
    for _ in range(5):
        shape = (1, 1, 4, 4, 4)
        buf.push(torch.rand(shape), torch.rand(shape), torch.rand(shape))

    batch = buf.sample_to(2, device=torch.device("cpu"))
    assert "T_in" in batch
    assert "Q" in batch
    assert "T_target" in batch


def test_sample_to_raises_when_insufficient_entries():
    """sample_to must raise RuntimeError when fewer entries than batch_size exist."""
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=4)
    shape = (1, 1, 4, 4, 4)
    buf.push(torch.rand(shape), torch.rand(shape), torch.rand(shape))  # only 1

    with pytest.raises(RuntimeError):
        buf.sample_to(5, device=torch.device("cpu"))


def test_mixed_t_lf_raises_or_skips():
    """Buffer must handle the case where some pushes have T_lf and some don't.

    When T_lf is mixed (some None, some not), sampling a batch must either:
    - raise an informative error, OR
    - return only T_lf for items that have it (partial batch handling), OR
    - omit 'T_lf' entirely if any entry is missing.

    We just assert sampling doesn't crash; the returned batch must be a dict.
    """
    from neural_pbf.models.replay_buffer import ExperienceReplayBuffer

    buf = ExperienceReplayBuffer(capacity=16, patch_size=8)

    T_in, Q, T_target = _make_patch(size=8)
    buf.push(T_in, Q, T_target, T_lf=torch.rand_like(T_in))
    T_in2, Q2, T_target2 = _make_patch(size=8)
    buf.push(T_in2, Q2, T_target2)  # No T_lf

    # Must not crash
    batch = buf.sample(2)
    assert isinstance(batch, dict)
    assert "T_in" in batch
