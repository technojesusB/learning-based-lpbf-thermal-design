"""ExperienceReplayBuffer — ring buffer of 3D patch experiences for surrogate training.

Patches are stored in CPU RAM to avoid GPU VRAM pressure.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ExperienceReplayBuffer:
    """
    Ring buffer storing volumetric patch experiences in CPU RAM.

    Each experience is a tuple of (T_in, Q, T_target) and optionally T_lf,
    all cropped to patch_size³. Oldest entries are evicted once capacity is
    reached (ring / circular buffer semantics).

    Args:
        capacity: Maximum number of patches to store.
        patch_size: Spatial side-length of each cubic 3D patch [voxels].
        device: Storage device (default CPU to avoid GPU VRAM pressure).
    """

    def __init__(
        self,
        capacity: int,
        patch_size: int,
        device: torch.device | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self._capacity = capacity
        self._patch_size = patch_size
        self._device = device if device is not None else torch.device("cpu")

        # Storage lists — use a fixed-length list as a ring buffer
        self._storage: list[dict[str, Tensor]] = []
        self._write_idx: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_patch(self, field: Tensor) -> Tensor:
        """Crop ``field`` to ``patch_size³`` if larger; otherwise return as-is.

        Field shape: (..., D, H, W).  Only the last three spatial dims are
        cropped.  If any spatial dimension is smaller than patch_size, the
        field is returned without cropping along that dim (keeps it intact).
        """
        ps = self._patch_size
        *leading, D, H, W = field.shape

        d_start = max(0, (D - ps) // 2) if ps < D else 0
        h_start = max(0, (H - ps) // 2) if ps < H else 0
        w_start = max(0, (W - ps) // 2) if ps < W else 0

        d_end = d_start + min(ps, D)
        h_end = h_start + min(ps, H)
        w_end = w_start + min(ps, W)

        return field[..., d_start:d_end, h_start:h_end, w_start:w_end]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        T_in: Tensor,
        Q: Tensor,
        T_target: Tensor,
        T_lf: Tensor | None = None,
        **extra_fields: Tensor,
    ) -> None:
        """Store one experience (crops to patch_size³ if necessary).

        Args:
            T_in:         Input temperature field. Shape: (1, 1, D, H, W).
            Q:            Heat source field.        Shape: (1, 1, D, H, W).
            T_target:     Target temperature field. Shape: (1, 1, D, H, W).
            T_lf:         Low-fidelity field.       Shape: (1, 1, D, H, W) or None.
            **extra_fields: Additional named tensors (e.g. ``mask_in``,
                            ``mask_target``, ``physics_ctx``). Each is cropped
                            and stored alongside the mandatory fields; they are
                            included in sampled batches when **all** sampled
                            experiences contain the same key.
        """
        experience: dict[str, Tensor] = {
            "T_in": self._crop_patch(T_in).to(dtype=torch.float32, device=self._device),
            "Q": self._crop_patch(Q).to(dtype=torch.float32, device=self._device),
            "T_target": self._crop_patch(T_target).to(
                dtype=torch.float32, device=self._device
            ),
        }
        if T_lf is not None:
            experience["T_lf"] = self._crop_patch(T_lf).to(
                dtype=torch.float32, device=self._device
            )
        for key, tensor in extra_fields.items():
            experience[key] = self._crop_patch(tensor).to(
                dtype=torch.float32, device=self._device
            )

        if len(self._storage) < self._capacity:
            self._storage.append(experience)
        else:
            self._storage[self._write_idx] = experience

        self._write_idx = (self._write_idx + 1) % self._capacity

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """Sample a random mini-batch from the buffer.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            Dictionary of stacked tensors with a leading batch dimension.
            Optional fields (``T_lf`` and any extra fields added via
            ``push(**extra_fields)``) are included only when **all** sampled
            experiences contain them.

        Raises:
            RuntimeError: If the buffer contains fewer entries than ``batch_size``.
        """
        n = len(self._storage)
        if n == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if n < batch_size:
            raise RuntimeError(
                f"Buffer has only {n} entries but batch_size={batch_size} "
                "was requested."
            )

        indices = torch.randperm(n)[:batch_size].tolist()
        samples = [self._storage[i] for i in indices]

        # Stack mandatory fields
        batch: dict[str, Tensor] = {
            "T_in": torch.cat([s["T_in"] for s in samples], dim=0),
            "Q": torch.cat([s["Q"] for s in samples], dim=0),
            "T_target": torch.cat([s["T_target"] for s in samples], dim=0),
        }

        # Include all optional fields present in every sampled experience
        optional_keys = set(samples[0].keys()) - {"T_in", "Q", "T_target"}
        for key in optional_keys:
            if all(key in s for s in samples):
                batch[key] = torch.cat([s[key] for s in samples], dim=0)

        return batch

    def sample_to(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Sample a mini-batch and move all tensors to ``device``.

        This is a convenience wrapper around :meth:`sample` that transfers
        every tensor in the returned dict to the requested device using
        non-blocking copies.

        Args:
            batch_size: Number of experiences to sample.
            device:     Target device for all returned tensors.

        Returns:
            Dictionary of stacked tensors, all residing on ``device``.

        Raises:
            RuntimeError: If the buffer contains fewer entries than ``batch_size``.
        """
        batch = self.sample(batch_size)
        return {key: t.to(device, non_blocking=True) for key, t in batch.items()}

    def estimate_memory_mb(self) -> float:
        """Return approximate CPU memory used by stored patches in megabytes."""
        if not self._storage:
            return 0.0

        total_bytes = 0
        for experience in self._storage:
            for tensor in experience.values():
                total_bytes += tensor.element_size() * tensor.numel()

        return total_bytes / (1024.0**2)

    def __len__(self) -> int:
        return len(self._storage)

    def is_ready(self, min_samples: int) -> bool:
        """Return True if the buffer contains at least ``min_samples`` entries."""
        return len(self._storage) >= min_samples
