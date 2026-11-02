"""sample_k and the optional generator on both noise paths."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neural_pbf.eval.benchmark.rollout import run_euler_rollout
from neural_pbf.eval.metrics.geometry import melt_pool_volume_vox
from neural_pbf.models.generative.fm.flow import sample_noise
from neural_pbf.uq.sampling import scalar_functionals, sample_k

GRID = {"dx_m": 1e-5, "dy_m": 1e-5, "dz_m": 1e-5}


class _Vel(nn.Module):
    patch_size = 4

    def forward(self, x, tau, cond, coords=None):
        return -x[:, :1] * 0.1


def _batch(rope: bool = False) -> dict:
    t = torch.zeros(1, 1, 1, 8, 8, 8)
    b = {"T_in": t, "T_target": t, "mask": t.clone(), "Q": t.clone(), "conditioning": torch.zeros(1, 12)}
    if rope:
        b["patch_origin"] = torch.zeros(1, 3, dtype=torch.long)
        b["h5_path"] = ["a.h5"]
    return b


@pytest.mark.unit
def test_sample_noise_default_uses_global_rng():
    x = torch.zeros(4)
    torch.manual_seed(3)
    a = sample_noise(x)
    torch.manual_seed(3)
    assert torch.equal(a, torch.randn_like(x))


@pytest.mark.unit
def test_rollout_default_unchanged_without_generator():
    torch.manual_seed(5)
    a = run_euler_rollout(_Vel(), nn.Identity(), _batch(), "net", torch.device("cpu"), n_steps=2)
    torch.manual_seed(5)
    x = torch.randn_like(torch.zeros(1, 1, 8, 8, 8))
    for _ in range(2):
        x = x + (-x * 0.1) * 0.5
    assert torch.allclose(a, x)


@pytest.mark.unit
@pytest.mark.parametrize("model_type", ["net", "dit"])
def test_generator_makes_net_path_reproducible_and_leaves_global_rng(model_type):
    run = lambda s: run_euler_rollout(  # noqa: E731
        _Vel(), nn.Identity(), _batch(), model_type, torch.device("cpu"), n_steps=2,
        generator=torch.Generator().manual_seed(s),
    )
    torch.manual_seed(1)
    state = torch.get_rng_state()
    a = run(9)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(a, run(9)) and not torch.equal(a, run(10))


@pytest.mark.unit
@pytest.mark.parametrize("model_type", ["rope", "triton"])
def test_generator_on_rope_path(model_type):
    run = lambda s: run_euler_rollout(  # noqa: E731
        _Vel(), nn.Identity(), _batch(rope=True), model_type, torch.device("cpu"), n_steps=2,
        grid_attrs=GRID, generator=torch.Generator().manual_seed(s),
    )
    assert torch.equal(run(9), run(9)) and not torch.equal(run(9), run(10))


@pytest.mark.unit
@pytest.mark.parametrize("model_type,rope", [("net", False), ("dit", False), ("rope", True), ("triton", True)])
def test_sample_k_shape_and_determinism(model_type, rope):
    kw = dict(model_type=model_type, K=5, base_seed=11, device=torch.device("cpu"), n_steps=2,
              grid_attrs=GRID if rope else None, draw_batch=2)
    a = sample_k(_Vel(), nn.Identity(), _batch(rope), **kw)
    assert a.shape == (5, 8, 8, 8)
    assert torch.equal(a, sample_k(_Vel(), nn.Identity(), _batch(rope), **kw))
    assert not torch.equal(a[0], a[1]), "draws must differ"
    assert not torch.equal(a, sample_k(_Vel(), nn.Identity(), _batch(rope), **{**kw, "base_seed": 12}))


@pytest.mark.unit
def test_sample_k_rejects_multi_sample_batch_and_nan():
    b = _batch()
    b = {k: torch.cat([v, v]) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
    with pytest.raises(ValueError, match="one input"):
        sample_k(_Vel(), nn.Identity(), b, "net", 2, 0, torch.device("cpu"))

    class _Nan(_Vel):
        def forward(self, x, tau, cond, coords=None):
            return x * float("nan")

    with pytest.raises(FloatingPointError):
        sample_k(_Nan(), nn.Identity(), _batch(), "net", 2, 0, torch.device("cpu"), n_steps=1)


@pytest.mark.unit
def test_melt_pool_volume():
    T = torch.zeros(1, 1, 4, 4, 4)
    T[0, 0, 0, :2, :2] = 2000.0
    assert melt_pool_volume_vox(T, 1500.0) == 4.0
    assert melt_pool_volume_vox(T, 2500.0) == 0.0


@pytest.mark.unit
def test_scalar_functionals_use_per_sample_liquidus():
    T = torch.full((3, 2, 2), 300.0)
    T[:2, 0, 0] = 1700.0
    assert scalar_functionals(T, T_l=1600.0) == (2.0, 1700.0, 2.0)
    assert scalar_functionals(T, T_l=1800.0) == (0.0, 1700.0, 0.0)
