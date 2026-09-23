# Copyright 2026 Black Forest Labs. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Power EMA against a direct transcription of the reference callback, plus sharded equality."""

import math

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flux_action.training.ema import (
    DROID_POWER_EMA_SIGMA_RELS,
    PowerEMA,
    power_ema_beta,
    sigma_rel_to_gamma,
)


def reference_gamma(sigma_rel):
    """bfl.modules.callbacks.ema.power_ema_sigma_rel_to_gamma, transcribed."""
    t = sigma_rel**-2
    return float(np.roots([1, 7, 16 - t, 12 - t]).real.max())


def reference_decay(gamma, n_averaged):
    """bfl.modules.callbacks.ema.power_ema_decay, transcribed."""
    if n_averaged == 0:
        return 0.0
    return (1 - 1 / (n_averaged + 1)) ** (gamma + 1)


def reference_update(ema_params, model_params, n_averaged, gamma):
    """AveragedModel.update_parameters with power_ema_sigma_rel set, transcribed."""
    if n_averaged == 0:
        for p_ema, p_model in zip(ema_params, model_params, strict=True):
            p_ema.copy_(p_model)
        return
    decay = reference_decay(gamma, n_averaged)
    torch._foreach_lerp_(ema_params, model_params, 1 - decay)  # get_ema_multi_avg_fn(decay)


@pytest.mark.parametrize("sigma_rel", DROID_POWER_EMA_SIGMA_RELS)
def test_gamma_matches_reference_and_edm2_profile(sigma_rel):
    gamma = sigma_rel_to_gamma(sigma_rel)
    assert gamma == reference_gamma(sigma_rel)
    assert math.sqrt((gamma + 1) / ((gamma + 2) ** 2 * (gamma + 3))) == pytest.approx(sigma_rel, abs=1e-12)


def test_recipe_gammas():
    assert sigma_rel_to_gamma(0.10) == pytest.approx(6.937203937601809, abs=1e-9)
    assert sigma_rel_to_gamma(0.05) == pytest.approx(16.972198602303443, abs=1e-9)
    for bad in (0.0, -0.1, 12**-0.5, 0.5):
        with pytest.raises(ValueError):
            sigma_rel_to_gamma(bad)


def test_beta_schedule():
    gamma = sigma_rel_to_gamma(0.10)
    assert power_ema_beta(gamma, 0) == 0.0
    assert power_ema_beta(gamma, 1) == pytest.approx(0.5 ** (gamma + 1))
    betas = [power_ema_beta(gamma, s) for s in range(1, 30001)]
    assert all(b1 < b2 for b1, b2 in zip(betas, betas[1:], strict=False)) and betas[-1] < 1.0
    assert betas[25999] == pytest.approx(reference_decay(gamma, 26000))
    with pytest.raises(ValueError):
        power_ema_beta(gamma, -1)


@pytest.mark.parametrize("sigma_rel", DROID_POWER_EMA_SIGMA_RELS)
def test_trace_matches_reference_update(sigma_rel):
    rng = torch.Generator().manual_seed(7)
    model = torch.nn.Linear(5, 3)
    ema = PowerEMA.from_module(model, sigma_rel)
    ref = [p.detach().clone() for p in model.parameters()]
    gamma = reference_gamma(sigma_rel)
    for step in range(40):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn(p.shape, generator=rng) * 0.1)
        beta = ema.update_from_module(model, step)
        assert beta == reference_decay(gamma, step)
        reference_update(ref, [p.detach() for p in model.parameters()], step, gamma)
        for ours, theirs in zip(ema.shadow, ref, strict=True):
            torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
    assert ema.last_step == 39
    # The first update was a copy of the post-update weights, not the initialization.
    assert not any(torch.equal(s, torch.zeros_like(s)) for s in ema.shadow)


def test_state_dict_round_trip_copy_to_and_consolidated():
    model = torch.nn.Linear(4, 2)
    ema = PowerEMA.from_module(model, 0.10)
    for step in range(3):
        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update_from_module(model, step)
    state = ema.state_dict()
    assert state["sigma_rel"] == 0.10 and state["last_step"] == 2
    other = PowerEMA.from_module(torch.nn.Linear(4, 2), 0.10)
    other.load_state_dict(state)
    for a, b in zip(ema.shadow, other.shadow, strict=True):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="sigma_rel"):
        PowerEMA.from_module(torch.nn.Linear(4, 2), 0.05).load_state_dict(state)
    target = torch.nn.Linear(4, 2)
    ema.copy_to(list(target.parameters()))
    assert torch.equal(target.weight.detach(), ema.shadow[0])
    full = ema.consolidated()
    assert set(full) == {"weight", "bias"} and torch.equal(full["weight"], ema.shadow[0])
    with pytest.raises(TypeError):
        PowerEMA([("ids", torch.zeros(2, dtype=torch.int64))], 0.10)
    with pytest.raises(ValueError):
        PowerEMA([], 0.10)


def _sharded_worker(rank, world_size, init_file):
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        rng = torch.Generator().manual_seed(11)
        full = torch.randn(8, 3, generator=rng)
        dense = PowerEMA([("w", full.clone())], 0.10)
        sharded = PowerEMA([("w", distribute_tensor(full.clone(), mesh, [Shard(0)]))], 0.10)
        for step in range(6):
            full = full + torch.randn(8, 3, generator=rng)  # identical on every rank
            dense.update([full], step)
            sharded.update([distribute_tensor(full, mesh, [Shard(0)])], step)
            gathered = sharded.consolidated()["w"]
            torch.testing.assert_close(gathered, dense.shadow[0], rtol=0, atol=0)
        local = sharded.state_dict()["tensors"]["w"].to_local()
        assert local.shape == (8 // world_size, 3)
    finally:
        dist.destroy_process_group()


def test_sharded_update_matches_unsharded(tmp_path):
    mp.spawn(_sharded_worker, args=(2, str(tmp_path / "pg")), nprocs=2, join=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_model_ema_accumulates_small_updates_and_resumes(dtype):
    source = torch.ones(4, dtype=dtype)
    ema = PowerEMA([("w", source)], 0.1)
    state = {"sigma_rel": 0.1, "last_step": 29999, "tensors": {"w": source.clone()}}
    ema.load_state_dict(state)
    assert ema.shadow[0].dtype == torch.float32
    source.fill_(2)
    beta = ema.update([source], 30000)
    expected = torch.ones(4).lerp(source.float(), 1 - beta)
    torch.testing.assert_close(ema.shadow[0], expected, rtol=0, atol=0)
    assert (ema.shadow[0] > 1).all()
    assert source.dtype == dtype and (source == 2).all()
    resumed = PowerEMA([("w", source)], 0.1)
    resumed.load_state_dict(ema.state_dict())
    ema.update([source], 30001)
    resumed.update([source], 30001)
    torch.testing.assert_close(resumed.shadow[0], ema.shadow[0], rtol=0, atol=0)
