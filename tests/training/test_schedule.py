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
"""Schedule factors against the reference formulas and the pinned call order."""

import pytest
import torch

from flux_action.training.schedule import (
    CooldownFrom,
    FrozenThenLinearWarmup,
    LinearWarmupLinearDecay,
    build_lr_scheduler,
    droid_lr_lambdas,
    resume_lr_scheduler,
)

# Values computed by hand from the reference classes with LR_SCALE folded out (scale one).
# Trunk: FrozenThenLinearWarmup(1000, 2000); heads: LinearWarmupLinearDecay(1000, None);
# both replaced from step 25000 by LinearWarmupLinearDecay(None, 5000, start_step=25000).
TABLE = {
    0: (0.0, 1 / 1001),
    999: (0.0, 1000 / 1001),
    1000: (0.0, 1.0),
    1001: (1 / 2000, 1.0),
    2999: (1999 / 2000, 1.0),
    3000: (1.0, 1.0),
    24999: (1.0, 1.0),
    25000: (1.0, 1.0),
    27500: (0.5, 0.5),
    29999: (1 / 5000, 1 / 5000),
    30000: (0.0, 0.0),
}


def test_droid_factor_table():
    trunk, heads = droid_lr_lambdas()
    for step, (t, h) in TABLE.items():
        assert trunk(step) == pytest.approx(t, abs=1e-15), step
        assert heads(step) == pytest.approx(h, abs=1e-15), step
    for f in (trunk, heads):
        with pytest.raises(RuntimeError, match="past the scheduled end"):
            f(30001)


def test_open_ended_base_launch_is_constant_after_warmup():
    trunk, heads = droid_lr_lambdas(cooldown_start=None)
    assert trunk(25000) == heads(25000) == trunk(49999) == heads(49999) == 1.0


def test_lr_scale_folding_matches_reference_peaks():
    # Reference: base LR 2e-4 / 1e-3 with LR_SCALE 0.96 in the factor. Ours: 1.92e-4 / 9.6e-4 at scale one.
    ref_trunk, ref_heads = droid_lr_lambdas(lr_scale=0.96)
    trunk, heads = droid_lr_lambdas()
    for step in TABLE:
        assert 2e-4 * ref_trunk(step) == pytest.approx(1.92e-4 * trunk(step), rel=1e-12)
        assert 1e-3 * ref_heads(step) == pytest.approx(9.6e-4 * heads(step), rel=1e-12)


def test_cooldown_from_any_step_matches_reference_overlay():
    base = LinearWarmupLinearDecay(1000, None)
    overlay = LinearWarmupLinearDecay(None, 5000, start_step=12345, lr_scale=0.96)
    combined = CooldownFrom(base, 12345, 5000, lr_scale=0.96)
    for step in (0, 12344, 12345, 12346, 15000, 17345):
        expected = base(step) if step < 12345 else overlay(step)
        assert combined(step) == expected
    with pytest.raises(RuntimeError):
        combined(17346)


def test_reference_class_edge_cases():
    with pytest.raises(ValueError):
        FrozenThenLinearWarmup(-1, 10)
    with pytest.raises(ValueError):
        FrozenThenLinearWarmup(0, 0)
    with pytest.raises(ValueError):
        LinearWarmupLinearDecay(-1, None)
    with pytest.raises(ValueError):
        CooldownFrom(lambda s: 1.0, 10, 0)
    # warmup + constant + decay, the full torchtitan shape, at scale 0.5
    f = LinearWarmupLinearDecay(4, 10, constant_steps=6, lr_scale=0.5)
    assert [f(s) for s in (0, 3, 4, 9, 10, 15, 20)] == pytest.approx(
        [0.5 / 5, 0.5 * 4 / 5, 0.5, 0.5, 0.5, 0.25, 0.0]
    )
    with pytest.raises(RuntimeError):
        f(21)


def _two_group_optimizer():
    trunk = torch.nn.Parameter(torch.ones(3))
    heads = torch.nn.Parameter(torch.ones(2))
    return torch.optim.AdamW([{"params": [trunk], "lr": 1.92e-4}, {"params": [heads], "lr": 9.6e-4}])


def _run(optimizer, scheduler, steps, start=0):
    """Reference loop: the LR consumed by update g must equal base * factor(g)."""
    trunk, heads = droid_lr_lambdas()
    consumed = []
    for g in range(start, start + steps):
        for p in (pg["params"][0] for pg in optimizer.param_groups):
            p.grad = torch.ones_like(p)
        consumed.append(tuple(pg["lr"] for pg in optimizer.param_groups))
        assert consumed[-1] == pytest.approx((1.92e-4 * trunk(g), 9.6e-4 * heads(g)), rel=1e-12), g
        optimizer.step()
        scheduler.step()
    return consumed


def test_lambda_lr_call_order_and_resume():
    optimizer = _two_group_optimizer()
    scheduler = build_lr_scheduler(optimizer, droid_lr_lambdas())
    assert scheduler.get_last_lr() == pytest.approx([0.0, 9.6e-4 / 1001])
    uninterrupted = _run(optimizer, scheduler, 6)
    assert scheduler.state_dict()["lr_lambdas"] == [None, None]  # config, not checkpoint, owns the schedule

    # Interrupt after three updates: a fresh optimizer loads the stale state, then the reference resume.
    optimizer2 = _two_group_optimizer()
    scheduler2 = build_lr_scheduler(optimizer2, droid_lr_lambdas())
    _run(optimizer2, scheduler2, 3)
    state = optimizer2.state_dict()
    optimizer3 = _two_group_optimizer()
    scheduler3 = build_lr_scheduler(optimizer3, droid_lr_lambdas())
    optimizer3.load_state_dict(state)
    resume_lr_scheduler(scheduler3, 3)
    resumed = _run(optimizer3, scheduler3, 3, start=3)
    assert resumed == uninterrupted[3:]
    with pytest.raises(ValueError):
        build_lr_scheduler(_two_group_optimizer(), [lambda s: 1.0])


def test_frozen_trunk_still_accumulates_adam_state():
    # The reference keeps the trunk in AdamW at LR zero: weights hold, moments and step advance.
    param = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([{"params": [param], "lr": 1.92e-4}], weight_decay=0.05)
    scheduler = build_lr_scheduler(optimizer, [FrozenThenLinearWarmup(1000, 2000)])
    for _ in range(3):
        param.grad = torch.full_like(param, 0.5)
        optimizer.step()
        scheduler.step()
    assert torch.equal(param.detach(), torch.ones(4))
    state = optimizer.state[param]
    assert int(state["step"]) == 3 and state["exp_avg"].abs().sum() > 0
