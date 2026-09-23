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
"""Learning-rate schedules of the DROID recipe, ported from the reference trainer.

Call order, pinned to the reference on 2026-09-12: a factor is evaluated at the zero-based index
``g`` of the optimizer update it applies to. ``LambdaLR`` primes ``factor(0)`` when it is built,
``optimizer.step()`` consumes that value for update 0, and ``scheduler.step()`` runs once after
every optimizer update, so update ``g`` uses ``factor(g)``. A checkpoint written after ``N``
updates resumes through :func:`resume_lr_scheduler`, the reference's ``last_epoch = N - 1`` plus
one ``step()``, so update ``N`` uses ``factor(N)`` with or without the interruption.

The trunk factor is exactly zero through update 1000. The reference keeps the trunk in the
optimizer during that phase: AdamW moments and step counts advance at learning rate zero.
Express freezing through the factor, never by skipping the parameter group.

The reference ran the cooldown as a second launch that replaced both schedules at step 25000.
:class:`CooldownFrom` reproduces that inside one schedule so a single run can cool down from
any step; the arithmetic is identical.
"""

import warnings
from collections.abc import Callable, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

LRLambda = Callable[[int], float]

# Reference recipe (GB200 runs 936673 / 938134 / 941584). LR_SCALE 0.96 is folded into the
# peak learning rates of PolicyConfig, so the factors below peak at exactly one.
DROID_FROZEN_STEPS = 1000
DROID_TRUNK_WARMUP_STEPS = 2000
DROID_HEADS_WARMUP_STEPS = 1000
DROID_COOLDOWN_START = 25000
DROID_COOLDOWN_STEPS = 5000


class FrozenThenLinearWarmup:
    """Zero through ``frozen_steps``, linear to ``lr_scale`` at ``frozen_steps + warmup_steps``, then flat.

    Reference ``bfl.modules.lr_scheduler.FrozenThenLinearWarmup``: the boundary step itself is
    still zero and the first non-zero factor is ``1 / warmup_steps``.
    """

    def __init__(self, frozen_steps: int, warmup_steps: int, lr_scale: float = 1.0):
        if frozen_steps < 0:
            raise ValueError(f"frozen_steps must be non-negative, got {frozen_steps}")
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
        self.frozen_steps = frozen_steps
        self.warmup_steps = warmup_steps
        self.lr_scale = lr_scale

    def __call__(self, step: int) -> float:
        if step <= self.frozen_steps:
            return 0.0
        progress = (step - self.frozen_steps) / self.warmup_steps
        return max(0.0, min(progress, 1.0)) * self.lr_scale


class LinearWarmupLinearDecay:
    """Reference ``bfl.modules.lr_scheduler.LinearWarmupLinearDecay`` (torchtitan lineage).

    Relative to ``start_step``: warmup returns ``(step + 1) / (warmup_steps + 1)`` (so the first
    update is not at zero), an optional constant phase returns one, the decay returns
    ``1 - (step - offset) / decay_steps`` and raises one step past its end. Steps before
    ``start_step`` return ``lr_scale``. Every phase is optional through ``None``.
    """

    def __init__(
        self,
        warmup_steps: int | None,
        decay_steps: int | None,
        constant_steps: int | None = None,
        lr_scale: float = 1.0,
        start_step: int = 0,
    ):
        for name, value in (
            ("warmup_steps", warmup_steps),
            ("decay_steps", decay_steps),
            ("constant_steps", constant_steps),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None, got {value}")
        if start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {start_step}")
        self.warmup_steps = warmup_steps
        self.warmup_offset = 0 if warmup_steps is None else warmup_steps
        self.constant_steps = constant_steps
        self.constant_offset = (
            self.warmup_offset if constant_steps is None else self.warmup_offset + constant_steps
        )
        self.decay_steps = decay_steps
        self.lr_scale = lr_scale
        self.start_step = start_step

    def __call__(self, step: int) -> float:
        effective = step - self.start_step
        if effective < 0:
            return self.lr_scale
        if self.warmup_steps is not None and effective < self.warmup_steps:
            factor = (effective + 1) / (self.warmup_steps + 1)
        elif self.constant_steps is not None and effective < self.constant_steps + self.warmup_offset:
            factor = 1.0
        elif self.decay_steps is not None:
            if effective > self.decay_steps + self.constant_offset:
                raise RuntimeError(
                    f"step {step} (effective {effective}) lies past the scheduled end "
                    f"{self.start_step + self.constant_offset + self.decay_steps}"
                )
            factor = 1 - (effective - self.constant_offset) / self.decay_steps
        else:
            factor = 1.0
        return max(min(factor, 1.0), 0.0) * self.lr_scale


class CooldownFrom:
    """``base`` before ``start_step``; from there the reference cooldown overlay.

    The overlay is ``LinearWarmupLinearDecay(None, decay_steps, start_step=start_step)``: it
    returns ``lr_scale`` at ``start_step`` and reaches zero at ``start_step + decay_steps``. With
    the recipe's constant phase both sides of the boundary equal ``lr_scale``. Relaunching the
    reference with its cooldown overlay at ``start_step`` yields the same factors.
    """

    def __init__(self, base: LRLambda, start_step: int, decay_steps: int, lr_scale: float = 1.0):
        if decay_steps <= 0:
            raise ValueError(f"decay_steps must be positive, got {decay_steps}")
        self.base = base
        self.start_step = start_step
        self.decay_steps = decay_steps
        self.cooldown = LinearWarmupLinearDecay(None, decay_steps, start_step=start_step, lr_scale=lr_scale)

    def __call__(self, step: int) -> float:
        return self.base(step) if step < self.start_step else self.cooldown(step)


def droid_lr_lambdas(
    *,
    cooldown_start: int | None = DROID_COOLDOWN_START,
    cooldown_steps: int = DROID_COOLDOWN_STEPS,
    lr_scale: float = 1.0,
    frozen_steps: int = DROID_FROZEN_STEPS,
    trunk_warmup_steps: int = DROID_TRUNK_WARMUP_STEPS,
    heads_warmup_steps: int = DROID_HEADS_WARMUP_STEPS,
) -> tuple[LRLambda, LRLambda]:
    """Factors for ``FluxActionPolicy.get_optim_params()`` in its order: ``(trunk, heads)``.

    Trunk: zero through ``frozen_steps``, linear to one over ``trunk_warmup_steps``, constant. Heads:
    ``(g + 1) / (heads_warmup_steps + 1)`` for the first ``heads_warmup_steps`` updates, then constant.
    Defaults are the DROID recipe (1000 / 2000 / 1000); a short finetune passes smaller phases or its trunk
    never trains. ``cooldown_start=None`` leaves both constant (the reference's open-ended base launch); the
    default cools both down from 25000 to zero at 30000.
    """
    trunk: LRLambda = FrozenThenLinearWarmup(frozen_steps, trunk_warmup_steps, lr_scale)
    heads: LRLambda = LinearWarmupLinearDecay(heads_warmup_steps, None, lr_scale=lr_scale)
    if cooldown_start is not None:
        trunk = CooldownFrom(trunk, cooldown_start, cooldown_steps, lr_scale)
        heads = CooldownFrom(heads, cooldown_start, cooldown_steps, lr_scale)
    return trunk, heads


def build_lr_scheduler(optimizer: Optimizer, lr_lambdas: Sequence[LRLambda]) -> LambdaLR:
    """One ``LambdaLR`` with one factor per parameter group.

    The factors are wrapped in plain functions, as in the reference, so ``state_dict()`` does not
    serialize them: a resumed run always uses the schedule of its current configuration.
    """
    if len(lr_lambdas) != len(optimizer.param_groups):
        raise ValueError(f"{len(optimizer.param_groups)} parameter groups but {len(lr_lambdas)} schedules")
    return LambdaLR(optimizer, lr_lambda=[lambda step, f=f: f(step) for f in lr_lambdas])


def resume_lr_scheduler(scheduler: LambdaLR, step: int) -> None:
    """Reference resume after ``step`` completed updates.

    Loading the optimizer state restores stale ``lr`` values; stepping once from ``step - 1``
    makes update ``step`` use ``factor(step)``. A fresh run (``step == 0``) needs nothing.
    """
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if step == 0:
        return
    scheduler.last_epoch = step - 1
    with warnings.catch_warnings():
        # The fresh optimizer has not stepped yet; the reference performs the same priming step.
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)`")
        scheduler.step()
