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
"""Karras power-function EMA, ported from the reference trainer's averaging callback.

Profile: ``sigma_rel`` maps to ``gamma``, the largest real root of
``g^3 + 7 g^2 + (16 - sigma_rel^-2) g + (12 - sigma_rel^-2)`` (EDM2 ``solve_gamma``; equivalently
``sigma_rel^2 = (g + 1) / ((g + 2)^2 (g + 3))``). The update after optimizer update ``g``
(zero-based; the reference passes its ``global_step`` before incrementing it) uses
``beta = (1 - 1 / (g + 1)) ** (gamma + 1)`` in ``ema = beta * ema + (1 - beta) * model``;
``g == 0`` copies the model instead. ``beta`` depends only on the global step, so a resumed run
continues the same profile.

The reference kept every EMA as a second FSDP-sharded module with the model's structure.
:class:`PowerEMA` keeps a shadow tensor per parameter in FP32 with the parameter's device and
(DTensor) placement, so no rank holds a full copy; :meth:`PowerEMA.consolidated` gathers full
tensors for export. Buffers are not averaged; the reference copied them from the model.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .distributed import clean_name

DROID_POWER_EMA_SIGMA_RELS = (0.10, 0.05)
SIGMA_REL_UPPER = 12**-0.5


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Reference ``power_ema_sigma_rel_to_gamma``: largest real root of the EDM2 cubic."""
    sigma_rel = float(sigma_rel)
    if not 0.0 < sigma_rel < SIGMA_REL_UPPER:
        raise ValueError(f"sigma_rel must lie in (0, 1 / sqrt(12)), got {sigma_rel}")
    t = sigma_rel**-2
    return float(np.roots([1, 7, 16 - t, 12 - t]).real.max())


def power_ema_beta(gamma: float, step: int) -> float:
    """Reference ``power_ema_decay``: zero at update 0 (copy), ``(1 - 1 / (step + 1)) ** (gamma + 1)`` after."""
    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if step == 0:
        return 0.0
    return (1 - 1 / (step + 1)) ** (gamma + 1)


def _copy_each(targets: Sequence[Tensor], sources: Sequence[Tensor]) -> None:
    """Per-tensor ``copy_`` as in the reference; ``_foreach_copy_`` has no DTensor sharding rule."""
    for target, source in zip(targets, sources, strict=True):
        target.copy_(source)


class PowerEMA:
    """Shadow copies of named tensors, updated with the power-function schedule."""

    def __init__(self, named_tensors: Iterable[tuple[str, Tensor]], sigma_rel: float):
        self.sigma_rel = float(sigma_rel)
        self.gamma = sigma_rel_to_gamma(sigma_rel)
        names, shadow = [], []
        for name, tensor in named_tensors:
            if not tensor.is_floating_point():
                raise TypeError(f"{name}: power EMA averages floating-point tensors only")
            names.append(name)
            shadow.append(tensor.detach().to(dtype=torch.float32, copy=True))
        if not names:
            raise ValueError("no tensors to average")
        self.names: tuple[str, ...] = tuple(names)
        self.shadow: list[Tensor] = shadow
        self.last_step: int | None = None

    @classmethod
    def from_module(cls, module: nn.Module, sigma_rel: float) -> "PowerEMA":
        """Average every trainable parameter of ``module`` (frozen encoders stay out).

        Names drop activation-checkpoint wrapper segments so they match the plain model's keys.
        """
        return cls(((clean_name(n), p) for n, p in module.named_parameters() if p.requires_grad), sigma_rel)

    @torch.no_grad()
    def update(self, tensors: Sequence[Tensor], step: int) -> float:
        """Fold the current ``tensors`` in after optimizer update ``step``; returns the beta used."""
        if len(tensors) != len(self.shadow):
            raise ValueError(f"expected {len(self.shadow)} tensors, got {len(tensors)}")
        current = [t.detach().float() for t in tensors]
        beta = power_ema_beta(self.gamma, step)
        if step == 0:
            _copy_each(self.shadow, current)
        else:
            torch._foreach_lerp_(self.shadow, current, 1 - beta)
        self.last_step = step
        return beta

    def update_from_module(self, module: nn.Module, step: int) -> float:
        params = {clean_name(n): p for n, p in module.named_parameters()}
        return self.update([params[n] for n in self.names], step)

    @torch.no_grad()
    def copy_to(self, tensors: Sequence[Tensor]) -> None:
        """Write the averaged values into ``tensors`` (evaluation swap or export into a model)."""
        if len(tensors) != len(self.shadow):
            raise ValueError(f"expected {len(self.shadow)} tensors, got {len(tensors)}")
        _copy_each([t.detach() for t in tensors], self.shadow)

    def state_dict(self) -> dict[str, object]:
        """Local shadow tensors (shards for DTensors) plus the profile and last step."""
        return {
            "sigma_rel": self.sigma_rel,
            "last_step": self.last_step,
            "tensors": dict(zip(self.names, self.shadow, strict=True)),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if float(state["sigma_rel"]) != self.sigma_rel:
            raise ValueError(
                f"checkpoint profile sigma_rel={state['sigma_rel']} differs from {self.sigma_rel}"
            )
        tensors = state["tensors"]
        if set(tensors) != set(self.names):
            raise ValueError("checkpoint tensor names differ from the averaged tensors")
        with torch.no_grad():
            _copy_each(self.shadow, [tensors[n] for n in self.names])
        self.last_step = state["last_step"]

    @torch.no_grad()
    def consolidated(self) -> dict[str, Tensor]:
        """Full tensors on the CPU, gathering DTensor shards; every rank must call this."""
        out = {}
        for name, tensor in zip(self.names, self.shadow, strict=True):
            full = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
            out[name] = full.detach().cpu().contiguous()
        return out
