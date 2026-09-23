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
"""Prepare a policy for the released CUDA inference path."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch


def prepare_for_serving(
    policy,
    *,
    device: str | torch.device | None = None,
    compile_dit: bool = False,
    offload_text_encoder: bool = False,
    warmup: int = 0,
    warmup_fn: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    assert warmup >= 0 and (not warmup or warmup_fn is not None), "warmup callback required"
    if offload_text_encoder:
        policy.set_text_encoder_offload(True)
    if device is not None and str(policy.device) != str(device):
        policy.to(device)

    prepared = False
    if policy.device.type == "cuda":
        prepared = policy.prepare_inference(compile=compile_dit)
    assert prepared or not compile_dit, "acceleration unavailable"

    report: dict[str, Any] = {
        "prepared": prepared,
        "compile_dit": compile_dit,
        "offload_text_encoder": offload_text_encoder,
    }
    if warmup:
        times = []
        for _ in range(warmup):
            started = time.perf_counter()
            warmup_fn()
            if policy.device.type == "cuda":
                torch.cuda.synchronize(policy.device)
            times.append(time.perf_counter() - started)
        report["warmup_seconds"] = times
    return report


__all__ = ["prepare_for_serving"]
