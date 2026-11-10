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
"""Absolute DROID end-effector commands from an EE-50 policy.

The policy predicts normalized EE-50 rows relative to the end-effector pose at request time. This module builds
the state token from the observed gripper and decodes the chunk with the observed pose, so every chunk is
anchored at the pose measured when it was requested. Use it with ``predict_action_chunk``; ``select_action``
queues commands without this decoding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import ee50


def predict_droid_commands(
    policy,
    observation: dict[str, Any],
    *,
    cartesian_position: np.ndarray,
    gripper_position: np.ndarray,
    stats: ee50.EE50Stats,
) -> np.ndarray:
    """One chunk of absolute commands per batch item.

    Args:
        policy: a ``FluxActionPolicy`` trained on EE-50 windows (``action_dim`` 50).
        observation: the cameras and ``task`` of the batch, as for ``predict_action_chunk``; ``state`` is
            built here.
        cartesian_position: ``(B, 6)`` observed ``[x, y, z, rx, ry, rz]`` at request time.
        gripper_position: ``(B,)`` observed gripper position (0 = open, 1 = closed).
        stats: the statistics the policy was trained with (``ee50_stats.json`` of its index).

    Returns:
        ``(B, chunk_size, 7)`` float64 ``[x, y, z, rx, ry, rz, gripper_position]`` commands.
    """
    anchors = np.asarray(cartesian_position, dtype=np.float64).reshape(-1, 6)
    grippers = np.asarray(gripper_position, dtype=np.float64).reshape(-1)
    if len(anchors) != len(grippers):
        raise ValueError("one Cartesian pose and one gripper position per batch item")
    state = torch.from_numpy(np.stack([ee50.droid_state_token(g, stats) for g in grippers]))
    reference = next((v for v in observation.values() if isinstance(v, torch.Tensor)), None)
    if reference is not None:
        state = state.to(reference.device)
    with torch.no_grad():
        chunk = policy.predict_action_chunk({**observation, "state": state})
    chunk = chunk.float().cpu().numpy()
    return np.stack([ee50.decode_droid_chunk(c, a, stats) for c, a in zip(chunk, anchors, strict=True)])
