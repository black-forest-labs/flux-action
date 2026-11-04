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
"""EE-50, the end-effector action layout of the base checkpoint's ``action_prediction_ee50`` heads.

A row has 50 values, a left block ``0:25`` and a right block ``25:50``, each
``[translation (3), rot6d (6), hand (16)]``. A single-arm robot fills the left block and leaves the right
block zero; a parallel gripper puts its open fraction (1 = open, 0 = closed) in the first hand channel
(dim 9) and leaves the other 15 zero.

Poses are relative to one anchor pose per window, ``T_rel = inv(T_anchor) @ T``:

* translation ``R_anchor^T (p - p_anchor)`` in metres, expressed in the anchor frame;
* rot6d, the first two rows of ``R_rel = R_anchor^T R``, row-major ``[r00 r01 r02 r10 r11 r12]``.

A window has ``K + 1`` rows at signed offsets ``0..K`` from the anchor. Row 0 is the conditioning state; rows
``1..K`` are the action targets. DROID windows are transition aligned: the anchor is the observed pose
``state[s]`` at the window's first frame, row 0 is ``state[s]`` itself (the identity pose), and row ``k > 0``
is the commanded pose ``action[s + k - 1]``, the command that leads to ``state[s + k]``.

Every value is z-scored with statistics per (offset, dim): ``clip((x - mean[o]) / std[o], -10, 10)`` with
``std < 1e-8`` replaced by 1, so constant and zero-padded dims stay 0. After normalization, the pose dims of
offset 0 (``0:9`` and ``25:34``) are set to exactly 0. The policy consumes these normalized values as they
are: the training config leaves ``action_normalization`` unset and uses action scale 1.

This module is numpy only, so the same code serves training, statistics and the robot-side decoding.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_DIM = 50
BLOCK = 25
GRIPPER_DIM = 9  # first hand channel of the left block: open fraction
POSE_DIMS = (*range(0, 9), *range(BLOCK, BLOCK + 9))
ACTIVE_DIMS = 10  # a single-arm robot with a parallel gripper uses dims 0..9
NORMALIZED_CLIP = 10.0
STD_FLOOR = 1e-8
STATS_FORMAT = "ee50_signed_offset_zscore"


# ---------------------------------------------------------------------------------------------- DROID poses
def droid_cartesian_to_matrices(cartesian: np.ndarray) -> np.ndarray:
    """DROID ``[x, y, z, rx, ry, rz]`` poses ``(N, 6)`` -> homogeneous ``(N, 4, 4)``, ``R = Rz(rz) Ry(ry) Rx(rx)``."""
    poses = np.asarray(cartesian, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 6 or not np.isfinite(poses).all():
        raise ValueError(f"Cartesian poses must be finite (N, 6), got {poses.shape}")
    sx, cx = np.sin(poses[:, 3]), np.cos(poses[:, 3])
    sy, cy = np.sin(poses[:, 4]), np.cos(poses[:, 4])
    sz, cz = np.sin(poses[:, 5]), np.cos(poses[:, 5])
    matrices = np.tile(np.eye(4), (len(poses), 1, 1))
    r = matrices[:, :3, :3]
    r[:, 0, 0], r[:, 0, 1], r[:, 0, 2] = cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx
    r[:, 1, 0], r[:, 1, 1], r[:, 1, 2] = sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx
    r[:, 2, 0], r[:, 2, 1], r[:, 2, 2] = -sy, cy * sx, cy * cx
    matrices[:, :3, 3] = poses[:, :3]
    return matrices


def matrices_to_droid_cartesian(matrices: np.ndarray) -> np.ndarray:
    """Inverse of :func:`droid_cartesian_to_matrices`: ``(N, 4, 4)`` -> ``(N, 6)``, angles in ``(-pi, pi]``."""
    m = np.asarray(matrices, dtype=np.float64)
    if m.ndim != 3 or m.shape[1:] != (4, 4):
        raise ValueError(f"matrices must be (N, 4, 4), got {m.shape}")
    r = m[:, :3, :3]
    ry = np.arctan2(-r[:, 2, 0], np.hypot(r[:, 0, 0], r[:, 1, 0]))
    rx = np.arctan2(r[:, 2, 1], r[:, 2, 2])
    rz = np.arctan2(r[:, 1, 0], r[:, 0, 0])
    return np.concatenate([m[:, :3, 3], np.stack([rx, ry, rz], axis=1)], axis=1)


def rot6d_rows_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """``(..., 6)`` -> ``(..., 3, 3)``: Gram-Schmidt on the two stored rows, third row their cross product."""
    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"rot6d must end in six values, got {values.shape}")
    row1 = values[..., :3] / np.clip(np.linalg.norm(values[..., :3], axis=-1, keepdims=True), 1e-8, None)
    row2 = values[..., 3:] - np.sum(row1 * values[..., 3:], axis=-1, keepdims=True) * row1
    row2 = row2 / np.clip(np.linalg.norm(row2, axis=-1, keepdims=True), 1e-8, None)
    return np.stack([row1, row2, np.cross(row1, row2)], axis=-2)


def gripper_open(closed_fraction: np.ndarray) -> np.ndarray:
    """DROID's gripper position (0 = open, 1 = closed) -> EE-50's open fraction."""
    return np.clip(1.0 - np.asarray(closed_fraction, dtype=np.float64), 0.0, 1.0)


# ------------------------------------------------------------------------------------------------ windows
def relative_rows(anchor: np.ndarray, targets: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Raw single-arm EE-50 rows ``(N, 50)`` of ``targets (N, 4, 4)`` relative to ``anchor (4, 4)``.

    ``gripper`` ``(N,)`` is the open fraction written to dim 9.
    """
    inverse = anchor[:3, :3].T
    rows = np.zeros((len(targets), ACTION_DIM), dtype=np.float64)
    rows[:, 0:3] = (targets[:, :3, 3] - anchor[:3, 3]) @ inverse.T
    rows[:, 3:9] = np.einsum("ij,njk->nik", inverse, targets[:, :3, :3])[:, :2, :].reshape(-1, 6)
    rows[:, GRIPPER_DIM] = gripper
    return rows


def droid_window(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """One transition-aligned DROID window in raw EE-50 units.

    Args:
        state: ``(7,)`` observed ``[x, y, z, rx, ry, rz, gripper_position]`` at the window's first frame ``s``.
        actions: ``(K, 7)`` the commands ``action[s .. s + K - 1]`` in the same layout.

    Returns:
        ``(K + 1, 50)`` float64 rows at offsets ``0..K``: the identity anchor row, then the commands relative
        to the observed anchor pose.
    """
    state = np.asarray(state, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if state.shape != (7,) or actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"expected state (7,) and actions (K, 7), got {state.shape} and {actions.shape}")
    poses = droid_cartesian_to_matrices(np.concatenate([state[None, :6], actions[:, :6]]))
    gripper = gripper_open(np.concatenate([state[None, 6], actions[:, 6]]))
    return relative_rows(poses[0], poses, gripper)


# ------------------------------------------------------------------------------------------ normalization
@dataclass(frozen=True)
class EE50Stats:
    """Per-offset z-score statistics: ``mean`` and ``std`` ``(len(offsets), 50)`` for contiguous ``offsets``."""

    offsets: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self):
        offsets = np.asarray(self.offsets, dtype=np.int64)
        if (
            offsets.ndim != 1
            or not len(offsets)
            or not np.array_equal(offsets, np.arange(offsets[0], offsets[-1] + 1))
        ):
            raise ValueError("offsets must be contiguous integers")
        for name in ("mean", "std"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (len(offsets), ACTION_DIM) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite ({len(offsets)}, {ACTION_DIM})")
            object.__setattr__(self, name, value)
        if np.any(self.std < 0):
            raise ValueError("std must be non-negative")
        object.__setattr__(self, "offsets", offsets)

    def rows(self, offsets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, std)`` for each offset; offsets outside the table use its nearest end, std floored to 1."""
        index = np.clip(np.asarray(offsets, dtype=np.int64) - self.offsets[0], 0, len(self.offsets) - 1)
        std = self.std[index]
        return self.mean[index], np.where(std < STD_FLOOR, 1.0, std)


def load_stats(path) -> EE50Stats:
    value = json.loads(Path(path).read_text())
    if value.get("format") != STATS_FORMAT or value.get("action_dim") != ACTION_DIM:
        raise ValueError(f"{path} is not an EE-50 statistics file")
    return EE50Stats(np.asarray(value["offsets"]), np.asarray(value["mean"]), np.asarray(value["std"]))


def save_stats(stats: EE50Stats, path, **provenance) -> None:
    """Write ``stats`` as JSON with free-form ``provenance`` fields (source, counts, filters), one row per line."""
    header = {
        "format": STATS_FORMAT,
        "action_dim": ACTION_DIM,
        **provenance,
        "offsets": stats.offsets.tolist(),
    }
    lines = [f" {json.dumps(key)}: {json.dumps(value)}," for key, value in header.items()]
    for name in ("mean", "std"):
        # shortest decimals that read back as the same float32 values
        rows = [" " * 2 + json.dumps([float(str(x)) for x in row]) for row in getattr(stats, name)]
        lines.append(f' "{name}": [\n' + ",\n".join(rows) + "\n ]" + ("," if name == "mean" else ""))
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("{\n" + "\n".join(lines) + "\n}\n")
    tmp.replace(path)


def normalize(rows: np.ndarray, offsets: Sequence[int], stats: EE50Stats) -> np.ndarray:
    """Raw rows ``(N, 50)`` at ``offsets (N,)`` -> normalized float32 rows (offset-0 pose dims set to 0)."""
    rows = np.asarray(rows, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != ACTION_DIM or offsets.shape != (len(rows),):
        raise ValueError(f"rows must be (N, {ACTION_DIM}) with one offset per row")
    mean, std = stats.rows(offsets)
    out = np.clip((rows - mean) / std, -NORMALIZED_CLIP, NORMALIZED_CLIP).astype(np.float32)
    out[np.ix_(offsets == 0, POSE_DIMS)] = 0.0
    return out


def denormalize(rows: np.ndarray, offsets: Sequence[int], stats: EE50Stats) -> np.ndarray:
    """Inverse of :func:`normalize` for offsets other than 0 (exact inside the clip range)."""
    rows = np.asarray(rows, dtype=np.float64)
    mean, std = stats.rows(offsets)
    return rows * std + mean


# ---------------------------------------------------------------------------------------------- inference
def droid_state_token(gripper_position: float, stats: EE50Stats) -> np.ndarray:
    """The normalized conditioning row ``(50,)`` for an observed DROID gripper position (0 = open).

    The anchor pose is the identity by construction, so only the gripper enters the state token.
    """
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_position])
    return normalize(droid_window(state, np.empty((0, 7)))[:1], [0], stats)[0]


def decode_droid_chunk(normalized: np.ndarray, anchor_cartesian: np.ndarray, stats: EE50Stats) -> np.ndarray:
    """Predicted normalized rows ``(K, 50)`` at offsets ``1..K`` -> absolute DROID commands ``(K, 7)``.

    Args:
        normalized: the policy's action chunk for one window.
        anchor_cartesian: ``(6,)`` the observed ``[x, y, z, rx, ry, rz]`` at the time of the request.
        stats: the statistics the policy was trained with.

    Returns:
        ``[x, y, z, rx, ry, rz, gripper_position]`` per command, in the base frame of the anchor pose and
        DROID's gripper convention (0 = open, 1 = closed).
    """
    normalized = np.asarray(normalized, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != ACTION_DIM:
        raise ValueError(f"normalized chunk must be (K, {ACTION_DIM}), got {normalized.shape}")
    rows = denormalize(normalized, np.arange(1, len(normalized) + 1), stats)
    anchor = droid_cartesian_to_matrices(np.asarray(anchor_cartesian, dtype=np.float64).reshape(1, 6))[0]
    relative = np.tile(np.eye(4), (len(rows), 1, 1))
    relative[:, :3, :3] = rot6d_rows_to_matrix(rows[:, 3:9])
    relative[:, :3, 3] = rows[:, 0:3]
    absolute = np.einsum("ij,njk->nik", anchor, relative)
    closed = 1.0 - np.clip(rows[:, GRIPPER_DIM], 0.0, 1.0)
    return np.concatenate([matrices_to_droid_cartesian(absolute), closed[:, None]], axis=1)


# ------------------------------------------------------------------------------------------------- stats
@dataclass
class Moments:
    """Running sums for :func:`finalize`, one row per offset ``0..max_offset``."""

    count: np.ndarray
    total: np.ndarray
    total2: np.ndarray

    @classmethod
    def empty(cls, max_offset: int) -> Moments:
        n = max_offset + 1
        return cls(np.zeros(n, np.int64), np.zeros((n, ACTION_DIM)), np.zeros((n, ACTION_DIM)))

    def add(self, other: Moments) -> None:
        self.count += other.count
        self.total += other.total
        self.total2 += other.total2

    def add_window(self, rows: np.ndarray) -> None:
        """Add the raw rows ``(k + 1, 50)`` of one anchor at offsets ``0..k``, for any embodiment.

        Use it with every anchor of your data and only the offsets whose target is valid; dims you leave
        zero get zero statistics and stay 0 after normalization.
        """
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != ACTION_DIM or len(rows) > len(self.count):
            raise ValueError(f"rows must be (k + 1, {ACTION_DIM}) with k <= {len(self.count) - 1}")
        self.count[: len(rows)] += 1
        self.total[: len(rows)] += rows
        self.total2[: len(rows)] += np.square(rows)


def droid_moments(
    state: np.ndarray, actions: np.ndarray, ranges: Sequence[Sequence[int]], max_offset: int
) -> Moments:
    """Sums of the raw rows of one episode for every anchor and offset ``0..max_offset``.

    Args:
        state: ``(N, 7)`` observed ``[x, y, z, rx, ry, rz, gripper_position]`` per frame.
        actions: ``(N, 7)`` commands in the same layout.
        ranges: half-open ``[start, end)`` valid frame ranges.
        max_offset: the largest offset (the chunk size).

    Every frame of a range is an anchor. Offset 0 uses the anchor's own state; offset ``k > 0`` uses
    ``action[a + k - 1]`` and only anchors with ``a + k`` inside the same range, so a command is paired with
    an anchor only when the frame it leads to is valid too, and the last command of a range is never used.
    """
    state_poses = droid_cartesian_to_matrices(np.asarray(state)[:, :6])
    action_poses = droid_cartesian_to_matrices(np.asarray(actions)[:, :6])
    state_grip, action_grip = gripper_open(np.asarray(state)[:, 6]), gripper_open(np.asarray(actions)[:, 6])
    moments = Moments.empty(max_offset)
    for start, end in ranges:
        for offset in range(min(max_offset, end - start - 1) + 1):
            anchors = np.arange(start, end - offset)
            if offset == 0:
                poses, grip = state_poses[anchors], state_grip[anchors]
            else:
                poses, grip = action_poses[anchors + offset - 1], action_grip[anchors + offset - 1]
            inverse = np.transpose(state_poses[anchors, :3, :3], (0, 2, 1))
            rows = np.zeros((len(anchors), ACTIVE_DIMS))
            rows[:, 0:3] = np.einsum("nij,nj->ni", inverse, poses[:, :3, 3] - state_poses[anchors, :3, 3])
            rows[:, 3:9] = np.einsum("nij,njk->nik", inverse, poses[:, :3, :3])[:, :2, :].reshape(-1, 6)
            rows[:, 9] = grip
            moments.count[offset] += len(rows)
            moments.total[offset, :ACTIVE_DIMS] += rows.sum(axis=0)
            moments.total2[offset, :ACTIVE_DIMS] += np.square(rows).sum(axis=0)
    return moments


def finalize(moments: Moments) -> EE50Stats:
    """Population mean and standard deviation per offset; dims that were always zero get zero for both."""
    if np.any(moments.count == 0):
        raise ValueError(f"offsets without samples: {np.flatnonzero(moments.count == 0).tolist()}")
    mean = moments.total / moments.count[:, None]
    std = np.sqrt(np.maximum(moments.total2 / moments.count[:, None] - np.square(mean), 0.0))
    return EE50Stats(np.arange(len(moments.count)), mean, std)
