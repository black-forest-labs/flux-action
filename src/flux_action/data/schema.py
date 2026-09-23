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
"""Versioned intermediate episode metadata. No implicit unit or gripper conversion."""

import math
from dataclasses import asdict, dataclass

from .windows import valid_window_spans


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    source_version: str
    split: str
    n_frames: int
    valid_ranges: tuple[tuple[int, int], ...]
    caption: str
    schema_version: int = 1
    fps: float = 15.0
    camera_order: tuple[str, ...] = ("wrist", "left", "right")
    action_space: str = "absolute_joint_position"
    joint_units: str = "radians"
    gripper_convention: str = "closed_fraction"

    def __post_init__(self):
        object.__setattr__(self, "valid_ranges", tuple(tuple(r) for r in self.valid_ranges))
        object.__setattr__(self, "camera_order", tuple(self.camera_order))
        if self.schema_version != 1 or not self.episode_id or not self.source_version or not self.split:
            raise ValueError("version 1 metadata requires episode_id, source_version and split")
        if not isinstance(self.caption, str):
            raise ValueError("caption must be a string")
        if not math.isfinite(self.fps) or self.fps != 15.0:
            raise ValueError("DROID intermediate requires 15 Hz timestamps")
        if self.camera_order != ("wrist", "left", "right"):
            raise ValueError("DROID cameras must be wrist, left, right")
        if (self.action_space, self.joint_units, self.gripper_convention) != (
            "absolute_joint_position",
            "radians",
            "closed_fraction",
        ):
            raise ValueError("expected absolute joints in radians and closed-fraction gripper")
        valid_window_spans(self.n_frames, self.valid_ranges)
        previous_end = 0
        for start, end in self.valid_ranges:
            if not 0 <= start < end <= self.n_frames or start < previous_end:
                raise ValueError("prepared valid ranges must be ordered, disjoint and within the episode")
            previous_end = end

    def to_dict(self):
        return asdict(self)
