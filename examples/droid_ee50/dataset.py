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
"""DROID windows in the EE-50 layout, as a trainer plug.

``TrainConfig.dataset = "examples.droid_ee50.dataset:build"``; ``index_dir`` is the output directory of
``python -m examples.droid_ee50.index``. Sampling, video decoding and resume are those of the DROID index
loader; only ``state`` and ``action`` change: ``state`` is the normalized anchor row ``(50,)`` and ``action``
the normalized chunk ``(32, 50)``, see :mod:`examples.droid_ee50.ee50`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from flux_action.data.droid.index import ROWS_FILENAME, load_manifest
from flux_action.training.data import WindowDataset

from . import ee50
from .index import STATS_FILENAME


class EE50WindowDataset(WindowDataset):
    """:class:`WindowDataset` over the Cartesian rows, with each window converted to normalized EE-50."""

    def __init__(self, *args: Any, stats: ee50.EE50Stats, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if (self.state_dim, self.action_dim) != (7, 7):
            raise ValueError("expected the Cartesian rows of examples.droid_ee50.index (7 + 7 per frame)")
        if stats.offsets[0] > 0 or stats.offsets[-1] < self.chunk_size:
            raise ValueError(f"statistics must cover offsets 0..{self.chunk_size}")
        self.stats = stats

    def window_at(self, episode_position: int, start: int, **kwargs: Any) -> dict[str, Any]:
        item = super().window_at(episode_position, start, **kwargs)
        raw = ee50.droid_window(item["state"].numpy(), item["action"].numpy())
        rows = ee50.normalize(raw, np.arange(len(raw)), self.stats)
        item["state"] = torch.from_numpy(rows[0])
        item["action"] = torch.from_numpy(rows[1:])
        return item


def build(config, **loader_kwargs: Any) -> EE50WindowDataset:
    index = Path(config.index_dir)
    return EE50WindowDataset(
        load_manifest(index / "manifest.json"),
        config.source_root,
        index / ROWS_FILENAME,
        stats=ee50.load_stats(index / STATS_FILENAME),
        decoder=config.decoder,
        frame_hw=config.frame_hw,
        visits_per_epoch=config.visits_per_epoch,
        split="train",
        **loader_kwargs,
    )
