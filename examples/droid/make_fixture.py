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
"""Write synthetic decoded arrays for exercising the intermediate preparation command."""

import argparse
import json
from pathlib import Path

import numpy as np

from flux_action.data.schema import EpisodeMetadata

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
metadata = EpisodeMetadata("synthetic-episode", "synthetic-v1", "test", 33, ((0, 33),), "synthetic test")
(args.output / "metadata.json").write_text(json.dumps(metadata.to_dict(), indent=2) + "\n")
np.savez_compressed(
    args.output / "arrays.npz",
    **{
        name: np.full((33, 360, 640, 3), value, dtype=np.uint8)
        for name, value in zip(("wrist", "left", "right"), (32, 96, 160), strict=True)
    },
    timestamps=np.arange(33, dtype=np.float64) / 15,
    state=np.zeros((33, 8), dtype=np.float32),
    action=np.zeros((33, 8), dtype=np.float32),
)
