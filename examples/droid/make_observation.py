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
"""Select a seeded valid frame from a prepared episode for offline inference."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from flux_action.data.droid.episodes import select_caption, validate_episode
from flux_action.data.windows import select_window_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    episode, output = Path(args.episode), Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("choose a new observation output path")
    metadata = validate_episode(episode)
    rng = random.Random(args.seed)
    start = select_window_start(metadata.n_frames, metadata.valid_ranges, rng)
    task = select_caption(metadata.caption, rng)
    with np.load(episode / "episode.npz", allow_pickle=False) as arrays:
        observation = {f"images.{k}": arrays[k][start] for k in metadata.camera_order}
        observation["state"] = arrays["state"][start]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez(stream, **observation)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "episode_id": metadata.episode_id,
                "frame_index": start,
                "seed": args.seed,
                "task": task,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
