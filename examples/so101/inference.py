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
"""Run a delivered SO-101 checkpoint without a LeRobot runtime dependency."""

import argparse
from pathlib import Path

import numpy as np

from flux_action.inference.offline import load_observation
from flux_action.inference.so101 import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Hub repo ID or local policy directory")
    parser.add_argument("--revision", help="Hub commit, branch or tag; defaults to the latest main")
    parser.add_argument(
        "--observation", type=Path, required=True, help="NPZ with images.scene/wrist and state"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/so101-actions.npy"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    policy = load_policy(args.weights, revision=args.revision).to(args.device)
    batch = load_observation(args.observation, policy.config, args.prompt, args.device)
    actions = policy.predict_action_chunk(batch).float().cpu().numpy()
    if (
        actions.shape != (1, policy.config.chunk_size, policy.config.action_dim)
        or not np.isfinite(actions).all()
    ):
        raise RuntimeError("policy returned invalid actions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.save(stream, actions, allow_pickle=False)
    print(f"Saved {actions.shape} actions to {args.output}")


if __name__ == "__main__":
    main()
