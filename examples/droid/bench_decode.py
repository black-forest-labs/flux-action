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
"""Measure window throughput of the streaming DROID loader on an indexed dataset."""

import argparse
import time

from flux_action.data.droid.index import ROWS_FILENAME, load_manifest
from flux_action.training.data import DroidWindowDataset, build_dataloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--index-dir", required=True, help="directory written by flux-action index-droid")
    parser.add_argument("--decoder", default="ffmpeg", choices=["ffmpeg", "pyav"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--windows-per-rank", type=int, default=8)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = load_manifest(f"{args.index_dir}/manifest.json")
    dataset = DroidWindowDataset(
        manifest,
        args.source_root,
        f"{args.index_dir}/{ROWS_FILENAME}",
        seed=args.seed,
        num_workers=args.workers,
        windows_per_rank=args.windows_per_rank,
        decoder=args.decoder,
    )
    loader = build_dataloader(dataset, in_process=args.workers == 1)
    frames_per_window = 3 * (manifest["chunk_size"] + 1)
    start = time.perf_counter()
    arrivals = []
    for i, _batch in enumerate(loader):
        arrivals.append(time.perf_counter())
        if i == 0:
            print(f"first batch after {arrivals[0] - start:.2f} s (worker start-up and prefetch included)")
        if i + 1 >= args.batches:
            break
    if len(arrivals) < 4:
        raise SystemExit("need at least four batches to report a steady-state rate")
    # Prefetched batches arrive together; measure the second half of the run.
    half = len(arrivals) // 2
    elapsed = arrivals[-1] - arrivals[half - 1]
    windows = (len(arrivals) - half) * args.windows_per_rank
    print(
        f"steady state over {len(arrivals) - half} batches: {windows / elapsed:.2f} windows/s, "
        f"{windows * frames_per_window / elapsed:.0f} frames/s with {args.workers} workers; "
        f"{elapsed / (len(arrivals) - half):.2f} s per batch of {args.windows_per_rank} windows; "
        f"{windows / elapsed / args.workers:.2f} windows/s per worker"
    )


if __name__ == "__main__":
    main()
