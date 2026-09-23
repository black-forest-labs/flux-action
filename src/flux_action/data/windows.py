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
"""Compact valid-start sampling: one window per episode visit."""

from collections.abc import Iterable, Sequence
from numbers import Integral
from random import Random


def valid_window_spans(n_frames: int, valid_ranges: Sequence[Sequence[int]], chunk_size: int = 32):
    """Return (first_start, count) spans for complete chunk_size+1 frame windows.

    Ranges are half-open. Out-of-bounds endpoints are clipped to the episode.
    Range order/multiplicity is preserved; prepared metadata should be disjoint.
    Missing valid ranges are an error, never a request to use the whole episode.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, Integral) or n_frames < 0:
        raise ValueError("n_frames must be a nonnegative integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if valid_ranges is None:
        raise ValueError("valid_ranges are required")
    spans = []
    for pair in valid_ranges:
        if len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, Integral) for x in pair):
            raise ValueError("valid_ranges must contain integer [start, end) pairs")
        start, end = pair
        if end < start:
            raise ValueError("valid range end precedes start")
        first = max(int(start), 0)
        count = max(0, min(int(end) - chunk_size, n_frames - chunk_size) - first)
        if count:
            spans.append((first, count))
    return spans


def select_window_start(n_frames, valid_ranges, rng: Random, chunk_size=32) -> int | None:
    spans = valid_window_spans(n_frames, valid_ranges, chunk_size)
    total = sum(count for _, count in spans)
    if total == 0:
        return None
    offset = rng.randrange(total)
    for start, count in spans:
        if offset < count:
            return start + offset
        offset -= count
    raise AssertionError("unreachable valid-window offset")


def sample_episode_visits(episodes: Iterable, rng: Random, chunk_size=32):
    """Yield one (episode, start) per eligible input visit; caller owns visit order.

    This deliberately does not flatten episodes into a globally window-weighted
    dataset. Distributed order/sharding and resumable prefetch remain external.
    """
    for episode in episodes:
        start = select_window_start(episode.n_frames, episode.valid_ranges, rng, chunk_size)
        if start is not None:
            yield episode, start
