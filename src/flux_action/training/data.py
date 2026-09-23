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
"""Streaming training windows over an indexed dataset: deterministic order, one window per episode visit.

The manifest comes from ``flux-action index-droid`` (the compact Cosmos3-DROID layout: three 360x640
cameras, 8-dimensional state and action) or ``flux-action index-lerobot`` (a LeRobot dataset: the
manifest names its camera streams, row widths and native camera sizes, and windows start at frame 1 so the
command before the window, ``action_prev``, exists for the joint-delta parameterization). Frames are
decoded at the manifest's native size and, for LeRobot cameras, rescaled to ``frame_hw``.

A visit of episode ``e`` in epoch ``k`` of a run with seed ``s`` draws its window start, caption
paraphrase and ``window_seed`` (the seed of the policy's augmentation and caption-dropout draws for that
window) from ``Random(f"{s}:{k}:{e}")``, so the set of windows of an epoch and their augmentations do not
depend on how many ranks or workers share it, nor on when the policy encodes the window; only the
assignment does. Episodes are shuffled per epoch,
split ``[rank::world_size]`` across ranks and ``[worker::num_workers]`` across loader workers.
``DataLoader`` batches ``windows_per_rank`` consecutive windows of one worker and serves workers in
order, so batch ``i`` comes from worker ``i % num_workers``.

An epoch has the same number of batches on every rank, a multiple of the accumulation steps:
``batches_per_rank = (n_episodes // (world_size * num_workers * windows_per_rank)) * num_workers``
rounded down to the accumulation multiple; the leftover episodes of the epoch are skipped. Ranks
therefore run in lockstep (FSDP collectives, epoch boundaries, ``max_epochs``) and one recorded
:class:`DataPosition` describes all of them. On an exact resume the dataset skips the consumed batches
and rotates the worker assignment so the loader, which always starts with worker 0, continues the
original order.

Frames stay uint8 until the policy moves them to the accelerator (2.2 GB per 32-window batch
instead of 8.7 GB as float32).
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from ..data.droid import index as droid_index
from ..data.droid.episodes import CAMERAS, select_caption
from ..data.droid.video import FRAME_HW, decode_window
from ..data.lerobot import index as lerobot_index
from ..data.windows import select_window_start, valid_window_spans

DROID_STATE_DIM = DROID_ACTION_DIM = 8
GRAY_LEVEL = 128  # the flat tile of a camera stream the index declared absent


def load_manifest(path) -> dict:
    """A DROID or LeRobot manifest, validated by its own loader (``kind`` tells them apart)."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("kind") == lerobot_index.KIND:
        return lerobot_index.load_manifest(path)
    return droid_index.load_manifest(path)


def manifest_dims(manifest: dict) -> tuple[int, int]:
    """``(state_dim, action_dim)`` of the rows array; DROID manifests predate the fields."""
    return int(manifest.get("state_dim", DROID_STATE_DIM)), int(manifest.get("action_dim", DROID_ACTION_DIM))


@dataclass
class DataPosition:
    """Loader position of one rank; exact only for the recorded topology."""

    epoch: int = 0
    batches_consumed: int = 0
    world_size: int = 1
    num_workers: int = 1
    windows_per_rank: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataPosition:
        return cls(**data)

    def matches(self, *, world_size: int, num_workers: int, windows_per_rank: int) -> bool:
        return (self.world_size, self.num_workers, self.windows_per_rank) == (
            world_size,
            num_workers,
            windows_per_rank,
        )


def epoch_order(n_episodes: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n_episodes))
    Random(f"{seed}:{epoch}").shuffle(order)
    return order


class WindowDataset(IterableDataset):
    """Windows of an indexed dataset (``load_manifest``): ``images.<camera>`` uint8 ``(chunk + 1, 3, H, W)``
    per camera, ``state`` ``(S,)`` at the window's first frame, ``action`` ``(chunk, A)`` commands, and, when
    the manifest provides it, ``action_prev`` ``(A,)``: the command before the window.

    With ``n_obs_steps=H``, the start is the current control tick: images span
    ``[start-H+1, start+chunk]``, states span ``[start-H+1, start]``, and
    ``command_history`` spans ``[start-H, start-1]``. Actions still start at ``start``.
    Only complete histories within a valid episode range are sampled.
    """

    def __init__(
        self,
        manifest: dict,
        source_root,
        rows_path,
        *,
        seed: int,
        epoch: int = 0,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 1,
        windows_per_rank: int = 1,
        skip_batches: int = 0,
        grad_accumulation: int = 1,
        decoder: str = "ffmpeg",
        frame_hw: tuple[int, int] = FRAME_HW,
        visits_per_epoch: int = 1,
        split: str | None = "train",
        n_obs_steps: int | None = None,
    ):
        if not 0 <= rank < world_size:
            raise ValueError("rank must lie in [0, world_size)")
        if num_workers < 1 or windows_per_rank < 1 or skip_batches < 0 or grad_accumulation < 1:
            raise ValueError(
                "num_workers, windows_per_rank and grad_accumulation must be positive, skip_batches non-negative"
            )
        if visits_per_epoch < 1:
            raise ValueError("visits_per_epoch must be positive")
        if n_obs_steps is not None and (n_obs_steps < 1 or not manifest.get("action_prev")):
            raise ValueError("history windows need positive n_obs_steps and an index with previous commands")
        self.n_obs_steps = n_obs_steps
        self.manifest = manifest
        # episodes carry a split when the index held some out (LeRobot ``--val-episodes``); None takes all
        self.episodes = [e for e in manifest["episodes"] if split is None or e.get("split", "train") == split]
        # Episodes that cannot supply one complete history window (n_obs_steps observed ticks plus the
        # chunk inside one keep range) are excluded; the count is kept for diagnostics.
        self.episodes_without_history = 0
        if n_obs_steps is not None:
            eligible = [
                e
                for e in self.episodes
                if valid_window_spans(e["n_frames"], self._valid_ranges(e), manifest["chunk_size"])
            ]
            self.episodes_without_history = len(self.episodes) - len(eligible)
            self.episodes = eligible
        self.split, self.visits_per_epoch = split, visits_per_epoch
        self.source_root = source_root
        self.rows_path = rows_path
        self.fps = int(manifest["fps"])
        self.chunk_size = int(manifest["chunk_size"])
        self.total_frames = int(manifest["total_frames"])
        self.state_dim, self.action_dim = manifest_dims(manifest)
        self.cameras = tuple(manifest.get("camera_order", CAMERAS))
        self.action_prev = bool(manifest.get("action_prev", False))
        self.seed, self.epoch = seed, epoch
        self.rank, self.world_size = rank, world_size
        self.num_workers, self.windows_per_rank = num_workers, windows_per_rank
        self.skip_batches = skip_batches
        self.decoder, self.frame_hw = decoder, tuple(frame_hw)
        # LeRobot manifests record the native size of every camera; frames of another size are rescaled to
        # frame_hw. DROID manifests do not, and their frames must already have frame_hw (never rescaled).
        native = manifest.get("camera_hw") or {}
        self.resize = {
            camera: native.get(camera) is not None and tuple(native[camera]) != self.frame_hw
            for camera in self.cameras
        }
        self._rows = None
        per_worker = self.n_positions // (world_size * num_workers * windows_per_rank)
        updates = per_worker * num_workers // grad_accumulation
        self.batches_per_rank = updates * grad_accumulation
        if self.batches_per_rank == 0:
            history = (
                f" ({self.episodes_without_history} more episodes were dropped because they are too short "
                f"for {n_obs_steps} observed ticks plus {manifest['chunk_size']} actions inside one keep range)"
                if self.episodes_without_history
                else ""
            )
            raise ValueError(
                f"{len(self.episodes)} episodes x {visits_per_epoch} visits cannot fill one update on "
                f"{world_size} ranks x {num_workers} workers x {windows_per_rank} windows x {grad_accumulation} "
                f"accumulation steps{history}; raise visits_per_epoch or shrink the update"
            )
        if skip_batches > self.batches_per_rank:
            raise ValueError(
                f"skip_batches {skip_batches} exceeds the {self.batches_per_rank} batches of an epoch"
            )

    @property
    def rows(self) -> np.ndarray:
        if self._rows is None:  # opened lazily so the memmap is created inside each worker
            rows = np.load(self.rows_path, mmap_mode="r")
            expected = (self.total_frames, self.state_dim + self.action_dim)
            if rows.dtype != np.float32 or rows.shape != expected:
                raise ValueError(f"rows array must be float32 {expected}, got {rows.dtype} {rows.shape}")
            self._rows = rows
        return self._rows

    @property
    def n_positions(self) -> int:
        """Episode visits per epoch: position ``p`` is episode ``p % n_episodes``, visit ``p // n_episodes``."""
        return len(self.episodes) * self.visits_per_epoch

    def rank_order(self) -> list[int]:
        return epoch_order(self.n_positions, self.seed, self.epoch)[self.rank :: self.world_size]

    def worker_batches(self, worker_id: int) -> int:
        """Batches worker ``worker_id`` contributes to the epoch (batch ``i`` comes from worker ``i % n``)."""
        remaining = self.batches_per_rank - worker_id
        return -(-remaining // self.num_workers) if remaining > 0 else 0  # ceil division

    def worker_positions(self, worker_id: int) -> list[int]:
        """Episode positions logical worker ``worker_id`` yields after skipping its consumed batches."""
        mine = self.rank_order()[worker_id :: self.num_workers]
        produced = 0
        if self.skip_batches > worker_id:
            produced = -(-(self.skip_batches - worker_id) // self.num_workers)
        return mine[produced * self.windows_per_rank : self.worker_batches(worker_id) * self.windows_per_rank]

    def logical_worker(self, physical_worker: int) -> int:
        """The loader serves physical workers from 0; after ``skip_batches`` the next batch is due from
        logical worker ``skip_batches % num_workers``, so the assignment rotates by that amount."""
        return (physical_worker + self.skip_batches) % self.num_workers

    def __iter__(self):
        info = get_worker_info()
        worker_id, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        if workers != self.num_workers:
            raise RuntimeError(f"dataset built for {self.num_workers} workers but the loader runs {workers}")
        for position in self.worker_positions(self.logical_worker(worker_id)):
            yield self.window(position)

    def _valid_ranges(self, episode: dict) -> list:
        # The index permits one preceding command outside each range. Extend that
        # requirement to a complete command/state/image history; never pad it.
        shift = (self.n_obs_steps or 1) - 1
        return [[lo + shift, hi] for lo, hi in episode["valid_ranges"] if lo + shift < hi]

    def window(self, position: int) -> dict[str, Any]:
        """The window of visit ``position`` (see :attr:`n_positions`) for this seed and epoch."""
        episode = self.episodes[position % len(self.episodes)]
        visit = position // len(self.episodes)
        key = f"{self.seed}:{self.epoch}:{episode['episode_index']}"  # visit 0: the DROID loader's key
        rng = Random(key if visit == 0 else f"{key}:{visit}")
        start = select_window_start(episode["n_frames"], self._valid_ranges(episode), rng, self.chunk_size)
        if start is None:
            raise ValueError(f"episode {episode['episode_id']} has no valid window")
        task = select_caption(episode["caption"], rng)
        return self.window_at(
            position % len(self.episodes), start, task=task, window_seed=rng.getrandbits(63)
        )

    def valid_starts(self, episode_position: int) -> list[int]:
        """Every allowed window start of an episode, in order (evaluation picks from these)."""
        episode = self.episodes[episode_position]
        spans = valid_window_spans(episode["n_frames"], self._valid_ranges(episode), self.chunk_size)
        return [s for first, count in spans for s in range(first, first + count)]

    def window_at(
        self, episode_position: int, start: int, *, task: str | None = None, window_seed: int = 0
    ) -> dict[str, Any]:
        """The window of episode ``episode_position`` starting at frame ``start`` (must be an allowed start);
        ``task`` defaults to the caption's first paraphrase."""
        episode = self.episodes[episode_position]
        if start not in self.valid_starts(episode_position):
            raise ValueError(f"episode {episode['episode_id']}: {start} is not an allowed window start")
        if task is None:
            task = select_caption(episode["caption"])
        first = episode["from_index"] + start
        before = self.n_obs_steps or (
            1 if self.action_prev else 0
        )  # the row before the window (start >= 1 by the manifest)
        if start - before < 0:
            raise ValueError(f"episode {episode['episode_id']}: window at {start} has no previous command")
        rows = np.array(self.rows[first - before : first + self.chunk_size + 1])
        if not np.isfinite(rows).all():
            raise ValueError(f"episode {episode['episode_id']}: rows missing from the rows array")
        s = self.state_dim
        item: dict[str, Any] = {
            "state": torch.from_numpy(rows[before, :s].copy()),
            "action": torch.from_numpy(rows[before : before + self.chunk_size, s:].copy()),
            "task": task,
            "episode_index": episode["episode_index"],
            "start": start,
            "window_seed": window_seed,
        }
        if self.action_prev:
            item["action_prev"] = torch.from_numpy(rows[before - 1, s:].copy())
        frame_start, frame_count = start, self.chunk_size + 1
        if self.n_obs_steps is not None:
            item["state"] = torch.from_numpy(rows[1 : before + 1, :s].copy())
            item["command_history"] = torch.from_numpy(rows[:before, s:].copy())
            frame_start = start - before + 1
            frame_count = before + self.chunk_size
        for camera in self.cameras:
            video = episode["videos"][camera]
            if video is None:  # a stream the index declared gray (no such camera in the dataset)
                item[f"images.{camera}"] = torch.full(
                    (frame_count, 3, *self.frame_hw), GRAY_LEVEL, dtype=torch.uint8
                )
                continue
            frames = decode_window(
                f"{self.source_root}/{video['file']}",
                video["first_frame"] + frame_start,
                frame_count,
                fps=self.fps,
                frame_hw=self.frame_hw,
                decoder=self.decoder,
                resize=self.resize[camera],
            )
            item[f"images.{camera}"] = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        return item


DroidWindowDataset = WindowDataset  # the name the DROID trainer and its tests were written against


def collate_windows(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {"task": [it["task"] for it in items]}
    for key in items[0]:
        if key == "task":
            continue
        values = [it[key] for it in items]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
    return batch


def build_dataloader(
    dataset: WindowDataset,
    *,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    in_process: bool = False,
) -> DataLoader:
    """Fixed ``windows_per_rank`` per batch, workers served in order, incomplete tail dropped.

    ``in_process`` runs the dataset in the calling process (tests, CPU runs); it needs ``num_workers == 1``.
    The loader gets its own generator seeded from the run seed, epoch and rank, so creating it never
    touches the global RNG (which an exact resume restores from the checkpoint).
    """
    if in_process and dataset.num_workers != 1:
        raise ValueError("in-process loading needs a dataset built for one worker")
    generator = torch.Generator().manual_seed(
        int.from_bytes(
            hashlib.sha256(f"{dataset.seed}:{dataset.epoch}:{dataset.rank}".encode()).digest()[:8], "big"
        )
        % (2**63 - 1)
    )
    kwargs: dict[str, Any] = {"generator": generator}
    if not in_process:
        kwargs.update(num_workers=dataset.num_workers, prefetch_factor=prefetch_factor, in_order=True)
        if sys.platform.startswith("linux"):
            # workers start after CUDA/NCCL initialization; a forked child inheriting that state can deadlock
            kwargs["multiprocessing_context"] = "forkserver"
    return DataLoader(
        dataset,
        batch_size=dataset.windows_per_rank,
        drop_last=True,
        collate_fn=collate_windows,
        pin_memory=pin_memory,
        **kwargs,
    )
