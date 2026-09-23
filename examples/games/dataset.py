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
"""GRUNT and VECTOR as one FLUX Action training stream through the trainer.

Each game keeps its own recorded episodes; this module gives them one contract so they train through the same
heads and the same loop as DROID:

    modality "game", action_dim 4:
        GRUNT   [move, strafe, turn, fire]
        VECTOR  [steer, throttle, nitro, 0]     4th dim padded, masked from the loss
    caption: the game's own line, the only thing that says which game a window is from.
    state: the last executed action, same padding.

Wired in with ``TrainConfig.dataset = "examples.games.dataset:build"`` (the trainer imports and calls
``build``). ``source_root`` is "grunt=/path,vector=/path"; ``index_dir`` is unused.

Sampling follows the DROID loader: one window per episode visit, episodes in a seeded epoch order, sliced by
rank then worker, fixed ``windows_per_rank`` per batch, exact-position resume via ``skip_batches``. The epoch
order shuffles the union of episodes, so the game mix is proportional to episode counts (800 / 800: half each).
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from random import Random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

CAMERA = "images.game"
MODALITY = "game"
ACTION_DIM = 4
WINDOW = 33
CHUNK = 32

GAMES: dict[str, dict[str, Any]] = {
    "grunt": dict(dim=4, caption="play the shooter: hunt the grunts, dodge the plasma, stay alive"),
    "vector": dict(
        dim=3,
        caption="drive the racer: read the road, brake before the bends, pass the traffic, take the fuel",
    ),
    # rotor (indoor drone, [fwd, lat, up, yaw]): the caption is per episode, the flight instruction the pilot
    # followed ("take off, fly under the table, and land behind the bookcase"). index.json carries it as
    # episodes[i].task; caption=None means read it from the episode.
    "rotor": dict(dim=4, caption=None, prefix="fly the drone: "),
}


def parse_roots(source_root: str) -> dict[str, list[Path]]:
    """``grunt=/a,vector=/b+/c`` -> {game: [roots]}; ``+`` joins several datasets of one game (e.g. two
    recording sessions). Episodes of all roots are pooled with equal weight per episode."""
    roots: dict[str, list[Path]] = {}
    for part in str(source_root).split(","):
        game, _, path = part.partition("=")
        if not _ or game not in GAMES:
            raise ValueError(
                f"source_root parts must be <game>=<path>[+<path>...] with game in {list(GAMES)}, got {part!r}"
            )
        roots[game] = [Path(p) for p in path.split("+") if p]
    return roots


def load_episodes(roots: dict[str, list[Path]]) -> list[dict[str, Any]]:
    """The union of every game's episodes, in a fixed order (game order of ``roots``, root order, then index order)."""
    episodes = []
    for game, game_roots in roots.items():
        for root in game_roots:
            index = json.loads((root / "index.json").read_text())
            if int(index.get("action_dim", GAMES[game]["dim"])) != GAMES[game]["dim"]:
                raise ValueError(
                    f"{game}: dataset action_dim {index.get('action_dim')} != {GAMES[game]['dim']}"
                )
            length = int(index["length"])
            if length < WINDOW:
                raise ValueError(f"{game}: episodes have {length} frames, need >= {WINDOW}")
            for e in index["episodes"]:
                ep = {"game": game, "file": str(root / e["file"]), "length": length}
                if GAMES[game]["caption"] is None:
                    if not e.get("task"):
                        raise ValueError(
                            f"{game}: per-episode captions required but episodes[].task missing in {root / 'index.json'}"
                        )
                    ep["caption"] = GAMES[game].get("prefix", "") + str(e["task"])
                episodes.append(ep)
    return episodes


def caption_of(e: dict[str, Any]) -> str:
    """The game's fixed line, or the episode's own instruction when the game has per-episode captions."""
    fixed = GAMES[e["game"]]["caption"]
    return fixed if fixed is not None else e["caption"]


def pad4(a: np.ndarray) -> np.ndarray:
    if a.shape[-1] == ACTION_DIM:
        return a
    out = np.zeros(a.shape[:-1] + (ACTION_DIM,), dtype=a.dtype)
    out[..., : a.shape[-1]] = a
    return out


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    Random(f"{seed}:{epoch}").shuffle(order)
    return order


class GameWindowDataset(IterableDataset):
    """Same slicing contract as ``training.data.DroidWindowDataset`` (see that class for the topology math)."""

    def __init__(
        self,
        episodes: list[dict[str, Any]],
        *,
        seed: int,
        epoch: int = 0,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 1,
        windows_per_rank: int = 1,
        skip_batches: int = 0,
        grad_accumulation: int = 1,
    ):
        if not 0 <= rank < world_size:
            raise ValueError("rank must lie in [0, world_size)")
        self.episodes = episodes
        self.seed, self.epoch = seed, epoch
        self.rank, self.world_size = rank, world_size
        self.num_workers, self.windows_per_rank = num_workers, windows_per_rank
        self.skip_batches = skip_batches
        per_worker = len(episodes) // (world_size * num_workers * windows_per_rank)
        updates = per_worker * num_workers // grad_accumulation
        self.batches_per_rank = updates * grad_accumulation
        if self.batches_per_rank == 0:
            raise ValueError(
                f"{len(episodes)} episodes cannot fill one update on {world_size} ranks x {num_workers} workers x "
                f"{windows_per_rank} windows x {grad_accumulation} accumulation steps"
            )
        if skip_batches > self.batches_per_rank:
            raise ValueError(
                f"skip_batches {skip_batches} exceeds the {self.batches_per_rank} batches of an epoch"
            )
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def rank_order(self) -> list[int]:
        return epoch_order(len(self.episodes), self.seed, self.epoch)[self.rank :: self.world_size]

    def worker_batches(self, worker_id: int) -> int:
        remaining = self.batches_per_rank - worker_id
        return -(-remaining // self.num_workers) if remaining > 0 else 0

    def worker_positions(self, worker_id: int) -> list[int]:
        mine = self.rank_order()[worker_id :: self.num_workers]
        produced = 0
        if self.skip_batches > worker_id:
            produced = -(-(self.skip_batches - worker_id) // self.num_workers)
        return mine[produced * self.windows_per_rank : self.worker_batches(worker_id) * self.windows_per_rank]

    def logical_worker(self, physical_worker: int) -> int:
        return (physical_worker + self.skip_batches) % self.num_workers

    def __iter__(self):
        info = get_worker_info()
        worker_id, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        if workers != self.num_workers:
            raise RuntimeError(f"dataset built for {self.num_workers} workers but the loader runs {workers}")
        for position in self.worker_positions(self.logical_worker(worker_id)):
            yield self.window(position)

    def _episode(self, path: str) -> dict[str, np.ndarray]:
        ep = self._cache.get(path)
        if ep is None:
            if len(self._cache) > 32:
                self._cache.clear()
            with np.load(path) as z:
                if "frames_png" in z.files:  # one PNG per frame, lossless, about 2.6x smaller than raw uint8
                    buf, off = z["frames_png"].tobytes(), z["frames_off"]
                    frames = np.stack(
                        [
                            np.asarray(Image.open(BytesIO(buf[off[i] : off[i + 1]])).convert("RGB"))
                            for i in range(len(off) - 1)
                        ]
                    )
                else:
                    frames = z["frames"]
                ep = {"frames": frames, "action": pad4(z["action"])}
            self._cache[path] = ep
        return ep

    def window(self, position: int) -> dict[str, Any]:
        e = self.episodes[position]
        rng = Random(f"{self.seed}:{self.epoch}:{position}")
        s = rng.randrange(0, e["length"] - WINDOW + 1)
        window_seed = rng.getrandbits(63)
        ep = self._episode(e["file"])
        act = ep["action"]
        frames = torch.from_numpy(np.ascontiguousarray(ep["frames"][s : s + WINDOW])).permute(0, 3, 1, 2)
        state = torch.from_numpy(act[s - 1].astype(np.float32)) if s > 0 else torch.zeros(ACTION_DIM)
        mask = torch.zeros(ACTION_DIM)
        mask[: GAMES[e["game"]]["dim"]] = 1.0
        return {
            CAMERA: frames,  # (33, 3, H, W) uint8
            "state": state,
            "action": torch.from_numpy(act[s : s + CHUNK].astype(np.float32)),
            "action_mask": mask,
            "task": caption_of(e),
            "episode_index": position,
            "start": s,
            "window_seed": window_seed,
        }


def build(config, **kwargs) -> GameWindowDataset:
    """The trainer's dataset hook: ``config`` is the TrainConfig, ``kwargs`` the topology/position arguments."""
    return GameWindowDataset(load_episodes(parse_roots(config.source_root)), **kwargs)
