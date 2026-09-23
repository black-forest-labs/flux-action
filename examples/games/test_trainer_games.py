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
"""The official trainer on the games dataset: a tiny policy, fake episodes for every game, six updates,
checkpoint + export, the mask keeps padded channels out of the loss, exact resume."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `examples.games.dataset` imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from conftest import TINY_DIT, FakeVideoVAE  # noqa: E402

from examples.games.dataset import (  # noqa: E402
    ACTION_DIM,
    GAMES,
    GameWindowDataset,
    load_episodes,
    parse_roots,
)
from flux_action.models.text_encoder import MockTextEncoder  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402
from flux_action.processing import packing  # noqa: E402
from flux_action.training.trainer import TrainConfig, Trainer  # noqa: E402


def fake_game(root: Path, game: str, n: int, length: int = 40, hw: int = 32):
    root.mkdir(parents=True)
    dim = GAMES[game]["dim"]
    eps = []
    rng = np.random.default_rng(hash(game) % 1000)
    for i in range(n):
        f = f"ep_{i:04d}.npz"
        np.savez(
            root / f,
            frames=rng.integers(0, 255, (length, hw, hw, 3), dtype=np.uint8),
            action=rng.uniform(-1, 1, (length, dim)).astype(np.float32),
        )
        e = {"file": f}
        if GAMES[game]["caption"] is None:
            e["task"] = f"take off, fly to place {i}, and land"  # per-episode instruction (ROTOR)
        eps.append(e)
    (root / "index.json").write_text(json.dumps({"episodes": eps, "length": length, "action_dim": dim}))


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    root = tmp_path_factory.mktemp("games")
    for g in GAMES:
        fake_game(root / g, g, 8)
    return ",".join(f"{g}={root / g}" for g in GAMES)


def tiny_policy(config):
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


def make_config(roots, output, **overrides):
    values = dict(
        source_root=roots,
        index_dir="unused",
        output_dir=str(output),
        dataset="examples.games.dataset:build",
        policy=dict(
            camera_layout="single",
            camera_keys=["images.game"],
            canvas_hw=[64, 64],
            action_dim=ACTION_DIM,
            action_modality="game",
            dit_config=TINY_DIT,
            augment=False,
            caption_dropout=0.0,
            gripper_flip_dims=[],
        ),
        steps=6,
        frozen_steps=1,
        trunk_warmup_steps=3,
        heads_warmup_steps=2,
        cooldown_start=4,
        cooldown_steps=2,
        windows_per_rank=2,
        num_workers=1,
        in_process_loader=True,
        param_dtype="float32",
        compute_dtype="float32",
        checkpoint_every=3,
        log_every=1,
        ema_sigma_rels=(0.10,),
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_dataset_mixes_games_and_masks_padding(roots):
    eps = load_episodes(parse_roots(roots))
    assert len(eps) == 8 * len(GAMES) and {e["game"] for e in eps} == set(GAMES)
    ds = GameWindowDataset(eps, seed=1, windows_per_rank=2)
    items = list(ds)
    assert len(items) == 8 * len(GAMES)
    fixed = {GAMES[g]["caption"]: g for g in GAMES if GAMES[g]["caption"] is not None}

    def game_of(task: str) -> str:
        if task in fixed:
            return fixed[task]
        assert task.startswith("fly the drone: take off, fly to place ")  # ROTOR: per-episode instruction
        return "rotor"

    seen = {game_of(it["task"]) for it in items}
    assert seen == set(GAMES)
    assert (
        len({it["task"] for it in items if game_of(it["task"]) == "rotor"}) == 8
    )  # every rotor episode has its own caption
    for it in items:
        g = game_of(it["task"])
        assert it["action"].shape == (32, ACTION_DIM) and it["state"].shape == (ACTION_DIM,)
        assert it["action_mask"].tolist() == [1.0] * GAMES[g]["dim"] + [0.0] * (ACTION_DIM - GAMES[g]["dim"])
        if g == "vector":
            assert float(it["action"][:, 3].abs().max()) == 0.0 and float(it["state"][3]) == 0.0
    # rank/worker slicing partitions the epoch like the DROID loader
    a = [
        it["episode_index"] for it in GameWindowDataset(eps, seed=1, rank=0, world_size=2, windows_per_rank=2)
    ]
    b = [
        it["episode_index"] for it in GameWindowDataset(eps, seed=1, rank=1, world_size=2, windows_per_rank=2)
    ]
    assert not set(a) & set(b) and len(a) + len(b) == 8 * len(GAMES)


def test_masked_flow_loss_ignores_padded_channel():
    B, K, D, Nv = 2, 32, 4, 10
    tgt = {"x_video": torch.zeros(B, Nv, 8), "x_game": torch.zeros(B, K, D)}
    pred = {"x_video": torch.zeros(B, Nv, 8), "x_game": torch.zeros(B, K, D)}
    pred["x_game"][1, :, 3] = 100.0  # garbage in the padded channel of a vector window
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)
    assert float(packing.flow_loss(pred, tgt, "game", 50.0, 1.0, action_mask=mask)["loss"]) == 0.0
    assert float(packing.flow_loss(pred, tgt, "game", 50.0, 1.0)["loss"]) > 0.0
    # the (n, D) mask applies unchanged to the packed per-token layout the trainer uses (window-major tokens)
    flat = packing.flatten_windows(tgt)
    flat_pred = packing.flatten_windows(pred)
    assert float(packing.flow_loss(flat_pred, flat, "game", 50.0, 1.0, action_mask=mask)["loss"]) == 0.0
    pred["x_game"][0, :, 3] = 1.0  # a REAL channel (grunt window) must count
    assert float(packing.flow_loss(pred, tgt, "game", 50.0, 1.0, action_mask=mask)["loss"]) > 0.0


def test_trainer_runs_games_end_to_end_and_resumes(roots, tmp_path):
    output = tmp_path / "run"
    Trainer(make_config(roots, output), policy_factory=tiny_policy).run()
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    steps = [r for r in lines if "loss" in r]
    assert [r["step"] for r in steps] == [1, 2, 3, 4, 5, 6]
    assert all(np.isfinite(r["loss"]) for r in steps)
    # our schedule knobs took: trunk frozen at step 1, warming after
    assert steps[0]["lr_trunk"] == 0.0 and steps[2]["lr_trunk"] > 0.0
    assert (output / "step-6" / "state.json").is_file()
    # exact resume: 3 steps, then 3 more, equals the 6-step run
    part = tmp_path / "part"
    Trainer(make_config(roots, part, steps=3), policy_factory=tiny_policy).run()
    Trainer(make_config(roots, part, steps=6, reseed_on_resume=False), policy_factory=tiny_policy).run()
    full = [r["loss"] for r in steps]
    resumed = [
        json.loads(line)["loss"]
        for line in (part / "metrics.jsonl").read_text().splitlines()
        if "loss" in line
    ]
    assert resumed[3:] == pytest.approx(full[3:], rel=1e-5, abs=1e-6)
