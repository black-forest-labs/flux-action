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
"""The trainer's embodiment settings: the dataset plug (``TrainConfig.dataset``), configurable LR phases, a
per-window action mask through ``prepare`` -> ``flow_loss``, the export dtype and the pre-rename export key.
"""

from __future__ import annotations

import json

import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE
from torch.utils.data import IterableDataset

from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.processing import packing
from flux_action.training.checkpoint import export_policy, latest_checkpoint
from flux_action.training.trainer import TrainConfig, Trainer, _import_dataset_builder

FRAMES, H, W, D, CHUNK = 33, 32, 48, 4, 32  # window = 1 conditioning frame + 32 future
TINY_POLICY = dict(
    camera_layout="single", camera_keys=["images.game"], canvas_hw=[64, 96], action_dim=D, dit_config=TINY_DIT
)


class TwoEmbodimentWindows(IterableDataset):
    """Synthetic windows of two 'games' sharing one 4-dim head: game A uses all 4 dims, game B the first 2
    (its last two are zero-padded and masked). Deterministic from (seed, epoch, rank, position)."""

    def __init__(
        self,
        seed,
        epoch,
        rank,
        world_size,
        num_workers,
        windows_per_rank,
        skip_batches,
        grad_accumulation,
        **_,
    ):
        self.seed, self.epoch, self.rank, self.num_workers = seed, epoch, rank, num_workers
        self.windows_per_rank, self.skip_batches = windows_per_rank, skip_batches
        self.batches_per_rank = 4 * grad_accumulation

    def __iter__(self):
        for b in range(self.skip_batches, self.batches_per_rank):
            for w in range(self.windows_per_rank):
                g = torch.Generator().manual_seed(hash((self.seed, self.epoch, self.rank, b, w)) % (2**31))
                game_b = (b + w) % 2 == 1
                mask = torch.tensor([1.0, 1.0, 0.0, 0.0]) if game_b else torch.ones(D)
                actions = torch.rand(CHUNK, D, generator=g) * mask
                yield {
                    "images.game": torch.randint(0, 255, (FRAMES, 3, H, W), dtype=torch.uint8, generator=g),
                    "state": torch.rand(D, generator=g) * mask,
                    "action": actions,
                    "action_mask": mask,
                    "task": "drive the racer" if game_b else "play the shooter",
                    "window_seed": int(torch.randint(0, 2**62, (), generator=g)),
                }


def build(config, **kwargs):
    return TwoEmbodimentWindows(**kwargs)


def tiny_policy(config):
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


def make_config(output, **overrides):
    values = dict(
        dataset=f"{__name__}:build",
        source_root="unused",
        index_dir="unused",
        output_dir=str(output),
        policy=TINY_POLICY,
        steps=6,
        cooldown_start=4,
        cooldown_steps=2,
        frozen_steps=1,
        trunk_warmup_steps=2,
        heads_warmup_steps=1,
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


def records(output):
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    return [r for r in lines if "loss" in r]


def test_dataset_plug_spec_parsing():
    assert _import_dataset_builder(f"{__name__}:build") is build
    with pytest.raises(ValueError, match="pkg.module:callable"):
        _import_dataset_builder("no_colon_here")


def test_plugged_dataset_trains_without_an_index_and_uses_the_lr_phases(tmp_path):
    output = tmp_path / "run"
    result = Trainer(make_config(output), policy_factory=tiny_policy).run()
    assert result["step"] == 6 and latest_checkpoint(output) == output / "step-6"
    rows = records(output)
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in rows)
    # frozen_steps=1, trunk_warmup_steps=2, reference semantics (boundary step still zero): update g uses
    # factor(g): g=0,1 -> 0; g=2 -> 1/2; g=3 -> 1 (then the cooldown overlay from g=4)
    assert rows[0]["lr_trunk"] == 0.0 and rows[1]["lr_trunk"] == 0.0
    assert rows[2]["lr_trunk"] == pytest.approx(0.5 * rows[3]["lr_trunk"]) and rows[3]["lr_trunk"] > 0
    # heads_warmup_steps=1: heads at full LR from the second update
    assert rows[1]["lr_heads"] == rows[2]["lr_heads"] == rows[3]["lr_heads"]


def test_resume_is_exact_with_a_plugged_dataset(tmp_path):
    full = tmp_path / "full"
    Trainer(make_config(full, checkpoint_every=100), policy_factory=tiny_policy).run()
    resumed = tmp_path / "resumed"
    Trainer(make_config(resumed, steps=3, reseed_on_resume=False), policy_factory=tiny_policy).run()
    Trainer(make_config(resumed, steps=6, reseed_on_resume=False), policy_factory=tiny_policy).run()
    a = [(r["step"], r["loss"]) for r in records(full)]
    b = [(r["step"], r["loss"]) for r in records(resumed)]
    assert a == b


def test_action_mask_removes_padded_dims_from_the_loss():
    pred = {"x_video": torch.zeros(1, 3, 2), "x_action": torch.ones(1, 4, D)}  # 2 windows x 2 action tokens
    target = {k: torch.zeros_like(v) for k, v in pred.items()}
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    plain = packing.flow_loss(pred, target, "action")
    masked = packing.flow_loss(pred, target, "action", action_mask=mask)
    # every action dim errs by 1: unmasked token MSE is 1 everywhere; masked, the 2 padded dims of window 2
    # leave its tokens at mean over 2 dims = 1 as well -> same MSE, but with errors only on padded dims the
    # masked loss must vanish for that window
    assert masked["action_mse"].item() == pytest.approx(plain["action_mse"].item())
    pred2 = {
        "x_video": torch.zeros(1, 3, 2),
        "x_action": torch.tensor([[0.0, 0.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 1.0, 1.0]] * 2)[None],
    }
    only_padded = packing.flow_loss(pred2, target, "action", action_mask=mask)
    assert only_padded["action_mse"].item() == 0.0
    assert packing.flow_loss(pred2, target, "action")["action_mse"].item() > 0.0


def test_masked_windows_train_and_the_mask_reaches_the_loss(tmp_path):
    seen = {}
    original = packing.flow_loss

    def spy(*args, **kwargs):
        seen["mask"] = kwargs.get("action_mask")
        return original(*args, **kwargs)

    packing.flow_loss = spy
    try:
        Trainer(make_config(tmp_path / "run", steps=1, cooldown_start=None), policy_factory=tiny_policy).run()
    finally:
        packing.flow_loss = original
    assert seen["mask"] is not None and seen["mask"].shape == (2, D)
    assert seen["mask"].sum().item() == 6.0  # one full window (4) + one game-B window (2)


def test_export_dtype(tmp_path):
    output = tmp_path / "run"
    Trainer(make_config(output), policy_factory=tiny_policy).run()
    export_policy(output / "step-6", tmp_path / "bf16", profile="model", dtype="bfloat16")
    cfg = json.loads((tmp_path / "bf16" / "config.json").read_text())
    assert cfg["torch_dtype"] == "bfloat16"
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "bf16", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert next(restored.dit.parameters()).dtype == torch.bfloat16
