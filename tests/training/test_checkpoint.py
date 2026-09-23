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
"""Checkpoint round trip, exact resume, export profiles and resharded loading."""

import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import FakeVideoVAE, make_policy, tiny_config  # noqa: F401
from safetensors.torch import load_file

from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.training.checkpoint import (
    TrainState,
    _dcp_load,
    _dcp_save,
    ema_folder,
    export_policy,
    latest_checkpoint,
    load_checkpoint,
    restore_rng,
    resume_seed,
    rng_state,
    save_checkpoint,
    seed_everything,
    set_rng_state,
)
from flux_action.training.ema import DROID_POWER_EMA_SIGMA_RELS, PowerEMA
from flux_action.training.schedule import build_lr_scheduler, droid_lr_lambdas, resume_lr_scheduler


def _setup(seed=0, **policy_overrides):
    seed_everything(seed)
    policy = make_policy(tiny_config(**policy_overrides) if policy_overrides else None).train()
    policy.freeze_streams(("image", "image_cond", "audio", "audio_cond"))  # no loss reaches them
    policy.freeze_conditioning_heads()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), betas=(0.9, 0.99), eps=1e-8, weight_decay=0.05)
    scheduler = build_lr_scheduler(optimizer, droid_lr_lambdas(cooldown_start=None))
    emas = {ema_folder(s): PowerEMA.from_module(policy, s) for s in DROID_POWER_EMA_SIGMA_RELS}
    return policy, optimizer, scheduler, emas


def _train(policy, optimizer, scheduler, emas, batch, start, n):
    losses = []
    for step in range(start, start + n):
        loss, _ = policy(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        for ema in emas.values():
            ema.update_from_module(policy, step)
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    return losses


@pytest.fixture
def train_batch():
    rng = torch.Generator().manual_seed(42)
    return {
        "images.top": torch.rand(2, 33, 3, 64, 96, generator=rng),
        "state": torch.rand(2, 6, generator=rng),
        "action": torch.rand(2, 32, 6, generator=rng),
        "task": ["move the object", "move the object"],
    }


def test_exact_resume_and_state(tmp_path, train_batch):
    policy, optimizer, scheduler, emas = _setup()
    _train(policy, optimizer, scheduler, emas, train_batch, 0, 3)
    state = TrainState(run_seed=0, data={"cursor": 17})
    path = save_checkpoint(
        tmp_path, step=3, model=policy, optimizer=optimizer, emas=emas, state=state, config=policy.config
    )
    assert path == tmp_path / "step-3" and latest_checkpoint(tmp_path) == path
    saved_ema = {name: [t.clone() for t in ema.shadow] for name, ema in emas.items()}
    reference = _train(policy, optimizer, scheduler, emas, train_batch, 3, 4)

    policy2, optimizer2, scheduler2, emas2 = _setup(seed=123)  # different init, different generator
    loaded = load_checkpoint(path, model=policy2, optimizer=optimizer2, emas=emas2)
    assert loaded.step == 3 and loaded.data == {"cursor": 17} and loaded.seed == 0 and loaded.world_size == 1
    n = len(emas2[ema_folder(0.10)].names)
    assert loaded.emas[ema_folder(0.10)] == {"sigma_rel": 0.10, "last_step": 2, "n_tensors": n}
    for name, ema in emas2.items():
        for a, b in zip(ema.shadow, saved_ema[name], strict=True):
            assert torch.equal(a, b)
        assert ema.last_step == 2
    assert all(int(optimizer2.state[p]["step"]) == 3 for p in policy2.parameters() if p.requires_grad)
    resume_lr_scheduler(scheduler2, loaded.step)
    assert restore_rng(path, loaded)
    resumed = _train(policy2, optimizer2, scheduler2, emas2, train_batch, 3, 4)
    assert resumed == reference
    for name, ema in emas2.items():
        for a, b in zip(ema.shadow, emas[name].shadow, strict=True):
            assert torch.equal(a, b)
    # The scheduler consumed the same factors: both optimizers hold the same learning rates.
    assert [g["lr"] for g in optimizer.param_groups] == [g["lr"] for g in optimizer2.param_groups]


def test_rng_state_round_trip_covers_all_generators():
    seed_everything(5)
    state = rng_state()
    a = (torch.rand(2).tolist(), __import__("random").random(), float(__import__("numpy").random.rand()))
    set_rng_state(state)
    b = (torch.rand(2).tolist(), __import__("random").random(), float(__import__("numpy").random.rand()))
    assert a == b


def test_world_size_mismatch_skips_rng(tmp_path):
    state = TrainState(world_size=2)
    assert not restore_rng(tmp_path, state)


def test_latest_ignores_incomplete_and_foreign_dirs(tmp_path):
    (tmp_path / "step-9").mkdir()
    (tmp_path / "step-5").mkdir()
    (tmp_path / "step-5" / "COMPLETE").write_text("5\n")
    (tmp_path / "step-x").mkdir()
    (tmp_path / "step-x" / "COMPLETE").write_text("")
    assert latest_checkpoint(tmp_path) == tmp_path / "step-5"
    assert latest_checkpoint(tmp_path / "missing") is None
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "step-9", model=torch.nn.Linear(1, 1))


def test_resume_seed_deterministic_and_distinct():
    seeds = [resume_seed(42, k) for k in (1, 2, 3)]
    assert seeds == [resume_seed(42, k) for k in (1, 2, 3)]
    assert len({42, *seeds}) == 4 and all(0 < s < 2**31 for s in seeds)
    with pytest.raises(ValueError):
        resume_seed(42, 0)
    state = TrainState(run_seed=42)
    state.seed_history.append(resume_seed(42, 1))
    restored = TrainState.from_json(state.to_json())
    assert restored.seed == seeds[0] and restored.seed_history == [42, seeds[0]]


def test_export_profiles(tmp_path, train_batch):
    policy, optimizer, scheduler, emas = _setup()
    _train(policy, optimizer, scheduler, emas, train_batch, 0, 2)
    path = save_checkpoint(
        tmp_path / "ckpt",
        step=2,
        model=policy,
        optimizer=optimizer,
        emas=emas,
        state=TrainState(),
        config=policy.config,
    )
    result = export_policy(path, tmp_path / "model_export", profile="model")
    assert result["step"] == 2 and result["tensors"] == len(policy.state_dict())
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "model_export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    for a, b in zip(policy.state_dict().values(), restored.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    manifest = json.loads((tmp_path / "model_export" / "manifest.json").read_text())
    assert manifest["weight_profile"] == "model"
    assert json.loads((tmp_path / "model_export" / "config.json").read_text())["torch_dtype"] == "float32"

    name = ema_folder(0.10)
    export_policy(path, tmp_path / "ema_export", profile=name)
    exported = load_file(str(tmp_path / "ema_export" / "model.safetensors"))
    for key, value in emas[name].consolidated().items():
        assert torch.equal(exported[key], value)
    assert not all(torch.equal(exported[k], v) for k, v in policy.state_dict().items())
    assert json.loads((tmp_path / "ema_export" / "manifest.json").read_text())["weight_profile"] == name
    with pytest.raises(ValueError, match="unknown profile"):
        export_policy(path, tmp_path / "bad", profile="ema_0p42")


def _shard_writer(rank, world_size, init_file, ckpt_dir):
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        full = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        ema = PowerEMA([("w", distribute_tensor(full, mesh, [Shard(0)]))], 0.10)
        ema.update([distribute_tensor(full * 2, mesh, [Shard(0)])], 0)
        _dcp_save({"tensors": ema.state_dict()["tensors"]}, ckpt_dir)
    finally:
        dist.destroy_process_group()


def test_sharded_save_loads_dense_in_one_process(tmp_path):
    mp.spawn(_shard_writer, args=(2, str(tmp_path / "pg"), str(tmp_path / "ema")), nprocs=2, join=True)
    template = {"tensors": {"w": torch.zeros(8, 3)}}
    _dcp_load(template, tmp_path / "ema")
    assert torch.equal(template["tensors"]["w"], torch.arange(24, dtype=torch.float32).reshape(8, 3) * 2)


def test_export_of_a_lean_policy_rebuilds_only_its_streams(tmp_path, train_batch):
    policy, optimizer, scheduler, emas = _setup(content_streams=("video", "video_cond"))
    assert "image" not in policy.dit.in_channels
    _train(policy, optimizer, scheduler, emas, train_batch, 0, 1)
    path = save_checkpoint(
        tmp_path / "ckpt",
        step=1,
        model=policy,
        optimizer=optimizer,
        emas=emas,
        state=TrainState(),
        config=policy.config,
    )
    result = export_policy(path, tmp_path / "lean_export", profile="model")
    assert result["tensors"] == len(policy.state_dict())
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "lean_export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert tuple(restored.config.content_streams) == ("video", "video_cond")
    ema = export_policy(path, tmp_path / "lean_ema_export", profile=ema_folder(0.10))
    assert ema["tensors"] == len(policy.state_dict())
