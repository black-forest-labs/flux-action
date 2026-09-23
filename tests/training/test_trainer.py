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
"""Trainer smoke and resume tests on the synthetic compact dataset with the tiny policy."""

import json
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import TINY_DIT, FakeVideoVAE
from synthetic_droid import FRAME_HW, build_compact_dataset

from flux_action.data.droid.index import ROWS_FILENAME, build_manifest, build_rows, write_manifest
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.training.checkpoint import export_policy, latest_checkpoint
from flux_action.training.trainer import TrainConfig, Trainer

EPISODES = [
    {"n_frames": 40, "caption": "pick up the cup | grab the cup", "keep": [[0, 40]]},
    {"n_frames": 38, "caption": "close the drawer", "keep": [[2, 38]]},
    {"n_frames": 44, "caption": "open the box", "keep": [[0, 44]]},
    {"n_frames": 39, "caption": "push the plate", "keep": [[0, 39]]},
]
TINY_POLICY = dict(
    camera_layout="single",
    camera_keys=["images.wrist"],
    canvas_hw=[64, 96],
    action_dim=8,
    dit_config=TINY_DIT,
    augment=True,
)


def tiny_policy(config):
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("trainer")
    filter_path = build_compact_dataset(root / "source", EPISODES)
    manifest = build_manifest(root / "source", filter_path)
    (root / "index").mkdir()
    build_rows(root / "source", manifest, root / "index" / ROWS_FILENAME)
    write_manifest(manifest, root / "index" / "manifest.json")
    return root


def make_config(root, output, **overrides):
    values = dict(
        source_root=str(root / "source"),
        index_dir=str(root / "index"),
        output_dir=str(output),
        policy=TINY_POLICY,
        steps=6,
        cooldown_start=4,
        cooldown_steps=2,
        windows_per_rank=2,
        num_workers=1,
        in_process_loader=True,
        decoder="pyav",
        frame_hw=FRAME_HW,
        param_dtype="float32",
        compute_dtype="float32",
        activation_checkpointing=True,
        checkpoint_every=3,
        log_every=1,
        ema_sigma_rels=(0.10,),
    )
    values.update(overrides)
    return TrainConfig(**values)


def losses(output):
    lines = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    return [(r["step"], r["loss"], r["lr_trunk"], r["lr_heads"]) for r in lines if "loss" in r]


def test_single_process_run_checkpoints_and_exports(indexed, tmp_path):
    output = tmp_path / "run"
    result = Trainer(make_config(indexed, output), policy_factory=tiny_policy).run()
    assert result["step"] == 6 and latest_checkpoint(output) == output / "step-6"
    assert (output / "step-3" / "COMPLETE").exists()
    records = losses(output)
    assert [r[0] for r in records] == [1, 2, 3, 4, 5, 6] and all(
        torch.isfinite(torch.tensor(r[1])) for r in records
    )
    # update g consumes factor(g): trunk frozen through g=3, the cooldown overlay from g=4 (1.0, then 0.5)
    assert records[3][2] == 0.0 and records[3][3] == pytest.approx(9.6e-4 * 4 / 1001)
    assert records[4][2] == pytest.approx(1.92e-4) and records[4][3] == pytest.approx(9.6e-4)
    assert records[5][2] == pytest.approx(0.96e-4) and records[5][3] == pytest.approx(4.8e-4)
    state = json.loads((output / "step-6" / "state.json").read_text())
    assert state["data"]["position"]["batches_consumed"] in (0, 1, 2) and state["seed_history"] == [42]
    export = export_policy(output / "step-6", tmp_path / "export", profile="ema_0p10")
    assert export["profile"] == "ema_0p10"
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert restored.config.action_dim == 8 and restored.config.torch_dtype == "float32"


def test_exact_resume_matches_uninterrupted(indexed, tmp_path):
    full = Trainer(
        make_config(indexed, tmp_path / "full", checkpoint_every=100), policy_factory=tiny_policy
    ).run()
    assert full["step"] == 6
    part = make_config(indexed, tmp_path / "part", steps=3, checkpoint_every=3, reseed_on_resume=False)
    Trainer(part, policy_factory=tiny_policy).run()
    rest = make_config(indexed, tmp_path / "part", steps=6, checkpoint_every=3, reseed_on_resume=False)
    Trainer(rest, policy_factory=tiny_policy).run()
    assert losses(tmp_path / "part") == losses(tmp_path / "full")
    state = json.loads((tmp_path / "part" / "step-6" / "state.json").read_text())
    assert state["extra"]["resumes"][0]["mode"] == "exact" and state["seed_history"] == [42]


def test_reseeded_resume_records_new_seed(indexed, tmp_path):
    Trainer(
        make_config(indexed, tmp_path / "r", steps=3, checkpoint_every=3), policy_factory=tiny_policy
    ).run()
    Trainer(
        make_config(indexed, tmp_path / "r", steps=5, checkpoint_every=5), policy_factory=tiny_policy
    ).run()
    state = json.loads((tmp_path / "r" / "step-5" / "state.json").read_text())
    assert state["extra"]["resumes"][0]["mode"] == "reseeded"
    assert len(state["seed_history"]) == 2 and state["seed_history"][1] != 42


def test_config_overrides_and_validation(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps(
            {"source_root": "s", "index_dir": "i", "output_dir": "o", "steps": 10, "cooldown_start": None}
        )
    )
    cfg = TrainConfig.from_file(
        path, ["steps=20", "policy.attn_mode=flash", "frame_hw=[36, 64]", "decoder=pyav"]
    )
    assert cfg.steps == 20 and cfg.policy == {"attn_mode": "flash"} and cfg.frame_hw == (36, 64)
    assert cfg.global_batch(64) == 64 * 32
    with pytest.raises(ValueError, match="cooldown_start"):
        TrainConfig(source_root="s", index_dir="i", output_dir="o", steps=40000)
    with pytest.raises(ValueError, match="key=value"):
        TrainConfig.from_file(path, ["steps"])


def _sharded_worker(rank, world_size, init_file, root, output, steps, reseed):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        from pathlib import Path

        config = make_config(
            Path(root),
            Path(output),
            steps=steps,
            windows_per_rank=1,
            checkpoint_every=2,
            reseed_on_resume=reseed,
        )
        Trainer(config, policy_factory=tiny_policy).run()
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp_run_resumes_exactly_and_exports(indexed, tmp_path):
    output = tmp_path / "fsdp"
    mp.spawn(
        _sharded_worker,
        args=(2, str(tmp_path / "pg1"), str(indexed), str(output), 2, False),
        nprocs=2,
        join=True,
    )
    assert latest_checkpoint(output) == output / "step-2"
    first = losses(output)
    mp.spawn(
        _sharded_worker,
        args=(2, str(tmp_path / "pg2"), str(indexed), str(output), 4, False),
        nprocs=2,
        join=True,
    )
    resumed = losses(output)
    assert [r[0] for r in resumed] == [1, 2, 3, 4] and resumed[:2] == first
    reference = tmp_path / "fsdp_full"
    mp.spawn(
        _sharded_worker,
        args=(2, str(tmp_path / "pg3"), str(indexed), str(reference), 4, False),
        nprocs=2,
        join=True,
    )
    assert losses(reference) == resumed
    export = export_policy(output / "step-4", tmp_path / "fsdp_export", profile="model")
    assert export["step"] == 4
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "fsdp_export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert sum(p.numel() for p in restored.parameters()) > 0


def test_resume_rejects_a_changed_global_batch(indexed, tmp_path):
    output = tmp_path / "run"
    Trainer(make_config(indexed, output, steps=3), policy_factory=tiny_policy).run()
    with pytest.raises(ValueError, match="windows per update"):
        Trainer(make_config(indexed, output, steps=6, windows_per_rank=1), policy_factory=tiny_policy).run()
    # checkpoints written before the field existed are checked from their saved configuration
    state_path = output / "step-3" / "state.json"
    state = json.loads(state_path.read_text())
    del state["extra"]["global_batch"]
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="windows per update"):
        Trainer(make_config(indexed, output, steps=6, windows_per_rank=1), policy_factory=tiny_policy).run()
    Trainer(
        make_config(indexed, output, steps=6, windows_per_rank=1, allow_global_batch_change=True),
        policy_factory=tiny_policy,
    ).run()


def test_checkpoint_retention_keeps_recent_and_milestones(indexed, tmp_path):
    output = tmp_path / "keep"
    config = make_config(indexed, output, checkpoint_every=1, keep_checkpoints=2, keep_every=3)
    Trainer(config, policy_factory=tiny_policy).run()
    kept = sorted(int(p.name.removeprefix("step-")) for p in output.glob("step-*"))
    assert kept == [3, 5, 6]
    events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [e["removed_steps"] for e in events if e.get("event") == "prune"] == [[1], [2], [4]]
    with pytest.raises(ValueError, match="keep_checkpoints"):
        make_config(indexed, output, keep_checkpoints=0)


def test_non_finite_update_aborts_before_any_state_changes(indexed, tmp_path, monkeypatch):
    from flux_action.policy import FluxActionPolicy

    calls = {"n": 0}
    original = FluxActionPolicy.forward

    def poisoned(self, batch, **kwargs):
        loss, info = original(self, batch, **kwargs)
        calls["n"] += 1
        return (loss * float("nan") if calls["n"] == 2 else loss), info

    monkeypatch.setattr(FluxActionPolicy, "forward", poisoned)
    output = tmp_path / "nan"
    with pytest.raises(RuntimeError, match="non-finite update at step 2"):
        Trainer(make_config(indexed, output, checkpoint_every=2), policy_factory=tiny_policy).run()
    steps = losses(output)
    assert [s for s, *_ in steps] == [1] and latest_checkpoint(output) is None
    first = next(
        r for r in map(json.loads, (output / "metrics.jsonl").read_text().splitlines()) if "step" in r
    )
    assert first["step"] == 1 and first["grad_norm"] > 0 and first["data_wait"] >= 0


def test_resume_rejects_a_different_run_seed(indexed, tmp_path):
    output = tmp_path / "seed"
    Trainer(make_config(indexed, output, steps=3, checkpoint_every=3), policy_factory=tiny_policy).run()
    with pytest.raises(ValueError, match="run seed 42"):
        Trainer(make_config(indexed, output, steps=6, seed=43), policy_factory=tiny_policy).run()
    result = Trainer(
        make_config(indexed, output, steps=6, seed=43, allow_seed_change=True), policy_factory=tiny_policy
    ).run()
    assert result["step"] == 6


def test_resume_preserves_explicit_full_stream_architecture(indexed, tmp_path):
    from flux_action.config import CONTENT_STREAMS

    output = tmp_path / "full_streams"
    full = {**TINY_POLICY, "content_streams": CONTENT_STREAMS}
    Trainer(
        make_config(indexed, output, policy=full, steps=3, reseed_on_resume=False), policy_factory=tiny_policy
    ).run()
    # The new default is lean, but an existing checkpoint owns its architecture.
    resumed = Trainer(
        make_config(indexed, output, reseed_on_resume=False), policy_factory=tiny_policy
    ).build()
    assert resumed.policy.config.content_streams == CONTENT_STREAMS
    assert resumed.step == 3
    with pytest.raises(ValueError, match="content_streams"):
        Trainer(
            make_config(indexed, output, policy={**TINY_POLICY, "content_streams": ("video", "video_cond")}),
            policy_factory=tiny_policy,
        ).build()
