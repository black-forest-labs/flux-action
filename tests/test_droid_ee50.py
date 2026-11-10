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
"""The EE-50 DROID example: layout and inverse, statistics, the Cartesian index and the trainer plug."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE
from synthetic_droid import FRAME_HW, build_compact_dataset, cartesian_pose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # repo root, so `examples.droid_ee50` imports

from examples.droid_ee50 import ee50  # noqa: E402
from examples.droid_ee50.dataset import build  # noqa: E402
from examples.droid_ee50.index import REFERENCE_STATS, STATS_FILENAME, compute_stats  # noqa: E402
from examples.droid_ee50.index import main as index_main  # noqa: E402
from examples.droid_ee50.inference import predict_droid_commands  # noqa: E402
from flux_action.config import PolicyConfig  # noqa: E402
from flux_action.data.droid.index import (  # noqa: E402
    ROWS_FILENAME,
    build_manifest,
    build_rows,
    write_manifest,
)
from flux_action.models.text_encoder import MockTextEncoder  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402
from flux_action.training.checkpoint import export_policy  # noqa: E402
from flux_action.training.trainer import TrainConfig, Trainer  # noqa: E402

CHUNK = 32
EPISODES = [
    {"n_frames": 40, "caption": "pick up the cup", "keep": [[0, 40]]},
    {"n_frames": 38, "caption": "close the drawer", "keep": [[2, 38]]},
    {"n_frames": 44, "caption": "open the box", "keep": [[0, 44]]},
    {"n_frames": 39, "caption": "push the plate", "keep": [[0, 39]]},
]


def smooth_episode(rng, n):
    """Observed states and the commands that lead to the next state, ``(n, 7)`` each."""
    t = np.linspace(0.0, 2.0, n + 1)[:, None]
    poses = np.concatenate(
        [
            0.4 + 0.1 * np.sin(t * np.array([1.0, 1.7, 2.3])),
            np.array([2.9, -0.3, 0.5]) + 0.4 * np.sin(t * np.array([0.9, 1.3, 2.1])),
        ],
        axis=1,
    )
    grip = rng.uniform(0.0, 1.0, n + 1)[:, None]
    state = np.concatenate([poses[:-1], grip[:-1]], axis=1)
    actions = np.concatenate([poses[1:] + 0.002 * rng.standard_normal((n, 6)), grip[1:]], axis=1)
    return state, actions


def test_droid_euler_round_trip():
    rng = np.random.default_rng(0)
    poses = np.concatenate(
        [
            rng.uniform(-1, 1, (64, 3)),
            rng.uniform(-math.pi, math.pi, (64, 1)),
            rng.uniform(-1.5, 1.5, (64, 1)),
            rng.uniform(-math.pi, math.pi, (64, 1)),
        ],
        axis=1,
    )
    back = ee50.matrices_to_droid_cartesian(ee50.droid_cartesian_to_matrices(poses))
    np.testing.assert_allclose(back, poses, atol=1e-9)


def test_window_rows_are_relative_to_the_observed_anchor():
    # anchor turned 90 degrees about z; the command moves 10 cm along world x and rolls 30 degrees
    state = [1.0, 2.0, 0.5, 0.0, 0.0, math.pi / 2, 0.0]
    command = [1.1, 2.0, 0.5, math.pi / 6, 0.0, math.pi / 2, 1.0]
    rows = ee50.droid_window(np.array(state), np.array([command]))
    anchor = np.zeros(50)
    anchor[[3, 7]] = 1.0
    anchor[9] = 1.0  # gripper position 0 (open) -> open fraction 1
    np.testing.assert_allclose(rows[0], anchor, atol=1e-12)
    c, s = math.cos(math.pi / 6), math.sin(math.pi / 6)
    np.testing.assert_allclose(rows[1, 0:3], [0.0, -0.1, 0.0], atol=1e-12)  # world x is the anchor's -y
    # the first two ROWS of R_rel = Rx(30 deg); its columns would put +sin in the last entry
    np.testing.assert_allclose(rows[1, 3:9], [1.0, 0.0, 0.0, 0.0, c, -s], atol=1e-12)
    assert rows[1, 9] == 0.0 and not rows[:, 10:].any()


def test_normalization_inverts_and_zeroes_the_anchor_pose():
    stats = ee50.load_stats(REFERENCE_STATS)
    state, actions = smooth_episode(np.random.default_rng(1), 60)
    raw = ee50.droid_window(state[10], actions[10 : 10 + CHUNK])
    offsets = np.arange(CHUNK + 1)
    rows = ee50.normalize(raw, offsets, stats)
    assert rows.dtype == np.float32 and not rows[0, list(ee50.POSE_DIMS)].any() and not rows[:, 10:].any()
    assert np.abs(rows).max() < ee50.NORMALIZED_CLIP
    np.testing.assert_allclose(ee50.denormalize(rows[1:], offsets[1:], stats), raw[1:], atol=1e-5)
    np.testing.assert_allclose(ee50.droid_state_token(state[10, 6], stats), rows[0], atol=1e-6)


def test_decoded_chunk_reproduces_the_commands():
    stats = ee50.load_stats(REFERENCE_STATS)
    state, actions = smooth_episode(np.random.default_rng(2), 60)
    start = 7
    raw = ee50.droid_window(state[start], actions[start : start + CHUNK])
    rows = ee50.normalize(raw, np.arange(CHUNK + 1), stats)
    commands = ee50.decode_droid_chunk(rows[1:], state[start, :6], stats)
    expected = ee50.droid_cartesian_to_matrices(actions[start : start + CHUNK, :6])
    decoded = ee50.droid_cartesian_to_matrices(commands[:, :6])
    np.testing.assert_allclose(decoded, expected, atol=1e-5)
    np.testing.assert_allclose(commands[:, 6], actions[start : start + CHUNK, 6], atol=1e-5)


def test_moments_equal_a_sum_over_windows():
    state, actions = smooth_episode(np.random.default_rng(3), 30)
    ranges, max_offset = [(0, 12), (15, 30)], 5
    moments = ee50.droid_moments(state, actions, ranges, max_offset)
    windows = ee50.Moments.empty(max_offset)  # the generic path for other embodiments
    for start, end in ranges:
        for anchor in range(start, end):
            k = min(max_offset, end - 1 - anchor)  # the frame a command leads to must be valid
            windows.add_window(ee50.droid_window(state[anchor], actions[anchor : anchor + k]))
    np.testing.assert_array_equal(moments.count, windows.count)
    np.testing.assert_allclose(moments.total, windows.total, atol=1e-9)
    np.testing.assert_allclose(moments.total2, windows.total2, atol=1e-9)
    stats = ee50.finalize(moments)
    np.testing.assert_allclose(stats.mean, windows.total / windows.count[:, None], atol=1e-6)
    assert not stats.mean[:, 10:].any() and not stats.std[:, 10:].any()


def test_reference_stats_cover_one_droid_window():
    stats = ee50.load_stats(REFERENCE_STATS)
    np.testing.assert_array_equal(stats.offsets, np.arange(CHUNK + 1))
    assert not stats.mean[:, 10:].any() and not stats.std[:, 10:].any()
    assert (stats.std[0, :9] < ee50.STD_FLOOR).all()  # the anchor pose is the identity
    assert (stats.std[1:, :10] > 0).all() and stats.std[0, 9] > 0
    count = json.loads(REFERENCE_STATS.read_text())["count"]
    assert len(set(np.diff(count))) == 1  # each range loses one anchor per offset


def test_config_is_a_valid_training_config():
    values = json.loads((ROOT / "configs/droid/train_ee50.json").read_text())
    config = TrainConfig(**values)
    policy = PolicyConfig(**config.policy)
    policy.validate_training()
    assert config.dataset == "examples.droid_ee50.dataset:build"
    assert (policy.action_modality, policy.action_dim, policy.chunk_size) == (
        "action_prediction_ee50",
        50,
        CHUNK,
    )
    assert policy.action_normalization is None and policy.action_scale == 1.0 and not policy.gripper_flip_dims


@pytest.fixture(scope="module")
def ee50_index(tmp_path_factory):
    root = tmp_path_factory.mktemp("ee50")
    filter_path = build_compact_dataset(root / "source", EPISODES)
    manifest = build_manifest(root / "source", filter_path)
    (root / "droid").mkdir()
    build_rows(root / "source", manifest, root / "droid" / ROWS_FILENAME)
    write_manifest(manifest, root / "droid" / "manifest.json")
    args = ["--source-root", str(root / "source"), "--droid-index", str(root / "droid")]
    summary = index_main([*args, "--output-dir", str(root / "index")])
    assert summary["episodes"] == len(EPISODES) and not summary["rejected"]
    return root


def test_index_writes_cartesian_rows_and_statistics(ee50_index):
    index = ee50_index / "index"
    manifest = json.loads((index / "manifest.json").read_text())
    assert (manifest["state_dim"], manifest["action_dim"]) == (7, 7)
    rows = np.load(index / ROWS_FILENAME)
    first = manifest["episodes"][2]["from_index"]
    np.testing.assert_allclose(rows[first + 5, :6], cartesian_pose(2, 5), atol=1e-6)
    np.testing.assert_allclose(rows[first + 5, 7:13], cartesian_pose(2, 6), atol=1e-6)
    np.testing.assert_allclose(rows[first + 5, [6, 13]], [0.25, 0.75])
    assert (index / STATS_FILENAME).read_bytes() == REFERENCE_STATS.read_bytes()
    stats, provenance = compute_stats(manifest, rows, CHUNK, min_range_frames=33)
    np.testing.assert_array_equal(stats.offsets, np.arange(CHUNK + 1))
    frames = sum(b - a for e in EPISODES for a, b in e["keep"])
    assert provenance["count"] == [frames - len(EPISODES) * k for k in range(CHUNK + 1)]
    assert compute_stats(manifest, rows, CHUNK, min_range_frames=41)[1]["episodes"] == 1


def plug_config(root, output=None, **overrides):
    values = dict(
        dataset="examples.droid_ee50.dataset:build",
        source_root=str(root / "source"),
        index_dir=str(root / "index"),
        output_dir=str(output or root / "unused"),
        decoder="pyav",
        frame_hw=FRAME_HW,
        windows_per_rank=2,
        num_workers=1,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_plug_yields_normalized_ee50_windows(ee50_index):
    loader = dict(seed=42, epoch=0, rank=0, world_size=1, num_workers=1, windows_per_rank=2)
    dataset = build(plug_config(ee50_index), skip_batches=0, grad_accumulation=1, **loader)
    stats = ee50.load_stats(ee50_index / "index" / STATS_FILENAME)
    for position in range(len(EPISODES)):
        item = dataset.window(position)
        episode = dataset.episodes[position]["episode_index"]
        start = item["start"]
        state = np.array([*cartesian_pose(episode, start), 0.25])
        actions = np.array([[*cartesian_pose(episode, start + k + 1), 0.75] for k in range(CHUNK)])
        raw = ee50.droid_window(state, actions)
        assert item["state"].shape == (50,) and item["action"].shape == (CHUNK, 50)
        np.testing.assert_allclose(item["state"].numpy(), ee50.droid_state_token(0.25, stats), atol=1e-6)
        decoded = ee50.denormalize(item["action"].numpy(), np.arange(1, CHUNK + 1), stats)
        np.testing.assert_allclose(decoded, raw[1:], atol=1e-6)
        assert item["images.wrist"].shape == (CHUNK + 1, 3, *FRAME_HW)


def test_trainer_trains_and_the_export_predicts_commands(ee50_index, tmp_path):
    policy = dict(
        action_modality="action_prediction_ee50",
        action_dim=50,
        camera_layout="single",
        camera_keys=["images.wrist"],
        canvas_hw=[64, 96],
        dit_config=TINY_DIT,
        action_scale=1.0,
        gripper_flip_dims=[],
    )
    config = plug_config(
        ee50_index,
        tmp_path / "run",
        policy=policy,
        steps=3,
        cooldown_start=None,
        frozen_steps=0,
        trunk_warmup_steps=1,
        heads_warmup_steps=1,
        in_process_loader=True,
        param_dtype="float32",
        compute_dtype="float32",
        checkpoint_every=3,
        log_every=1,
        ema_sigma_rels=(0.10,),
    )

    def tiny(policy_config):
        return FluxActionPolicy(policy_config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))

    result = Trainer(config, policy_factory=tiny).run()
    assert result["step"] == 3
    records = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(r["loss"]) for r in records if "loss" in r)
    export_policy(tmp_path / "run" / "step-3", tmp_path / "export", profile="model")
    restored = FluxActionPolicy.from_pretrained(
        tmp_path / "export", video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    for name, value in dict(
        sampler="euler", num_inference_steps=2, guidance_scale=1.0, sampler_shift=5.0
    ).items():
        setattr(restored.config, name, value)
    observation = {"images.wrist": torch.rand(1, 3, 64, 96), "task": ["pick up the cup"]}
    commands = predict_droid_commands(
        restored,
        observation,
        cartesian_position=np.array([cartesian_pose(0, 3)]),
        gripper_position=np.array([0.25]),
        stats=ee50.load_stats(ee50_index / "index" / STATS_FILENAME),
    )
    assert commands.shape == (1, CHUNK, 7) and np.isfinite(commands).all()
