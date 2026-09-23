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
"""Validate the runnable SO-101 training config and the separate historical corpus inventory."""

import json
import math
from pathlib import Path

from flux_action.config import PolicyConfig
from flux_action.training.trainer import TrainConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "so101"
CORPUS = json.loads((CONFIGS / "community_corpus.json").read_text())


def test_policy_config_geometry():
    cfg = PolicyConfig(**TrainConfig.from_file(CONFIGS / "train.json").policy)
    assert cfg.window_frames == 50 and cfg.chunk_size == 42 and cfg.fps == 30.0
    assert cfg.inference_profile == "history" and cfg.n_obs_steps == 8
    assert cfg.history_snapshots == 2 and cfg.condition_on_past_actions
    assert cfg.n_action_steps == 32 and cfg.text_fixed_length == 320 and cfg.video_position_fps == 24.0
    assert cfg.latent_hw == (8, 16)
    assert cfg.camera_layout == "side_by_side" and cfg.camera_keys == ("images.scene", "images.wrist")
    assert cfg.gripper_flip_dims == () and not cfg.augment
    assert cfg.action_parameterization == "joint_delta" and cfg.absolute_action_dims == (5,)
    assert cfg.action_normalization is None and cfg.state_normalization is None  # filled from the index
    assert cfg.normalization_clip == 6.0 and cfg.camera_dropout == {"images.wrist": 0.2}
    assert cfg.caption_dropout == 0.1 and cfg.train_timestep_shift == 42.0 and cfg.action_scale == 2.0
    assert cfg.optimizer_lr == 1.3e-4 and cfg.optimizer_lr_heads_multiplier == 5.0
    assert cfg.canvas_hw == (256, 512)


def test_history_settings_match_current_checkpoint():
    cfg = PolicyConfig(**TrainConfig.from_file(CONFIGS / "train.json").policy)
    delivered = json.loads((CONFIGS.parents[1] / "tests/fixtures/so101_history/config.json").read_text())
    for key in (
        "n_obs_steps",
        "history_snapshots",
        "condition_on_past_actions",
        "chunk_size",
        "n_action_steps",
        "fps",
        "video_position_fps",
        "text_fixed_length",
        "separate_timesteps",
        "video_logit_mean",
        "video_logit_std",
        "conditioning_noise_max",
        "loss_reduction",
        "action_channel_weights",
        "action_loss_weight",
        "video_loss_weight",
        "train_timestep_width",
        "train_timestep_shift",
    ):
        assert getattr(cfg, key) == delivered[key], key


def test_train_config_and_global_batch():
    cfg = TrainConfig.from_file(CONFIGS / "train.json")
    assert cfg.global_batch(16) == 512
    assert cfg.steps == 10000 and cfg.cooldown_start is None and cfg.checkpoint_every == 500
    assert cfg.keep_checkpoints is None and cfg.keep_every is None  # every checkpoint kept
    assert cfg.ema_sigma_rels == (0.1, 0.05) and cfg.betas == (0.9, 0.99) and cfg.weight_decay == 0.05
    assert cfg.frame_hw == (256, 256) and cfg.max_grad_norm == 1.0
    assert (cfg.frozen_steps, cfg.trunk_warmup_steps, cfg.heads_warmup_steps) == (1000, 2000, 1000)
    assert not cfg.reseed_on_resume


def test_learning_rates_follow_batch_scaling():
    train = TrainConfig.from_file(CONFIGS / "train.json")
    cfg = PolicyConfig(**train.policy)
    assert math.isclose(cfg.optimizer_lr, 2e-4 * 1.3 * math.sqrt(train.global_batch(16) / 2048))
    assert math.isclose(cfg.optimizer_lr * cfg.optimizer_lr_heads_multiplier, 6.5e-4)


def test_corpus_reference():
    counts = CORPUS["counts"]
    parts = CORPUS["parts"]
    assert len(parts) == counts["parts_built"] == 1021
    assert sum(p["train_windows"] for p in parts) == counts["train_windows"] == 631391
    assert sum(p["val_windows"] for p in parts) == counts["val_windows"] == 119
    assert sum(1 for p in parts if p["train_windows"]) == counts["parts_in_train"] == 1016
    assert [p["part"] for p in parts if p["val_windows"]] == CORPUS["val_parts"]
    origins = {p["origin"] for p in parts}
    assert origins == {"molmoact", "community_dataset_v3"}
    assert CORPUS["sources"]["molmoact"]["parts"] == sum(p["origin"] == "molmoact" for p in parts)
    assert CORPUS["sources"]["community_dataset_v3"]["parts"] == sum(
        p["origin"] == "community_dataset_v3" for p in parts
    )
    assert len(CORPUS["sources"]["community_dataset_v3"]["revision"]) == 40
    for p in parts:
        assert p["fps"] == 30.0 and p["scene_camera"].startswith("observation.images.")
        assert "/" in p["dataset"] and not p["dataset"].startswith("/")
        dim = p["action_dim"]
        assert len(p["action_q01"]) == len(p["action_q99"]) == dim
        assert len(p["state_q01"]) == len(p["state_q99"]) == dim
        # names come from the source dataset; a few declare one group label ("motors") instead of six names
        assert len(p["action_names"]) in (dim, 1) and len(p["state_names"]) in (dim, 1)
        # q99 == q01 happens (a gripper that never moved); the builder then normalizes with span 1.0
        assert all(hi >= lo for lo, hi in zip(p["action_q01"], p["action_q99"]))
        assert all(hi >= lo for lo, hi in zip(p["state_q01"], p["state_q99"]))
        if dim != 6:
            assert "excluded" in p and p["train_windows"] == 0 and p["val_windows"] == 0
        else:
            assert "excluded" not in p
    assert [p["part"] for p in parts] == sorted(p["part"] for p in parts)
    assert "degenerate_channels" in CORPUS["builder"]
