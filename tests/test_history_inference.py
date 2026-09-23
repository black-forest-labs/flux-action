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
"""History checkpoints: token contract, per-tick command feedback and export/reload."""

import pytest
import torch
from conftest import FakeVideoVAE, make_policy, tiny_config

from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy
from flux_action.processing.history import observation_window, pack_conditioning


def history_config(**kwargs):
    return tiny_config(
        **{
            "inference_profile": "history",
            "n_obs_steps": 8,
            "history_snapshots": 2,
            "condition_on_past_actions": True,
            "chunk_size": 42,
            "n_action_steps": 32,
            "fps": 30.0,
            "video_position_fps": 24.0,
            "action_parameterization": "joint_delta",
            "absolute_action_dims": (5,),
            **kwargs,
        }
    )


def observation(tick=0):
    return {
        "images.top": torch.full((1, 3, 64, 96), tick / 255),
        "state": torch.full((1, 6), float(tick)),
        "task": "move",
    }


def test_history_tokens_follow_snapshot_and_command_clocks():
    config = history_config(
        state_normalization={"q01": [0.0] * 6, "q99": [10.0] * 6},
        action_normalization={"q01": [-2.0] * 6, "q99": [2.0] * 6},
    )
    states = torch.arange(8).float()[None, :, None].expand(1, 8, 6)
    commands = states + 1
    cameras = torch.zeros(1, 8, 3, 64, 96)
    cameras[:, -1] = 1
    vae = FakeVideoVAE()
    cond = pack_conditioning(vae, cameras, states, commands, config)
    assert vae.frame_calls == 2
    assert cond["x_video_cond"].shape == (1, 12, 96)
    assert cond["x_video_cond_ids"][0, :, 0].tolist() == [0] * 6 + [29] * 6
    assert not torch.equal(cond["x_video_cond"][:, :6], cond["x_video_cond"][:, 6:])
    ids = cond["x_action_cond_ids"][0]
    assert ids[:, 0].tolist() == [-24, -20, -17, -14, -10, -7, -4, 0]
    assert ids[:, -1].tolist() == [-1] * 8
    expected_state = states / 5 - 1
    torch.testing.assert_close(cond["x_action_cond"][..., 6:], expected_state, rtol=0, atol=0)
    past = cond["x_action_cond"][..., :6]
    assert (past[:, 0] == 0).all()
    assert (past[:, 1:, :5] == 0.5).all()  # consecutive command deltas, unscaled
    torch.testing.assert_close(past[:, 1:, 5], commands[:, 1:, 5] / 2)


def test_history_select_records_every_tick_and_only_executed_commands(monkeypatch):
    policy = make_policy(history_config(n_action_steps=3))
    windows = []

    def predict(batch):
        windows.append(
            {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
        )
        return torch.ones(1, 42, 6)

    monkeypatch.setattr(policy, "predict_normalized_targets", predict)
    for tick in range(7):
        command = policy.select_action(observation(tick))
        assert command[0, 0] == tick + 1
        assert command[0, 5] == 1
        command.fill_(-999)  # callers must not corrupt the command history
    assert len(windows) == 3
    assert windows[1]["state"][0, :, 0].tolist() == [0, 0, 0, 0, 0, 1, 2, 3]
    assert windows[1]["command_history"][0, :, 0].tolist() == [0, 0, 0, 0, 0, 1, 2, 3]
    assert windows[2]["images.top"][0, :, 0, 0, 0].tolist() == pytest.approx(
        [0, 0, 1 / 255, 2 / 255, 3 / 255, 4 / 255, 5 / 255, 6 / 255]
    )
    assert len(policy._action_queue) == 2  # unexecuted predictions do not enter history
    policy.reset()
    assert not policy._observation_history.observations and policy._last_command is None
    assert policy.select_action(observation(10))[0, 0] == 11
    assert (windows[-1]["command_history"] == 10).all()


def test_history_predict_export_reload_and_explicit_window(tmp_path):
    policy = make_policy(history_config())
    assert policy.dit.emb_in["action_cond"].in_features == 12
    captures = []
    policy.dit.register_forward_pre_hook(lambda m, args, kwargs: captures.append(kwargs), with_kwargs=True)
    obs = observation(3)
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (1, 42, 6) and torch.isfinite(chunk).all()
    assert not policy._observation_history.observations  # stateless prediction
    first = captures[0]
    assert first["x_video"].shape[1] == 11 * 6
    assert first["x_video_ids"][0, ::6, 0].tolist() == [33, 50, 66, 83, 100, 116, 133, 150, 166, 183, 200]
    assert first["x_action_ids"][0, :4, 0].tolist() == [0, 3, 6, 10]
    assert first["ctx"].shape[1] == 320
    explicit = observation_window(obs, policy.config)
    torch.testing.assert_close(policy.predict_action_chunk(explicit), chunk, rtol=0, atol=0)
    del explicit["command_history"]
    with pytest.raises(ValueError, match="command_history"):
        policy.predict_action_chunk(explicit)
    policy.save_pretrained(tmp_path)
    restored = FluxActionPolicy.from_pretrained(
        tmp_path, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    torch.testing.assert_close(restored.predict_action_chunk(obs), chunk, rtol=0, atol=0)
    with pytest.raises(ValueError, match="training needs 50 frames"):
        policy.prepare(obs)


def test_history_requires_reset_after_batch_size_changes():
    policy = make_policy(history_config())
    policy.select_action(observation())
    obs = observation()
    obs["state"] = obs["state"].expand(2, -1)
    obs["images.top"] = obs["images.top"].expand(2, -1, -1, -1)
    with pytest.raises(ValueError, match="reset"):
        policy.select_action(obs)
