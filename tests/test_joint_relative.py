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
"""Request-state relative actions: one anchor per chunk, absolute dimensions untouched."""

import pytest
import torch
from conftest import make_policy, tiny_config

from flux_action.processing import normalization
from flux_action.processing.history import conditioning_values


def relative_config(**kwargs):
    return tiny_config(
        **{
            "action_parameterization": "joint_relative",
            "absolute_action_dims": (5,),
            "action_normalization": {"q01": [-2.0] * 6, "q99": [2.0] * 6},
            **kwargs,
        }
    )


def test_relative_round_trip_keeps_absolute_dims():
    rng = torch.Generator().manual_seed(0)
    actions = torch.randn(3, 32, 6, generator=rng)
    state = torch.randn(3, 6, generator=rng)
    targets = normalization.relative(actions, state, (5,))
    torch.testing.assert_close(targets[..., :5], actions[..., :5] - state[:, None, :5])
    assert torch.equal(targets[..., 5], actions[..., 5])
    for i in range(3):
        torch.testing.assert_close(normalization.from_relative(targets[i], state[i], (5,)), actions[i])
    with pytest.raises(ValueError, match="reference state"):
        normalization.relative(actions, state[0], (5,))
    with pytest.raises(ValueError, match="reference state"):
        normalization.from_relative(targets[0], state, (5,))


def test_relative_config_keeps_grippers_absolute():
    assert tiny_config().action_parameterization == "absolute"
    with pytest.raises(AssertionError, match="gripper"):
        relative_config(gripper_flip_dims=(4,))
    relative_config(gripper_flip_dims=(5,))


def test_training_targets_invert_to_the_commands(batch):
    policy = make_policy(relative_config())
    batch["state"] = torch.full((1, 6), 0.5)
    targets = policy.training_targets(batch)
    expected = batch["action"].clone()
    expected[..., :5] -= 0.5
    torch.testing.assert_close(targets, expected / 2)  # range normalization over [-2, 2]
    torch.testing.assert_close(policy.actions_from_targets(targets, batch["state"]), batch["action"])


def test_queue_anchors_each_chunk_at_its_request_state(monkeypatch):
    policy = make_policy(relative_config(n_action_steps=2))

    def predict(batch):
        targets = torch.full((1, 32, 6), 0.5)  # denormalizes to +1 per arm joint
        targets[..., -1] = 0.25  # absolute gripper command 0.5
        return targets

    monkeypatch.setattr(policy, "predict_normalized_targets", predict)
    expected = [11.0, 11.0, 103.0, 103.0]  # no integration and no carry-over from the last command
    for tick, value in enumerate(expected):
        state = torch.full((1, 6), 10.0 if tick == 0 else 100.0 + tick)
        action = policy.select_action({"state": state})
        torch.testing.assert_close(action[0, :5], torch.full((5,), value))
        assert action[0, -1] == 0.5


def test_history_queue_anchors_at_the_last_observed_state(monkeypatch):
    config = relative_config(
        inference_profile="history",
        n_obs_steps=4,
        history_snapshots=1,
        condition_on_past_actions=True,
        chunk_size=8,
        n_action_steps=3,
    )
    policy = make_policy(config)
    windows = []

    def predict(batch):
        windows.append(batch["state"].clone())
        targets = torch.full((1, 8, 6), 0.5)
        targets[..., -1] = 0.25
        return targets

    monkeypatch.setattr(policy, "predict_normalized_targets", predict)
    for tick in range(4):
        obs = {
            "images.top": torch.zeros(1, 3, 64, 96),
            "state": torch.full((1, 6), 10.0 * tick),
            "task": "move",
        }
        action = policy.select_action(obs)
        anchor = 0.0 if tick < 3 else 30.0
        torch.testing.assert_close(action[0, :5], torch.full((5,), anchor + 1.0))
        assert action[0, -1] == 0.5
    assert len(windows) == 2 and windows[1][0, -1, 0] == 30.0


def test_history_past_commands_are_relative_to_the_last_state():
    config = relative_config(
        inference_profile="history", n_obs_steps=4, history_snapshots=1, condition_on_past_actions=True
    )
    states = torch.arange(4).float()[None, :, None].expand(1, 4, 6)
    commands = states + 1
    past = conditioning_values(states, commands, config)[..., :6]
    assert (past[:, 0] == 0).all()
    expected = (commands[:, 1:, :5] - states[:, -1:, :5]) / 2
    torch.testing.assert_close(past[:, 1:, :5], expected)
    torch.testing.assert_close(past[:, 1:, 5], commands[:, 1:, 5] / 2)
