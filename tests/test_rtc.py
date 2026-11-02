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
"""Committed-prefix RTC: clean prefix tokens in training, masked loss, pinned prefix at inference."""

import pytest
import torch
from conftest import make_policy, tiny_config

from flux_action.processing import packing


def history_config(**kwargs):
    return tiny_config(
        **{
            "inference_profile": "history",
            "n_obs_steps": 8,
            "history_snapshots": 2,
            "condition_on_past_actions": True,
            "chunk_size": 32,
            "n_action_steps": 32,
            "fps": 30.0,
            "action_parameterization": "joint_relative",
            "absolute_action_dims": (5,),
            "rtc_max_prefix": 15,
            "caption_dropout": 0.0,
            **kwargs,
        }
    )


def absolute_config(**kwargs):
    return tiny_config(
        **{
            "rtc_max_prefix": 15,
            "caption_dropout": 0.0,
            "action_normalization": {"q01": [-2.0] * 6, "q99": [2.0] * 6},
            **kwargs,
        }
    )


def history_batch():
    rng = torch.Generator().manual_seed(4)
    return {
        "images.top": torch.rand(3, 40, 3, 64, 96, generator=rng),
        "state": torch.randn(3, 8, 6, generator=rng),
        "command_history": torch.randn(3, 8, 6, generator=rng) + 5,
        "action": torch.randn(3, 32, 6, generator=rng),
        "task": ["move"] * 3,
        "window_seed": torch.tensor([40, 41, 42]),
        "rtc_prefix_lengths": torch.tensor([0, 6, 15]),
    }


def absolute_batch():
    rng = torch.Generator().manual_seed(5)
    return {
        "images.top": torch.rand(3, 33, 3, 64, 96, generator=rng),
        "state": torch.randn(3, 6, generator=rng),
        "action": torch.randn(3, 32, 6, generator=rng),
        "task": ["move"] * 3,
        "window_seed": torch.tensor([40, 41, 42]),
        "rtc_prefix_lengths": torch.tensor([0, 6, 15]),
    }


def test_config_validation():
    assert tiny_config().rtc_max_prefix == 0
    for bad in (-1, 32, 1.5):
        with pytest.raises(ValueError, match="rtc_max_prefix"):
            tiny_config(rtc_max_prefix=bad)
    with pytest.raises(ValueError, match="joint_relative"):
        tiny_config(rtc_max_prefix=4, action_parameterization="joint_delta", absolute_action_dims=(5,))


@pytest.mark.parametrize(
    "make_config,make_batch", [(history_config, history_batch), (absolute_config, absolute_batch)]
)
def test_prefix_reaches_model_clean_at_timestep_zero(make_config, make_batch):
    policy = make_policy(make_config()).train()
    data = make_batch()
    prepared = policy.prepare(data)
    calls = []
    policy.dit.register_forward_pre_hook(lambda m, a, kw: calls.append(kw), with_kwargs=True)
    loss, metrics = policy(data, prepared)
    loss.backward()
    assert torch.isfinite(loss) and metrics["n_valid_windows"] == 3
    acts = calls[0]["x_action"].reshape(3, 32, 6)
    times = calls[0]["x_action_timesteps"].reshape(3, 32)
    clean = (prepared.actions * policy.config.action_scale).to(torch.bfloat16).float()
    for i, p in enumerate([0, 6, 15]):
        torch.testing.assert_close(acts[i, :p], clean[i, :p], rtol=0, atol=0)
        assert times[i, :p].eq(0).all() and times[i, p:].gt(0).all()
    assert calls[0]["x_video_timesteps"].gt(0).all()
    assert policy.dit.final_layer["action"].linear.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("packed", [True, False])
def test_prefix_has_zero_gradient_and_window_weight_is_length_invariant(packed):
    shape = (1, 96, 6) if packed else (3, 32, 6)
    pred = {
        "x_video": torch.ones(1, 12, 96, requires_grad=True),
        "x_action": torch.ones(shape, requires_grad=True),
    }
    targets = {k: torch.zeros_like(v) for k, v in pred.items()}
    mask = torch.arange(32)[None] >= torch.tensor([0, 6, 15])[:, None]
    result = packing.flow_loss(pred, targets, "action", action_token_mask=mask)
    base = packing.flow_loss(pred, targets, "action")
    torch.testing.assert_close(result["loss"], base["loss"])
    torch.testing.assert_close(result["action_mse"], torch.tensor(1.0))
    result["loss"].backward()
    grad = pred["x_action"].grad.reshape(3, 32, 6)
    assert grad[~mask].eq(0).all() and grad[mask].gt(0).all()


def test_zero_prefix_matches_a_policy_without_rtc():
    data = absolute_batch()
    data["rtc_prefix_lengths"] = torch.zeros(3, dtype=torch.long)
    plain = {k: v for k, v in data.items() if k != "rtc_prefix_lengths"}
    losses = []
    for config, batch in ((absolute_config(), data), (absolute_config(rtc_max_prefix=0), plain)):
        torch.manual_seed(0)
        policy = make_policy(config).train()
        losses.append(policy(batch)[0])
        obs = {"images.top": batch["images.top"][:1, :1], "state": batch["state"][:1], "task": "move"}
        losses.append(policy.predict_action_chunk(obs))
    torch.testing.assert_close(losses[0], losses[2], rtol=0, atol=0)
    torch.testing.assert_close(losses[1], losses[3], rtol=0, atol=0)


def test_prefix_sampling_is_seeded_and_covers_the_range():
    policy = make_policy(history_config()).train()
    data = history_batch()
    data.pop("rtc_prefix_lengths")
    first = policy.prepare(data).action_prefix_lengths
    torch.testing.assert_close(policy.prepare(data).action_prefix_lengths, first)
    assert policy.eval().prepare(data).action_prefix_lengths.eq(15).all()
    seen = {
        int(torch.randint(16, (), generator=torch.Generator().manual_seed(seed ^ 0x525443)))
        for seed in range(500)
    }
    assert seen == set(range(16))


@pytest.mark.parametrize("sampler", ["euler", "cosmos_unipc"])
def test_history_inference_pins_the_committed_prefix(sampler):
    policy = make_policy(history_config(sampler=sampler, num_inference_steps=4)).eval()
    data = history_batch()
    obs = {
        "images.top": data["images.top"][:1, :8],
        "state": data["state"][:1],
        "command_history": data["command_history"][:1],
        "task": "move",
    }
    prefix = data["action"][:1, :15]
    for offset in (0, 3):
        request = {**obs, "state": obs["state"] + offset}
        calls = []
        handle = policy.dit.register_forward_pre_hook(lambda m, a, kw: calls.append(kw), with_kwargs=True)
        actions = policy.predict_action_chunk(request, committed_actions=prefix)
        handle.remove()
        torch.testing.assert_close(actions[:, :15], prefix, rtol=0, atol=0)
        expected = policy._action_targets(prefix, request["state"][:, -1]) * policy.config.action_scale
        for call in calls:
            torch.testing.assert_close(call["x_action"][:, :15], expected)
            assert call["x_action_timesteps"][:, :15].eq(0).all()
            assert call["x_action_timesteps"][:, 15:].gt(0).all()
        assert torch.isfinite(actions).all()
    assert policy.predict_action_chunk(obs).shape == (1, 32, 6)


def test_absolute_inference_pins_normalized_prefix():
    policy = make_policy(absolute_config(num_inference_steps=3)).eval()
    data = absolute_batch()
    obs = {"images.top": data["images.top"][:1, :1], "state": data["state"][:1], "task": "move"}
    prefix = data["action"][:1, :4]
    calls = []
    policy.dit.register_forward_pre_hook(lambda m, a, kw: calls.append(kw), with_kwargs=True)
    actions = policy.predict_action_chunk(obs, committed_actions=prefix)
    torch.testing.assert_close(actions[:, :4], prefix, rtol=0, atol=0)
    expected = prefix / 2 * policy.config.action_scale  # range normalization over [-2, 2]
    for call in calls:
        torch.testing.assert_close(call["x_action"][:, :4], expected)
    empty = policy.predict_action_chunk(obs, committed_actions=prefix[:, :0])
    torch.testing.assert_close(empty, policy.predict_action_chunk(obs), rtol=0, atol=0)


def test_invalid_prefixes_are_rejected():
    policy = make_policy(history_config())
    data = history_batch()
    with pytest.raises(ValueError, match="rtc_prefix_lengths"):
        policy.prepare({**data, "rtc_prefix_lengths": torch.tensor([0, 6, 16])})
    obs = {"images.top": data["images.top"][:1, :8], "state": data["state"][:1], "task": "move"}
    obs["command_history"] = data["command_history"][:1]
    with pytest.raises(ValueError, match="committed_actions"):
        policy.predict_action_chunk(obs, committed_actions=data["action"][:1, :16])
    plain = make_policy(tiny_config())
    batch = absolute_batch()
    with pytest.raises(ValueError, match="rtc_max_prefix"):
        plain.prepare(batch)
    obs = {"images.top": batch["images.top"][:1, :1], "state": batch["state"][:1], "task": "move"}
    with pytest.raises(ValueError, match="rtc_max_prefix"):
        plain.predict_action_chunk(obs, committed_actions=batch["action"][:1, :2])


def test_overlap_validation():
    absolute_config(rtc_overlap=15)
    with pytest.raises(ValueError, match="rtc_overlap"):
        absolute_config(rtc_overlap=16)
    with pytest.raises(ValueError, match="rtc_overlap"):
        tiny_config(rtc_overlap=1)
    with pytest.raises(ValueError, match="n_action_steps"):
        absolute_config(rtc_overlap=4, n_action_steps=4)
    config = absolute_config()
    config.rtc_overlap = 20  # changed after loading: caught before the next prediction
    with pytest.raises(ValueError, match="rtc_overlap"):
        config.validate_inference()


def chunk_stub(policy, calls):
    """predict_action_chunk stand-in: chunk n holds 100 * n + k, with the committed commands as its start."""

    def predict(batch, committed_actions=None):
        calls.append(None if committed_actions is None else committed_actions.clone())
        chunk = 100.0 * len(calls) + torch.arange(32.0)[None, :, None].expand(1, 32, 6).clone()
        if committed_actions is not None:
            chunk[:, : committed_actions.shape[1]] = committed_actions
        return chunk

    return predict


@pytest.mark.parametrize(
    "make_config",
    [lambda **kw: absolute_config(**kw), lambda **kw: history_config(**kw)],
    ids=["absolute", "history_relative"],
)
def test_select_action_commits_the_queued_commands(make_config, monkeypatch):
    policy = make_policy(make_config(rtc_overlap=5, n_action_steps=8))
    calls = []
    monkeypatch.setattr(policy, "predict_action_chunk", chunk_stub(policy, calls))
    obs = {"images.top": torch.zeros(1, 3, 64, 96), "state": torch.zeros(1, 6), "task": "move"}
    returned = [float(policy.select_action(obs)[0, 0]) for _ in range(9)]
    # chunk 1 gives 8 commands; at 5 left a new chunk starts with those 5 and adds 3 new ones, so the
    # robot sees one continuous sequence: chunk 1 to its end, then chunk 2's new commands
    assert returned == [100, 101, 102, 103, 104, 105, 106, 107, 205]
    assert calls[0] is None
    torch.testing.assert_close(calls[1][0, :, 0], torch.tensor([103.0, 104, 105, 106, 107]))
    torch.testing.assert_close(calls[2][0, :, 0], torch.tensor([106.0, 107, 205, 206, 207]))
    assert len(calls) == 3


def test_select_action_without_overlap_is_unchanged(monkeypatch):
    policy = make_policy(absolute_config(n_action_steps=4))
    calls = []
    monkeypatch.setattr(policy, "predict_action_chunk", chunk_stub(policy, calls))
    obs = {"images.top": torch.zeros(1, 3, 64, 96), "state": torch.zeros(1, 6), "task": "move"}
    returned = [float(policy.select_action(obs)[0, 0]) for _ in range(6)]
    assert returned == [100, 101, 102, 103, 200, 201]
    assert calls == [None, None]


def test_select_action_overlap_runs_the_real_model():
    policy = make_policy(absolute_config(rtc_overlap=3, n_action_steps=6)).eval()
    obs = {"images.top": torch.rand(1, 3, 64, 96), "state": torch.zeros(1, 6), "task": "move"}
    first = policy.predict_action_chunk(obs)[0, :6]
    returned = torch.stack([policy.select_action(obs)[0] for _ in range(7)])
    torch.testing.assert_close(returned[:6], first, rtol=0, atol=0)  # 3 committed commands come back exactly
    assert torch.isfinite(returned).all()
