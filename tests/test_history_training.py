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
"""History training boundaries, conditioning parity, loss, and backward execution."""

import pytest
import torch
from conftest import make_policy, tiny_config

from flux_action.processing import history, packing


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
            "action_parameterization": "joint_delta",
            "absolute_action_dims": (-1,),
            "caption_dropout": 0.0,
            "separate_timesteps": True,
            "conditioning_noise_max": 0.2,
            "loss_reduction": "modalities",
            "action_loss_weight": 0.5,
            "action_channel_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            **kwargs,
        }
    )


def batch():
    rng = torch.Generator().manual_seed(24)
    commands = torch.randn(2, 50, 6, generator=rng).cumsum(1)
    return {
        "images.top": torch.randint(256, (2, 50, 3, 64, 96), dtype=torch.uint8, generator=rng),
        "state": torch.randn(2, 8, 6, generator=rng),
        "command_history": commands[:, :8],
        "action": commands[:, 8:],
        "task": ["move the object", "move"],
        "window_seed": torch.tensor([42, 43]),
    }


def test_history_training_matches_inference_conditioning_and_has_no_future_leak(monkeypatch):
    policy = make_policy(history_config()).eval()
    data = batch()
    lengths = []
    encode = policy.video_vae.encode_task
    monkeypatch.setattr(
        policy.video_vae, "encode_task", lambda clip: (lengths.append(clip.shape[2]), encode(clip))[1]
    )
    prepared = policy.prepare(data)
    assert lengths == [1, 1, 1, 1, 42, 42]
    assert all(ctx.shape[1] == 320 for ctx in prepared.ctxs)
    streams = {
        **history.pack_video(prepared.latents, policy.config),
        **history.pack_actions(prepared.state, prepared.actions, policy.config),
    }
    assert streams["x_video"].shape == (2, 66, 96)
    assert streams["x_video_cond"].shape == (2, 12, 96)
    assert streams["x_action_cond"].shape == (2, 8, 12)
    for i in range(2):
        expected = history.pack_conditioning(
            policy.video_vae,
            data["images.top"][i : i + 1, :8],
            data["state"][i : i + 1],
            data["command_history"][i : i + 1],
            policy.config,
        )
        for key, value in expected.items():
            torch.testing.assert_close(streams[key][i : i + 1], value, rtol=0, atol=0)
    assert streams["x_video_cond_ids"][0, ::6, 0].tolist() == [0, 29]
    assert streams["x_video_ids"][0, ::6, 0].tolist() == [33, 50, 66, 83, 100, 116, 133, 150, 166, 183, 200]
    assert streams["x_action_ids"][0, :4, 0].tolist() == [0, 3, 6, 10]
    torch.testing.assert_close(
        prepared.actions[:, 0, :5], data["action"][:, 0, :5] - data["command_history"][:, -1, :5]
    )
    torch.testing.assert_close(prepared.state[:, 0, :6], torch.zeros(2, 6))
    altered = {**data, "images.top": data["images.top"].clone(), "action": data["action"] + 100}
    altered["images.top"][:, 8:] = 255 - altered["images.top"][:, 8:]
    again = policy.prepare(altered)
    torch.testing.assert_close(prepared.latents[:, :, :2], again.latents[:, :, :2], rtol=0, atol=0)
    torch.testing.assert_close(prepared.state, again.state, rtol=0, atol=0)
    assert not torch.equal(prepared.latents[:, :, 2:], again.latents[:, :, 2:])


def test_separate_noise_and_weighted_modality_loss_backward():
    policy = make_policy(history_config()).train()
    calls = []
    policy.dit.register_forward_pre_hook(lambda m, a, k: calls.append(k), with_kwargs=True)
    loss, metrics = policy(batch())
    loss.backward()
    assert torch.isfinite(loss) and metrics["n_valid_windows"] == 2
    assert len(calls) == 1 and calls[0]["seqlens"]["x_action_cond"] == [8, 8]
    assert calls[0]["x_video_cond_timesteps"].gt(0).all()
    assert calls[0]["x_action_cond_timesteps"].eq(0).all()
    assert calls[0]["x_video_timesteps"][0, 0] != calls[0]["x_action_timesteps"][0, 0]
    assert policy.dit.final_layer["action"].linear.weight.grad.abs().sum() > 0
    policy.eval()
    policy(batch())
    assert calls[-1]["x_video_cond_timesteps"].eq(0).all()


def test_modality_loss_gripper_weight_and_mask():
    pred = {"x_video": torch.full((1, 2, 96), 2.0), "x_action": torch.tensor([[[1.0, 3.0], [1.0, 3.0]]])}
    target = {k: torch.zeros_like(v) for k, v in pred.items()}
    result = packing.flow_loss(
        pred, target, "action", 0.5, 1.0, reduction="modalities", channel_weights=[1.0, 2.0]
    )
    torch.testing.assert_close(result["loss"], torch.tensor(4.0 + 0.5 * (1.0 * 0.4 + 9.0 * 1.6) / 2))
    torch.testing.assert_close(result["action_mse"], torch.tensor(5.0))
    masked = packing.flow_loss(
        pred,
        target,
        "action",
        0.5,
        1.0,
        reduction="modalities",
        channel_weights=[1.0, 2.0],
        action_mask=torch.tensor([[1.0, 0.0]]),
    )
    torch.testing.assert_close(masked["loss"], torch.tensor(4.2))


@pytest.mark.parametrize(
    "settings",
    [{"conditioning_noise_max": 1.1}, {"loss_reduction": "sum"}, {"action_channel_weights": [0.0] * 6}],
)
def test_invalid_history_loss_contract(settings):
    with pytest.raises(ValueError):
        history_config(**settings)
