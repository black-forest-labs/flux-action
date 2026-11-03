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
"""The prepared BF16 backend on non-DROID geometry, against the reference DiT (tiny model, CPU, eager)."""

import pytest
import torch
from conftest import make_policy, tiny_config

from flux_action.config import PolicyConfig
from flux_action.models.transformer_inf_bf16 import BF16InferenceDiT
from flux_action.policy import _prepared_bf16_dims


def bf16_config(**overrides):
    values = dict(torch_dtype="bfloat16", action_parameterization="absolute")
    return tiny_config(**{**values, **overrides})


def observation():
    rng = torch.Generator().manual_seed(3)
    return {
        "images.top": torch.rand(1, 3, 64, 96, generator=rng),
        "state": torch.rand(1, 6, generator=rng),
        "task": "move",
    }


def prepared_pair(config):
    """A reference policy and a copy whose DiT is the prepared BF16 backend (installed without CUDA)."""
    torch.manual_seed(0)
    reference = make_policy(config).eval()
    fast = make_policy(config).eval()
    fast.load_state_dict(reference.state_dict())
    dims = _prepared_bf16_dims(fast.config)
    assert dims is not None
    fast.dit = BF16InferenceDiT.from_state_dict(fast.dit.state_dict(), dims)
    fast.reset()
    fast._inference_prepared = True
    return reference, fast


def assert_close_bf16(actual, expected):
    # Eager CPU runs the same bf16 operations in the same order (exact today); allow kernel-level noise.
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-3)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"sampler": "cosmos_unipc", "num_inference_steps": 3},
        {"guidance_scale": 2.0, "guidance_scale_action": 1.5},
        {"action_parameterization": "joint_relative"},
        {"action_modality": "action_custom", "action_dim": 6},
    ],
)
def test_prepared_matches_reference(overrides):
    reference, fast = prepared_pair(bf16_config(**overrides))
    assert fast.dit.action == reference.modality
    assert_close_bf16(fast.predict_action_chunk(observation()), reference.predict_action_chunk(observation()))


@pytest.mark.parametrize("sampler", ["euler", "cosmos_unipc"])
def test_prepared_committed_prefix_matches_reference(sampler):
    reference, fast = prepared_pair(bf16_config(rtc_max_prefix=8, sampler=sampler))
    free = reference.predict_action_chunk(observation())
    committed = free[:, :5] + 0.05
    expected = reference.predict_action_chunk(observation(), committed_actions=committed)
    actual = fast.predict_action_chunk(observation(), committed_actions=committed)
    torch.testing.assert_close(actual[:, :5], committed, rtol=0, atol=0)
    assert_close_bf16(actual, expected)


def test_prepared_prefix_needs_clean_timestep(monkeypatch):
    """Control: committed tokens modulated with the noisy timestep instead of 0 no longer match the reference."""
    reference, fast = prepared_pair(bf16_config(rtc_max_prefix=8))
    committed = reference.predict_action_chunk(observation())[:, :5] + 0.05
    expected = reference.predict_action_chunk(observation(), committed_actions=committed)
    prepare_steps = fast.dit.prepare_steps
    monkeypatch.setattr(
        fast.dit, "prepare_steps", lambda request, v, a, action_prefix=0: prepare_steps(request, v, a)
    )
    actual = fast.predict_action_chunk(observation(), committed_actions=committed)
    assert (actual - expected)[:, 5:].abs().max() > 0


def test_prepared_prefix_keeps_step_shapes_fixed():
    _, fast = prepared_pair(bf16_config(rtc_max_prefix=8))
    shapes = set()
    for prefix in (1, 4, 8):
        committed = torch.zeros(1, prefix, 6)
        fast.predict_action_chunk(observation(), committed_actions=committed)
        text = fast._prepared_text_cache["move"]
        request = fast.dit.prepare_observation(
            text,
            video_ids=torch.zeros(1, 4, 4, dtype=torch.long),
            video_cond=torch.zeros(1, 4, 96, dtype=torch.bfloat16),
            video_cond_ids=torch.zeros(1, 4, 4, dtype=torch.long),
            action_ids=torch.zeros(1, 32, 4, dtype=torch.long),
            action_cond=torch.zeros(1, 1, 6, dtype=torch.bfloat16),
            action_cond_ids=torch.zeros(1, 1, 4, dtype=torch.long),
        )
        times = torch.tensor([0.9, 0.5])
        steps = fast.dit.prepare_steps(request, times, times, action_prefix=prefix)
        shapes.add(tuple(steps[0].early_action[0].shape))
    assert shapes == {(1, 32, 64)}  # a compiled hot path sees one shape for every prefix length


def test_prepared_dims_gate():
    assert _prepared_bf16_dims(tiny_config()) is None  # float32
    config = bf16_config()
    dims = _prepared_bf16_dims(config)
    assert dims.hidden_size == 64 and dims.joint_depth == 1 and dims.context_dim == 32
    for dit_override in ({"depth_late_blocks": 1}, {"qkv_bias": True}, {"gate_type": "tanh"}):
        dit_config = {**config.dit_config, **dit_override}
        assert _prepared_bf16_dims(PolicyConfig(**{**config.to_dict(), "dit_config": dit_config})) is None
    released = PolicyConfig(torch_dtype="bfloat16")
    assert _prepared_bf16_dims(released) is not None


def test_prepared_history_profile_matches_reference():
    from test_history_inference import history_config
    from test_history_inference import observation as history_observation

    reference, fast = prepared_pair(history_config(torch_dtype="bfloat16"))
    for tick in range(3):  # the history window fills over several ticks
        expected = reference.select_action(history_observation(tick))
        actual = fast.select_action(history_observation(tick))
        assert_close_bf16(actual, expected)
    assert_close_bf16(
        fast.predict_action_chunk(history_observation(3)),
        reference.predict_action_chunk(history_observation(3)),
    )


def test_prepare_inference_twice_keeps_custom_geometry(monkeypatch):
    _, fast = prepared_pair(bf16_config(action_modality="action_custom"))
    monkeypatch.setattr(type(fast), "device", property(lambda self: torch.device("cuda")))
    monkeypatch.setattr(fast.dit, "validate_runtime_device", lambda: None)
    assert fast.prepare_inference(compile=False)  # already prepared: no DROID-only assertion
