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
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import FakeVideoVAE, make_policy, tiny_config

from flux_action.inference.offline import run_inference
from flux_action.models.text_encoder import MockTextEncoder, Qwen3VLEmbedder
from flux_action.policy import FluxActionPolicy


def test_recorded_export_inference(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    make_policy().save_pretrained(checkpoint)
    restore = FluxActionPolicy.from_pretrained
    monkeypatch.setattr(
        FluxActionPolicy,
        "from_pretrained",
        lambda directory, device: restore(
            directory, device=device, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
        ),
    )
    observation = tmp_path / "observation.npz"
    np.savez(observation, **{"images.top": np.zeros((64, 96, 3), np.uint8), "state": np.zeros(6, np.float32)})
    first = run_inference(checkpoint, observation, tmp_path / "first", task="pick up", device="cpu")
    second = run_inference(
        tmp_path, observation, tmp_path / "second", subfolder="checkpoint", task="pick up", device="cpu"
    )
    first_actions = np.load(tmp_path / "first/actions.npy")
    second_actions = np.load(tmp_path / "second/actions.npy")
    assert np.isfinite(first_actions).all() and np.isfinite(second_actions).all(), "nonfinite actions"
    print(f"reload max_abs={np.abs(first_actions - second_actions).max():.9g}")
    assert second["checkpoint_manifest"] == json.loads((checkpoint / "manifest.json").read_text())
    assert first["actions_shape"] == [1, 32, 6]
    assert first["policy_config"]["sampler"] == "euler"
    assert not first["serving_setup"]["prepared"], "CPU inference uses the reference backend"
    assert first["gpu_memory_bytes"] == {
        "peak_during_load": None,
        "allocated_after_load": None,
        "peak_during_chunk": None,
    }
    assert json.loads((tmp_path / "first/report.json").read_text()) == json.loads(json.dumps(first))
    with pytest.raises(FileExistsError):
        run_inference(checkpoint, observation, tmp_path / "first", task="pick up", device="cpu")


def test_settings_supply_sampling_for_exports_without_them(tmp_path, monkeypatch):
    # a training export carries no sampler settings; --setting must supply them and be recorded
    checkpoint = tmp_path / "checkpoint"
    make_policy(
        tiny_config(sampler=None, num_inference_steps=None, guidance_scale=None, sampler_shift=None)
    ).save_pretrained(checkpoint)
    restore = FluxActionPolicy.from_pretrained
    monkeypatch.setattr(
        FluxActionPolicy,
        "from_pretrained",
        lambda directory, device: restore(
            directory, device=device, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
        ),
    )
    observation = tmp_path / "observation.npz"
    np.savez(observation, **{"images.top": np.zeros((64, 96, 3), np.uint8), "state": np.zeros(6, np.float32)})
    with pytest.raises(AssertionError, match="invalid sampler"):
        run_inference(checkpoint, observation, tmp_path / "none", task="pick up", device="cpu")
    settings = {"sampler": "euler", "num_inference_steps": 2, "guidance_scale": 1.0, "sampler_shift": 5.0}
    report = run_inference(
        checkpoint, observation, tmp_path / "set", task="pick up", device="cpu", settings=settings
    )
    assert report["settings"] == settings and report["policy_config"]["sampler"] == "euler"
    assert report["actions_shape"] == [1, 32, 6]


@pytest.mark.parametrize("length,expected", [(21, 80), (80, 80), (81, 160), (8100, 8160), (9000, 8192)])
def test_text_bucket_inputs_and_padding_hidden_states(length, expected):
    calls = []

    def tokenize(text, **kwargs):
        calls.append(kwargs)
        real = min(length, kwargs["max_length"])
        width = kwargs["max_length"] if kwargs["padding"] == "max_length" else real
        return {
            "input_ids": torch.ones(1, width, dtype=torch.long),
            "attention_mask": (torch.arange(width)[None] < real).long(),
        }

    def model(**kwargs):
        assert kwargs["use_cache"] is False
        assert kwargs["attention_mask"].sum() == min(length, 8192)
        # Preserve padded hidden states: they are part of the reference conditioning.
        return SimpleNamespace(hidden_states=(torch.ones(1, expected, 2),))

    encoder = Qwen3VLEmbedder.__new__(Qwen3VLEmbedder)
    torch.nn.Module.__init__(encoder)
    encoder.model = SimpleNamespace(model=model, device="cpu")
    encoder.processor = SimpleNamespace(apply_chat_template=lambda *a, **k: "formatted", tokenizer=tokenize)
    encoder.output_layer = [0]
    encoder.dtype = torch.bfloat16
    result = encoder.forward_bucketed("caption")
    assert result.shape == (1, expected, 2)
    assert result.sum() == 2 * expected
    assert calls[0]["max_length"] == 8192
    assert calls[1]["padding_side"] == "right"
