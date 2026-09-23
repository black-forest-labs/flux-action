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
"""Regression coverage for current history delivery packages."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE, make_policy, tiny_config
from safetensors.torch import save_file

from flux_action.inference import so101
from flux_action.models.text_encoder import MockTextEncoder, Qwen3VLEmbedder
from flux_action.policy import FluxActionPolicy

spec = importlib.util.spec_from_file_location(
    "so101_inference_example", Path(__file__).parents[1] / "examples/so101/inference.py"
)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


@pytest.mark.parametrize("length", [0, 21, 400])
def test_fixed_text_length_encodes_padding_and_truncates(length):
    def tokenize(texts, **kwargs):
        assert kwargs["padding"] == "max_length" and kwargs["max_length"] == 320
        return {
            "input_ids": torch.ones(len(texts), 320, dtype=torch.long),
            "attention_mask": (torch.arange(320)[None].expand(len(texts), -1) < length).long(),
        }

    def model(**kwargs):
        assert kwargs["attention_mask"].sum() == 2 * min(length, 320)
        # Hidden states on padding tokens must be encoded, not zero-padded after encoding.
        return SimpleNamespace(hidden_states=(torch.ones(2, 320, 2),))

    encoder = Qwen3VLEmbedder.__new__(Qwen3VLEmbedder)
    torch.nn.Module.__init__(encoder)
    encoder.model = SimpleNamespace(model=model, device="cpu")
    encoder.processor = SimpleNamespace(apply_chat_template=lambda *a, **k: "formatted", tokenizer=tokenize)
    encoder.output_layer = [0]
    encoder.dtype = torch.bfloat16
    result = encoder.forward_bucketed_batch(["caption", ""], fixed_length=320)
    assert len(result) == 2 and all(c.shape == (1, 320, 2) and c.sum() == 640 for c in result)


@pytest.fixture
def history_package(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures/so101_history"
    bundle = tmp_path
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        (bundle / name).write_bytes((fixture / name).read_bytes())
    config_path = bundle / "config.json"
    raw = json.loads(config_path.read_text())
    raw["video_vae_id"] = "org/base:video_vae.safetensors@fixed"
    raw["text_encoder_id"] = "org/base:text_encoder@fixed"
    config_path.write_text(json.dumps(raw))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in json.loads((bundle / name).read_text())["steps"]:
            if "state_file" not in step:
                continue
            streams = ("action", "state") if "observation_history" in step["registry_name"] else ("action",)
            tensors = {
                f"{stream}.{q}": torch.full((6,), bound)
                for stream in streams
                for q, bound in (("q01", -2.0), ("q99", 2.0))
            }
            save_file(tensors, bundle / step["state_file"])
    reference = make_policy(
        tiny_config(
            inference_profile="history",
            n_obs_steps=8,
            history_snapshots=2,
            condition_on_past_actions=True,
            chunk_size=42,
            n_action_steps=32,
            torch_dtype="bfloat16",
        )
    )
    save_file(reference.state_dict(), bundle / "model.safetensors")

    def tiny_policy(config, *, _restore):
        assert _restore
        config.dit_config = TINY_DIT
        policy = FluxActionPolicy(
            config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32), _restore=_restore
        )
        assert all(parameter.is_meta for parameter in policy.parameters())
        return policy

    monkeypatch.setattr(so101, "FluxActionPolicy", tiny_policy)
    return tmp_path


@pytest.mark.parametrize("remote", [False, True])
def test_root_package_resolves_shared_encoder_specs(history_package, monkeypatch, remote):
    bundle = history_package
    config_path = bundle / "config.json"
    raw = json.loads(config_path.read_text())
    raw["video_vae_id"] = "org/base:video_vae.safetensors@fixed"
    raw["text_encoder_id"] = "org/base:text_encoder@fixed"
    config_path.write_text(json.dumps(raw))

    def download(**kwargs):
        assert kwargs["repo_id"] == "org/so101" and kwargs["revision"] == "robot-commit"
        assert "policy_preprocessor_step_*.safetensors" in kwargs["allow_patterns"]
        return str(bundle)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    policy = (
        inference.load_policy("org/so101", revision="robot-commit")
        if remote
        else inference.load_policy(bundle)
    )
    assert policy.config.video_vae_id == raw["video_vae_id"]
    assert policy.config.text_encoder_id == raw["text_encoder_id"]
    assert policy.config.n_action_steps == 32


def test_current_loader_uses_checkpoint_settings_and_saved_processors(history_package):
    policy = inference.load_policy(history_package)
    cfg = policy.config
    assert all(
        parameter.device.type == "cpu" and parameter.dtype == torch.bfloat16
        for parameter in policy.parameters()
    )
    assert policy.dtype_ == torch.bfloat16
    assert (cfg.n_obs_steps, cfg.history_snapshots, cfg.condition_on_past_actions) == (8, 2, True)
    assert (cfg.chunk_size, cfg.n_action_steps) == (42, 32)
    assert cfg.sampler == "euler" and cfg.sampler_shift == 6.93 and cfg.inference_seed == 42
    assert cfg.action_normalization == {"q01": [-2.0] * 6, "q99": [2.0] * 6}
    assert cfg.camera_keys == ("images.scene", "images.wrist")
    obs = {key: torch.zeros(1, 3, 256, 256) for key in cfg.camera_keys}
    obs.update(state=torch.zeros(1, 6), task="move")
    actions = policy.predict_action_chunk(obs)
    assert actions.shape == (1, 42, 6) and torch.isfinite(actions).all()


@pytest.mark.parametrize("defect", ["normalization", "settings", "missing_state", "pipeline"])
def test_current_loader_rejects_inconsistent_processors_before_loading_model(
    history_package, monkeypatch, defect
):
    from safetensors.torch import load_file

    bundle = history_package
    path = bundle / "policy_preprocessor.json"
    pre = json.loads(path.read_text())
    state = bundle / pre["steps"][4]["state_file"]
    if defect == "normalization":
        values = load_file(state)
        values["action.q99"][0] += 1
        save_file(values, state)
    elif defect == "settings":
        pre["steps"][3]["config"]["n_obs_steps"] = 1
    elif defect == "pipeline":
        pre["steps"][3], pre["steps"][4] = pre["steps"][4], pre["steps"][3]
    else:
        state.unlink()
    path.write_text(json.dumps(pre))

    def unexpected_model(*args, **kwargs):
        pytest.fail("invalid package must be rejected before loading weights")

    monkeypatch.setattr(so101, "FluxActionPolicy", unexpected_model)
    with pytest.raises((ValueError, FileNotFoundError)):
        inference.load_policy(history_package)


def test_old_combined_processors_are_rejected(history_package):
    bundle = history_package
    path = bundle / "policy_preprocessor.json"
    pre = json.loads(path.read_text())
    pre["steps"] = pre["steps"][:4]
    pre["steps"][3]["registry_name"] = "flux3_action_history_normalizer"
    path.write_text(json.dumps(pre))
    with pytest.raises(ValueError, match="unsupported SO-101 processor pipeline"):
        inference.load_policy(history_package)


def test_package_requires_saved_processors(history_package):
    (history_package / "policy_preprocessor.json").unlink()
    with pytest.raises(FileNotFoundError):
        inference.load_policy(history_package)


def test_removed_profile_is_rejected():
    with pytest.raises(ValueError, match="inference_profile"):
        tiny_config(inference_profile="so101_r62")


@pytest.mark.parametrize("defect", ["dtype", "missing_key"])
def test_loader_rejects_inconsistent_weights(history_package, defect):
    from safetensors.torch import load_file

    path = history_package / "model.safetensors"
    state = load_file(path)
    key = next(iter(state))
    if defect == "dtype":
        state[key] = state[key].float()
    else:
        del state[key]
    save_file(state, path)
    with pytest.raises((AssertionError, RuntimeError), match="dtype|Missing key"):
        inference.load_policy(history_package)
