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
"""The DROID entry point restores the root package's native config and verified weights."""

import builtins
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from conftest import FakeVideoVAE, make_policy, tiny_config
from safetensors.torch import save_file

from flux_action import cli
from flux_action.config import DROID_INFERENCE_SETTINGS
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy


def test_infer_help_does_not_require_serving_extras(monkeypatch, capsys):
    original_import = builtins.__import__

    def without_serving(name, *args, **kwargs):
        if name.startswith(("flux_action.serving", "msgpack", "websockets")):
            raise ModuleNotFoundError(f"serving extra unavailable: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_serving)
    monkeypatch.setattr(sys, "argv", ["flux-action", "infer", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert "--observation" in capsys.readouterr().out


def test_delivered_weights_load_strictly_and_predict(tmp_path, monkeypatch):
    reference = make_policy(
        tiny_config(
            **DROID_INFERENCE_SETTINGS,
            camera_layout="droid",
            camera_keys=("images.wrist", "images.left", "images.right"),
            canvas_hw=(544, 736),
            chunk_size=32,
            fps=15.0,
            action_scale=2.0,
            action_parameterization="absolute",
            action_normalization=None,
            state_normalization=None,
            single_frame_encode=False,
            inference_seed=0,
            action_dim=8,
            action_modality="action_prediction_droid",
            gripper_flip_dims=(-1,),
            content_streams=("video", "video_cond"),
            torch_dtype="bfloat16",
        )
    )
    bundle = tmp_path
    reference.save_pretrained(bundle)
    native = bundle / "config.native.json"
    (bundle / "config.json").rename(native)
    (bundle / "config.json").write_text('{"type": "flux3"}')
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"][native.name] = hashlib.sha256(native.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    weights = bundle / "model.safetensors"

    restore = FluxActionPolicy.from_pretrained

    def tiny_restore(directory, **kwargs):
        return restore(directory, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32), **kwargs)

    monkeypatch.setattr(FluxActionPolicy, "from_pretrained", tiny_restore)
    loaded = FluxActionPolicy.from_pretrained(tmp_path)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    for key, value in DROID_INFERENCE_SETTINGS.items():
        assert getattr(loaded.config, key) == value
    for key in (
        "camera_layout",
        "camera_keys",
        "canvas_hw",
        "chunk_size",
        "fps",
        "action_scale",
        "gripper_flip_dims",
        "action_parameterization",
        "action_normalization",
        "state_normalization",
        "single_frame_encode",
        "inference_seed",
    ):
        assert getattr(loaded.config, key) == getattr(reference.config, key)
    observation = tmp_path / "observation.npz"
    np.savez(
        observation,
        **{
            **{key: np.zeros((360, 640, 3), np.uint8) for key in reference.config.camera_keys},
            "state": np.zeros(8, np.float32),
        },
    )
    output = tmp_path / "inference"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flux-action",
            "infer",
            "--checkpoint",
            str(tmp_path),
            "--observation",
            str(observation),
            "--task",
            "move",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    cli.main()
    actions = np.load(output / "actions.npy")
    assert actions.shape == (1, 32, 8) and actions.dtype == np.float32 and np.isfinite(actions).all()
    state = reference.state_dict()
    state.pop(next(iter(state)))
    save_file(state, weights)
    manifest["sha256"][weights.name] = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Missing key"):
        FluxActionPolicy.from_pretrained(tmp_path)


@pytest.mark.parametrize("missing", ["config.json", "manifest.json", "model.safetensors"])
def test_missing_delivery_files_fail_before_model_loading(tmp_path, monkeypatch, missing):
    make_policy().save_pretrained(tmp_path)
    (tmp_path / missing).unlink()

    def unexpected_load(*args, **kwargs):
        pytest.fail("Must reject incomplete snapshots before loading the model or contacting the Hub")

    monkeypatch.setattr(FluxActionPolicy, "__init__", unexpected_load)
    with pytest.raises(FileNotFoundError, match=missing):
        FluxActionPolicy.from_pretrained(tmp_path)
