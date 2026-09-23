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
"""Pinned, selective downloads and strict native-policy restores from the Hub."""

import fnmatch
import sys
from types import SimpleNamespace

import pytest
import torch
from conftest import FakeVideoVAE, make_policy

from flux_action.hub import resolve_policy_directory
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy


@pytest.mark.parametrize("subfolder", [None, "variants/gd"])
def test_hub_restore_downloads_only_the_selected_policy(tmp_path, monkeypatch, subfolder):
    policy = make_policy()
    package = tmp_path / subfolder if subfolder else tmp_path
    policy.save_pretrained(package)
    prefix = f"{subfolder}/" if subfolder else ""
    files = [
        "model.safetensors",
        "config.native.json",
        "variants/gd/model.safetensors",
        "variants/gd/manifest.json",
        "variants/sd/model.safetensors",
        "variants/fp8r/model.safetensors",
        "video_vae.safetensors",
        "text_encoder/model.safetensors",
        "policy_preprocessor_step_3_flux3_observation_history_normalizer.safetensors",
    ]

    def download(**kwargs):
        assert kwargs["repo_id"] == "org/droid" and kwargs["revision"] == "immutable"
        selected = {
            name for name in files if any(fnmatch.fnmatchcase(name, p) for p in kwargs["allow_patterns"])
        }
        expected = (
            {"variants/gd/model.safetensors", "variants/gd/manifest.json"}
            if subfolder
            else {files[0], files[1], files[-1]}
        )
        assert selected == expected
        for name in ("model.safetensors", "config.json", "config.native.json", "manifest.json"):
            assert prefix + name in kwargs["allow_patterns"]
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    loaded = FluxActionPolicy.from_pretrained(
        "org/droid",
        revision="immutable",
        subfolder=subfolder,
        video_vae=FakeVideoVAE(),
        text_encoder=MockTextEncoder(32),
    )
    assert loaded.config.to_dict() == policy.config.to_dict()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    assert resolve_policy_directory(tmp_path, subfolder=subfolder) == package


@pytest.mark.parametrize("subfolder", ["../other", "/absolute", "variants/../../other"])
def test_rejects_subfolders_outside_the_repository(tmp_path, subfolder):
    with pytest.raises(ValueError, match="subfolder"):
        resolve_policy_directory(tmp_path, subfolder=subfolder)


def test_missing_local_path_and_local_revision_fail_without_hub_access(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_policy_directory(tmp_path / "missing")
    with pytest.raises(ValueError, match="revision"):
        resolve_policy_directory(tmp_path, revision="ignored-revision")


@pytest.mark.parametrize(
    "source", ["outputs/missing-export", "runs/x/export", "export.safetensors", "./nope", "~/nope"]
)
def test_path_like_sources_are_missing_files_not_hub_repositories(tmp_path, monkeypatch, source):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    # huggingface_hub is an optional extra (CI runs without it): stub the module so any Hub lookup fails loudly
    hub_stub = SimpleNamespace(snapshot_download=lambda **k: pytest.fail("asked the Hub for a path"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_stub)
    with pytest.raises(FileNotFoundError, match="no such policy directory"):
        resolve_policy_directory(source)


def test_missing_local_text_encoder_directory_is_a_missing_file(tmp_path, monkeypatch):
    from flux_action.models.text_encoder import load_text_encoder

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    with pytest.raises(FileNotFoundError, match="no such text encoder directory"):
        load_text_encoder("outputs/weights/text_encoder")
