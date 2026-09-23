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
"""Shared encoders must resolve the same pinned repo/subfolder for all components."""

import sys
from types import SimpleNamespace

import pytest
import torch

from flux_action.models.text_encoder import Qwen3VLEmbedder


@pytest.mark.parametrize("local", [False, True])
def test_text_model_and_processor_share_location(monkeypatch, tmp_path, local):
    calls = []

    def model(spec, **kwargs):
        calls.append(("model", spec, kwargs))
        return torch.nn.Identity()

    def processor(spec, **kwargs):
        calls.append(("processor", spec, kwargs))
        return SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=model),
            AutoProcessor=SimpleNamespace(from_pretrained=processor),
        ),
    )
    spec = str(tmp_path) if local else "org/base:text_encoder@immutable"
    Qwen3VLEmbedder(spec)
    expected = {} if local else {"revision": "immutable", "subfolder": "text_encoder"}
    repo = spec if local else "org/base"
    assert calls == [
        ("model", repo, {"torch_dtype": torch.bfloat16, **expected}),
        ("processor", repo, expected),
    ]
