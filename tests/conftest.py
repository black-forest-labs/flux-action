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
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from flux_action.config import PolicyConfig
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.policy import FluxActionPolicy

TINY_DIT = {
    "hidden_size": 64,
    "num_heads": 2,
    "depth": 1,
    "depth_single_blocks": 1,
    "axes_dim": [8, 8, 8, 8],
    "context_in_dim": 32,
}


class FakeVideoVAE(nn.Module):
    """Shape-only CPU fixture; does not establish real VAE parity."""

    def __init__(self):
        super().__init__()
        self.register_buffer("proj", torch.randn(96, 3, generator=torch.Generator().manual_seed(0)) * 0.1)

    def encode(self, video):
        b, c, t, h, w = video.shape
        x = video.float()[:, :, ::4]
        tp = x.shape[2]
        x = F.avg_pool2d(x.transpose(1, 2).reshape(-1, c, h, w), 32)
        x = torch.einsum("nchw,oc->nohw", x, self.proj.float())
        return x.reshape(b, tp, 96, h // 32, w // 32).transpose(1, 2)

    def encode_task(self, video):
        return self.encode(video)

    def encode_frame(self, frame):
        """The single-frame path of the real VAE; counted so tests can assert which path ran."""
        self.frame_calls = getattr(self, "frame_calls", 0) + 1
        return self.encode(frame[:, :, None])


def tiny_config(**kwargs):
    values = dict(
        camera_layout="single",
        camera_keys=("images.top",),
        canvas_hw=(64, 96),
        action_dim=6,
        gripper_flip_dims=(),
        dit_config=TINY_DIT,
        torch_dtype="float32",
        sampler="euler",
        num_inference_steps=2,
        guidance_scale=1.0,
        sampler_shift=5.0,
        augment=False,
    )
    values.update(kwargs)
    return PolicyConfig(**values)


def make_policy(config=None):
    return FluxActionPolicy(
        config or tiny_config(), video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )


@pytest.fixture
def batch():
    rng = torch.Generator().manual_seed(42)
    return {
        "images.top": torch.rand(1, 33, 3, 64, 96, generator=rng),
        "state": torch.rand(1, 6, generator=rng),
        "action": torch.rand(1, 32, 6, generator=rng),
        "task": ["move the object"],
    }


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
