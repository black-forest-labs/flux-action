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
"""The looped video-VAE decode equals the full decode; its halo arithmetic matches NATTEN's edge clamping."""

import pytest
import torch

pytest.importorskip("natten")  # the VAE imports NATTEN; on CPU it runs NATTEN's flex-attention backend

from flux_action.models import video_vae as vv  # noqa: E402
from flux_action.models.runtime import random_init_  # noqa: E402


def natten_window(i: int, length: int, kernel: int, causal: bool) -> range:
    """Frames that frame ``i`` attends to under NATTEN's neighborhood definition."""
    if causal:
        return range(max(0, i - kernel + 1), i + 1)
    start = min(max(i - kernel // 2, 0), length - kernel)
    return range(start, start + kernel)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kernel", [3, 5])
def test_temporal_core_halo_reproduces_full_sequence_neighborhoods(kernel, causal):
    for length in range(kernel, 4 * kernel):
        for max_t in range(kernel, kernel + 6):
            for core_start in range(0, length, max_t):
                core_end = min(core_start + max_t, length)
                hs, he = vv._temporal_core_halo(core_start, core_end, length, kernel, causal)
                assert 0 <= hs <= core_start and core_end <= he <= length
                assert he - hs >= kernel
                # Never reaches into the window before the previous one, whose output is already written.
                assert hs >= core_start - max_t
                for i in range(core_start, core_end):
                    inside_halo = natten_window(i - hs, he - hs, kernel, causal)
                    assert [f + hs for f in inside_halo] == list(natten_window(i, length, kernel, causal))


def tiny_params(**overrides) -> vv.ViTNormInferenceParams:
    # head_dim 8 at every stage; spatial kernel 3 keeps the 3x3 latent legal for NATTEN and the CPU cheap.
    return vv.ViTNormInferenceParams(
        z_dim=8,
        embed_dim=8,
        num_heads=[1, 2, 4, 8],
        window_size=[5, 3, 3],
        enc_depths=[1, 1, 1, 1],
        dec_depths=[1, 1, 2, 1],
        dtype="float32",
        use_compile=False,
        **overrides,
    )


@pytest.fixture
def cpu_backend(monkeypatch):
    monkeypatch.setenv(vv._NATTEN_BACKEND_ENV, "flex-fna")
    monkeypatch.setattr(vv, "_natten_backends", {})


def tiny_vae(decoder_max_t: int | None) -> vv.ViTNormInference:
    model = vv.ViTNormInference(tiny_params(decoder_max_t=decoder_max_t))
    random_init_(model, dtype=torch.float32)
    return model.eval()


def test_looped_decode_matches_full_decode(cpu_backend, monkeypatch):
    model = tiny_vae(decoder_max_t=5)
    regions = []
    forward_region = vv.Natten3D.forward_region
    monkeypatch.setattr(
        vv.Natten3D,
        "forward_region",
        lambda self, x, **kw: regions.append(x.shape[1]) or forward_region(self, x, **kw),
    )
    torch.manual_seed(1)
    # 5 latent frames -> 9 -> 17 through the decoder's two temporal expansions (NATTEN needs T >= kernel).
    z = torch.randn(1, 8, 5, 3, 3)
    with torch.inference_mode():
        looped = model.decode(z)
        for block in model.model.decoder.features.modules():
            if isinstance(block, vv.SwinTransformerBlock):
                block.max_t = None
        full = model.decode(z, chunked_decode=False)
    assert looped.shape == full.shape == (1, 3, 17, 96, 96)
    # The 5-frame stage runs in one pass. 9 frames loop as windows [0,5) [5,9) with halos of 7 and 6 frames;
    # 17 frames as [0,5) [5,10) [10,15) [15,17) with halos of 7, 9, 9 and 5 frames.
    assert regions and set(regions) == {7, 6, 9, 5}
    torch.testing.assert_close(looped, full, atol=1e-5, rtol=1e-4)


def test_looped_decoder_handles_a_single_frame_latent(cpu_backend):
    model = tiny_vae(decoder_max_t=5)
    with torch.inference_mode():
        out = model.decode(torch.randn(1, 8, 1, 3, 3))
    assert out.shape == (1, 3, 1, 96, 96)


def test_looped_decode_rejects_windows_below_the_kernel_and_gradients(cpu_backend):
    with pytest.raises(ValueError, match="temporal kernel"):
        vv.ViTNormInference(tiny_params(decoder_max_t=4))
    model = tiny_vae(decoder_max_t=5)
    with pytest.raises(RuntimeError, match="in place"):
        model.model.decode(torch.randn(1, 8, 9, 3, 3))
