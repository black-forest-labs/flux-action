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
"""Packing the windows of a micro-batch into one DiT forward must equal separate per-window forwards."""

import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE, make_policy, tiny_config

from flux_action.models.text_encoder import MockTextEncoder
from flux_action.models.transformer import (
    JointSingleSeq,
    JointSingleSeqParams,
    packed_attention,
    window_major_order,
)
from flux_action.models.wiring import action_dit_params
from flux_action.processing import packing

MODALITY = "action"
CTX_DIM = TINY_DIT["context_in_dim"]


def tiny_dit(attn_mode="torch", seed=0):
    params = action_dit_params(JointSingleSeqParams(**TINY_DIT), MODALITY, 6, attn_mode=attn_mode)
    torch.manual_seed(seed)
    return JointSingleSeq(params).eval()


def window_inputs(n_windows, ctx_lens, latent_frames=3, latent_hw=(2, 3), chunk=8, seed=0):
    """Per-window DiT kwargs (``B = 1`` each, dense) and the same windows batched (``B = n``) + contexts."""
    gen = torch.Generator().manual_seed(seed)
    latents = torch.randn(n_windows, 96, latent_frames, *latent_hw, generator=gen)
    state = torch.randn(n_windows, 1, 6, generator=gen)
    actions = torch.randn(n_windows, chunk, 6, generator=gen)
    t = torch.rand(n_windows, generator=gen)
    ctxs = [torch.randn(1, n, CTX_DIM, generator=gen) for n in ctx_lens]
    video = packing.pack_video(latents)
    action = packing.pack_actions(state, actions, packing.default_action_times(n_windows, chunk), MODALITY)
    kwargs, targets, _ = packing.build_forward_kwargs(
        video, action, {}, t, MODALITY, torch.Generator().manual_seed(seed + 1)
    )
    kwargs = {k: v.float() if v.is_floating_point() else v for k, v in kwargs.items()}
    per_window = []
    for i in range(n_windows):
        text = packing.pack_text(ctxs[i], 768)
        per_window.append({**{k: v[i : i + 1] for k, v in kwargs.items()}, **text})
    return per_window, kwargs, ctxs


def unpack(out, n_windows):
    return {k: v.reshape(n_windows, v.shape[1] // n_windows, v.shape[2]) for k, v in out.items()}


@pytest.mark.parametrize("attn_mode", ["torch", "cudnn"])
@pytest.mark.parametrize("ctx_lens", [[4, 4, 4], [4, 8, 4, 12]])
def test_packed_forward_matches_per_window_forwards(attn_mode, ctx_lens):
    dit = tiny_dit(attn_mode)
    n = len(ctx_lens)
    per_window, kwargs, ctxs = window_inputs(n, ctx_lens)
    with torch.no_grad():
        dense = [dit(**kw) for kw in per_window]
        packed, seqlens = packing.pack_windows(kwargs, ctxs)
        out = unpack(dit(**packed, seqlens=seqlens), n)
    assert set(out) == set(dense[0]) == {"x_video", "x_video_cond", f"x_{MODALITY}", f"x_{MODALITY}_cond"}
    for name in out:
        expected = torch.cat([d[name] for d in dense])
        torch.testing.assert_close(out[name], expected, rtol=1e-4, atol=1e-4)


def test_packed_backward_matches_the_mean_of_per_window_losses():
    dit = tiny_dit()
    n = 3
    per_window, kwargs, ctxs = window_inputs(n, [4, 8, 4])
    targets = {k: torch.randn_like(v) for k, v in per_window[0].items() if k in ("x_video", f"x_{MODALITY}")}

    def loss_of(out, i):
        return sum(((out[k][0] - targets[k][0]) ** 2).mean() for k in targets) / n

    for p in dit.parameters():
        p.grad = None
    sum(loss_of(dit(**kw), 0) for kw in per_window).backward()
    reference = {name: p.grad.clone() for name, p in dit.named_parameters() if p.grad is not None}
    for p in dit.parameters():
        p.grad = None
    packed, seqlens = packing.pack_windows(kwargs, ctxs)
    out = unpack(dit(**packed, seqlens=seqlens), n)
    sum(sum(((out[k][i] - targets[k][0]) ** 2).mean() for k in targets) / n for i in range(n)).backward()
    assert reference
    for name, p in dit.named_parameters():
        if name in reference:
            torch.testing.assert_close(p.grad, reference[name], rtol=1e-4, atol=1e-5)


def test_unequal_stream_lengths_across_windows():
    """The general grouping path: windows whose content streams differ in length (not produced by the policy)."""
    dit = tiny_dit()
    a, _, ctx_a = window_inputs(1, [4], latent_frames=3, seed=1)
    b, _, ctx_b = window_inputs(1, [8], latent_frames=2, seed=2)
    dense = [dit(**a[0]), dit(**b[0])]
    packed, seqlens = {}, {}
    for key in a[0]:
        if key in ("vector",):
            packed[key] = a[0][key]
            continue
        packed[key] = torch.cat([a[0][key], b[0][key]], dim=1)
        if key.startswith("x_") and not key.endswith(("_ids", "_timesteps")):
            seqlens[key] = [a[0][key].shape[1], b[0][key].shape[1]]
    seqlens["ctx"] = [4, 8]
    out = dit(**packed, seqlens=seqlens)
    for name, value in out.items():
        expected = torch.cat([dense[0][name], dense[1][name]], dim=1)
        torch.testing.assert_close(value, expected, rtol=1e-4, atol=1e-4)


def test_window_major_order_is_a_permutation_with_inverse():
    order, inverse, lens = window_major_order([[2, 3], [4, 1], [1, 1]], torch.device("cpu"))
    assert lens == [7, 5] and order.tolist() == [0, 1, 5, 6, 7, 8, 10, 2, 3, 4, 9, 11]
    x = torch.arange(12)
    assert torch.equal(x[order][inverse], x)


def test_packed_attention_validates_its_inputs():
    q = torch.randn(1, 2, 6, 4)
    with pytest.raises(ValueError):
        packed_attention(q, q, q, "torch", [4, 4])
    with pytest.raises(ValueError):
        packed_attention(torch.randn(2, 2, 6, 4), q, q, "torch", [3, 3])
    dit = tiny_dit()
    per_window, kwargs, ctxs = window_inputs(2, [4, 4])
    packed, seqlens = packing.pack_windows(kwargs, ctxs)
    with pytest.raises(ValueError):
        dit(**packed, seqlens={**seqlens, "ctx": [4, 3]})


class VariableLengthMock(MockTextEncoder):
    """Caption buckets of 4 or 8 tokens, so a batch mixes context lengths as DROID captions do."""

    def forward_bucketed(self, text):
        return self.encode(1, n_tokens=8 if len(text) > 12 else 4, seed=len(text))[0]


def test_policy_trains_a_mixed_caption_batch_in_one_dit_forward():
    calls = []
    policy = make_policy(tiny_config(vae_batch_windows=2)).train()
    policy.frozen.text_encoder = VariableLengthMock(CTX_DIM)  # the encoders live in FrozenComponents
    policy.dit.register_forward_pre_hook(
        lambda m, args, kwargs: calls.append(kwargs["ctx"].shape), with_kwargs=True
    )
    rng = torch.Generator().manual_seed(0)
    batch = {
        "images.top": torch.rand(3, 33, 3, 64, 96, generator=rng),
        "state": torch.rand(3, 6, generator=rng),
        "action": torch.rand(3, 32, 6, generator=rng),
        "task": ["short", "a much longer caption", "short"],
    }
    loss, info = policy(batch)
    assert torch.isfinite(loss) and info["n_valid_windows"] == 3
    assert calls == [(1, 4 + 8 + 4, CTX_DIM)]  # one forward, contexts concatenated unpadded
    loss.backward()
    assert all(p.grad is not None for p in policy.dit.emb_in[MODALITY].parameters())
    assert len(policy._ctx_cache) == 2  # the repeated caption was encoded once


def test_batched_contexts_match_single_encodes_and_are_cached():
    encoder = MockTextEncoder(CTX_DIM)
    singles = [encoder.forward_bucketed(c) for c in ("a", "b", "a")]
    batched = encoder.forward_bucketed_batch(["a", "b", "a"])
    for s, b in zip(singles, batched, strict=True):
        assert torch.equal(s, b)
    policy = make_policy()
    encoded = []
    original = policy.text_encoder.forward_bucketed_batch
    object.__setattr__(
        policy.text_encoder,
        "forward_bucketed_batch",
        lambda texts: encoded.append(list(texts)) or original(texts),
    )
    first = policy._contexts(["x", "y", "x"], torch.device("cpu"))
    again = policy._contexts(["y", "z"], torch.device("cpu"))
    assert encoded == [["x", "y"], ["z"]]
    assert torch.equal(first[1][0], again[0][0])


def test_caption_cache_rollover_keeps_the_captions_of_the_batch():
    policy = make_policy()
    dev = torch.device("cpu")
    for start in range(0, 256, 32):
        policy._contexts([str(i) for i in range(start, start + 32)], dev)
    assert len(policy._ctx_cache) == 256
    before = policy._contexts(["0"], dev)[0][0]
    # "0" and "255" are cached, "new" overflows the cache: all three must come back, none may raise
    ctxs = policy._contexts(["0", "new", "255", "0"], dev)
    assert len(ctxs) == 4 and set(policy._ctx_cache) == {"0", "new", "255"}
    assert torch.equal(ctxs[0][0], before)
    assert torch.equal(ctxs[3][0], before)


def test_batched_vae_encode_matches_per_window_encode():
    vae = FakeVideoVAE()
    videos = torch.rand(3, 3, 33, 64, 96) * 2 - 1
    batched = packing.encode_videos(vae, videos, (2, 3))
    singles = torch.cat([packing.encode_video(vae, v, (2, 3)) for v in videos])
    assert batched.shape == (3, 96, 9, 2, 3)
    torch.testing.assert_close(batched, singles)


def _seeded_batch(n, seeds):
    rng = torch.Generator().manual_seed(5)
    return {
        "images.top": torch.rand(n, 33, 3, 64, 96, generator=rng),
        "state": torch.rand(n, 6, generator=rng),
        "action": torch.rand(n, 32, 6, generator=rng),
        "task": ["pick"] * n,
        "window_seed": torch.tensor(seeds),
    }


def test_prepared_windows_give_the_same_loss_as_the_inline_path():
    policy = make_policy(tiny_config(augment=True, vae_batch_windows=2)).train()
    batch = _seeded_batch(3, [11, 22, 33])
    torch.manual_seed(1)
    inline, _ = policy(batch)
    torch.manual_seed(1)
    prepared = policy.prepare(batch)
    assert prepared.idx == [0, 1, 2] and prepared.event is None
    ahead, info = policy(batch, prepared=prepared)
    assert torch.equal(inline, ahead) and info["n_valid_windows"] == 3


def test_window_seeds_fix_augmentation_and_caption_dropout_independently_of_the_global_rng():
    policy = make_policy(tiny_config(augment=True, caption_dropout=0.5)).train()
    seeded = _seeded_batch(4, [1, 2, 3, 4])
    torch.manual_seed(0)
    first = policy.prepare(seeded)
    torch.manual_seed(999)
    second = policy.prepare(seeded)
    assert torch.equal(first.latents, second.latents)
    assert all(torch.equal(a, b) for a, b in zip(first.ctxs, second.ctxs, strict=True))
    unseeded = {k: v for k, v in seeded.items() if k != "window_seed"}
    torch.manual_seed(0)
    third = policy.prepare(unseeded)
    torch.manual_seed(999)
    fourth = policy.prepare(unseeded)
    assert not torch.equal(third.latents, fourth.latents)  # augmentation from the global RNG differs
