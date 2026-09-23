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
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import TINY_DIT, FakeVideoVAE, make_policy, tiny_config
from safetensors.torch import load_file, save_file

import flux_action
from flux_action.config import CONTENT_STREAMS, DROID_INFERENCE_SETTINGS, PolicyConfig
from flux_action.models.text_encoder import MockTextEncoder
from flux_action.models.transformer import JointSingleSeq, JointSingleSeqParams
from flux_action.models.wiring import action_dit_params, restrict_content_streams, stream_of_key
from flux_action.policy import FluxActionPolicy, _released_inference_config
from flux_action.processing import packing


def report_prediction_drift(actual, expected):
    assert actual.shape == expected.shape, "prediction shape mismatch"
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), "nonfinite prediction"
    print(f"prediction max_abs={(actual.float() - expected.float()).abs().max().item():.9g}")


def test_train_export_reload_predict_and_queue(batch, tmp_path):
    policy = make_policy().train()
    before = policy.dit.emb_in["action"].weight.detach().clone()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    loss, info = policy(batch)
    assert torch.isfinite(loss) and info["n_valid_windows"] == 1
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, policy.dit.emb_in["action"].weight)
    observation = {**batch, "images.top": batch["images.top"][:, 0]}
    predicted = policy.predict_action_chunk(observation)
    assert predicted.shape == (1, 32, 6) and torch.isfinite(predicted).all()
    policy.config.trunk_weights = "/nonexistent/initialization.safetensors"
    policy.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["trunk_weights"] is None
    assert all(key.startswith("dit.") for key in load_file(str(tmp_path / "model.safetensors")))
    restored = FluxActionPolicy.from_pretrained(
        tmp_path, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    report_prediction_drift(restored.predict_action_chunk(observation), predicted)
    report_prediction_drift(restored.select_action(observation), predicted[:, 0])
    assert len(restored._action_queue) == 31
    restored.reset()
    assert not restored._action_queue and not restored._ctx_cache
    with pytest.raises(FileExistsError):
        policy.save_pretrained(tmp_path)


def test_single_frame_encode_uses_the_frame_path_and_survives_export(batch, tmp_path):
    observation = {**batch, "images.top": batch["images.top"][:, 0]}
    reference = make_policy(tiny_config(single_frame_encode=False))
    reference.predict_action_chunk(observation)
    assert getattr(reference.video_vae, "frame_calls", 0) == 0
    reference.prepare_inference(compile=False)
    reference.predict_action_chunk(observation)
    assert reference.video_vae.frame_calls == 1
    fast = make_policy()
    fast.load_state_dict(reference.state_dict())
    predicted = fast.predict_action_chunk(observation)
    assert fast.video_vae.frame_calls == 1 and predicted.shape == (1, 32, 6)
    fast.save_pretrained(tmp_path)
    restored = FluxActionPolicy.from_pretrained(
        tmp_path, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert restored.config.single_frame_encode is True
    report_prediction_drift(restored.predict_action_chunk(observation), predicted)


def test_full_restore_rejects_missing_head(batch, tmp_path):
    policy = make_policy()
    policy.save_pretrained(tmp_path)
    file = tmp_path / "model.safetensors"
    state = load_file(str(file))
    del state["dit.emb_in.action.weight"]
    save_file(state, str(file))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["sha256"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Missing key"):
        FluxActionPolicy.from_pretrained(tmp_path, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


def test_delta_queue_continues_from_last_command_despite_tracking_lag(monkeypatch):
    policy = make_policy(
        tiny_config(
            action_parameterization="joint_delta",
            absolute_action_dims=(5,),
            gripper_flip_dims=(),
            n_action_steps=2,
            action_normalization={"q01": [-2.0] * 6, "q99": [2.0] * 6},
        )
    )
    observations = []

    def predict(batch):
        observations.append(batch["state"].clone())
        targets = torch.full((1, 32, 6), 0.75)  # denormalizes to +1.5 per arm joint
        targets[..., -1] = 0.25  # absolute gripper command 0.5
        return targets

    monkeypatch.setattr(policy, "predict_normalized_targets", predict)
    for tick in range(5):
        state = torch.full((1, 6), 10.0 if tick == 0 else 100.0 + tick)
        action = policy.select_action({"state": state})
        torch.testing.assert_close(action[0, :5], torch.full((5,), 10.0 + (tick + 1) * 1.5))
        assert action[0, -1] == 0.5
        action.fill_(-999)  # caller mutation must not change the retained command anchor
    assert [value[0, 0].item() for value in observations] == [10.0, 102.0, 104.0]
    assert len(policy._action_queue) == 1
    policy.reset()
    assert not policy._action_queue and policy._last_command is None
    fresh = policy.select_action({"state": torch.full((1, 6), 20.0)})
    torch.testing.assert_close(fresh[0, :5], torch.full((5,), 21.5))
    assert fresh[0, -1] == 0.5


def test_native_config_alongside_lerobot_config_is_verified(batch, tmp_path):
    policy = make_policy()
    policy.save_pretrained(tmp_path)
    native = tmp_path / "config.native.json"
    raw = json.loads((tmp_path / "config.json").read_text())
    raw["quantization"] = None
    native.write_text(json.dumps(raw))
    (tmp_path / "config.json").write_text('{"type": "flux3"}')
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["sha256"][native.name] = hashlib.sha256(native.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    restored = FluxActionPolicy.from_pretrained(
        tmp_path, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    observation = {**batch, "images.top": batch["images.top"][:, 0]}
    report_prediction_drift(
        restored.predict_action_chunk(observation), policy.predict_action_chunk(observation)
    )
    native.write_text(native.read_text() + " ")
    with pytest.raises(AssertionError, match="checkpoint checksum mismatch: config.native.json"):
        FluxActionPolicy.from_pretrained(tmp_path)


def test_restore_checksum(tmp_path):
    make_policy().save_pretrained(tmp_path)
    with (tmp_path / "config.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(AssertionError):
        FluxActionPolicy.from_pretrained(tmp_path)


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("modality", ["action", "action_prediction_droid"])
def test_base_remapping_and_fresh_heads(tmp_path, history, modality):
    config = (
        dict(inference_profile="history", n_obs_steps=8, history_snapshots=2, condition_on_past_actions=True)
        if history
        else {}
    )
    config["action_modality"] = modality
    params = action_dit_params(
        restrict_content_streams(JointSingleSeqParams(**TINY_DIT), CONTENT_STREAMS), "action_prediction", 16
    )
    base = JointSingleSeq(params)
    path = tmp_path / "base.safetensors"
    save_file({k: v.contiguous() for k, v in base.state_dict().items()}, str(path))
    policy = make_policy(tiny_config(trunk_weights=str(path), **config))
    torch.testing.assert_close(
        policy.dit.content_mode_blocks[modality][0].q_proj.weight,
        base.content_mode_blocks["action_prediction"][0].q_proj.weight,
    )
    assert policy.dit.emb_in[modality].weight.shape == (64, 6)
    cond_width = 12 if history else 6
    assert policy.dit.emb_in[f"{modality}_cond"].weight.shape == (64, cond_width)
    assert policy.dit.final_layer[f"{modality}_cond"].linear.weight.shape == (cond_width, 64)
    # fresh heads carry the reference initialization: xavier input projections, zero final layers
    emb = policy.dit.emb_in[modality].weight
    bound = (6 / (6 + 64)) ** 0.5
    assert 0.5 * bound < emb.abs().max() <= bound
    head = policy.dit.final_layer[modality]
    assert not head.linear.weight.any() and not head.adaLN_modulation[1].weight.any()
    assert not policy.dit.final_layer[f"{modality}_cond"].linear.weight.any()
    # the draw depends on head_init_seed only, not on the caller's RNG state (ranks seed differently)
    torch.manual_seed(7)
    again = make_policy(tiny_config(trunk_weights=str(path), **config))
    assert torch.equal(again.dit.emb_in[modality].weight, emb)
    torch.testing.assert_close(
        again.dit.emb_in[f"{modality}_cond"].weight,
        policy.dit.emb_in[f"{modality}_cond"].weight,
        rtol=0,
        atol=0,
    )
    other = make_policy(tiny_config(trunk_weights=str(path), head_init_seed=3, **config))
    assert not torch.equal(other.dit.emb_in[modality].weight, emb)


def test_droid_inference_settings_are_valid_and_explicit():
    assert PolicyConfig().sampler is None and PolicyConfig().caption_dropout == 0.1
    cfg = tiny_config(**DROID_INFERENCE_SETTINGS)
    cfg.validate_inference()
    assert (cfg.sampler, cfg.num_inference_steps, cfg.sampler_shift) == ("cosmos_unipc", 4, 5.0)
    assert (cfg.guidance_scale, cfg.guidance_scale_action, cfg.n_action_steps) == (4.0, 1.0, 32)


@pytest.mark.parametrize("dit_config", [{}, {"hidden_size": 3072}, {"theta": 10000, "axes_dim": [32] * 4}])
def test_prepared_inference_accepts_resolved_defaults(dit_config):
    config = PolicyConfig(
        **DROID_INFERENCE_SETTINGS,
        action_modality="action_prediction_droid",
        content_streams=("video", "video_cond"),
        dit_config=dit_config,
    )
    assert _released_inference_config(config)
    for overrides in (
        {"dit_config": {"theta": 20000}},
        {"dit_config": {"axes_dim": [16, 16, 48, 48]}},
        {"dit_config": {"depth": 4}},
        {"action_modality": "action"},
        {"torch_dtype": "float32"},
        {"sampler": "euler"},
        {"inference_profile": "history", "gripper_flip_dims": ()},
        {"chunk_size": 16, "n_action_steps": 16},
    ):
        assert not _released_inference_config(PolicyConfig(**{**config.to_dict(), **overrides}))


def test_joint_token_loss():
    # Unequal numbers of video/action tokens distinguish this from a modality mean.
    pred = {"x_video": torch.full((2, 5, 3), 2.0), "x_action": torch.full((2, 2, 7), 3.0)}
    target = {k: torch.zeros_like(v) for k, v in pred.items()}
    result = packing.flow_loss(pred, target, "action")
    assert result["loss"].item() == pytest.approx((10 * 4 + 50 * 4 * 9) / 14)


def test_no_framework_dependency():
    # Isolated mode drops PYTHONPATH, so point the child at this checkout explicitly; the check
    # must hold whether the package is installed or only on the path (as on the cluster).
    src = str(Path(flux_action.__file__).resolve().parents[1])
    code = (
        f"import sys; sys.path.insert(0, {src!r}); from flux_action.policy import FluxActionPolicy; "
        "assert not any(k.startswith(('lerobot', 'bfl.')) for k in sys.modules)"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_frozen_components_and_dtype(batch):
    policy = make_policy().train()
    assert not policy.video_vae.training and not policy.text_encoder.training
    assert not any("video_vae" in k or "text_encoder" in k for k in policy.state_dict())
    policy._context("test", torch.device("cpu"))
    policy.to(dtype=torch.bfloat16)
    assert policy.dtype_ == torch.bfloat16 and not policy._ctx_cache


def test_strict_component_loader_rejects_partial_heads(tmp_path):
    from flux_action.models.wiring import load_action_checkpoint

    model = make_policy().dit
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    del state["emb_in.action_cond.weight"]
    path = tmp_path / "partial.safetensors"
    save_file(state, str(path))
    with pytest.raises(ValueError, match="missing required embodiment heads"):
        load_action_checkpoint(model, str(path), "action", strict_heads=True)


def test_optimizer_groups_are_disjoint_and_cover_trainable_parameters():
    policy = make_policy()
    trunk, heads = policy.get_optim_params()
    trunk_ids, head_ids = ({id(p) for p in group["params"]} for group in (trunk, heads))
    assert not trunk_ids & head_ids
    assert trunk_ids | head_ids == {id(p) for p in policy.parameters() if p.requires_grad}
    assert trunk["lr"] == 1.92e-4 and heads["lr"] == pytest.approx(5 * 1.92e-4)


def test_frozen_streams_leave_the_optimizer_and_untouched_streams_get_no_gradient(batch):
    policy = make_policy(tiny_config(content_streams=CONTENT_STREAMS)).train()
    loss, _ = policy(batch)
    loss.backward()
    grads = {n: p.grad for n, p in policy.named_parameters()}
    dead = [n for n in grads if any(f".{m}." in n for m in ("image", "image_cond", "audio", "audio_cond"))]
    assert dead and all(grads[n] is None for n in dead)  # by design: these streams carry no loss
    assert grads["dit.emb_in.action.weight"].abs().sum() > 0
    cond_heads = [
        n for n in grads if n.startswith(("dit.final_layer.video_cond.", "dit.final_layer.action_cond."))
    ]
    assert cond_heads and all(grads[n] is None for n in cond_heads)  # conditioning predictions are discarded
    policy = make_policy(tiny_config(content_streams=CONTENT_STREAMS))
    frozen = policy.freeze_streams(("image", "image_cond", "audio", "audio_cond"))
    assert set(frozen) == set(dead)
    heads = policy.freeze_conditioning_heads()
    assert set(heads) >= set(cond_heads) and set(frozen) | set(heads) == set(dead) | set(cond_heads)
    params = dict(policy.named_parameters())
    trainable = {id(p) for group in policy.get_optim_params() for p in group["params"]}
    assert not any(id(params[n]) in trainable for n in frozen + heads)
    loss, _ = policy.train()(batch)
    loss.backward()
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)


def test_uint8_frames_train_like_float_frames(batch):
    policy = make_policy().train()
    as_bytes = {**batch, "images.top": (batch["images.top"] * 255).round().to(torch.uint8)}
    as_float = {**batch, "images.top": as_bytes["images.top"].float() / 255}
    torch.manual_seed(3)
    loss_bytes, _ = policy(as_bytes)
    torch.manual_seed(3)
    loss_float, _ = policy(as_float)
    assert torch.equal(loss_bytes, loss_float)


LEAN_STREAMS = ("video", "video_cond")
DROPPED_STREAMS = {"image", "image_cond", "audio", "audio_cond"}


def _tiny_trunks(tmp_path):
    params = action_dit_params(
        restrict_content_streams(JointSingleSeqParams(**TINY_DIT), CONTENT_STREAMS), "action_prediction", 16
    )
    torch.manual_seed(11)
    full_sd = {k: v.contiguous() for k, v in JointSingleSeq(params).state_dict().items()}
    lean_sd = {k: v for k, v in full_sd.items() if stream_of_key(k) not in DROPPED_STREAMS}
    assert len(lean_sd) < len(full_sd)
    full, lean = tmp_path / "full.safetensors", tmp_path / "lean.safetensors"
    save_file(full_sd, str(full))
    save_file(lean_sd, str(lean))
    return full, lean


def _same_predictions_and_loss(p_a, p_b, batch):
    observation = {**batch, "images.top": batch["images.top"][:, 0]}
    p_a.eval()
    p_b.eval()
    torch.testing.assert_close(
        p_a.predict_action_chunk(observation), p_b.predict_action_chunk(observation), rtol=0, atol=0
    )
    p_a.train()
    p_b.train()
    torch.manual_seed(3)
    loss_a, _ = p_a(batch)
    torch.manual_seed(3)
    loss_b, _ = p_b(batch)
    assert torch.equal(loss_a, loss_b)


def test_lean_trunk_without_image_audio_streams_predicts_identically(batch, tmp_path):
    """The policy never feeds image / audio tokens; a trunk without those streams loads and matches exactly."""
    full, lean = _tiny_trunks(tmp_path)
    p_full = make_policy(tiny_config(trunk_weights=str(full), content_streams=CONTENT_STREAMS))
    p_lean = make_policy(tiny_config(trunk_weights=str(lean)))
    assert p_full.config.content_streams == CONTENT_STREAMS
    assert p_lean.config.content_streams == LEAN_STREAMS
    assert set(p_lean.dit.in_channels) == {"video", "video_cond", "action", "action_cond"}
    assert not any(stream_of_key(k) in DROPPED_STREAMS for k in p_lean.state_dict())
    assert sum(p.numel() for p in p_lean.parameters()) < sum(p.numel() for p in p_full.parameters())
    full_sd = p_full.state_dict()
    for k, v in p_lean.state_dict().items():
        if not k.startswith(("dit.emb_in.action", "dit.final_layer.action")):
            assert torch.equal(v, full_sd[k]), k
    p_lean.load_state_dict({k: full_sd[k] for k in p_lean.state_dict()})  # identical fresh heads
    _same_predictions_and_loss(p_full, p_lean, batch)
    # the trainer's freeze list names streams a lean policy does not have: nothing to freeze, no error
    assert p_lean.freeze_streams(("image", "image_cond", "audio", "audio_cond")) == []


def test_lean_policy_exports_its_streams_and_full_trunks_load_into_lean_policies(batch, tmp_path):
    full, lean = _tiny_trunks(tmp_path)
    p_lean = make_policy(tiny_config(trunk_weights=str(lean)))
    export = tmp_path / "export"
    p_lean.save_pretrained(export)
    assert json.loads((export / "config.json").read_text())["content_streams"] == list(LEAN_STREAMS)
    restored = FluxActionPolicy.from_pretrained(
        export, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32)
    )
    assert restored.config.content_streams == LEAN_STREAMS
    _same_predictions_and_loss(p_lean, restored, batch)
    explicit = make_policy(tiny_config(trunk_weights=str(full), content_streams=LEAN_STREAMS))
    explicit.load_state_dict(p_lean.state_dict())  # fresh heads carry each build's own init
    _same_predictions_and_loss(p_lean, explicit, batch)


def test_bf16_noising_rounds_timesteps_and_noise_like_the_reference():
    x0 = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    t = torch.tensor([0.9988, 0.150390625])  # 1 - 2^-9 < t rounds to 1; (1 - t) needs 9 bits for the second
    x_t, target = packing.add_noise(x0, t, torch.Generator().manual_seed(1))
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float32).to(
        torch.bfloat16
    )
    tb = t.to(torch.bfloat16).view(-1, 1, 1)
    assert x_t.dtype == target.dtype == torch.bfloat16
    assert torch.equal(x_t, tb * eps + (1 - tb) * x0) and torch.equal(target, eps - x0)
    assert torch.equal(x_t[0], eps[0])  # t = 1 exactly: pure noise, as in the reference's bf16 packing
    ids = torch.zeros(2, 4, 4, dtype=torch.int64)
    video = {"x_video": x0, "x_video_ids": ids, "x_video_cond": x0[:, :1], "x_video_cond_ids": ids[:, :1]}
    action = {
        "x_action": x0[:, :3, :6],
        "x_action_ids": ids[:, :3],
        "x_action_cond": x0[:, :1, :6],
        "x_action_cond_ids": ids[:, :1],
    }
    text = {
        "ctx": torch.zeros(2, 3, 5),
        "ctx_ids": ids[:, :3],
        "vector": torch.zeros(2, 7),
        "timesteps_ctx": torch.zeros(2, 3),
    }
    kw, targets, _ = packing.build_forward_kwargs(
        video, action, text, t, "action", torch.Generator().manual_seed(1)
    )
    assert kw["x_video_timesteps"].dtype == torch.float32
    assert kw["x_video_timesteps"][0].unique().tolist() == [1.0]
    assert kw["x_action_timesteps"][1].unique().item() == pytest.approx(float(tb[1]))
    assert torch.equal(kw["x_video"], x_t) and torch.equal(targets["x_video"], target)
    # fp32 latents keep the exact timestep
    kw32, _, _ = packing.build_forward_kwargs(
        {k: v.float() for k, v in video.items()}, {k: v.float() for k, v in action.items()}, text, t, "action"
    )
    assert kw32["x_video_timesteps"][0].unique().item() == pytest.approx(0.9988)


def test_timestep_mlp_runs_in_its_weight_precision_outside_autocast():
    from flux_action.models.transformer import timestep_embedding

    dit = make_policy().dit  # fp32 parameters
    seen = []
    dit.time_in.register_forward_pre_hook(lambda m, args: seen.append(args[0].dtype))
    ts = torch.tensor([0.9988, 0.0])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        vec = dit.embed_timesteps(ts, torch.bfloat16)
    assert seen == [torch.float32] and vec.dtype == torch.bfloat16
    assert torch.equal(vec, dit.time_in(timestep_embedding(ts, 256)).to(torch.bfloat16))
    dit.time_in.to(torch.bfloat16)  # a bf16 inference model computes it in bf16, as before
    assert dit.embed_timesteps(ts, torch.bfloat16).dtype == torch.bfloat16 and seen[-1] == torch.bfloat16


@pytest.mark.parametrize("conditioning_channels", [6, 12])
def test_fresh_heads_preserve_global_rng(conditioning_channels):
    from flux_action.models.wiring import fresh_head_state_dict

    before = torch.random.get_rng_state().clone()
    first = fresh_head_state_dict(32, "action", 6, seed=42, conditioning_channels=conditioning_channels)
    assert torch.equal(torch.random.get_rng_state(), before)
    second = fresh_head_state_dict(32, "action", 6, seed=42, conditioning_channels=conditioning_channels)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert all(torch.count_nonzero(v) == 0 for k, v in first.items() if k.startswith("final_layer."))


def test_conditioning_only_packing_matches_previous_target_packer():
    policy = make_policy(tiny_config())
    canvas = torch.linspace(-1, 1, 3 * 64 * 96).reshape(3, 1, 64, 96)
    state = torch.arange(6).float()[None]
    cond = policy._encode_canvas(canvas, state)
    latent = packing.encode_single_frame(
        policy.video_vae, canvas[:, 0], policy.config.latent_hw, single_frame=False
    )
    video = packing.pack_video(torch.cat([latent, torch.zeros_like(latent)], 2))
    action = packing.pack_actions(state[None], torch.zeros(1, 32, 6), torch.zeros(1, 32), "action")
    for key, value in cond.items():
        torch.testing.assert_close(value, {**video, **action}[key], rtol=0, atol=0)


def test_full_trunk_defaults_to_lean_construction_without_loading_unused_tensors(tmp_path, monkeypatch):
    import safetensors

    full, lean = _tiny_trunks(tmp_path)
    original = safetensors.safe_open
    loaded = []

    class RecordingFile:
        def __init__(self, *args, **kwargs):
            self.file = original(*args, **kwargs)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def keys(self):
            return self.file.keys()

        def get_tensor(self, key):
            loaded.append(key)
            return self.file.get_tensor(key)

    monkeypatch.setattr(safetensors, "safe_open", RecordingFile)
    policy = make_policy(tiny_config(trunk_weights=str(full)))
    assert policy.config.content_streams == LEAN_STREAMS
    assert not any(stream_of_key(key) in DROPPED_STREAMS for key in loaded)
    assert not any(stream_of_key(key) in DROPPED_STREAMS for key in policy.state_dict())
