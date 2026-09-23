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
"""Load SO-101 policy packages without importing LeRobot."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..config import PolicyConfig
from ..hub import resolve_policy_directory
from ..policy import FluxActionPolicy


def _processor_statistics(bundle: Path, raw: dict) -> dict:
    """Validate saved processors and read their checkpoint-owned quantiles."""
    pre = json.loads((bundle / "policy_preprocessor.json").read_text())["steps"]
    post = json.loads((bundle / "policy_postprocessor.json").read_text())["steps"]
    expected_pre = [
        "rename_observations_processor",
        "to_batch_processor",
        "device_processor",
        "flux3_observation_history_normalizer",
        "flux3_action_target_normalizer",
        "flux3_camera_resize",
    ]
    expected_post = ["flux3_action_history_unnormalizer", "device_processor"]
    names = [step["registry_name"] for step in pre]
    if names != expected_pre or [step["registry_name"] for step in post] != expected_post:
        raise ValueError("unsupported SO-101 processor pipeline; use the documented package revision")
    if pre[0]["config"].get("rename_map"):
        raise ValueError("delivery loader requires checkpoint camera keys without a saved rename_map")
    if any(step["config"].get("float_dtype") is not None for step in (pre[2], post[1])):
        raise ValueError("delivery loader requires processors that preserve observation dtype")
    shared = {
        "action_dim": 6,
        "action_representation": raw["action_representation"],
        "absolute_dims": raw["delta_absolute_dims"],
        "normalization_clip": raw["normalization_clip"],
    }
    temporal = {name: raw[name] for name in ("n_obs_steps", "chunk_size")}
    checks = [
        (
            pre[3],
            {
                **shared,
                **temporal,
                "camera_keys": raw["camera_keys"],
                "condition_on_past_actions": raw["condition_on_past_actions"],
            },
        ),
        (post[0], shared),
    ]
    checks.extend(
        [
            (pre[4], {**shared, **temporal}),
            (pre[5], {"camera_keys": raw["camera_keys"], "camera_layout": raw["camera_layout"]}),
        ]
    )
    for step, expected in checks:
        if step["config"] != expected:
            raise ValueError(f"processor {step['registry_name']} disagrees with the policy config")
    states = []
    for step in (pre[3], pre[4], post[0]):
        filename = step["state_file"]
        if Path(filename).name != filename:
            raise ValueError("processor state_file must name a file inside the package")
        states.append(load_file(str(bundle / filename)))
    observation, output = states[0], states[-1]
    action_keys = {"action.q01", "action.q99"}
    target = states[1]
    if (
        set(observation) != action_keys | {"state.q01", "state.q99"}
        or set(target) != action_keys
        or set(output) != action_keys
    ):
        raise ValueError("processor states must contain exactly the action/state quantiles")
    for key in action_keys:
        if not torch.equal(observation[key], target[key]) or not torch.equal(target[key], output[key]):
            raise ValueError("observation/action/output processors disagree on normalization")
    for key, value in observation.items():
        if value.shape != (6,) or not torch.isfinite(value).all():
            raise ValueError(f"{key} must contain six finite quantiles")
    return {
        stream: {q: observation[f"{stream}.{q}"].float().tolist() for q in ("q01", "q99")}
        for stream in ("state", "action")
    }


def load_policy(weights: str | Path, *, revision: str | None = None) -> FluxActionPolicy:
    """Load model, robot contract and normalization from a local package or Hub repository.

    The package must supply the current split processors and their saved quantiles.
    """
    bundle = resolve_policy_directory(weights, revision=revision)
    for path in (
        bundle / "model.safetensors",
        bundle / "config.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    raw = json.loads((bundle / "config.json").read_text())
    expected = {
        "conditioning": "history",
        "action_representation": "delta",
        "delta_absolute_dims": [-1],
        "camera_layout": "side_by_side",
        "action_modality": "action",
        "gripper_flip_dims": [],
        "use_relative_actions": False,
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"},
    }
    mismatches = [name for name, value in expected.items() if raw.get(name) != value]
    if raw.get("packer") is not None:
        mismatches.append("packer")
    if raw.get("output_features", {}).get("action", {}).get("shape") != [6]:
        mismatches.append("output_features.action")
    if raw.get("input_features", {}).get("observation.state", {}).get("shape") != [6]:
        mismatches.append("input_features.observation.state")
    if mismatches:
        raise ValueError(f"unsupported SO-101 package: {', '.join(mismatches)}")
    cameras = raw["camera_keys"]
    if any(not key.startswith("observation.images.") for key in cameras):
        raise ValueError("SO-101 camera keys must start with observation.images.")
    stats = _processor_statistics(bundle, raw)
    fields = {
        name: raw[name]
        for name in (
            "n_obs_steps",
            "history_snapshots",
            "condition_on_past_actions",
            "text_fixed_length",
            "video_position_fps",
            "chunk_size",
            "n_action_steps",
            "fps",
            "action_scale",
            "canvas_hw",
            "normalization_clip",
            "sampler",
            "num_inference_steps",
            "guidance_scale",
            "guidance_scale_action",
            "sampler_shift",
            "inference_seed",
            "separate_timesteps",
            "video_logit_mean",
            "video_logit_std",
            "conditioning_noise_max",
            "loss_reduction",
            "action_channel_weights",
            "action_loss_weight",
            "video_loss_weight",
            "train_timestep_width",
            "train_timestep_shift",
        )
    }
    config = PolicyConfig(
        **fields,
        inference_profile="history",
        action_dim=6,
        action_modality="action",
        camera_layout="side_by_side",
        camera_keys=tuple(key.removeprefix("observation.") for key in cameras),
        gripper_flip_dims=(),
        action_parameterization="joint_delta",
        absolute_action_dims=(5,),
        action_normalization={q: stats["action"][q] for q in ("q01", "q99")},
        state_normalization={q: stats["state"][q] for q in ("q01", "q99")},
        single_frame_encode=True,
        torch_dtype=raw["dtype"],
        attn_mode=raw["attn_mode"],
        dit_config=raw.get("dit_config") or {},
        video_vae_id=raw["video_vae_id"],
        text_encoder_id=raw["text_encoder_id"],
    )
    config.validate_inference()
    state = load_file(str(bundle / "model.safetensors"))
    assert all(
        value.dtype == getattr(torch, config.torch_dtype)
        for value in state.values()
        if value.is_floating_point()
    ), "SO-101 checkpoint dtype disagrees with the policy config"
    policy = FluxActionPolicy(config, _restore=True)
    policy.load_state_dict(state, strict=True, assign=True)
    return policy.eval()
