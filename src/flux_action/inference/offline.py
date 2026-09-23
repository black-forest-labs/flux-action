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
"""Recorded offline inference on a decoded observation, without robot execution."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ..checkpoints.artifacts import sha256
from ..config import PolicyConfig
from ..hub import resolve_policy_directory
from ..policy import FluxActionPolicy
from .precision import prepare_for_serving


def load_observation(path, config, task, device):
    """Read one synchronized observation: HWC uint8 cameras and a float state vector."""
    observation = {"task": [task]}
    with np.load(path, allow_pickle=False) as arrays:
        for key in config.camera_order:
            frame = arrays[key]
            assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] == 3, (
                "expected RGB uint8 frame"
            )
            assert config.camera_layout != "droid" or frame.shape[:2] == (360, 640), "DROID frame size"
            pixels = torch.from_numpy(frame.copy()).permute(2, 0, 1)[None]
            # Keep bytes for the SO-101 delivery profile: its policy normalizes on CPU
            # before moving float pixels to the VAE device.
            observation[key] = (
                pixels.to(device=device)
                if config.inference_profile == "history"
                else pixels.to(device=device, dtype=torch.float32) / 255
            )
        state = arrays["state"]
        assert state.shape == (config.action_dim,) and state.dtype.kind == "f" and np.isfinite(state).all(), (
            "invalid state"
        )
        observation["state"] = torch.from_numpy(state.copy())[None].to(device=device, dtype=torch.float32)
    return observation


def run_inference(
    checkpoint,
    observation,
    output,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    task,
    device="cuda",
    compile_dit: bool = False,
    offload_text_encoder: bool = False,
    settings: dict | None = None,
):
    """Restore a verified export and save actions alongside the effective run settings.

    ``settings`` are explicit ``PolicyConfig`` overrides applied before validation, for exports that carry no
    sampling settings (training exports); they are recorded in the report.

    The serve-time knobs (``inference.precision``) are applied on the CPU-restored policy before it moves
    to ``device``; with ``compile_dit`` the reported first-chunk time includes the compilation. The report's
    ``gpu_memory_bytes`` gives the peak while loading and preparing, the bytes allocated afterwards, and the
    peak during the chunk."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new output directory for each inference run")
    checkpoint = resolve_policy_directory(checkpoint, revision=revision, subfolder=subfolder)
    device = torch.device(device)
    policy = FluxActionPolicy.from_pretrained(checkpoint, device="cpu")
    if settings:
        policy.config = PolicyConfig(**{**policy.config.to_dict(), **settings})
    policy.config.validate_inference()
    memory = {"peak_during_load": None, "allocated_after_load": None, "peak_during_chunk": None}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup = prepare_for_serving(
        policy,
        device=device,
        compile_dit=compile_dit,
        offload_text_encoder=offload_text_encoder,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory["peak_during_load"] = torch.cuda.max_memory_allocated(device)
        memory["allocated_after_load"] = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    batch = load_observation(observation, policy.config, task, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory["peak_during_chunk"] = torch.cuda.max_memory_allocated(device)
    elapsed = time.perf_counter() - started
    expected = (1, policy.config.chunk_size, policy.config.action_dim)
    assert tuple(actions.shape) == expected and torch.isfinite(actions).all(), "invalid action output"
    report = {
        "format_version": 1,
        "checkpoint_manifest": json.loads((checkpoint / "manifest.json").read_text()),
        "observation_sha256": sha256(observation),
        "task": task,
        "policy_config": policy.config.to_dict(),
        "settings": dict(settings or {}),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "inference_seconds": elapsed,
        "timing_scope": "first chunk including conditioning; excludes checkpoint loading"
        + ("; includes DiT compilation" if compile_dit else ""),
        "serving_setup": setup,
        "gpu_memory_bytes": memory,
        "actions_shape": list(actions.shape),
        "encoder_integrity_verified": False,
    }
    output.mkdir(parents=True)
    np.save(output / "actions.npy", actions.float().cpu().numpy(), allow_pickle=False)
    report["actions_sha256"] = sha256(output / "actions.npy")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
