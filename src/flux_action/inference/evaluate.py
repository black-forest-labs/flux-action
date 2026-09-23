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
"""Offline evaluation of a policy export on the held-out windows of an indexed dataset.

For every episode of the chosen split (``val`` by default: the episodes ``index-lerobot --val-episodes``
held out), a fixed number of window starts spread evenly over the allowed starts is taken, the policy
predicts from the window's first frame, state and caption, and the prediction is scored against the
window's recorded commands two ways:

* ``action_mse_normalized``: mean squared error between the predicted and the recorded targets in the
  trained action space (normalized deltas for the SO-101 recipe). This is the quantity checkpoint
  selection ranks by; it does not depend on the units of the dataset.
* ``action_mse_raw``: mean squared error of the absolute commands in dataset units (degrees for
  SO-101), per channel and overall, after denormalization and integration onto the observed state.
  Under ``joint_delta`` this includes the offset between the observed state and the command before the
  window (the leader-follower gap of teleoperated data): the recorded deltas are relative to that
  command, deployment integrates onto the state, so even a perfect prediction scores that offset here.
  History policies instead receive the recorded preceding commands and integrate onto the last one,
  matching their explicit offline history inference contract.

Windows and captions are deterministic (first paraphrase, no dropout), the sampler seed is the policy's
``inference_seed``; two exports evaluated with the same arguments see identical inputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from ..checkpoints.artifacts import sha256
from ..data.droid.index import ROWS_FILENAME
from ..hub import resolve_policy_directory
from ..policy import FluxActionPolicy
from ..training.data import WindowDataset, collate_windows, load_manifest


def spread(n: int, k: int) -> list[int]:
    """``k`` indices spread evenly over ``range(n)``, first and last included, distinct and sorted."""
    if n <= 0 or k <= 0:
        return []
    if k == 1 or n == 1:
        return [0]
    return sorted({round(j * (n - 1) / (k - 1)) for j in range(k)})


def evaluate_policy(
    policy: FluxActionPolicy,
    source_root,
    index_dir,
    *,
    split: str = "val",
    windows_per_episode: int = 4,
    max_windows: int | None = None,
    decoder: str = "pyav",
    frame_hw: tuple[int, int] = (256, 256),
    seed: int = 0,
) -> dict[str, Any]:
    """Score ``policy`` on the split's windows; returns the report (per-window records included)."""
    if windows_per_episode < 1:
        raise ValueError("windows_per_episode must be positive")
    index_dir = Path(index_dir)
    manifest = load_manifest(index_dir / "manifest.json")
    dataset = WindowDataset(
        manifest,
        source_root,
        index_dir / ROWS_FILENAME,
        seed=seed,
        frame_hw=frame_hw,
        decoder=decoder,
        split=split,
        n_obs_steps=policy.config.n_obs_steps if policy.config.inference_profile == "history" else None,
    )
    device = policy.device
    records: list[dict[str, Any]] = []
    raw_sum = torch.zeros(policy.config.action_dim, dtype=torch.float64)
    started = time.perf_counter()
    for position, episode in enumerate(dataset.episodes):
        starts = dataset.valid_starts(position)
        for i in spread(len(starts), windows_per_episode):
            if max_windows is not None and len(records) >= max_windows:
                break
            batch = collate_windows([dataset.window_at(position, starts[i])])
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            observation = {
                **{key: batch[key][:, 0] for key in policy.config.camera_order},
                "state": batch["state"],
                "task": batch["task"],
            }
            anchor = batch["state"]
            if policy.config.inference_profile == "history":
                observation.update(
                    {key: batch[key][:, : policy.config.n_obs_steps] for key in policy.config.camera_order}
                )
                observation["command_history"] = batch["command_history"]
                anchor = batch["command_history"][:, -1]
            predicted = policy.predict_normalized_targets(observation)  # (1, K, D)
            recorded = policy.training_targets(batch).to(predicted)
            actions = policy.actions_from_targets(predicted, anchor)
            raw_error = ((actions - batch["action"].float()) ** 2).mean(dim=(0, 1)).double().cpu()
            raw_sum += raw_error
            records.append(
                {
                    "episode_index": episode["episode_index"],
                    "start": starts[i],
                    "task": batch["task"][0],
                    "action_mse_normalized": float(((predicted - recorded) ** 2).mean()),
                    "action_mse_raw": float(raw_error.mean()),
                }
            )
    if not records:
        raise ValueError(f"no windows in split {split!r}")
    normalized = torch.tensor([r["action_mse_normalized"] for r in records], dtype=torch.float64)
    raw = torch.tensor([r["action_mse_raw"] for r in records], dtype=torch.float64)
    return {
        "format_version": 1,
        "split": split,
        "dataset_id": manifest.get("dataset_id"),
        "episodes": len(dataset.episodes),
        "n_windows": len(records),
        "windows_per_episode": windows_per_episode,
        "action_mse_normalized": float(normalized.mean()),
        "action_mse_normalized_median": float(normalized.median()),
        "action_mse_raw": float(raw.mean()),
        "action_mse_raw_per_channel": (raw_sum / len(records)).tolist(),
        "action_names": manifest.get("action_names"),
        "inference": {
            k: getattr(policy.config, k)
            for k in (
                "sampler",
                "num_inference_steps",
                "sampler_shift",
                "guidance_scale",
                "guidance_scale_action",
                "inference_seed",
                "action_parameterization",
            )
        },
        "seconds": time.perf_counter() - started,
        "windows": records,
    }


def evaluate_export(
    checkpoint,
    source_root,
    index_dir,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    output=None,
    device: str = "cuda",
    settings: dict[str, Any] | None = None,
    video_vae=None,
    text_encoder=None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Restore a policy export (optionally overriding inference settings), evaluate it, write ``output``."""
    checkpoint = resolve_policy_directory(checkpoint, revision=revision, subfolder=subfolder)
    policy = FluxActionPolicy.from_pretrained(
        checkpoint, video_vae=video_vae, text_encoder=text_encoder, device=device
    )
    for key, value in (settings or {}).items():
        if not hasattr(policy.config, key):
            raise ValueError(f"unknown inference setting {key!r}")
        setattr(policy.config, key, value)
    policy.config.validate_inference()
    policy.prepare_inference()
    report = evaluate_policy(policy, source_root, index_dir, **kwargs)
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_manifest"] = json.loads((checkpoint / "manifest.json").read_text())
    report["index_manifest_sha256"] = sha256(Path(index_dir) / "manifest.json")
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1) + "\n")
    return report
