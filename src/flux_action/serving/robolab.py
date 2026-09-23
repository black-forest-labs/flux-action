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
"""RoboLab policy adapter and WebSocket server (OpenPI protocol).

Request keys (the RoboLab Cosmos client): ``observation/image`` uint8 ``(540, 640, 3)`` composite
(wrist on top, the two exteriors at half resolution below), or the three RoBoArena views
``observation/wrist_image_left``, ``observation/exterior_image_1_left``,
``observation/exterior_image_2_left`` which are composed the same way; ``observation/joint_position``
``(7,)`` (the last row of a history); ``observation/gripper_position`` scalar or ``(1,)``, in the
dataset's convention (the policy flips it internally, as in training and in the source server);
``prompt``. Response: ``{"action": float32 (32, 8)}`` plus ``server_timing``.

Protocol (OpenPI ``WebsocketPolicyServer``): on connection the server sends the packed metadata
dict; every following client frame is one packed observation and gets one packed response. On an
exception the traceback is sent as a text frame and the connection closes with code 1011.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PolicyConfig
from ..models.transformer import JointSingleSeq
from . import protocol

COMPOSITE_HW = (540, 640)
IMAGE_KEY = "observation/image"
VIEW_KEYS = (
    "observation/wrist_image_left",
    "observation/exterior_image_1_left",
    "observation/exterior_image_2_left",
)
INTERNAL_ERROR = 1011


def _rgb_uint8(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    assert image.ndim == 3 and image.shape[-1] == 3, "image shape"
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    # msgpack arrays view immutable bytes; torch.from_numpy requires writable storage.
    return np.require(image, requirements=["C", "W"])


def _resize_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Cosmos server resize: bilinear on float, back to uint8 by truncation."""
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float()
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).numpy().astype(np.uint8)


def compose_views(wrist: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """RoBoArena views -> the 540x640 composite, as the Cosmos RoboLab server composes them."""
    half = (wrist.shape[0] // 2, wrist.shape[1] // 2)
    bottom = np.concatenate([_resize_uint8(left, half), _resize_uint8(right, half)], axis=1)
    return np.concatenate([wrist, bottom], axis=0)


def observation_image(obs: dict[str, Any]) -> np.ndarray:
    if IMAGE_KEY in obs:
        image = _rgb_uint8(obs[IMAGE_KEY], IMAGE_KEY)
    elif all(key in obs for key in VIEW_KEYS):
        image = compose_views(*(_rgb_uint8(obs[key], key) for key in VIEW_KEYS))
    else:
        raise KeyError(IMAGE_KEY)
    assert image.shape[:2] == COMPOSITE_HW, "composite size"
    return image


def observation_state(obs: dict[str, Any], action_dim: int) -> np.ndarray:
    """``[joint_position (7), gripper_position (1)]`` from the last history row, in dataset units."""
    joints = np.asarray(obs["observation/joint_position"], dtype=np.float32)
    if joints.ndim == 2:
        joints = joints[-1]
    gripper = np.asarray(obs["observation/gripper_position"], dtype=np.float32).reshape(-1)
    assert gripper.size > 0, "gripper required"
    state = np.concatenate([joints.reshape(-1), gripper[-1:]])
    assert state.shape == (action_dim,) and np.isfinite(state).all(), "invalid state"
    return state


class RoboLabPolicy:
    """``infer(obs)`` for the OpenPI server; wraps a FluxActionPolicy on its device."""

    def __init__(self, policy, *, log_dir=None, seed_base: int = 0):
        self.policy = policy
        self.log_dir = Path(log_dir) if log_dir else None
        self.seed_base = seed_base
        self.queries = 0
        self.chunk_size = policy.config.chunk_size
        self.action_dim = policy.config.action_dim

    def infer(self, obs: dict[str, Any], *, seed: int | None = None) -> dict[str, Any]:
        image = observation_image(obs)
        state = observation_state(obs, self.action_dim)
        prompt = str(obs.get("prompt", "") or "")
        self.queries += 1
        query_seed = self.seed_base + self.queries if seed is None else seed
        started = time.perf_counter()
        actions = self.policy.predict_from_composite(
            torch.from_numpy(image), torch.from_numpy(state), prompt, seed=query_seed
        )
        actions = actions.detach().cpu().numpy().astype(np.float32)
        elapsed = time.perf_counter() - started
        if actions.shape != (self.chunk_size, self.action_dim):
            raise RuntimeError(f"policy returned {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("policy returned non-finite actions")
        if self.log_dir is not None:
            self._log(image, state, prompt, query_seed, actions, elapsed)
        return {"action": actions}

    def _log(self, image, state, prompt, seed, actions, elapsed) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stem = self.log_dir / f"query_{self.queries:06d}"
        np.savez(
            stem.with_suffix(".npz"),
            **{
                IMAGE_KEY: image,
                "observation/joint_position": state[:-1],
                "observation/gripper_position": state[-1:],
            },
        )
        stem.with_suffix(".json").write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "seed": seed,
                    "predicted_actions": actions.tolist(),
                    "inference_time_s": elapsed,
                    "serving_setup": self.policy.serving_setup,
                },
                indent=2,
            )
            + "\n"
        )


def prepare_serving_policy(
    policy,
    *,
    device="cuda",
    dtype: str = "keep",
    settings: dict[str, Any] | None = None,
    compile_dit: bool = True,
    offload_text_encoder: bool = False,
    warmup: int = 1,
):
    """Apply explicit sampler overrides and the serve-time knobs of ``inference.precision`` to a policy
    that still sits on the CPU, then move it to ``device``.

    Published sampler settings remain unchanged unless ``settings`` explicitly overrides them. The optimized
    inference backend is installed after placement and compiled by default; ``warmup`` requests go through
    :meth:`FluxActionPolicy.predict_from_composite` exactly as :meth:`RoboLabPolicy.infer` does, so the cached
    compiled paths are the ones serving uses. ``policy.serving_setup`` records what was applied."""
    from ..inference.precision import prepare_for_serving

    assert dtype in ("keep", "bfloat16", "float32"), "serving dtype"
    prepared = policy._inference_prepared
    requested_dtype = (
        str(next(policy.dit.parameters()).dtype).removeprefix("torch.") if dtype == "keep" else dtype
    )
    assert not prepared or requested_dtype == policy.config.torch_dtype, "prepared dtype mismatch"
    values = dict(settings or {})
    values["torch_dtype"] = requested_dtype
    policy.config = PolicyConfig(**{**policy.config.to_dict(), **values})
    policy.config.validate_inference()
    target_dtype = getattr(torch, requested_dtype)
    current_dtype = next(policy.dit.parameters()).dtype
    if dtype != "keep" and target_dtype != current_dtype:
        assert isinstance(policy.dit, JointSingleSeq), "cannot recast packed inference weights"
        # Frozen encoders manage their own precision (including fp32 buffers).
        policy.dit.to(dtype=target_dtype)  # CPU cast before GPU placement and inference preparation
        policy.set_compute_dtype(None)
    policy.reset()

    def warm_request():
        image = torch.zeros((*COMPOSITE_HW, 3), dtype=torch.uint8)
        state = torch.zeros(policy.config.action_dim, dtype=torch.float32)
        policy.predict_from_composite(image, state, "warm-up request", seed=0)

    policy.serving_setup = prepare_for_serving(
        policy,
        device=device,
        compile_dit=compile_dit,
        offload_text_encoder=offload_text_encoder,
        warmup=warmup,
        warmup_fn=warm_request,
    )
    policy.reset()  # the warm-up caption and queue state must not leak into the first request
    return policy


def load_serving_policy(
    checkpoint, *, revision: str | None = None, subfolder: str | None = None, device="cuda", **kwargs
):
    """Restore an export on the CPU and :func:`prepare_serving_policy` it for ``device`` (same keywords)."""
    from ..policy import FluxActionPolicy

    return prepare_serving_policy(
        FluxActionPolicy.from_pretrained(checkpoint, revision=revision, subfolder=subfolder, device="cpu"),
        device=device,
        **kwargs,
    )


async def _handle(websocket, adapter: RoboLabPolicy, metadata: dict[str, Any]) -> None:
    import websockets

    packer = protocol.Packer()
    await websocket.send(packer.pack(metadata))
    prev_total = None
    while True:
        try:
            started = time.monotonic()
            obs = protocol.unpackb(await websocket.recv())
            infer_started = time.monotonic()
            response = adapter.infer(obs)
            response["server_timing"] = {"infer_ms": (time.monotonic() - infer_started) * 1000}
            if prev_total is not None:
                response["server_timing"]["prev_total_ms"] = prev_total * 1000
            await websocket.send(packer.pack(response))
            prev_total = time.monotonic() - started
        except websockets.ConnectionClosed:
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=INTERNAL_ERROR, reason="Internal server error. Traceback included in previous frame."
            )
            raise


async def serve_async(
    adapter: RoboLabPolicy, *, host: str = "0.0.0.0", port: int = 8000, metadata=None, ready=None
):
    """Serve until cancelled; ``ready`` (an ``asyncio.Event``) is set once the port is bound."""
    import websockets.asyncio.server

    async def handler(websocket):
        await _handle(websocket, adapter, metadata or {})

    async with websockets.asyncio.server.serve(
        handler, host, port, compression=None, max_size=None
    ) as server:
        if ready is not None:
            ready.set()
        await server.serve_forever()


def serve(adapter: RoboLabPolicy, *, host: str = "0.0.0.0", port: int = 8000, metadata=None) -> None:
    asyncio.run(serve_async(adapter, host=host, port=port, metadata=metadata))


def collection_entries(path) -> list[dict[str, Any]]:
    """Fixture entries from a ``collection.json`` or from a directory of server query logs.

    Query logs (``query_NNNNNN.npz`` + ``.json`` + ``_actions.json``, the format the source server
    and this adapter write) get the query index as seed, which is what the source server used.
    """
    path = Path(path)
    if path.is_dir():
        entries = []
        for npz in sorted(path.glob("query_*.npz")):
            meta = json.loads(npz.with_suffix(".json").read_text())
            actions = npz.with_name(npz.stem + "_actions.json")
            if not actions.is_file():
                actions = npz.with_suffix(".json")  # this adapter stores the actions in the same json
            entries.append(
                {
                    "id": npz.stem,
                    "fixture": npz.name,
                    "expected_actions": actions.name,
                    "prompt": meta.get("prompt", ""),
                    "seed": int(meta.get("seed", int(npz.stem.split("_")[-1]))),
                }
            )
        assert entries, "empty fixture collection"
        return entries
    collection = json.loads(path.read_text())
    assert collection.get("kind") == "policy-fixture-collection", "fixture collection kind"
    return list(collection["fixtures"])


def replay_collection(
    collection_path, adapter: RoboLabPolicy, output_dir, *, atol: float = 1e-3
) -> dict[str, Any]:
    """Run recorded fixtures and compare the actions with the recorded ones.

    Each entry names ``fixture`` (npz with the observation keys), ``expected_actions`` (json with
    ``predicted_actions``), ``prompt`` and ``seed``. Reports per-fixture max absolute and relative
    L2 differences; ``passed`` means every fixture stays within ``atol``.
    """
    collection_path, output_dir = Path(collection_path), Path(output_dir)
    base = collection_path if collection_path.is_dir() else collection_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for entry in collection_entries(collection_path):
        fixture = base / entry["fixture"]
        expected_path = base / entry["expected_actions"]
        expected = np.asarray(json.loads(expected_path.read_text())["predicted_actions"], dtype=np.float32)
        with np.load(fixture, allow_pickle=False) as arrays:
            obs = {key: arrays[key] for key in arrays.files}
        obs["prompt"] = entry.get("prompt", "")
        actions = adapter.infer(obs, seed=int(entry["seed"]))["action"]
        diff = np.abs(actions - expected)
        rel = float(np.linalg.norm(actions - expected) / max(np.linalg.norm(expected), 1e-12))
        case = {
            "id": entry["id"],
            "max_abs": float(diff.max()),
            "rel_l2": rel,
            "passed": bool(diff.max() <= atol),
        }
        (output_dir / f"{entry['id']}.json").write_text(
            json.dumps({**case, "predicted_actions": actions.tolist()}, indent=2) + "\n"
        )
        cases.append(case)
    report = {
        "collection": str(collection_path),
        "atol": atol,
        "passed": all(c["passed"] for c in cases),
        "cases": cases,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
