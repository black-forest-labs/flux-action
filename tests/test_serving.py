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
"""OpenPI wire format, RoboLab adapter, WebSocket server and fixture replay."""

import asyncio
import contextlib
import functools
import json
import socket
import threading

import numpy as np
import pytest
import torch
from conftest import FakeVideoVAE, tiny_config

msgpack = pytest.importorskip("msgpack")  # the OpenPI wire format; flux_action.serving needs the serve extra

from flux_action.models.text_encoder import MockTextEncoder  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402
from flux_action.processing import packing  # noqa: E402
from flux_action.serving import protocol  # noqa: E402
from flux_action.serving.robolab import (  # noqa: E402
    RoboLabPolicy,
    compose_views,
    observation_state,
    replay_collection,
    serve_async,
)


# --- reference codec: openpi_client.msgpack_numpy 0.1.0 (Apache-2.0), copied for a byte-level check ---
def _ref_pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _ref_unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


ref_packb = functools.partial(msgpack.packb, default=_ref_pack_array)
ref_unpackb = functools.partial(msgpack.unpackb, object_hook=_ref_unpack_array)


def sample_obs():
    rng = np.random.default_rng(0)
    return {
        "observation/image": rng.integers(0, 256, (540, 640, 3), dtype=np.uint8),
        "observation/joint_position": rng.standard_normal(7).astype(np.float32),
        "observation/gripper_position": np.array([0.25], dtype=np.float32),
        "prompt": "put the cup in the bowl",
    }


def test_codec_is_byte_identical_to_openpi_and_round_trips():
    obs = sample_obs()
    obs["nested"] = {"count": np.int64(3), "flag": True, "ratio": np.float32(0.5)}
    ours, theirs = protocol.packb(obs), ref_packb(obs)
    assert ours == theirs
    back = protocol.unpackb(theirs)
    assert back["prompt"] == obs["prompt"] and np.array_equal(
        back["observation/image"], obs["observation/image"]
    )
    assert back["observation/joint_position"].dtype == np.float32 and back["nested"]["count"] == 3
    assert ref_unpackb(ours)["observation/gripper_position"].tolist() == [0.25]
    with pytest.raises(ValueError, match="Unsupported dtype"):
        protocol.packb({"x": np.array([object()])})


def droid_policy():
    config = tiny_config(
        camera_layout="droid",
        camera_keys=("images.wrist", "images.left", "images.right"),
        canvas_hw=(544, 736),
        action_dim=8,
        gripper_flip_dims=(-1,),
    )
    return FluxActionPolicy(config, video_vae=FakeVideoVAE(), text_encoder=MockTextEncoder(32))


@pytest.fixture(scope="module")
def adapter():
    return RoboLabPolicy(droid_policy(), seed_base=100)


def test_adapter_returns_chunk_and_matches_the_three_camera_path(adapter):
    obs = sample_obs()
    out = adapter.infer(obs)
    assert set(out) == {"action"} and out["action"].shape == (32, 8) and out["action"].dtype == np.float32
    assert np.isfinite(out["action"]).all() and adapter.queries == 1
    # A composite assembled from three cameras in float (the training composition) must give the
    # actions the three-camera path gives for those cameras.
    rng = torch.Generator().manual_seed(1)
    cams = {
        k: torch.rand(1, 3, 360, 640, generator=rng) for k in ("images.wrist", "images.left", "images.right")
    }
    stacked = torch.stack([cams[k][0] for k in ("images.wrist", "images.left", "images.right")])[:, None]
    canvas = packing.compose_canvas(stacked, "droid", (544, 736))  # (3, 1, 544, 736) in [-1, 1]
    composite = canvas[:, 0, :540, :640].add(1).div(2)  # the un-padded content, back to [0, 1]
    state = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.25])
    policy = adapter.policy
    direct = policy.predict_action_chunk({**cams, "state": state[None], "task": ["move"]})[0]
    via_composite = policy.predict_from_composite(composite, state, "move", seed=policy.config.inference_seed)
    torch.testing.assert_close(via_composite, direct, rtol=0, atol=0)
    # uint8 composites take the same path after scaling
    as_uint8 = (composite * 255).round().to(torch.uint8).permute(1, 2, 0)
    assert policy.predict_from_composite(as_uint8, state, "move").shape == (32, 8)


def test_observation_parsing_and_views(adapter):
    obs = sample_obs()
    obs["observation/joint_position"] = np.stack([obs["observation/joint_position"]] * 3)  # history
    obs["observation/gripper_position"] = np.array([[0.1], [0.7]], dtype=np.float32)
    state = observation_state(obs, 8)
    assert state.shape == (8,) and state[-1] == np.float32(0.7)  # last row, gripper passed as given
    views = {
        "observation/wrist_image_left": obs["observation/image"][:360],
        "observation/exterior_image_1_left": obs["observation/image"][:360],
        "observation/exterior_image_2_left": obs["observation/image"][:360],
    }
    composite = compose_views(*(views[k] for k in views))
    assert composite.shape == (540, 640, 3) and composite.dtype == np.uint8
    out = adapter.infer(
        {**views, "observation/joint_position": state[:7], "observation/gripper_position": 0.3}
    )
    assert out["action"].shape == (32, 8)


def test_replay_collection_reports_matches_and_mismatches(adapter, tmp_path):
    obs = sample_obs()
    recorded = adapter.infer(obs, seed=7)["action"]
    np.savez(tmp_path / "fixture.npz", **{k: v for k, v in obs.items() if k != "prompt"})
    (tmp_path / "predicted_actions.json").write_text(json.dumps({"predicted_actions": recorded.tolist()}))
    collection = {
        "kind": "policy-fixture-collection",
        "fixtures": [
            {
                "id": "a",
                "fixture": "fixture.npz",
                "expected_actions": "predicted_actions.json",
                "prompt": obs["prompt"],
                "seed": 7,
            }
        ],
    }
    (tmp_path / "collection.json").write_text(json.dumps(collection))
    report = replay_collection(tmp_path / "collection.json", adapter, tmp_path / "out")
    assert report["passed"] and report["cases"][0]["max_abs"] == 0.0
    collection["fixtures"][0]["seed"] = 8  # a different seed draws different noise
    (tmp_path / "collection.json").write_text(json.dumps(collection))
    assert not replay_collection(tmp_path / "collection.json", adapter, tmp_path / "out2")["passed"]
    # server query logs replay as well
    logged = RoboLabPolicy(adapter.policy, log_dir=tmp_path / "logs")
    logged.infer(obs)
    metadata = json.loads((tmp_path / "logs/query_000001.json").read_text())
    assert metadata["serving_setup"] == adapter.policy.serving_setup
    report = replay_collection(tmp_path / "logs", logged, tmp_path / "out3")
    assert report["passed"] and report["cases"][0]["id"] == "query_000001"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Stub:
    chunk_size, action_dim = 32, 8

    def infer(self, obs, *, seed=None):
        if "explode" in obs:
            raise ValueError("boom")
        return {"action": np.full((32, 8), float(obs["observation/gripper_position"][0]), np.float32)}


def test_server_speaks_the_openpi_protocol():
    import websockets
    import websockets.sync.client

    port = _free_port()
    loop = asyncio.new_event_loop()
    ready = asyncio.Event()
    stop = loop.create_future()

    async def main():
        task = asyncio.ensure_future(
            serve_async(_Stub(), host="127.0.0.1", port=port, metadata={"name": "stub"}, ready=ready)
        )
        await stop
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
        loop.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for _ in range(200):
        if ready.is_set():
            break
        threading.Event().wait(0.02)
    packer = protocol.Packer()
    with websockets.sync.client.connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as conn:
        assert protocol.unpackb(conn.recv()) == {"name": "stub"}
        conn.send(packer.pack({"observation/gripper_position": np.array([0.5], np.float32)}))
        response = protocol.unpackb(conn.recv())
        assert response["action"].shape == (32, 8) and float(response["action"][0, 0]) == 0.5
        assert "infer_ms" in response["server_timing"]
        conn.send(packer.pack({"observation/gripper_position": np.array([1.0], np.float32)}))
        assert "prev_total_ms" in protocol.unpackb(conn.recv())["server_timing"]
        conn.send(packer.pack({"explode": True}))
        error = conn.recv()
        assert isinstance(error, str) and "boom" in error
        with pytest.raises(websockets.ConnectionClosed) as info:
            conn.recv()
        assert info.value.rcvd.code == 1011
    loop.call_soon_threadsafe(stop.set_result, None)
    thread.join(timeout=10)
    assert not thread.is_alive()


@pytest.mark.parametrize("views", [False, True])
def test_msgpack_observations_are_writable_before_torch_conversion(views):
    import warnings

    adapter = RoboLabPolicy(droid_policy())
    obs = sample_obs()
    if views:
        frame = obs.pop("observation/image")[:360]
        for key in (
            "observation/wrist_image_left",
            "observation/exterior_image_1_left",
            "observation/exterior_image_2_left",
        ):
            obs[key] = frame
    decoded = protocol.unpackb(protocol.packb(obs))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        actual = adapter.infer(decoded, seed=17)["action"]
    expected = adapter.infer(obs, seed=17)["action"]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_actions_are_rejected_before_logging(tmp_path, monkeypatch, value):
    policy = droid_policy()
    monkeypatch.setattr(policy, "predict_from_composite", lambda *a, **k: torch.full((32, 8), value))
    adapter = RoboLabPolicy(policy, log_dir=tmp_path / "logs")
    with pytest.raises(RuntimeError, match="non-finite actions"):
        adapter.infer(sample_obs())
    assert not (tmp_path / "logs").exists()
