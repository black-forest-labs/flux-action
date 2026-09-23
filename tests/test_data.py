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
import json
from random import Random

import numpy as np
import pytest
import torch

from flux_action.data.droid.episodes import (
    load_window,
    prepare_episode,
    select_caption,
    validate_arrays,
    validate_episode,
)
from flux_action.data.schema import EpisodeMetadata
from flux_action.data.windows import sample_episode_visits, select_window_start, valid_window_spans


@pytest.mark.parametrize(
    "n,ranges,expected",
    [
        (32, [(0, 32)], []),
        (33, [(0, 33)], [(0, 1)]),
        (34, [(0, 34)], [(0, 2)]),
        (100, [(0, 34), (60, 100)], [(0, 2), (60, 8)]),
        (33, [(-5, 50)], [(0, 1)]),
        (100, [], []),
    ],
)
def test_valid_boundaries(n, ranges, expected):
    assert valid_window_spans(n, ranges) == expected


def test_selection_weights_valid_starts_not_ranges():
    # Probe every possible RNG offset: two starts in the first range, eight in the second.
    class Offset:
        def __init__(self, value):
            self.value = value

        def randrange(self, total):
            assert total == 10
            return self.value

    assert [select_window_start(100, [(0, 34), (60, 100)], Offset(i)) for i in range(10)] == [
        0,
        1,
        *range(60, 68),
    ]


def test_one_window_per_episode_visit_and_rng_resume():
    short = EpisodeMetadata("short", "test", "train", 33, ((0, 33),), "a")
    long = EpisodeMetadata("long", "test", "train", 200, ((0, 200),), "b")
    visits = [short, long] * 4
    rng = Random(42)
    saved = rng.getstate()
    chosen = list(sample_episode_visits(visits, rng))
    assert [episode.episode_id for episode, _ in chosen] == ["short", "long"] * 4
    rng.setstate(saved)
    assert chosen == list(sample_episode_visits(visits, rng))


@pytest.mark.parametrize("ranges", [None, [(0.5, 33)], [(False, 33)], [(10, 2)], [(0, 1, 2)]])
def test_invalid_range_metadata(ranges):
    with pytest.raises(ValueError):
        valid_window_spans(33, ranges)


def test_schema_rejects_overlap_and_convention():
    with pytest.raises(ValueError, match="disjoint"):
        EpisodeMetadata("e", "test", "train", 100, ((0, 50), (40, 90)), "a")
    with pytest.raises(ValueError, match="closed-fraction"):
        EpisodeMetadata("e", "test", "train", 33, ((0, 33),), "a", gripper_convention="open_fraction")


@pytest.fixture
def episode_input(tmp_path):
    n = 34
    metadata = EpisodeMetadata("fixture", "synthetic-v1", "test", n, ((0, n),), "move")
    arrays = {
        name: np.full((n, 360, 640, 3), i * 60, dtype=np.uint8)
        for i, name in enumerate(("wrist", "left", "right"), 1)
    }
    arrays["timestamps"] = np.arange(n, dtype=np.float64) / 15
    arrays["state"] = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 8))
    arrays["action"] = arrays["state"] + 100
    arrays["state"][:, -1] = 0.2
    arrays["action"][:, -1] = 0.7
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata.to_dict()))
    arrays_path = tmp_path / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    return metadata, arrays, metadata_path, arrays_path


def test_prepare_resume_and_alignment(episode_input, tmp_path):
    metadata, arrays, metadata_path, arrays_path = episode_input
    out = tmp_path / "prepared"
    manifest = prepare_episode(metadata_path, arrays_path, out)
    assert manifest["n_valid_starts"] == 2
    assert prepare_episode(metadata_path, arrays_path, out) == manifest
    assert validate_episode(out) == metadata
    batch = load_window(out, 1)
    assert batch["images.wrist"].shape == (1, 33, 3, 360, 640)
    assert batch["images.wrist"][0, 0, 0, 0, 0] == pytest.approx(60 / 255)
    assert batch["images.left"][0, 0, 0, 0, 0] == pytest.approx(120 / 255)
    assert batch["images.right"][0, 0, 0, 0, 0] == pytest.approx(180 / 255)
    assert batch["state"][0, 0] == 1
    torch.testing.assert_close(batch["action"][0, :, 0], torch.arange(101, 133, dtype=torch.float32))
    assert batch["state"][0, -1] == pytest.approx(0.2)
    assert batch["action"][0, 0, -1] == pytest.approx(0.7)  # no premature gripper flip
    with pytest.raises(ValueError, match="valid window"):
        load_window(out, 2)
    with (out / "episode.npz").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_episode(out)


def test_caption_paraphrase_selection(episode_input, tmp_path):
    metadata, _, metadata_path, arrays_path = episode_input
    metadata_path.write_text(json.dumps({**metadata.to_dict(), "caption": "pick it up | grab it | take it"}))
    out = tmp_path / "prepared"
    prepare_episode(metadata_path, arrays_path, out)
    assert load_window(out, 0)["task"] == ["pick it up"]
    drawn = {load_window(out, 0, rng=Random(seed))["task"][0] for seed in range(20)}
    assert drawn == {"pick it up", "grab it", "take it"}
    assert select_caption("single") == "single"


def test_bad_timestamps_and_gripper(episode_input):
    metadata, arrays, _, _ = episode_input
    arrays["timestamps"][5] += 0.01
    with pytest.raises(ValueError, match="timestamps"):
        validate_arrays(metadata, arrays)
    arrays["timestamps"] = np.arange(metadata.n_frames) / 15
    arrays["action"][0, -1] = 1.2
    with pytest.raises(ValueError, match="gripper"):
        validate_arrays(metadata, arrays)


def test_prepared_droid_window_train_and_infer(episode_input, tmp_path):
    from conftest import make_policy, tiny_config

    _, _, metadata_path, arrays_path = episode_input
    out = tmp_path / "prepared"
    prepare_episode(metadata_path, arrays_path, out)
    batch = load_window(out, 0)
    config = tiny_config(
        camera_layout="droid",
        camera_keys=("images.wrist", "images.left", "images.right"),
        canvas_hw=(544, 736),
        action_dim=8,
        gripper_flip_dims=(-1,),
    )
    policy = make_policy(config).train()
    loss, info = policy(batch)
    loss.backward()
    assert torch.isfinite(loss) and info["n_valid_windows"] == 1
    observation = {**batch, **{key: batch[key][:, 0] for key in config.camera_keys}}
    chunk = policy.predict_action_chunk(observation)
    assert chunk.shape == (1, 32, 8) and torch.isfinite(chunk).all()
    flipped = policy._flip(batch["action"])
    torch.testing.assert_close(flipped[..., -1], torch.full((1, 32), 0.3))
    torch.testing.assert_close(policy._flip(flipped), batch["action"])
