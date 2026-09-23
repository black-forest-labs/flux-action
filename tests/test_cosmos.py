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

import numpy as np
import pytest

from flux_action.checkpoints.artifacts import sha256
from flux_action.data.droid.cosmos import CAMERA_FEATURES, convert_episode, filter_key

pa = pytest.importorskip("pyarrow")  # the data extra; skip these tests without it
pq = pytest.importorskip("pyarrow.parquet")


@pytest.fixture
def compact_source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "success/meta/episodes/chunk-000").mkdir(parents=True)
    (root / "success/data").mkdir()
    info = {
        "codebase_version": "v3.0",
        "fps": 15,
        "data_path": "data/rows.parquet",
        "video_path": "videos/{video_key}.mp4",
    }
    (root / "success/meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pylist([{"task": "task", "task_index": 0}]), root / "success/meta/tasks.parquet"
    )
    row = {
        "episode_index": 2,
        "episode_id": "lab/episode",
        "tasks": ["task"],
        "length": 34,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 100,
        "dataset_to_index": 134,
    }
    for feature in CAMERA_FEATURES.values():
        for key, value in [
            ("chunk_index", 0),
            ("file_index", 0),
            ("from_timestamp", 10.0),
            ("to_timestamp", 10.0 + 34 / 15),
        ]:
            row[f"videos/{feature}/{key}"] = value
        p = root / f"success/videos/{feature}.mp4"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"fixture")
    pq.write_table(pa.Table.from_pylist([row]), root / "success/meta/episodes/chunk-000/file-000.parquet")
    rows = {
        "episode_index": [2] * 34,
        "frame_index": list(range(34)),
        "index": list(range(100, 134)),
        "timestamp": pa.array(np.arange(34, dtype=np.float32) / np.float32(15)),
        "task_index": [0] * 34,
        "action.joint_position": [[0.2] * 7] * 34,
        "action.gripper_position": [[0.7]] * 34,
        "observation.state.joint_positions": [[0.1] * 7] * 34,
        "observation.state.gripper_position": [[0.3]] * 34,
    }
    pq.write_table(pa.table(rows), root / "success/data/rows.parquet")
    (root / "filter.json").write_text(json.dumps({filter_key("lab/episode"): [[0, 34]]}))
    lock = {
        "format_version": 1,
        "dataset_id": "nvidia/Cosmos3-DROID",
        "dataset_revision": "a" * 40,
        "filter_file": "filter.json",
        "files": {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()},
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))

    def decode(path, start, length):
        assert start == 150 and length == 34
        return np.zeros((length, 360, 640, 3), np.uint8)

    monkeypatch.setattr("flux_action.data.droid.cosmos.decode_frames", decode)
    return root, path, tmp_path / "output"


def test_compact_episode_alignment(compact_source):
    root, lock, output = compact_source
    result = convert_episode(root, lock, 2, output)
    assert result["n_valid_starts"] == 2
    with np.load(output / "episode.npz") as arrays:
        np.testing.assert_array_equal(arrays["state"][:, -1], np.full(34, 0.3, np.float32))
        np.testing.assert_array_equal(arrays["action"][:, -1], np.full(34, 0.7, np.float32))
    assert json.loads((output / "source.json").read_text())["source_lock_sha256"] == sha256(lock)


def test_source_checksum_rejected(compact_source):
    root, lock, output = compact_source
    (root / "success/meta/info.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        convert_episode(root, lock, 2, output)


def test_missing_source_file_is_named(compact_source):
    root, lock, output = compact_source
    (root / "success/meta/info.json").unlink()
    with pytest.raises(FileNotFoundError, match="success/meta/info.json"):
        convert_episode(root, lock, 2, output)


def test_missing_filter_episode_not_treated_as_valid(compact_source):
    root, path, output = compact_source
    (root / "filter.json").write_text("{}")
    lock = json.loads(path.read_text())
    lock["files"]["filter.json"] = sha256(root / "filter.json")
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="no complete window"):
        convert_episode(root, path, 2, output)
