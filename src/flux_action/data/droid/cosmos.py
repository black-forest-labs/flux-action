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
"""Convert a checksum-pinned local Cosmos3-DROID episode to the decoded contract."""

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ...checkpoints.artifacts import sha256
from ..schema import EpisodeMetadata
from ..windows import valid_window_spans
from .episodes import prepare_episode

CAMERA_FEATURES = {
    "wrist": "observation.image.wrist_image_left",
    "left": "observation.image.exterior_image_1_left",
    "right": "observation.image.exterior_image_2_left",
}
FILTER_PREFIX = "gs://xembodiment_data/r2d2/r2d2-data-full/"


def filter_key(episode_id):
    base = FILTER_PREFIX + episode_id.strip("/")
    return f"{base}/recordings/MP4--{base}/trajectory.h5"


def decode_frames(path, start, length, frame_hw=(360, 640)):
    """Decode by integer frame index from the file start, without seeking or a lossy re-encode."""
    h, w = frame_hw
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(path),
            "-vf",
            f"trim=start_frame={start}:end_frame={start + length},setpts=PTS-STARTPTS",
            "-frames:v",
            str(length),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if len(result.stdout) != length * h * w * 3:
        raise ValueError("decoded video does not match the exact episode frame count and dimensions")
    return np.frombuffer(result.stdout, np.uint8).reshape(length, h, w, 3)


def convert_episode(source_root, lock_path, episode_index, output):
    """Require every consumed source file to match the supplied artifact lock."""
    import pyarrow.parquet as pq

    root, lock_path, output = map(Path, (source_root, lock_path, output))
    lock = json.loads(lock_path.read_text())
    if lock.get("format_version") != 1 or lock.get("dataset_id") != "nvidia/Cosmos3-DROID":
        raise ValueError("unsupported DROID source lock")
    revision = lock["dataset_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("dataset revision must be an immutable commit hash")
    checked = {}

    def verified(relative):
        relative = str(relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("source path escapes the dataset directory")
        expected = lock["files"].get(relative)
        if expected is None:
            raise ValueError(f"source file not covered by the lock: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"locked source file missing under {root}: {relative}")
        if relative not in checked:
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"source checksum mismatch: {relative}")
            checked[relative] = actual
        return path

    info = json.loads(verified("success/meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 15:
        raise ValueError("expected compact LeRobot v3 DROID at 15 Hz")
    columns = [
        "episode_index",
        "episode_id",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for feature in CAMERA_FEATURES.values():
        columns += [
            f"videos/{feature}/{suffix}"
            for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]
        ]
    matches = []
    for name in lock["files"]:
        if name.startswith("success/meta/episodes/") and name.endswith(".parquet"):
            matches += pq.read_table(
                verified(name), columns=columns, filters=[("episode_index", "=", episode_index)]
            ).to_pylist()
    if len(matches) != 1:
        raise ValueError("episode must appear exactly once in the locked metadata files")
    row = matches[0]
    n = int(row["length"])
    if row["dataset_to_index"] - row["dataset_from_index"] != n:
        raise ValueError("episode data boundaries disagree with its length")
    mapping = json.loads(verified(lock["filter_file"]).read_text())
    ranges = mapping.get(filter_key(row["episode_id"]), [])
    metadata = EpisodeMetadata(
        episode_id=row["episode_id"],
        source_version=f"nvidia/Cosmos3-DROID@{revision}",
        split="success/train",
        n_frames=n,
        valid_ranges=ranges,
        caption=row["tasks"][0],
    )
    if not valid_window_spans(n, ranges):
        raise ValueError("episode has no complete window allowed by the pinned filter")
    tasks = pq.read_table(verified("success/meta/tasks.parquet")).to_pylist()
    task_lookup = {r.get("task", r.get("__index_level_0__")): r["task_index"] for r in tasks}
    if metadata.caption not in task_lookup:
        raise ValueError("episode caption absent from task metadata")
    relative = "success/" + info["data_path"].format(
        chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
    )
    state_columns = [
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
        "action.joint_position",
        "action.gripper_position",
        "observation.state.joint_positions",
        "observation.state.gripper_position",
    ]
    rows = pq.read_table(
        verified(relative), columns=state_columns, filters=[("episode_index", "=", episode_index)]
    )
    if rows.num_rows != n:
        raise ValueError("data row count does not match the episode")
    for key, expected in [
        ("frame_index", np.arange(n)),
        ("index", np.arange(row["dataset_from_index"], row["dataset_to_index"])),
    ]:
        if not np.array_equal(rows[key].to_numpy(), expected):
            raise ValueError(f"non-contiguous {key}")
    if not np.all(rows["task_index"].to_numpy() == task_lookup[metadata.caption]):
        raise ValueError("data task_index disagrees with the episode caption")
    times = rows["timestamp"].to_numpy()
    expected_times = np.arange(n, dtype=np.float32) / np.float32(15)
    if times.dtype != np.float32 or not np.array_equal(times.view(np.uint32), expected_times.view(np.uint32)):
        raise ValueError("timestamps disagree with the native float32 15 Hz grid")
    arrays = {"timestamps": times.astype(np.float64)}
    for name, joints, gripper in [
        ("state", "observation.state.joint_positions", "observation.state.gripper_position"),
        ("action", "action.joint_position", "action.gripper_position"),
    ]:
        joint_array = np.asarray(rows[joints].to_pylist(), dtype=np.float32)
        grip = np.asarray(rows[gripper].to_pylist(), dtype=np.float32)
        if joint_array.shape != (n, 7) or grip.size != n:
            raise ValueError("invalid joint/gripper shape")
        arrays[name] = np.concatenate([joint_array, grip.reshape(n, 1)], axis=1)
    for camera, feature in CAMERA_FEATURES.items():
        prefix = f"videos/{feature}"
        first, last = [float(row[f"{prefix}/{suffix}"]) * 15 for suffix in ["from_timestamp", "to_timestamp"]]
        if (
            not all(math.isclose(x, round(x), rel_tol=0, abs_tol=2e-5) for x in [first, last])
            or round(last) - round(first) != n
        ):
            raise ValueError(f"{camera}: video boundaries disagree with the frame grid")
        relative = "success/" + info["video_path"].format(
            video_key=feature,
            chunk_index=row[f"{prefix}/chunk_index"],
            file_index=row[f"{prefix}/file_index"],
        )
        arrays[camera] = decode_frames(verified(relative), round(first), n)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".droid-convert-", dir=output.parent) as tmp:
        tmp = Path(tmp)
        np.savez_compressed(tmp / "arrays.npz", **arrays)
        (tmp / "metadata.json").write_text(json.dumps(metadata.to_dict()))
        result = prepare_episode(tmp / "metadata.json", tmp / "arrays.npz", output)
    provenance = {
        "format_version": 1,
        "source_lock_sha256": sha256(lock_path),
        "consumed_sha256": checked,
        "episode_index": episode_index,
        "decoded_rgb_sha256": {k: hashlib.sha256(arrays[k].tobytes()).hexdigest() for k in CAMERA_FEATURES},
    }
    (output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return result
