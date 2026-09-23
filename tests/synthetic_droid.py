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
"""Synthetic compact Cosmos3-DROID layout with real AV1 clips for the data-path tests."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from flux_action.data.droid.cosmos import CAMERA_FEATURES, filter_key

pa = pytest.importorskip("pyarrow")  # the data extra; skip these tests without it
pq = pytest.importorskip("pyarrow.parquet")

FRAME_HW = (32, 64)
FPS = 15


def frame_image(episode: int, frame: int, camera: int, frame_hw=FRAME_HW) -> np.ndarray:
    """Flat colors that survive AV1: red = episode, green = frame index, blue = camera."""
    img = np.empty((*frame_hw, 3), np.uint8)
    img[..., 0] = 30 + 50 * episode
    img[..., 1] = (6 * frame) % 250
    img[..., 2] = 60 + 60 * camera
    return img


def write_av1_clip(path, frames: np.ndarray, fps: int = FPS) -> None:
    """Encode uint8 ``(N, H, W, 3)`` frames as AV1 with a keyframe every second frame."""
    av = pytest.importorskip("av")  # the data extra

    n, h, w, _ = frames.shape
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libsvtav1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"g": "2", "preset": "12"}
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frames[i]), format="rgb24").reformat(
                format="yuv420p"
            )
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    with av.open(str(path)) as container:  # the fixture must carry exact frame timestamps
        stream = container.streams.video[0]
        indices = [round(float(f.pts * stream.time_base) * fps) for f in container.decode(stream)]
    assert indices == list(range(n)), indices[:5]


def build_compact_dataset(root, episodes, frame_hw=FRAME_HW):
    """``episodes``: dicts with ``n_frames``, ``caption`` and ``keep`` (list of [a, b) ranges or None).

    Writes info.json, tasks/episodes/data parquet files, three concatenated AV1 files and the
    keep-ranges filter. Returns the filter path.
    """
    root = Path(root)
    for sub in ("success/meta/episodes/chunk-000", "success/data/chunk-000"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    total = sum(e["n_frames"] for e in episodes)
    info = {
        "codebase_version": "v3.0",
        "fps": FPS,
        "total_episodes": len(episodes),
        "total_frames": total,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "success/meta/info.json").write_text(json.dumps(info))
    captions = sorted({e["caption"] for e in episodes})
    pq.write_table(
        pa.Table.from_pylist([{"task_index": i, "task": c} for i, c in enumerate(captions)]),
        root / "success/meta/tasks.parquet",
    )
    episode_rows, data_rows, keep, cursor = [], [], {}, 0
    for e, episode in enumerate(episodes):
        n, caption = episode["n_frames"], episode["caption"]
        row = {
            "episode_index": e,
            "episode_id": f"lab/episode-{e}",
            "tasks": [caption],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + n,
        }
        for feature in CAMERA_FEATURES.values():
            row[f"videos/{feature}/chunk_index"] = 0
            row[f"videos/{feature}/file_index"] = 0
            row[f"videos/{feature}/from_timestamp"] = cursor / FPS
            row[f"videos/{feature}/to_timestamp"] = (cursor + n) / FPS
        episode_rows.append(row)
        for i in range(n):
            data_rows.append(
                {
                    "episode_index": e,
                    "frame_index": i,
                    "index": cursor + i,
                    "timestamp": np.float32(i) / np.float32(FPS),
                    "task_index": captions.index(caption),
                    "action.joint_position": [float(e) + 100 + i / 1000] * 7,
                    "action.gripper_position": [0.75],
                    "observation.state.joint_positions": [float(e) + i / 1000] * 7,
                    "observation.state.gripper_position": [0.25],
                }
            )
        if episode.get("keep") is not None:
            keep[filter_key(row["episode_id"])] = episode["keep"]
        cursor += n
    pq.write_table(
        pa.Table.from_pylist(episode_rows), root / "success/meta/episodes/chunk-000/file-000.parquet"
    )
    table = pa.Table.from_pylist(data_rows)
    table = table.set_column(
        table.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(table["timestamp"].to_numpy(), pa.float32()),
    )
    pq.write_table(table, root / "success/data/chunk-000/file-000.parquet")
    for c, feature in enumerate(CAMERA_FEATURES.values()):
        frames = np.stack(
            [frame_image(e, i, c, frame_hw) for e, ep in enumerate(episodes) for i in range(ep["n_frames"])]
        )
        path = root / f"success/videos/{feature}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_av1_clip(path, frames)
    filter_path = root / "keep_ranges.json"
    filter_path.write_text(json.dumps(keep))
    return filter_path
