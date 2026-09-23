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
"""Synthetic LeRobot datasets (v2.1 and v3.0 layouts) with real AV1 clips for the data-path tests.

Two cameras of different native sizes (``top`` 48x64, ``wrist`` 32x48), 6-dimensional state and action at
30 Hz. Frame ``i`` of episode ``e`` holds ``state = e + i / 1000`` on the joints, ``action = state + 100 +
i / 100`` (so consecutive deltas are ``0.011``), a constant gripper state 20 and a gripper command ``30 + i``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from synthetic_droid import frame_image, write_av1_clip

pa = pytest.importorskip("pyarrow")  # the data extra; skip these tests without it
pq = pytest.importorskip("pyarrow.parquet")

FPS = 30
DIM = 6
CAMERAS = {"top": "observation.images.top", "wrist": "observation.images.front"}
CAMERA_HW = {"top": (48, 64), "wrist": (32, 48)}
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _motion_frame(frame: int, spec: dict | None) -> tuple[float, float]:
    """``(joint offset, gripper offset)`` of a frame: the default slow drift (1/1000 per frame), or, with a
    spec, ``stride`` per frame up to ``freeze_from`` and constant afterwards (a dead stretch)."""
    spec = spec or {}
    effective = min(frame, spec["freeze_from"]) if "freeze_from" in spec else frame
    return effective * spec.get("stride", 1 / 1000), float(effective)


def state_row(episode: int, frame: int, spec: dict | None = None) -> list[float]:
    joints, _ = _motion_frame(frame, spec)
    return [float(episode) + joints] * (DIM - 1) + [20.0]


def action_row(episode: int, frame: int, spec: dict | None = None) -> list[float]:
    joints, gripper = _motion_frame(frame, spec)
    return [float(episode) + 100 + joints + gripper / 100] * (DIM - 1) + [30.0 + gripper]


def _features() -> dict:
    features = {
        "action": {"dtype": "float32", "shape": [DIM], "names": JOINT_NAMES},
        "observation.state": {"dtype": "float32", "shape": [DIM], "names": JOINT_NAMES},
    }
    for stream, feature in CAMERAS.items():
        h, w = CAMERA_HW[stream]
        features[feature] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {"video.height": h, "video.width": w, "video.codec": "av1", "video.fps": FPS},
        }
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[name] = {"dtype": "float32" if name == "timestamp" else "int64", "shape": [1], "names": None}
    return features


def _data_rows(episodes, first_index=0, first_episode=0):
    rows, cursor = [], first_index
    for e, episode in enumerate(episodes, start=first_episode):
        for i in range(episode["n_frames"]):
            rows.append(
                {
                    "action": action_row(e, i, episode),
                    "observation.state": state_row(e, i, episode),
                    "timestamp": np.float32(i) / np.float32(FPS),
                    "frame_index": i,
                    "episode_index": e,
                    "index": cursor + i,
                    "task_index": episode["task_index"],
                }
            )
        cursor += episode["n_frames"]
    return rows


def _table(rows):
    table = pa.Table.from_pylist(rows)
    for name in ("action", "observation.state"):
        column = pa.array([r[name] for r in rows], pa.list_(pa.float32(), DIM))
        table = table.set_column(table.schema.get_field_index(name), name, column)
    return table.set_column(
        table.schema.get_field_index("timestamp"),
        "timestamp",
        pa.array(table["timestamp"].to_numpy(), pa.float32()),
    )


def _camera_frames(episodes, stream: str, camera_index: int) -> np.ndarray:
    return np.stack(
        [
            frame_image(e, i, camera_index, CAMERA_HW[stream])
            for e, episode in enumerate(episodes)
            for i in range(episode["n_frames"])
        ]
    )


def build_v3_dataset(root, episodes) -> Path:
    """``episodes``: dicts with ``n_frames`` and ``caption``. Concatenated data and video files."""
    root = Path(root)
    for sub in ("meta/episodes/chunk-000", "data/chunk-000"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    captions = sorted({e["caption"] for e in episodes})
    for episode in episodes:
        episode["task_index"] = captions.index(episode["caption"])
    total = sum(e["n_frames"] for e in episodes)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so101_follower",
        "fps": FPS,
        "total_episodes": len(episodes),
        "total_frames": total,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": _features(),
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.Table.from_pylist([{"task_index": i, "task": c} for i, c in enumerate(captions)]),
        root / "meta/tasks.parquet",
    )
    episode_rows, cursor = [], 0
    for e, episode in enumerate(episodes):
        n = episode["n_frames"]
        row = {
            "episode_index": e,
            "tasks": [episode["caption"]],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": cursor,
            "dataset_to_index": cursor + n,
        }
        for feature in CAMERAS.values():
            row[f"videos/{feature}/chunk_index"] = 0
            row[f"videos/{feature}/file_index"] = 0
            row[f"videos/{feature}/from_timestamp"] = cursor / FPS
            row[f"videos/{feature}/to_timestamp"] = (cursor + n) / FPS
        episode_rows.append(row)
        cursor += n
    pq.write_table(pa.Table.from_pylist(episode_rows), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(_table(_data_rows(episodes)), root / "data/chunk-000/file-000.parquet")
    for c, (stream, feature) in enumerate(CAMERAS.items()):
        path = root / f"videos/{feature}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_av1_clip(path, _camera_frames(episodes, stream, c), fps=FPS)
    return root


def build_v21_dataset(root, episodes) -> Path:
    """Same content in the v2.1 layout: one parquet and one mp4 per episode and camera, jsonl metadata."""
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    captions = sorted({e["caption"] for e in episodes})
    for episode in episodes:
        episode["task_index"] = captions.index(episode["caption"])
    total = sum(e["n_frames"] for e in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so101",
        "fps": FPS,
        "total_episodes": len(episodes),
        "total_frames": total,
        "total_tasks": len(captions),
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(),
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": i, "task": c}) + "\n" for i, c in enumerate(captions))
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": e, "tasks": [ep["caption"]], "length": ep["n_frames"]}) + "\n"
            for e, ep in enumerate(episodes)
        )
    )
    cursor = 0
    for e, episode in enumerate(episodes):
        rows = _data_rows([episode], first_index=cursor, first_episode=e)
        path = root / f"data/chunk-000/episode_{e:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_table(rows), path)
        for c, (stream, feature) in enumerate(CAMERAS.items()):
            clip = root / f"videos/chunk-000/{feature}/episode_{e:06d}.mp4"
            clip.parent.mkdir(parents=True, exist_ok=True)
            frames = np.stack([frame_image(e, i, c, CAMERA_HW[stream]) for i in range(episode["n_frames"])])
            write_av1_clip(clip, frames, fps=FPS)
        cursor += episode["n_frames"]
    return root
