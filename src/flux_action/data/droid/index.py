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
"""Whole-dataset manifest for the compact Cosmos3-DROID layout.

``build_manifest`` lists every episode of the ``success`` split, joins the keep-ranges filter and
keeps the episodes with at least one complete window and aligned camera streams. ``build_rows``
extracts state and action for every frame into one float32 array indexed by the dataset's
global frame ``index`` (``rows[i] = state(8) + action(8)``), validated the way ``prepare-droid``
validates a single episode. Training windows then read three video seeks and one array slice.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

from ...checkpoints.artifacts import sha256
from ..windows import valid_window_spans
from .cosmos import CAMERA_FEATURES, filter_key

MANIFEST_FORMAT = 1
ROWS_FILENAME = "rows.f32.npy"
ROW_WIDTH = 16  # 7 joints + gripper for the state, then the same for the action
EPISODE_COLUMNS = [
    "episode_index",
    "episode_id",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
] + [
    f"videos/{feature}/{suffix}"
    for feature in CAMERA_FEATURES.values()
    for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
]
ROW_COLUMNS = [
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


def _read_info(root: Path) -> dict:
    info = json.loads((root / "success/meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 15:
        raise ValueError("expected compact LeRobot v3 DROID at 15 Hz")
    return info


def build_manifest(
    source_root,
    filter_path,
    *,
    chunk_size: int = 32,
    only_available: bool = False,
    hash_files: bool = False,
    revision: str | None = None,
) -> dict:
    """Scan ``success/meta/episodes`` and the filter into one JSON-ready manifest."""
    import pyarrow.parquet as pq

    root, filter_path = Path(source_root), Path(filter_path)
    info = _read_info(root)
    fps = int(info["fps"])
    ranges_by_key = json.loads(filter_path.read_text())
    counts = {"listed": 0, "eligible": 0, "no_valid_window": 0, "misaligned": 0, "unavailable": 0}
    episodes, files, valid_total = [], {}, 0
    for parquet in sorted((root / "success/meta/episodes").rglob("*.parquet")):
        for row in pq.read_table(parquet, columns=EPISODE_COLUMNS).to_pylist():
            counts["listed"] += 1
            n = int(row["length"])
            if row["dataset_to_index"] - row["dataset_from_index"] != n:
                counts["misaligned"] += 1
                continue
            ranges = [list(map(int, r)) for r in ranges_by_key.get(filter_key(row["episode_id"]), [])]
            n_valid = sum(c for _, c in valid_window_spans(n, ranges, chunk_size))
            if n_valid == 0:
                counts["no_valid_window"] += 1
                continue
            videos, aligned = {}, True
            for camera, feature in CAMERA_FEATURES.items():
                prefix = f"videos/{feature}"
                first, last = (float(row[f"{prefix}/{s}"]) * fps for s in ("from_timestamp", "to_timestamp"))
                if not all(math.isclose(x, round(x), rel_tol=0, abs_tol=2e-5) for x in (first, last)):
                    aligned = False
                if round(last) - round(first) != n:
                    aligned = False
                videos[camera] = {
                    "file": "success/"
                    + info["video_path"].format(
                        video_key=feature,
                        chunk_index=row[f"{prefix}/chunk_index"],
                        file_index=row[f"{prefix}/file_index"],
                    ),
                    "first_frame": int(round(first)),
                }
            if not aligned:
                counts["misaligned"] += 1
                continue
            data_file = "success/" + info["data_path"].format(
                chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
            )
            needed = [data_file, *(v["file"] for v in videos.values())]
            if only_available and not all((root / f).is_file() for f in needed):
                counts["unavailable"] += 1
                continue
            for f in needed:
                files.setdefault(f, None)
            counts["eligible"] += 1
            valid_total += n_valid
            episodes.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "episode_id": row["episode_id"],
                    "n_frames": n,
                    "caption": row["tasks"][0],
                    "valid_ranges": ranges,
                    "n_valid_starts": n_valid,
                    "data_file": data_file,
                    "from_index": int(row["dataset_from_index"]),
                    "videos": videos,
                }
            )
    file_records = {}
    for name in sorted(files):
        path = root / name
        record = {"bytes": path.stat().st_size if path.is_file() else None}
        if hash_files and path.is_file():
            record["sha256"] = sha256(path)
        file_records[name] = record
    counts["valid_starts_total"] = valid_total
    return {
        "format_version": MANIFEST_FORMAT,
        "dataset_id": "nvidia/Cosmos3-DROID",
        "dataset_revision": revision,
        "filter_file": filter_path.name,
        "filter_sha256": sha256(filter_path),
        "fps": fps,
        "chunk_size": chunk_size,
        "total_frames": int(info["total_frames"]),
        "camera_order": list(CAMERA_FEATURES),
        "counts": counts,
        "files": file_records,
        "episodes": episodes,
    }


def write_manifest(manifest: dict, path) -> None:
    """Write the manifest atomically: readers waiting for the file never see a partial one."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(tmp, path)


def load_manifest(path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if (
        manifest.get("format_version") != MANIFEST_FORMAT
        or manifest.get("dataset_id") != "nvidia/Cosmos3-DROID"
    ):
        raise ValueError("unsupported DROID manifest")
    return manifest


def build_rows(source_root, manifest: dict, output) -> dict:
    """Write ``rows.f32.npy`` (``total_frames x 16``) and drop episodes whose rows fail validation.

    Rows are written at their global ``index``; unlisted frames stay NaN. Checks per episode:
    contiguous ``frame_index`` and ``index``, float32 15 Hz timestamps, one ``task_index`` equal to
    the caption's, finite values, gripper in ``[0, 1]``.
    """
    import pyarrow.parquet as pq

    root, output = Path(source_root), Path(output)
    info = _read_info(root)
    tasks = pq.read_table(root / "success/meta/tasks.parquet").to_pylist()
    task_index = {r.get("task", r.get("__index_level_0__")): r["task_index"] for r in tasks}
    total = int(manifest["total_frames"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(total, ROW_WIDTH))
    rows[:] = np.nan
    by_file: dict[str, list[dict]] = {}
    for episode in manifest["episodes"]:
        by_file.setdefault(episode["data_file"], []).append(episode)
    kept, rejected = [], {}
    for data_file, group in sorted(by_file.items()):
        table = pq.read_table(root / data_file, columns=ROW_COLUMNS)
        episode_col = table["episode_index"].to_numpy()
        for episode in group:
            n, start = episode["n_frames"], episode["from_index"]
            mask = episode_col == episode["episode_index"]
            part = table.filter(mask)
            reason = _validate_rows(part, n, start, task_index.get(episode["caption"]), info["fps"])
            if reason is not None:
                rejected[episode["episode_id"]] = reason
                continue
            joints = np.asarray(part["observation.state.joint_positions"].to_pylist(), np.float32)
            grip = np.asarray(part["observation.state.gripper_position"].to_pylist(), np.float32).reshape(
                n, 1
            )
            act = np.asarray(part["action.joint_position"].to_pylist(), np.float32)
            act_grip = np.asarray(part["action.gripper_position"].to_pylist(), np.float32).reshape(n, 1)
            rows[start : start + n] = np.concatenate([joints, grip, act, act_grip], axis=1)
            kept.append(episode)
    rows.flush()
    manifest["episodes"] = kept
    manifest["counts"]["eligible"] = len(kept)
    manifest["counts"]["rows_rejected"] = len(rejected)
    manifest["counts"]["valid_starts_total"] = sum(e["n_valid_starts"] for e in kept)
    manifest["rows"] = {"file": output.name, "sha256": sha256(output), "width": ROW_WIDTH}
    return {"kept": len(kept), "rejected": rejected}


def _validate_rows(part, n, start, expected_task, fps):
    if part.num_rows != n:
        return f"{part.num_rows} rows for {n} frames"
    if not np.array_equal(part["frame_index"].to_numpy(), np.arange(n)):
        return "non-contiguous frame_index"
    if not np.array_equal(part["index"].to_numpy(), np.arange(start, start + n)):
        return "non-contiguous index"
    if expected_task is None or not np.all(part["task_index"].to_numpy() == expected_task):
        return "task_index disagrees with the caption"
    times = part["timestamp"].to_numpy()
    grid = np.arange(n, dtype=np.float32) / np.float32(fps)
    if times.dtype != np.float32 or not np.array_equal(times.view(np.uint32), grid.view(np.uint32)):
        return "timestamps off the float32 15 Hz grid"
    for joints, grip in (
        ("observation.state.joint_positions", "observation.state.gripper_position"),
        ("action.joint_position", "action.gripper_position"),
    ):
        j = np.asarray(part[joints].to_pylist(), np.float32)
        g = np.asarray(part[grip].to_pylist(), np.float32)
        if j.shape != (n, 7) or g.size != n or not (np.isfinite(j).all() and np.isfinite(g).all()):
            return f"invalid {joints} shape or values"
        if np.any((g < 0) | (g > 1)):
            return f"{grip} outside [0, 1]"
    return None


def open_rows(path, total_frames: int) -> np.ndarray:
    rows = np.load(path, mmap_mode="r")
    if rows.dtype != np.float32 or rows.shape != (total_frames, ROW_WIDTH):
        raise ValueError(
            f"rows array must be float32 ({total_frames}, {ROW_WIDTH}), got {rows.dtype} {rows.shape}"
        )
    return rows
