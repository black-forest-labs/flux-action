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
"""Whole-dataset manifest, rows array and normalization statistics for a LeRobot dataset.

Two on-disk layouts are read, both as LeRobot writes them:

* **v2.1** (also v2.0): ``meta/episodes.jsonl`` and ``meta/tasks.jsonl``, one parquet and one mp4 per
  episode and camera (``data/chunk-000/episode_000012.parquet``,
  ``videos/chunk-000/<camera feature>/episode_000012.mp4``). The community SO-100/SO-101 datasets
  are in this layout.
* **v3.0**: ``meta/episodes/chunk-*/file-*.parquet`` and ``meta/tasks.parquet``, frames of many episodes
  concatenated into data parquet files and one mp4 per camera and file, each episode addressed by its
  ``dataset_from_index`` and the video's ``from_timestamp``.

``build_manifest`` lists the episodes with at least one complete window, ``build_rows`` extracts the
state and the action of every frame into one float32 array (``rows[i] = state(S) + action(A)`` at the
dataset's global frame ``index``), ``build_statistics`` computes the per-channel 1st / 99th percentiles
of the training targets and of the state that the policy normalizes with. A training window is then one
video seek per camera and one slice of the rows array, as for DROID (``data/droid/index.py``).

Windows start at frame 1 at the earliest (``MIN_START``): the recipe trains on per-frame joint deltas
``a[t] - a[t-1]`` computed within the episode, so the action before the window's first action must exist.
The window dataset hands it to the policy as ``action_prev``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ...checkpoints.artifacts import sha256
from ..droid.episodes import CAPTION_SEPARATOR
from ..droid.index import ROWS_FILENAME, write_manifest
from ..windows import valid_window_spans

MANIFEST_FORMAT = 1
KIND = "lerobot"
STATISTICS_FILENAME = "statistics.json"
STATISTICS_FORMAT = 1
MIN_START = 1  # the first action of a window is a[s] - a[s-1]
GRAY = "gray"  # a camera stream the dataset lacks: the window dataset serves a flat mid-gray tile for it
NORMALIZATION_CLIP = 6.0
# Dead-window stripping, the corpus builder's definition: a frame is dead inside a stretch of at least
# DEAD_SECONDS in which no joint of the state moves more than DEAD_THRESHOLD per frame; episodes moving less
# than MIN_MOVING_FRACTION of their frames, or longer than LONG_EPISODE_SECONDS with more than
# LONG_DEAD_FRACTION dead, are dropped; a window is stripped when more than the given fraction of its
# frames is dead (the recipe used 0.6).
DEAD_THRESHOLD = 0.1
DEAD_SECONDS = 2.0
MIN_MOVING_FRACTION = 0.02
LONG_EPISODE_SECONDS = 300.0
LONG_DEAD_FRACTION = 0.5
DEGENERATE_SPAN = 1e-6  # a channel whose q99 - q01 is at most this uses span 1 (a gripper that never moved)
ACTION_PARAMETERIZATIONS = ("absolute", "joint_delta")
SUPPORTED_MAJOR_VERSIONS = (2, 3)

__all__ = [
    "KIND",
    "GRAY",
    "MANIFEST_FORMAT",
    "MIN_START",
    "NORMALIZATION_CLIP",
    "ROWS_FILENAME",
    "STATISTICS_FILENAME",
    "apply_dead_window_filter",
    "build_manifest",
    "build_rows",
    "build_statistics",
    "load_manifest",
    "load_statistics",
    "read_info",
    "write_manifest",
    "write_statistics",
]


# ---- metadata -------------------------------------------------------------------------------
def read_info(root) -> dict:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    version = major_version(info)
    if version not in SUPPORTED_MAJOR_VERSIONS:
        raise ValueError(
            f"unsupported LeRobot codebase_version {info.get('codebase_version')!r}; expected v2.x or v3.x"
        )
    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0 or not float(fps).is_integer():
        raise ValueError(f"LeRobot fps must be a positive whole number, got {fps!r}")
    if "features" not in info or "data_path" not in info or "video_path" not in info:
        raise ValueError("meta/info.json must list features, data_path and video_path")
    return info


def major_version(info: dict) -> int:
    text = str(info.get("codebase_version", "")).strip().lstrip("vV")
    try:
        return int(text.split(".")[0])
    except ValueError as error:
        raise ValueError(f"unreadable codebase_version {info.get('codebase_version')!r}") from error


def feature_names(feature: dict, dim: int) -> list[str] | None:
    """``names`` of a 1-D feature: a list of ``dim`` names, or the single list some datasets nest under one
    key (``{"motors": [...]}``); anything else is unnamed."""
    names = feature.get("names")
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    if isinstance(names, list) and len(names) == dim and all(isinstance(n, str) for n in names):
        return list(names)
    return None


def vector_dim(features: dict, key: str) -> int:
    feature = features.get(key)
    if feature is None:
        raise ValueError(f"feature {key!r} is not in the dataset; available: {sorted(features)}")
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or int(shape[0]) < 1:
        raise ValueError(f"feature {key!r} must be a 1-D vector, got shape {shape}")
    return int(shape[0])


def camera_hw(features: dict, key: str) -> tuple[int, int]:
    feature = features.get(key)
    if feature is None:
        cams = sorted(k for k, f in features.items() if f.get("dtype") in ("video", "image"))
        raise ValueError(f"camera feature {key!r} is not in the dataset; cameras: {cams}")
    if feature.get("dtype") != "video":
        raise ValueError(
            f"camera feature {key!r} has dtype {feature.get('dtype')!r}; only video streams are read"
        )
    shape, names = list(feature["shape"]), feature.get("names")
    if isinstance(names, list) and {"height", "width"} <= set(names):
        return int(shape[names.index("height")]), int(shape[names.index("width")])
    if len(shape) == 3 and shape[-1] in (1, 3):
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] in (1, 3):
        return int(shape[1]), int(shape[2])
    raise ValueError(f"camera feature {key!r}: cannot read height and width from shape {shape}")


def load_tasks(root: Path, version: int) -> dict[int, str]:
    if version >= 3:
        import pyarrow.parquet as pq

        table = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
        if "task_index" not in table:
            return {i: str(t) for i, t in enumerate(table["task"])}
        column = "task" if "task" in table else next((k for k in table if k != "task_index"), None)
        if column is None:
            raise ValueError("meta/tasks.parquet has no task column")
        return {int(i): str(t) for i, t in zip(table["task_index"], table[column], strict=True)}
    tasks = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def _episode_caption(tasks: list[str]) -> str:
    """Several tasks of one episode become paraphrases the loader picks from (``select_caption``)."""
    return CAPTION_SEPARATOR.join(str(t).strip() for t in tasks if str(t).strip())


def _list_episodes_v3(root: Path, info: dict, cameras: dict[str, str], fps: int, counts: dict) -> list[dict]:
    import pyarrow.parquet as pq

    columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index"]
    columns += ["dataset_from_index", "dataset_to_index"]
    columns += [
        f"videos/{feature}/{suffix}"
        for feature in cameras.values()
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ]
    episodes = []
    for parquet in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        for row in pq.read_table(parquet, columns=columns).to_pylist():
            counts["listed"] += 1
            n = int(row["length"])
            if row["dataset_to_index"] - row["dataset_from_index"] != n:
                counts["misaligned"] += 1
                continue
            videos, aligned = {}, True
            for stream, feature in cameras.items():
                prefix = f"videos/{feature}"
                first, last = (float(row[f"{prefix}/{s}"]) * fps for s in ("from_timestamp", "to_timestamp"))
                if not all(math.isclose(x, round(x), rel_tol=0, abs_tol=1e-3) for x in (first, last)):
                    aligned = False
                if round(last) - round(first) != n:
                    aligned = False
                videos[stream] = {
                    "file": info["video_path"].format(
                        video_key=feature,
                        chunk_index=row[f"{prefix}/chunk_index"],
                        file_index=row[f"{prefix}/file_index"],
                    ),
                    "first_frame": int(round(first)),
                }
            if not aligned:
                counts["misaligned"] += 1
                continue
            episodes.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "n_frames": n,
                    "caption": _episode_caption(list(row["tasks"] or [])),
                    "data_file": info["data_path"].format(
                        chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
                    ),
                    "from_index": int(row["dataset_from_index"]),
                    "videos": videos,
                }
            )
    return episodes


def _list_episodes_v2(root: Path, info: dict, cameras: dict[str, str], counts: dict) -> list[dict]:
    chunks_size = int(info.get("chunks_size", 1000))
    episodes = []
    for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        counts["listed"] += 1
        index = int(row["episode_index"])
        chunk = index // max(1, chunks_size)
        episodes.append(
            {
                "episode_index": index,
                "n_frames": int(row["length"]),
                "caption": _episode_caption(list(row.get("tasks") or [])),
                "data_file": info["data_path"].format(episode_chunk=chunk, episode_index=index),
                "from_index": None,  # the global frame index of the episode's parquet; read by build_rows
                "videos": {
                    stream: {
                        "file": info["video_path"].format(
                            episode_chunk=chunk, video_key=feature, episode_index=index
                        ),
                        "first_frame": 0,
                    }
                    for stream, feature in cameras.items()
                },
            }
        )
    return episodes


def build_manifest(
    source_root,
    cameras: dict[str, str],
    *,
    state_key: str = "observation.state",
    action_key: str = "action",
    chunk_size: int = 32,
    val_episodes: int = 0,
    dataset_id: str | None = None,
    revision: str | None = None,
    only_available: bool = False,
    hash_files: bool = False,
) -> dict:
    """List the episodes of a LeRobot dataset into one JSON-ready manifest.

    ``cameras`` maps the stream names the policy sees (``images.<stream>``; layout order) to the dataset's
    video features, e.g. ``{"top": "observation.images.top", "wrist": "observation.images.front"}``; the
    value ``"gray"`` declares a stream the dataset lacks, served as a flat mid-gray tile (a single-camera
    dataset trained with the two-tile layout of the SO-101 checkpoints). Episodes without a complete
    ``chunk_size + 1`` frame window from frame ``MIN_START`` on, and v3 episodes whose video segments do not
    start and end on whole frames spanning the episode, are counted and left out. ``only_available``
    additionally drops episodes whose files are not on disk. The last ``val_episodes`` eligible episodes (by
    index) are marked ``split: val`` and held out of training; the rest are ``train``.
    """
    root = Path(source_root)
    if not cameras:
        raise ValueError("name at least one camera: {stream: dataset feature}")
    if val_episodes < 0:
        raise ValueError("val_episodes must be non-negative")
    info = read_info(root)
    version, fps = major_version(info), int(info["fps"])
    features = info["features"]
    state_dim, action_dim = vector_dim(features, state_key), vector_dim(features, action_key)
    video_cameras = {stream: feature for stream, feature in cameras.items() if feature != GRAY}
    if not video_cameras:
        raise ValueError("at least one camera must be a video feature of the dataset")
    hw = {
        stream: None if feature == GRAY else list(camera_hw(features, feature))
        for stream, feature in cameras.items()
    }
    counts = {"listed": 0, "eligible": 0, "no_valid_window": 0, "misaligned": 0, "unavailable": 0}
    listed = (
        _list_episodes_v3(root, info, video_cameras, fps, counts)
        if version >= 3
        else _list_episodes_v2(root, info, video_cameras, counts)
    )
    episodes, files, valid_total = [], {}, 0
    for episode in sorted(listed, key=lambda e: e["episode_index"]):
        n = episode["n_frames"]
        ranges = [[MIN_START, n]]
        n_valid = sum(c for _, c in valid_window_spans(n, ranges, chunk_size))
        if n_valid == 0:
            counts["no_valid_window"] += 1
            continue
        needed = [episode["data_file"], *(v["file"] for v in episode["videos"].values())]
        if only_available and not all((root / f).is_file() for f in needed):
            counts["unavailable"] += 1
            continue
        for f in needed:
            files.setdefault(f, None)
        counts["eligible"] += 1
        valid_total += n_valid
        episodes.append(
            {
                "episode_index": episode["episode_index"],
                "episode_id": f"episode_{episode['episode_index']:06d}",
                "split": "train",
                "n_frames": n,
                "caption": episode["caption"],
                "valid_ranges": ranges,
                "n_valid_starts": n_valid,
                "data_file": episode["data_file"],
                "from_index": episode["from_index"],
                "videos": {stream: episode["videos"].get(stream) for stream in cameras},
            }
        )
    if val_episodes >= len(episodes) and episodes:
        raise ValueError(f"val_episodes {val_episodes} would leave no training episode of {len(episodes)}")
    for episode in episodes[len(episodes) - val_episodes :] if val_episodes else []:
        episode["split"] = "val"
    counts["train_episodes"] = sum(e["split"] == "train" for e in episodes)
    counts["val_episodes"] = sum(e["split"] == "val" for e in episodes)
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
        "kind": KIND,
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": fps,
        "chunk_size": chunk_size,
        "min_start": MIN_START,
        "action_prev": True,
        "total_frames": int(info["total_frames"]),
        "state_key": state_key,
        "action_key": action_key,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_names": feature_names(features[state_key], state_dim),
        "action_names": feature_names(features[action_key], action_dim),
        "cameras": dict(cameras),
        "camera_order": list(cameras),
        "camera_hw": hw,
        "counts": counts,
        "files": file_records,
        "episodes": episodes,
    }


def load_manifest(path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("format_version") != MANIFEST_FORMAT or manifest.get("kind") != KIND:
        raise ValueError("unsupported LeRobot manifest")
    return manifest


# ---- rows -----------------------------------------------------------------------------------
def _vectors(column, n: int, dim: int, name: str) -> np.ndarray | str:
    values = np.asarray(column.to_pylist(), dtype=np.float32)
    if values.shape != (n, dim):
        return f"{name} has shape {values.shape}, expected {(n, dim)}"
    if not np.isfinite(values).all():
        return f"{name} holds non-finite values"
    return values


def _validate_episode_rows(part, n: int, start: int, fps: int) -> str | None:
    if part.num_rows != n:
        return f"{part.num_rows} rows for {n} frames"
    if not np.array_equal(part["frame_index"].to_numpy(), np.arange(n)):
        return "non-contiguous frame_index"
    if not np.array_equal(part["index"].to_numpy(), np.arange(start, start + n)):
        return "non-contiguous index"
    times = np.asarray(part["timestamp"].to_numpy(), dtype=np.float64)
    if np.abs(times - np.arange(n) / fps).max() > 0.25 / fps:
        return f"timestamps off the {fps} Hz grid"
    return None


def build_rows(source_root, manifest: dict, output) -> dict:
    """Write ``rows.f32.npy`` (``total_frames x (state_dim + action_dim)``); drop episodes that fail validation.

    Checks per episode: ``frame_index`` runs ``0 .. n-1``, ``index`` runs from the episode's global offset
    (v3: ``dataset_from_index``; v2.1: the first ``index`` of its parquet), timestamps sit on the frame
    grid, state and action are finite vectors of the declared width. Rows of dropped or unlisted frames
    stay NaN. The manifest's episode list, counts, ``total_frames`` and ``rows`` record are updated.
    """
    import pyarrow.parquet as pq

    root, output = Path(source_root), Path(output)
    fps = int(manifest["fps"])
    state_key, action_key = manifest["state_key"], manifest["action_key"]
    state_dim, action_dim = int(manifest["state_dim"]), int(manifest["action_dim"])
    width = state_dim + action_dim
    columns = ["frame_index", "index", "timestamp", state_key, action_key]
    v3 = all(e["from_index"] is not None for e in manifest["episodes"])
    if not v3:  # v2.1: one parquet per episode; its first global index is the episode's offset
        for episode in manifest["episodes"]:
            first = pq.read_table(root / episode["data_file"], columns=["index"])["index"]
            episode["from_index"] = int(first[0].as_py()) if len(first) else 0
    total = max(
        int(manifest["total_frames"]), max(e["from_index"] + e["n_frames"] for e in manifest["episodes"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(total, width))
    rows[:] = np.nan
    by_file: dict[str, list[dict]] = {}
    for episode in manifest["episodes"]:
        by_file.setdefault(episode["data_file"], []).append(episode)
    kept, rejected = [], {}
    for data_file, group in sorted(by_file.items()):
        table = pq.read_table(root / data_file, columns=columns + ["episode_index"])
        episode_col = table["episode_index"].to_numpy()
        for episode in group:
            n, start = episode["n_frames"], episode["from_index"]
            part = table.filter(episode_col == episode["episode_index"])
            reason = _validate_episode_rows(part, n, start, fps)
            if reason is None:
                state = _vectors(part[state_key], n, state_dim, state_key)
                action = _vectors(part[action_key], n, action_dim, action_key)
                reason = state if isinstance(state, str) else action if isinstance(action, str) else None
            if reason is not None:
                rejected[episode["episode_id"]] = reason
                continue
            rows[start : start + n] = np.concatenate([state, action], axis=1)
            kept.append(episode)
    rows.flush()
    manifest["episodes"] = kept
    manifest["total_frames"] = total
    _recount(manifest)
    manifest["counts"]["rows_rejected"] = len(rejected)
    manifest["rows"] = {"file": output.name, "sha256": sha256(output), "width": width}
    return {"kept": len(kept), "rejected": rejected}


def _recount(manifest: dict) -> None:
    counts, episodes = manifest["counts"], manifest["episodes"]
    counts["eligible"] = len(episodes)
    counts["valid_starts_total"] = sum(e["n_valid_starts"] for e in episodes)
    counts["train_episodes"] = sum(e.get("split", "train") == "train" for e in episodes)
    counts["val_episodes"] = sum(e.get("split", "train") == "val" for e in episodes)


# ---- dead-window stripping ----------------------------------------------------------------
def dead_mask(
    states: np.ndarray, fps: float, *, threshold: float = DEAD_THRESHOLD, seconds: float = DEAD_SECONDS
):
    """``(mask, moving_fraction)``: frames inside stretches of at least ``seconds`` in which no joint moves
    more than ``threshold`` per frame; frame 0 counts as moving (the corpus builder's definition)."""
    step = np.abs(np.diff(np.asarray(states, dtype=np.float64), axis=0)).max(axis=1)
    moving = np.concatenate([[True], step > threshold])
    mask = np.zeros(len(states), dtype=bool)
    min_run = max(1, int(round(seconds * fps)))
    run_start = None
    for i, m in enumerate(moving):
        if not m and run_start is None:
            run_start = i
        elif m and run_start is not None:
            if i - run_start >= min_run:
                mask[run_start:i] = True
            run_start = None
    if run_start is not None and len(moving) - run_start >= min_run:
        mask[run_start:] = True
    return mask, float(moving.mean())


def _ranges_of_starts(starts: list[int], chunk_size: int) -> list[list[int]]:
    """Allowed window starts -> half-open frame ranges that ``valid_window_spans`` maps back onto exactly them."""
    ranges: list[list[int]] = []
    for s in starts:
        if ranges and ranges[-1][1] == s + chunk_size:  # consecutive start: extend the range by one frame
            ranges[-1][1] = s + chunk_size + 1
        else:
            ranges.append([s, s + chunk_size + 1])
    return ranges


def apply_dead_window_filter(manifest: dict, rows: np.ndarray, *, max_dead_fraction: float) -> dict:
    """Drop static episodes and strip the windows whose frames are dead beyond ``max_dead_fraction``.

    Uses the state columns of the rows array. Episodes are dropped when fewer than ``MIN_MOVING_FRACTION`` of
    their frames move, or when they last longer than ``LONG_EPISODE_SECONDS`` with more than
    ``LONG_DEAD_FRACTION`` dead; the remaining episodes keep exactly the window starts whose ``chunk_size + 1``
    frames are dead for at most ``max_dead_fraction`` (as ``valid_ranges``). Counts are recorded in the
    manifest; returns them.
    """
    if not 0 <= max_dead_fraction <= 1:
        raise ValueError("max_dead_fraction must lie in [0, 1]")
    fps, chunk_size, state_dim = (
        float(manifest["fps"]),
        int(manifest["chunk_size"]),
        int(manifest["state_dim"]),
    )
    kept, dropped = [], {"dead_static": 0, "dead_long": 0, "dead_no_window": 0}
    removed_starts = 0
    for episode in manifest["episodes"]:
        start, n = episode["from_index"], episode["n_frames"]
        states = np.asarray(rows[start : start + n, :state_dim], dtype=np.float64)
        mask, moving = dead_mask(states, fps)
        if moving < MIN_MOVING_FRACTION:
            dropped["dead_static"] += 1
            continue
        if n > LONG_EPISODE_SECONDS * fps and mask.mean() > LONG_DEAD_FRACTION:
            dropped["dead_long"] += 1
            continue
        before = [
            s for f, c in valid_window_spans(n, episode["valid_ranges"], chunk_size) for s in range(f, f + c)
        ]
        starts = [s for s in before if mask[s : s + chunk_size + 1].mean() <= max_dead_fraction]
        removed_starts += len(before) - len(starts)
        if not starts:
            dropped["dead_no_window"] += 1
            continue
        episode["valid_ranges"] = _ranges_of_starts(starts, chunk_size)
        episode["n_valid_starts"] = len(starts)
        assert sum(c for _, c in valid_window_spans(n, episode["valid_ranges"], chunk_size)) == len(starts), (
            "valid window count mismatch"
        )
        kept.append(episode)
    manifest["episodes"] = kept
    _recount(manifest)
    manifest["counts"].update(dropped)
    manifest["counts"]["dead_starts_removed"] = removed_starts
    manifest["dead_window_filter"] = {
        "max_dead_fraction": max_dead_fraction,
        "threshold": DEAD_THRESHOLD,
        "seconds": DEAD_SECONDS,
        "min_moving_fraction": MIN_MOVING_FRACTION,
        "long_episode_seconds": LONG_EPISODE_SECONDS,
        "long_dead_fraction": LONG_DEAD_FRACTION,
    }
    return {**dropped, "dead_starts_removed": removed_starts}


def open_rows(path, total_frames: int, width: int) -> np.ndarray:
    rows = np.load(path, mmap_mode="r")
    if rows.dtype != np.float32 or rows.shape != (total_frames, width):
        raise ValueError(
            f"rows array must be float32 ({total_frames}, {width}), got {rows.dtype} {rows.shape}"
        )
    return rows


# ---- normalization statistics ----------------------------------------------------------------
def normalize_dims(dims, dim: int) -> tuple[int, ...]:
    """Non-negative, sorted, distinct channel indices (negative values count from the end)."""
    out = sorted({int(d) % dim for d in dims})
    if any(not -dim <= int(d) < dim for d in dims):
        raise ValueError(f"channel index outside the {dim} action channels: {list(dims)}")
    return tuple(out)


def action_targets(actions: np.ndarray, parameterization: str, absolute_dims: tuple[int, ...]) -> np.ndarray:
    """The training targets of one episode's actions ``(n, A)``: the actions themselves (``absolute``), or
    the per-frame deltas ``a[t] - a[t-1]`` with the first row zero and ``absolute_dims`` kept absolute
    (``joint_delta``, the SO-101 recipe's parameterization and its builder's convention)."""
    if parameterization == "absolute":
        return actions.astype(np.float64)
    if parameterization != "joint_delta":
        raise ValueError(
            f"unknown action parameterization {parameterization!r}; choose from {ACTION_PARAMETERIZATIONS}"
        )
    actions = actions.astype(np.float64)
    targets = np.zeros_like(actions)
    targets[1:] = actions[1:] - actions[:-1]
    if absolute_dims:
        targets[:, list(absolute_dims)] = actions[:, list(absolute_dims)]
    return targets


def build_statistics(
    manifest: dict,
    rows: np.ndarray,
    *,
    action_parameterization: str = "joint_delta",
    absolute_action_dims=(-1,),
    clip: float = NORMALIZATION_CLIP,
) -> dict:
    """Per-channel q01 / q99 of the action targets and of the state over every frame of the kept episodes.

    The percentiles are ``numpy.percentile`` (linear interpolation) over float64 values, the targets of
    ``action_targets`` per episode, over the train and the held-out episodes alike; this is the
    computation the SO-101 community corpus reference (``configs/so101/community_corpus.json``) was built
    with, so re-indexing one of its datasets reproduces the listed bounds.
    """
    if action_parameterization not in ACTION_PARAMETERIZATIONS:
        raise ValueError(f"unknown action parameterization {action_parameterization!r}")
    state_dim, action_dim = int(manifest["state_dim"]), int(manifest["action_dim"])
    dims = (
        normalize_dims(absolute_action_dims, action_dim) if action_parameterization == "joint_delta" else ()
    )
    states, targets, frames = [], [], 0
    for episode in manifest["episodes"]:
        start, n = episode["from_index"], episode["n_frames"]
        block = np.asarray(rows[start : start + n], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError(f"episode {episode['episode_id']}: rows missing from the rows array")
        states.append(block[:, :state_dim])
        targets.append(action_targets(block[:, state_dim:], action_parameterization, dims))
        frames += n
    if not targets:
        raise ValueError("no episodes to compute statistics from")
    state, target = np.concatenate(states), np.concatenate(targets)

    def bounds(values: np.ndarray) -> dict[str, list[float]]:
        return {
            "q01": [float(v) for v in np.percentile(values, 1, axis=0)],
            "q99": [float(v) for v in np.percentile(values, 99, axis=0)],
        }

    return {
        "format_version": STATISTICS_FORMAT,
        "normalization": "range",
        "formula": (
            "x_norm = clip(2 * (x - q01) / span - 1, -clip, clip) with span = q99 - q01, or 1 where "
            f"q99 - q01 <= {DEGENERATE_SPAN}"
        ),
        "clip": float(clip),
        "episodes": len(targets),
        "frames": frames,
        "action": {
            "parameterization": action_parameterization,
            "absolute_dims": list(dims),
            "names": manifest.get("action_names"),
            **bounds(target),
        },
        "state": {"names": manifest.get("state_names"), **bounds(state)},
    }


def write_statistics(statistics: dict, path) -> None:
    Path(path).write_text(json.dumps(statistics, indent=1) + "\n")


def load_statistics(path) -> dict:
    statistics = json.loads(Path(path).read_text())
    if statistics.get("format_version") != STATISTICS_FORMAT or statistics.get("normalization") != "range":
        raise ValueError("unsupported statistics file")
    return statistics


def index_dataset(
    source_root,
    output_dir,
    cameras: dict[str, str],
    *,
    action_parameterization: str = "joint_delta",
    absolute_action_dims=(-1,),
    clip: float = NORMALIZATION_CLIP,
    strip_dead_windows: float | None = None,
    **manifest_kwargs: Any,
) -> dict:
    """``build_manifest`` + ``build_rows`` (+ ``apply_dead_window_filter``) + ``build_statistics`` into
    ``output_dir``; returns a summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(source_root, cameras, **manifest_kwargs)
    rows_result = build_rows(source_root, manifest, out / ROWS_FILENAME)
    rows = open_rows(out / ROWS_FILENAME, manifest["total_frames"], manifest["rows"]["width"])
    if strip_dead_windows is not None:
        apply_dead_window_filter(manifest, rows, max_dead_fraction=strip_dead_windows)
    statistics = build_statistics(
        manifest,
        rows,
        action_parameterization=action_parameterization,
        absolute_action_dims=absolute_action_dims,
        clip=clip,
    )
    write_statistics(statistics, out / STATISTICS_FILENAME)
    write_manifest(manifest, out / "manifest.json")
    return {
        "manifest": str(out / "manifest.json"),
        "statistics": str(out / STATISTICS_FILENAME),
        "counts": manifest["counts"],
        "rows_rejected": rows_result["rejected"],
        "action": {k: statistics["action"][k] for k in ("parameterization", "absolute_dims", "q01", "q99")},
        "state": {k: statistics["state"][k] for k in ("q01", "q99")},
    }
