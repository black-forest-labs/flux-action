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
"""Lossless intermediate episode preparation and aligned window loading.

Input is decoded, synchronized DROID arrays with supplied valid-range metadata.
The public Cosmos3-DROID adapter is implemented separately in ``cosmos.py``.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from ...checkpoints.artifacts import sha256
from ..schema import EpisodeMetadata
from ..windows import valid_window_spans

CAMERAS = ("wrist", "left", "right")
CAPTION_SEPARATOR = " | "  # Cosmos3-DROID stores several paraphrases of one instruction in a single string


def select_caption(caption, rng=None):
    """One paraphrase of a stored caption: a random one with ``rng`` (the training recipe), else the first."""
    alternatives = caption.split(CAPTION_SEPARATOR)
    return rng.choice(alternatives) if rng is not None else alternatives[0]


def validate_arrays(metadata, arrays):
    n = metadata.n_frames
    required = {*CAMERAS, "timestamps", "state", "action"}
    if set(arrays) != required:
        raise ValueError(f"episode arrays must have exactly {sorted(required)}")
    for camera in CAMERAS:
        if arrays[camera].shape != (n, 360, 640, 3) or arrays[camera].dtype != np.uint8:
            raise ValueError(f"{camera} must be uint8 (N,360,640,3) RGB")
    timestamps = arrays["timestamps"]
    if timestamps.shape != (n,) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite (N,)")
    if n > 1 and not np.allclose(np.diff(timestamps), 1 / metadata.fps, atol=2e-4, rtol=0):
        raise ValueError("timestamps must be ordered and synchronized at 15 Hz")
    for name in ("state", "action"):
        value = arrays[name]
        if (
            value.shape != (n, 8)
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"{name} must be finite floating-point (N,8)")
        if np.any((value[:, -1] < 0) | (value[:, -1] > 1)):
            raise ValueError(f"{name} gripper must be a closed fraction in [0,1]")
    if not valid_window_spans(n, metadata.valid_ranges):
        raise ValueError("episode has no complete valid 33-frame window")


def prepare_episode(metadata_path, arrays_path, output):
    """Validate/copy a canonical NPZ episode. Same-input retries verify and reuse it."""
    metadata_path, arrays_path, output = map(Path, (metadata_path, arrays_path, output))
    metadata = EpisodeMetadata(**json.loads(metadata_path.read_text()))
    source_hashes = {"metadata": sha256(metadata_path), "arrays": sha256(arrays_path)}
    if output.exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous["source_sha256"] != source_hashes:
            raise FileExistsError("destination belongs to different source inputs")
        validate_episode(output)
        return previous
    with np.load(arrays_path, allow_pickle=False) as arrays:
        validate_arrays(metadata, arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shutil.copyfile(arrays_path, staging / "episode.npz")
        (staging / "metadata.json").write_text(json.dumps(metadata.to_dict(), indent=2) + "\n")
        manifest = {
            "format_version": 1,
            "source_sha256": source_hashes,
            "sha256": {name: sha256(staging / name) for name in ("metadata.json", "episode.npz")},
            "n_valid_starts": sum(c for _, c in valid_window_spans(metadata.n_frames, metadata.valid_ranges)),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def validate_episode(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format_version"] != 1:
        raise ValueError("unsupported prepared episode format")
    for name in ("metadata.json", "episode.npz"):
        if sha256(directory / name) != manifest["sha256"].get(name):
            raise ValueError(f"episode checksum mismatch: {name}")
    metadata = EpisodeMetadata(**json.loads((directory / "metadata.json").read_text()))
    with np.load(directory / "episode.npz", allow_pickle=False) as arrays:
        validate_arrays(metadata, arrays)
    return metadata


def load_window(directory, start, chunk_size=32, rng=None):
    """Return a batch of one: observations s:s+33, commands s:s+32, state at s.

    Values remain in source units. The policy flips the gripper once and scales
    the representation internally; preparation does neither transformation.
    The caption is one paraphrase of the stored task, drawn with ``rng`` as in the
    reference finetune, or the first one when no ``rng`` is given.
    """
    directory = Path(directory)
    metadata = validate_episode(directory)
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or not any(
            first <= start < first + count
            for first, count in valid_window_spans(metadata.n_frames, metadata.valid_ranges, chunk_size)
        )
    ):
        raise ValueError("start is not a complete valid window")
    with np.load(directory / "episode.npz", allow_pickle=False) as arrays:
        batch = {
            f"images.{camera}": torch.from_numpy(arrays[camera][start : start + chunk_size + 1].copy())
            .permute(0, 3, 1, 2)
            .float()
            .div_(255)
            .unsqueeze(0)
            for camera in CAMERAS
        }
        batch["state"] = torch.from_numpy(arrays["state"][start].copy()).float()[None]
        batch["action"] = torch.from_numpy(arrays["action"][start : start + chunk_size].copy()).float()[None]
        batch["task"] = [select_caption(metadata.caption, rng)]
    return batch
