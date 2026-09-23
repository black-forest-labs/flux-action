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
"""Manifest, rows array and the streaming window dataset on a synthetic compact layout."""

import json

import numpy as np
import pytest
import torch
from synthetic_droid import FRAME_HW, build_compact_dataset

from flux_action.data.droid.index import (
    ROWS_FILENAME,
    build_manifest,
    build_rows,
    load_manifest,
    open_rows,
    write_manifest,
)
from flux_action.data.droid.video import decode_window
from flux_action.training.data import DataPosition, DroidWindowDataset, build_dataloader, epoch_order

pq = pytest.importorskip("pyarrow.parquet")  # the data extra; skip these tests without it

EPISODES = [
    {"n_frames": 40, "caption": "pick up the cup | grab the cup", "keep": [[0, 40]]},  # 8 starts
    {"n_frames": 38, "caption": "close the drawer", "keep": [[2, 38]]},  # starts 2..5
    {"n_frames": 36, "caption": "no filter entry", "keep": None},  # excluded
    {"n_frames": 20, "caption": "too short", "keep": [[0, 20]]},  # excluded
]


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("compact")
    filter_path = build_compact_dataset(root / "source", EPISODES)
    manifest = build_manifest(root / "source", filter_path, revision="f" * 40)
    out = root / "index"
    out.mkdir()
    rows_result = build_rows(root / "source", manifest, out / ROWS_FILENAME)
    write_manifest(manifest, out / "manifest.json")
    return root / "source", out, rows_result


def test_manifest_counts_and_rows(indexed):
    source, out, rows_result = indexed
    manifest = load_manifest(out / "manifest.json")
    counts = manifest["counts"]
    assert counts["listed"] == 4 and counts["eligible"] == 2 and counts["no_valid_window"] == 2
    assert counts["valid_starts_total"] == 8 + 4 and rows_result == {"kept": 2, "rejected": {}}
    first, second = manifest["episodes"]
    assert first["videos"]["left"]["first_frame"] == 0 and second["videos"]["wrist"]["first_frame"] == 40
    assert second["from_index"] == 40 and manifest["dataset_revision"] == "f" * 40
    assert set(manifest["files"]) == {first["data_file"], *(v["file"] for v in first["videos"].values())}
    rows = open_rows(out / ROWS_FILENAME, manifest["total_frames"])
    table = pq.read_table(source / first["data_file"]).to_pylist()
    assert rows[41, :7].tolist() == pytest.approx([1 + 1 / 1000] * 7, rel=1e-6)
    assert (
        rows[41, 7] == 0.25
        and rows[41, 15] == 0.75
        and rows[41, 8] == pytest.approx(table[41]["action.joint_position"][0])
    )
    assert np.isnan(rows[80:]).all()  # frames of excluded episodes are never read


def test_windows_are_topology_invariant_and_exact(indexed):
    source, out, _ = indexed
    manifest = load_manifest(out / "manifest.json")
    rows_path = out / ROWS_FILENAME
    common = dict(seed=7, epoch=0, frame_hw=FRAME_HW, decoder="pyav")
    single = DroidWindowDataset(manifest, source, rows_path, **common)
    windows = list(single)
    assert [w["episode_index"] for w in windows] == epoch_order(2, 7, 0)
    split = [
        w
        for rank in range(2)
        for w in DroidWindowDataset(manifest, source, rows_path, rank=rank, world_size=2, **common)
    ]
    key = lambda w: (w["episode_index"], w["start"], w["task"])  # noqa: E731
    assert sorted(map(key, windows)) == sorted(map(key, split))
    for w in windows:
        episode = next(e for e in manifest["episodes"] if e["episode_index"] == w["episode_index"])
        start = w["start"]
        assert start in range(*episode["valid_ranges"][0][:1], episode["n_frames"] - 32)
        assert w["task"] in episode["caption"].split(" | ")
        assert w["images.wrist"].dtype == torch.uint8 and w["images.wrist"].shape == (33, 3, *FRAME_HW)
        for camera in ("wrist", "left", "right"):
            video = episode["videos"][camera]
            expected = decode_window(
                source / video["file"], video["first_frame"] + start, 33, frame_hw=FRAME_HW, decoder="pyav"
            )
            assert torch.equal(w[f"images.{camera}"], torch.from_numpy(expected).permute(0, 3, 1, 2))
        assert w["state"].tolist() == pytest.approx(
            [w["episode_index"] + start / 1000] * 7 + [0.25], rel=1e-6
        )
        assert w["action"].shape == (32, 8) and w["action"][0, 0] == pytest.approx(
            100 + w["episode_index"] + start / 1000
        )
    different_epoch = list(DroidWindowDataset(manifest, source, rows_path, **{**common, "epoch": 1}))
    assert sorted(map(key, different_epoch)) != sorted(map(key, windows))


def test_loader_batches_and_exact_resume(indexed):
    source, out, _ = indexed
    manifest = load_manifest(out / "manifest.json")
    rows_path = out / ROWS_FILENAME
    common = dict(seed=3, frame_hw=FRAME_HW, decoder="pyav", windows_per_rank=1)
    full = list(build_dataloader(DroidWindowDataset(manifest, source, rows_path, **common), in_process=True))
    assert (
        len(full) == 2 and full[0]["images.left"].shape == (1, 33, 3, *FRAME_HW) and len(full[0]["task"]) == 1
    )
    resumed = list(
        build_dataloader(
            DroidWindowDataset(manifest, source, rows_path, skip_batches=1, **common), in_process=True
        )
    )
    assert len(resumed) == 1 and resumed[0]["episode_index"].tolist() == full[1]["episode_index"].tolist()
    assert torch.equal(resumed[0]["images.right"], full[1]["images.right"])
    position = DataPosition(epoch=0, batches_consumed=1, world_size=1, num_workers=1, windows_per_rank=1)
    assert DataPosition.from_dict(json.loads(json.dumps(position.to_dict()))) == position
    assert position.matches(world_size=1, num_workers=1, windows_per_rank=1)
    assert not position.matches(world_size=2, num_workers=1, windows_per_rank=1)
    with pytest.raises(ValueError):
        build_dataloader(
            DroidWindowDataset(manifest, source, rows_path, num_workers=2, **common), in_process=True
        )


def test_two_worker_loader_and_skip(indexed):
    source, out, _ = indexed
    manifest = load_manifest(out / "manifest.json")
    rows_path = out / ROWS_FILENAME
    common = dict(seed=5, frame_hw=FRAME_HW, decoder="pyav", windows_per_rank=1, num_workers=2)
    batches = list(
        build_dataloader(DroidWindowDataset(manifest, source, rows_path, **common), prefetch_factor=1)
    )
    order = epoch_order(2, 5, 0)
    assert [b["episode_index"].item() for b in batches] == order  # worker 0 then worker 1
    resumed = list(
        build_dataloader(
            DroidWindowDataset(manifest, source, rows_path, skip_batches=1, **common), prefetch_factor=1
        )
    )
    assert [b["episode_index"].item() for b in resumed] == order[1:]


def _round_robin(per_worker: list[list[int]]) -> list[int]:
    """The DataLoader's in-order service of workers 0..n-1 until each is exhausted."""
    out, i = [], 0
    while any(per_worker):
        w = i % len(per_worker)
        if per_worker[w]:
            out.append(per_worker[w].pop(0))
        i += 1
    return out


def test_epochs_have_equal_batches_per_rank_and_resume_rotates_workers():
    manifest = {
        "episodes": [{"episode_index": i} for i in range(600)],
        "fps": 15,
        "chunk_size": 32,
        "total_frames": 0,
    }

    def make(**kw):
        return DroidWindowDataset(manifest, "/nowhere", "/nowhere", seed=1, **kw)

    tops = dict(world_size=8, num_workers=4, windows_per_rank=4)
    counts = [sum(len(make(rank=r, **tops).worker_positions(w)) for w in range(4)) for r in range(8)]
    assert len(set(counts)) == 1 and counts[0] == 16 * 4  # 600 // (8*4*4) = 4 batches per worker
    acc = make(rank=0, grad_accumulation=3, **tops)
    assert acc.batches_per_rank == 15 and sum(len(acc.worker_positions(w)) for w in range(4)) == 60
    with pytest.raises(ValueError):
        make(rank=0, world_size=64, num_workers=8, windows_per_rank=4)
    # two workers, one window per batch, seven episodes: six batches, three per worker
    small = {**manifest, "episodes": manifest["episodes"][:7]}
    full = DroidWindowDataset(small, "/nowhere", "/nowhere", seed=1, num_workers=2, windows_per_rank=1)
    order = _round_robin([full.worker_positions(w) for w in range(2)])
    assert len(order) == 6
    for skip in range(1, 6):
        resumed = DroidWindowDataset(
            small, "/nowhere", "/nowhere", seed=1, num_workers=2, windows_per_rank=1, skip_batches=skip
        )
        served = _round_robin([resumed.worker_positions(resumed.logical_worker(p)) for p in range(2)])
        assert served == order[skip:], skip
