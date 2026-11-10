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
"""Cartesian rows and EE-50 statistics next to a ``flux-action index-droid`` index.

``build_cartesian_rows`` reads the end-effector columns of the indexed episodes and writes one float32 array
indexed by the dataset's global frame ``index``: ``rows[i] = state(7) + action(7)``, each
``[x, y, z, rx, ry, rz, gripper_position]`` in DROID's units. The output directory gets its own
``manifest.json`` (the DROID manifest with ``state_dim = action_dim = 7``), the rows and ``ee50_stats.json``,
either the shipped reference statistics or statistics computed from the indexed rows.

    python -m examples.droid_ee50.index --source-root outputs/droid/source \
        --droid-index outputs/droid/index --output-dir outputs/droid_ee50/index
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from flux_action.checkpoints.artifacts import sha256
from flux_action.data.droid.index import ROWS_FILENAME, load_manifest, write_manifest

from . import ee50

ROW_WIDTH = 14
STATS_FILENAME = "ee50_stats.json"
REFERENCE_STATS = Path(__file__).with_name("droid_reference_stats.json")
# The reference statistics count only keep ranges of at least this many frames.
REFERENCE_MIN_RANGE_FRAMES = 73
COLUMNS = [
    "episode_index",
    "index",
    "observation.state.cartesian_position",
    "observation.state.gripper_position",
    "action.cartesian_position",
    "action.gripper_position",
]


def build_cartesian_rows(source_root, manifest: dict, output) -> dict:
    """Write ``(total_frames, 14)`` rows for the manifest's episodes; drop episodes whose rows fail checks.

    Unlisted frames stay NaN. An episode is dropped when its rows are not contiguous in ``index``, a pose is
    not finite or a gripper position lies outside ``[0, 1]``.
    """
    import pyarrow.parquet as pq

    root, output = Path(source_root), Path(output)
    total = int(manifest["total_frames"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(total, ROW_WIDTH))
    rows[:] = np.nan
    by_file: dict[str, list[dict]] = {}
    for episode in manifest["episodes"]:
        by_file.setdefault(episode["data_file"], []).append(episode)
    kept, rejected = [], {}
    for data_file, group in sorted(by_file.items()):
        table = pq.read_table(root / data_file, columns=COLUMNS)
        episode_col = table["episode_index"].to_numpy()
        for episode in group:
            n, start = episode["n_frames"], episode["from_index"]
            part = table.filter(episode_col == episode["episode_index"])
            values, reason = _episode_rows(part, n, start)
            if reason is not None:
                rejected[episode["episode_id"]] = reason
                continue
            rows[start : start + n] = values
            kept.append(episode)
    rows.flush()
    return {"kept": kept, "rejected": rejected}


def _episode_rows(part, n: int, start: int) -> tuple[np.ndarray | None, str | None]:
    if part.num_rows != n:
        return None, f"{part.num_rows} rows for {n} frames"
    if not np.array_equal(part["index"].to_numpy(), np.arange(start, start + n)):
        return None, "non-contiguous index"
    columns = []
    for pose, grip in (
        ("observation.state.cartesian_position", "observation.state.gripper_position"),
        ("action.cartesian_position", "action.gripper_position"),
    ):
        p = np.asarray(part[pose].to_pylist(), np.float32)
        g = np.asarray(part[grip].to_pylist(), np.float32).reshape(-1)
        if p.shape != (n, 6) or g.shape != (n,) or not (np.isfinite(p).all() and np.isfinite(g).all()):
            return None, f"invalid {pose} shape or values"
        if np.any((g < 0) | (g > 1)):
            return None, f"{grip} outside [0, 1]"
        columns += [p, g[:, None]]
    return np.concatenate(columns, axis=1), None


def compute_stats(
    manifest: dict, rows: np.ndarray, max_offset: int, min_range_frames: int
) -> tuple[ee50.EE50Stats, dict]:
    """Statistics over every frame of every keep range with at least ``min_range_frames`` frames."""
    moments = ee50.Moments.empty(max_offset)
    episodes = ranges = 0
    for episode in manifest["episodes"]:
        kept = [(a, b) for a, b in episode["valid_ranges"] if b - a >= min_range_frames]
        if not kept:
            continue
        first = episode["from_index"]
        values = np.asarray(rows[first : first + episode["n_frames"]], dtype=np.float64)
        moments.add(ee50.droid_moments(values[:, :7], values[:, 7:], kept, max_offset))
        episodes, ranges = episodes + 1, ranges + len(kept)
    provenance = {"episodes": episodes, "ranges": ranges, "count": moments.count.tolist()}
    return ee50.finalize(moments), provenance


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-root", required=True, help="directory holding success/ of nvidia/Cosmos3-DROID"
    )
    parser.add_argument("--droid-index", required=True, help="output directory of flux-action index-droid")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stats",
        choices=("reference", "compute"),
        default="reference",
        help="copy the statistics of the reference fine-tune, or compute them from the indexed rows",
    )
    parser.add_argument("--min-range-frames", type=int, default=REFERENCE_MIN_RANGE_FRAMES)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(Path(args.droid_index) / "manifest.json")
    result = build_cartesian_rows(args.source_root, manifest, out / ROWS_FILENAME)
    manifest["episodes"] = result["kept"]
    manifest["counts"]["eligible"] = len(result["kept"])
    manifest["counts"]["cartesian_rows_rejected"] = len(result["rejected"])
    manifest["counts"]["valid_starts_total"] = sum(e["n_valid_starts"] for e in result["kept"])
    manifest["state_dim"] = manifest["action_dim"] = 7
    manifest["rows"] = {"file": ROWS_FILENAME, "sha256": sha256(out / ROWS_FILENAME), "width": ROW_WIDTH}
    if args.stats == "reference":
        shutil.copyfile(REFERENCE_STATS, out / STATS_FILENAME)
    else:
        rows = np.load(out / ROWS_FILENAME, mmap_mode="r")
        stats, provenance = compute_stats(manifest, rows, manifest["chunk_size"], args.min_range_frames)
        ee50.save_stats(
            stats,
            out / STATS_FILENAME,
            source=f"computed from {args.droid_index}",
            min_range_frames=args.min_range_frames,
            **provenance,
        )
    write_manifest(manifest, out / "manifest.json")
    summary = {
        "manifest": str(out / "manifest.json"),
        "stats": str(out / STATS_FILENAME),
        "episodes": len(result["kept"]),
        "rejected": result["rejected"],
    }
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
