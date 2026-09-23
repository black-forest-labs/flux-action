"""Select a seeded valid frame from an indexed LeRobot dataset for offline SO-101 inference.

Index the dataset first with ``flux-action index-lerobot`` using the checkpoint's stream names
(``--camera scene=... --camera wrist=...``); the NPZ keys are ``images.<stream>`` plus ``state``.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from flux_action.data.droid.episodes import select_caption
from flux_action.data.droid.index import ROWS_FILENAME
from flux_action.data.lerobot.index import KIND as LEROBOT_KIND
from flux_action.training.data import WindowDataset, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", required=True, help="dataset directory holding meta/, data/, videos/"
    )
    parser.add_argument("--index-dir", required=True, help="output of flux-action index-lerobot")
    parser.add_argument("--output", required=True, help="NPZ to write; a JSON sidecar is written next to it")
    parser.add_argument("--seed", type=int, default=0, help="selects the episode, window start and caption")
    parser.add_argument("--episode-index", type=int, default=None, help="restrict to one dataset episode")
    parser.add_argument(
        "--split", default=None, help="val or train to restrict to an index split; default: all"
    )
    parser.add_argument("--decoder", default="pyav", help="pyav (default) or ffmpeg")
    parser.add_argument(
        "--frame-hw",
        default="256,256",
        help="frames are rescaled to H,W (the SO-101 canvas uses 256 x 256 views)",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("choose a new observation output path")
    index_dir = Path(args.index_dir)
    if json.loads((index_dir / "manifest.json").read_text()).get("kind") != LEROBOT_KIND:
        raise ValueError(f"{index_dir} is not a LeRobot index; run flux-action index-lerobot")
    manifest = load_manifest(index_dir / "manifest.json")
    h, w = (int(x) for x in args.frame_hw.split(","))
    dataset = WindowDataset(
        manifest,
        args.source_root,
        index_dir / ROWS_FILENAME,
        seed=args.seed,
        frame_hw=(h, w),
        decoder=args.decoder,
        split=args.split,
    )
    positions = list(range(len(dataset.episodes)))
    if args.episode_index is not None:
        positions = [p for p in positions if dataset.episodes[p]["episode_index"] == args.episode_index]
        if not positions:
            raise ValueError(
                f"episode {args.episode_index} is not in the index (or not in split {args.split!r})"
            )
    rng = random.Random(args.seed)
    position = rng.choice(positions)
    episode = dataset.episodes[position]
    starts = dataset.valid_starts(position)
    start = rng.choice(starts)
    task = select_caption(episode["caption"], rng)
    window = dataset.window_at(position, start, task=task)
    observation = {
        f"images.{camera}": window[f"images.{camera}"][0].permute(1, 2, 0).numpy().copy()
        for camera in dataset.cameras
    }
    observation["state"] = window["state"].numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez(stream, **observation)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset_id": manifest.get("dataset_id"),
                "episode_index": episode["episode_index"],
                "frame_index": start,
                "seed": args.seed,
                "cameras": list(dataset.cameras),
                "frame_hw": [h, w],
                "task": task,
            },
            indent=2,
        )
        + "\n"
    )
    shapes = {k: list(v.shape) for k, v in observation.items()}
    print(f"Saved {output} {shapes}; task: {task!r}")


if __name__ == "__main__":
    main()
