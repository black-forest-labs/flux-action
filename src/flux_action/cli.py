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
"""Commands for data preparation and indexing, training, checkpoint export and offline inference."""

import argparse
import json

from .data.droid.episodes import prepare_episode, validate_episode


def _parse_settings(items: list[str]) -> dict:
    """``KEY=VALUE`` overrides, VALUE as JSON when it parses and as a plain string otherwise.

    The fallback is what lets ``--setting sampler=euler`` work alongside ``--setting guidance_scale=3.0``;
    ``TrainConfig.from_file`` reads ``--override`` the same way.
    """
    settings = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--setting {item!r} is not KEY=VALUE")
        try:
            settings[key] = json.loads(raw)
        except json.JSONDecodeError:
            settings[key] = raw
    return settings


def _add_checkpoint_arguments(parser) -> None:
    parser.add_argument("--checkpoint", required=True, help="local policy directory or Hugging Face repo ID")
    parser.add_argument("--revision", help="Hugging Face commit, tag or branch")
    parser.add_argument("--subfolder", help="policy package within the repository, e.g. variants/gd-fp8r")


def _add_precision_arguments(parser, *, compile_default: bool) -> None:
    """Serve-time precision knobs shared by ``infer`` and ``serve-robolab`` (see inference/precision.py)."""
    parser.add_argument(
        "--compile-dit",
        action=argparse.BooleanOptionalAction,
        default=compile_default,
        help="torch.compile prepared DiT inference; the first call includes compilation",
    )
    parser.add_argument(
        "--offload-text-encoder",
        action="store_true",
        help="keep the text encoder on the CPU between new captions (frees about 9 GB of GPU memory)",
    )


def main():
    parser = argparse.ArgumentParser(
        description="FLUX Action: DROID preparation, checkpoint export and offline inference"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate/package decoded arrays and supplied valid ranges")
    prepare.add_argument("--metadata", required=True)
    prepare.add_argument("--arrays", required=True)
    prepare.add_argument("--output", required=True)
    validate = commands.add_parser("validate-data", help="verify an intermediate episode")
    validate.add_argument("directory")
    cosmos = commands.add_parser("prepare-droid", help="convert a pinned local Cosmos3-DROID episode")
    cosmos.add_argument("--source-root", required=True)
    cosmos.add_argument("--lock", required=True)
    cosmos.add_argument("--episode-index", required=True, type=int)
    cosmos.add_argument("--output", required=True)
    infer = commands.add_parser("infer", help="record offline inference from a policy package")
    _add_checkpoint_arguments(infer)
    infer.add_argument("--observation", required=True)
    infer.add_argument("--task", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="cuda")
    infer.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="inference override for exports without saved sampling settings, e.g. sampler=euler; "
        "VALUE is JSON when it parses, otherwise a plain string",
    )
    _add_precision_arguments(infer, compile_default=False)
    index = commands.add_parser("index-droid", help="build the whole-dataset manifest and rows array")
    index.add_argument(
        "--source-root", required=True, help="directory holding success/ of nvidia/Cosmos3-DROID"
    )
    index.add_argument("--filter", required=True, help="keep_ranges json from KarlP/droid")
    index.add_argument("--output-dir", required=True)
    index.add_argument("--revision", default=None, help="dataset commit hash to record")
    index.add_argument("--only-available", action="store_true", help="keep only episodes whose files exist")
    index.add_argument("--hash-files", action="store_true", help="record SHA-256 of every referenced file")
    lerobot = commands.add_parser(
        "index-lerobot",
        help="index a LeRobot dataset (v2.1 or v3.0) for training: manifest, rows array, normalization statistics",
    )
    lerobot.add_argument(
        "--source-root", required=True, help="dataset directory holding meta/, data/, videos/"
    )
    lerobot.add_argument("--output-dir", required=True)
    lerobot.add_argument(
        "--camera",
        action="append",
        required=True,
        metavar="STREAM=FEATURE",
        help="policy stream and dataset video feature, in layout order, e.g. top=observation.images.top "
        "wrist=observation.images.front (the batch key is images.<STREAM>)",
    )
    lerobot.add_argument("--state-key", default="observation.state")
    lerobot.add_argument("--action-key", default="action")
    lerobot.add_argument(
        "--action-parameterization",
        default="joint_delta",
        choices=("absolute", "joint_delta"),
        help="targets the statistics describe: the commands, or per-frame deltas (the SO-101 recipe)",
    )
    lerobot.add_argument(
        "--absolute-action-dims",
        default="-1",
        help="comma-separated channels kept absolute under joint_delta (default: the last, the gripper); "
        "'' for none",
    )
    lerobot.add_argument("--normalization-clip", type=float, default=6.0)
    lerobot.add_argument("--chunk-size", type=int, default=32)
    lerobot.add_argument(
        "--val-episodes", type=int, default=0, help="hold out the last N eligible episodes (split: val)"
    )
    lerobot.add_argument(
        "--strip-dead-windows",
        type=float,
        default=None,
        metavar="FRACTION",
        help="drop static episodes and windows whose frames are dead beyond this fraction (the recipe: 0.6)",
    )
    lerobot.add_argument("--dataset-id", default=None, help="Hub repo id to record")
    lerobot.add_argument("--revision", default=None, help="dataset commit hash to record")
    lerobot.add_argument("--only-available", action="store_true", help="keep only episodes whose files exist")
    lerobot.add_argument("--hash-files", action="store_true", help="record SHA-256 of every referenced file")
    evaluate = commands.add_parser(
        "evaluate", help="score a policy export on the held-out windows of an indexed LeRobot dataset"
    )
    _add_checkpoint_arguments(evaluate)
    evaluate.add_argument("--source-root", required=True)
    evaluate.add_argument("--index-dir", required=True)
    evaluate.add_argument("--split", default="val", help="val (default), train, or all")
    evaluate.add_argument("--windows-per-episode", type=int, default=4)
    evaluate.add_argument("--max-windows", type=int, default=None)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--decoder", default="pyav")
    evaluate.add_argument("--frame-hw", default="256,256", help="frames are rescaled to H,W before tiling")
    evaluate.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="inference override; VALUE is JSON when it parses, otherwise a plain string",
    )
    evaluate.add_argument("--output", default=None, help="report JSON (per-window records included)")
    train = commands.add_parser(
        "train", help="run a finetune over an indexed dataset (single process or under torchrun)"
    )
    train.add_argument(
        "--config",
        required=True,
        help="training JSON, see configs/droid/train.json and configs/so101/train.json",
    )
    train.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field (JSON values)",
    )
    serve = commands.add_parser(
        "serve-robolab", help="serve a policy export to RoboLab over the OpenPI protocol"
    )
    _add_checkpoint_arguments(serve)
    serve.add_argument(
        "--dtype",
        choices=("bfloat16", "float32", "keep"),
        default="bfloat16",
        help="DiT serving dtype; cast on CPU before GPU placement (default: bfloat16)",
    )
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", default="cuda")
    serve.add_argument("--log-dir", default=None, help="record every query (npz + json) here")
    serve.add_argument(
        "--setting", action="append", default=[], metavar="KEY=VALUE", help="inference override"
    )
    _add_precision_arguments(serve, compile_default=True)
    serve.add_argument(
        "--warmup", type=int, default=1, help="requests answered before listening (compiles the graphs)"
    )
    replay = commands.add_parser(
        "replay-fixtures", help="replay a recorded fixture collection through the adapter"
    )
    _add_checkpoint_arguments(replay)
    replay.add_argument("--collection", required=True, help="collection.json")
    replay.add_argument("--output", required=True)
    replay.add_argument("--device", default="cuda")
    replay.add_argument(
        "--dtype",
        choices=("bfloat16", "float32", "keep"),
        default="keep",
        help="DiT replay dtype; select bfloat16 to match the server default (default: keep)",
    )
    replay.add_argument("--atol", type=float, default=1e-3)
    replay.add_argument("--setting", action="append", default=[], metavar="KEY=VALUE")
    export = commands.add_parser(
        "export-checkpoint", help="write an inference export from a training checkpoint's model or EMA"
    )
    export.add_argument("--checkpoint", required=True, help="a complete <root>/step-<N> directory")
    export.add_argument("--output", required=True)
    export.add_argument("--profile", default="model", help="model, ema_0p10 or ema_0p05")
    export.add_argument(
        "--dtype",
        default="keep",
        choices=("keep", "bfloat16", "float32"),
        help="export weight dtype: keep (the checkpoint's fp32 masters, default) | bfloat16 (for serving, about 7x faster plans) | float32",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_episode(args.metadata, args.arrays, args.output)
    elif args.command == "validate-data":
        result = validate_episode(args.directory).to_dict()
    elif args.command == "prepare-droid":
        from .data.droid.cosmos import convert_episode

        result = convert_episode(args.source_root, args.lock, args.episode_index, args.output)
    elif args.command == "index-droid":
        from pathlib import Path

        from .data.droid.index import ROWS_FILENAME, build_manifest, build_rows, write_manifest

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(
            args.source_root,
            args.filter,
            only_available=args.only_available,
            hash_files=args.hash_files,
            revision=args.revision,
        )
        rows = build_rows(args.source_root, manifest, out / ROWS_FILENAME)
        write_manifest(manifest, out / "manifest.json")
        result = {
            "manifest": str(out / "manifest.json"),
            "counts": manifest["counts"],
            "rows_rejected": rows["rejected"],
        }
    elif args.command == "index-lerobot":
        from .data.lerobot.index import index_dataset

        cameras = {}
        for item in args.camera:
            stream, _, feature = item.partition("=")
            if not _ or not stream or not feature:
                raise SystemExit(f"--camera {item!r} is not STREAM=FEATURE")
            cameras[stream] = feature
        dims = [int(d) for d in args.absolute_action_dims.split(",") if d.strip()]
        result = index_dataset(
            args.source_root,
            args.output_dir,
            cameras,
            action_parameterization=args.action_parameterization,
            absolute_action_dims=dims if args.action_parameterization == "joint_delta" else (),
            clip=args.normalization_clip,
            strip_dead_windows=args.strip_dead_windows,
            state_key=args.state_key,
            action_key=args.action_key,
            chunk_size=args.chunk_size,
            val_episodes=args.val_episodes,
            dataset_id=args.dataset_id,
            revision=args.revision,
            only_available=args.only_available,
            hash_files=args.hash_files,
        )
    elif args.command == "evaluate":
        from .inference.evaluate import evaluate_export

        settings = _parse_settings(args.setting)
        h, w = (int(x) for x in args.frame_hw.split(","))
        report = evaluate_export(
            args.checkpoint,
            args.source_root,
            args.index_dir,
            output=args.output,
            device=args.device,
            revision=args.revision,
            subfolder=args.subfolder,
            settings=settings,
            split=None if args.split == "all" else args.split,
            windows_per_episode=args.windows_per_episode,
            max_windows=args.max_windows,
            decoder=args.decoder,
            frame_hw=(h, w),
            seed=args.seed,
        )
        result = {k: v for k, v in report.items() if k != "windows"}
    elif args.command == "train":
        from .training.trainer import TrainConfig, Trainer

        result = Trainer(TrainConfig.from_file(args.config, args.override)).run()
    elif args.command in ("serve-robolab", "replay-fixtures"):
        from .serving.robolab import RoboLabPolicy, load_serving_policy, replay_collection, serve

        settings = _parse_settings(args.setting)
        precision = dict(dtype=args.dtype, compile_dit=False, warmup=0)  # replay compares eager actions
        if args.command == "serve-robolab":
            precision = dict(
                dtype=args.dtype,
                compile_dit=args.compile_dit,
                offload_text_encoder=args.offload_text_encoder,
                warmup=args.warmup,
            )
        policy = load_serving_policy(
            args.checkpoint,
            revision=args.revision,
            subfolder=args.subfolder,
            device=args.device,
            settings=settings,
            **precision,
        )
        if args.command == "serve-robolab":
            adapter = RoboLabPolicy(policy, log_dir=args.log_dir)
            print(
                json.dumps(
                    {
                        "serving": args.checkpoint,
                        "host": args.host,
                        "port": args.port,
                        "serving_setup": policy.serving_setup,
                        **policy.config.to_dict(),
                    }
                )
            )
            serve(adapter, host=args.host, port=args.port, metadata={"serving_setup": policy.serving_setup})
            return
        result = replay_collection(args.collection, RoboLabPolicy(policy), args.output, atol=args.atol)
    elif args.command == "export-checkpoint":
        from .training.checkpoint import export_policy

        result = export_policy(
            args.checkpoint,
            args.output,
            profile=args.profile,
            dtype=None if args.dtype == "keep" else args.dtype,
        )
    else:
        from .inference.offline import run_inference

        settings = _parse_settings(args.setting)
        result = run_inference(
            args.checkpoint,
            args.observation,
            args.output,
            task=args.task,
            revision=args.revision,
            subfolder=args.subfolder,
            device=args.device,
            compile_dit=args.compile_dit,
            offload_text_encoder=args.offload_text_encoder,
            settings=settings,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
