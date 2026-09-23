# Prepare a small public DROID sample

Convert episodes 0 and 1 from the [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID)
success subset (compact LeRobot v3), using the [DROID keep-ranges filter](https://huggingface.co/KarlP/droid).
[The sample lock](../configs/droid/public_sample.lock.json) pins revisions and file checksums for
these two episodes. The filter is hosted in a **model** repository.

## Download and prepare

Complete the [shared setup](setup.md), including the data dependencies and FFmpeg. For CPU-only preparation,
`uv sync --locked --extra data --extra encoders` is sufficient; no NATTEN or GPU is needed.

Download the pinned source files with the Hugging Face CLI:

```sh
uv run hf download nvidia/Cosmos3-DROID --repo-type dataset \
  --revision 5c11a20accb11497270a5247a7f1e66ad04c956c \
  --local-dir outputs/public-droid/source \
  --include success/meta/info.json \
  --include success/meta/tasks.parquet \
  --include success/meta/episodes/chunk-000/file-000.parquet \
  --include success/data/chunk-000/file-000.parquet \
  --include success/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4 \
  --include success/videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4 \
  --include success/videos/observation.image.exterior_image_2_left/chunk-000/file-000.mp4
uv run hf download KarlP/droid keep_ranges_1_0_1.json --repo-type model \
  --revision bcb840c3b496533e0adf548a54b51f2f00057837 \
  --local-dir outputs/public-droid/source
```

The download is about 800 MB because source files contain multiple episodes.
Allow extra disk and memory for decoded RGB arrays, then prepare each episode:

```sh
uv run flux-action prepare-droid --source-root outputs/public-droid/source \
  --lock configs/droid/public_sample.lock.json --episode-index 0 \
  --output outputs/public-droid/episode-000000
uv run flux-action prepare-droid --source-root outputs/public-droid/source \
  --lock configs/droid/public_sample.lock.json --episode-index 1 \
  --output outputs/public-droid/episode-000001
uv run flux-action validate-data outputs/public-droid/episode-000000
```

## Data contract

A prepared episode is a directory with four files:

| File | Content |
|---|---|
| `episode.npz` | `wrist`, `left`, `right`: uint8 `(N, 360, 640, 3)` RGB; `timestamps` `(N,)` at 15 Hz; `state`, `action`: float32 `(N, 8)` |
| `metadata.json` | Episode id, caption, `valid_ranges`, fps, camera order, units |
| `manifest.json` | SHA-256 of the two files above, hashes of the inputs, number of valid window starts |
| `source.json` | Hashes of every consumed source file and of the decoded RGB per camera |

State and action are seven joint positions in radians plus the gripper as a closed fraction in
`[0, 1]`, exactly as stored in the source. The policy applies the gripper flip and the
representation scale itself.

| Camera | Source stream |
|---|---|
| Wrist | `wrist_image_left` |
| Left | `exterior_image_1_left` |
| Right | `exterior_image_2_left` |

**How it is enforced.** `prepare-droid` refuses to write anything unless every check passes:

- every source file it reads matches the SHA-256 in the lock file;
- the episode appears exactly once in the metadata, and its row range matches its length;
- `frame_index` and `index` are contiguous, every row carries the caption's `task_index`, and
  the timestamps equal `arange(N) / 15` in float32 bit for bit;
- each camera's start and end timestamps land on whole frames and span exactly `N` frames, and
  `ffmpeg` returns exactly `N` decoded 360x640 frames;
- the keep-ranges filter has an entry for the episode and it allows at least one complete
  33-frame window. A missing or empty entry is an error.

`prepare` (for your own decoded arrays) and `validate-data` run the same array checks: key set,
shapes, dtypes, 15 Hz spacing, gripper range, at least one valid window. `validate-data` also
recomputes the file checksums. Every read of a training window repeats this validation.

**Training windows.** A window is 33 consecutive frames: observations `s:s+33`, state at `s`,
commanded actions `s:s+32`. A keep range `[a, b)` allows the starts `a .. b-33`. Training draws one
start uniformly from all allowed starts each time it visits an episode. Cosmos3-DROID stores several
paraphrases of the instruction in one string separated by `" | "`; a window uses one of them, drawn at
random in training and fixed by the seed in `make_observation.py`. Never pass the joined string to the model. Episode 0 has 471 frames
and 379 allowed starts; episode 1 has 457 frames and 383.

## Create an inference observation

`flux-action infer` needs a single frame, not a window. This command picks one allowed window
start from a prepared episode with a fixed seed and saves that frame's three cameras and state:

```sh
uv run python examples/droid/make_observation.py outputs/public-droid/episode-000000 \
  --seed 0 --output outputs/public-droid/observation.npz
```

The same episode and seed always give the same frame; episode 0 with seed 0 gives frame 232, a
fixed recorded observation. The JSON written next to the NPZ records the episode,
frame index, seed and caption. Pass that caption unchanged as `--task` to
[`flux-action infer`](setup.md#inference-from-an-existing-export).

## Bringing your own data

If you already have decoded, frame-aligned DROID arrays and know the allowed ranges, skip the
download and run:

```sh
uv run flux-action prepare --metadata metadata.json --arrays arrays.npz --output episode
```

`arrays.npz` must hold the six arrays from the table above and `metadata.json` must follow
`EpisodeMetadata`. [`examples/droid/make_fixture.py`](../examples/droid/make_fixture.py) generates
synthetic inputs for this path. It does not download anything or compute valid ranges for you.

The one-directory-per-episode format stores decoded RGB and is meant for a few episodes and for
inference observations. Training reads the whole dataset directly from the compact files.

## Index the whole dataset for training

Training does not convert episodes. Download the complete `success` split and the filter, then
build the index once:

```sh
uv run hf download nvidia/Cosmos3-DROID --repo-type dataset \
  --revision 5c11a20accb11497270a5247a7f1e66ad04c956c \
  --local-dir outputs/droid/source --include "success/*"
uv run hf download KarlP/droid keep_ranges_1_0_1.json --repo-type model \
  --revision bcb840c3b496533e0adf548a54b51f2f00057837 --local-dir outputs/droid/source
uv run flux-action index-droid --source-root outputs/droid/source \
  --filter outputs/droid/source/keep_ranges_1_0_1.json --output-dir outputs/droid/index \
  --revision 5c11a20accb11497270a5247a7f1e66ad04c956c
```

`index-droid` writes two files:

| File | Content |
|---|---|
| `manifest.json` | Every eligible episode: id, length, caption, keep ranges, number of allowed starts, data file and global row offset, video file and first frame per camera; counts of listed, eligible, filtered and misaligned episodes; the referenced files with sizes (`--hash-files` adds SHA-256) |
| `rows.f32.npy` | float32 `(total_frames, 16)`: state (7 joints, gripper) and action (7 joints, gripper) for every frame at its global `index`; rows of excluded episodes are NaN |

Eligibility and validation are the ones `prepare-droid` applies per episode: a keep-ranges entry
with at least one complete 33-frame window, camera streams that start and end on whole frames
and span the episode, contiguous `frame_index` and `index`, float32 timestamps on the 15 Hz grid,
one `task_index` matching the caption, finite values and a gripper in `[0, 1]`. Episodes that fail
are counted and left out; `--only-available` additionally drops episodes whose files are not on
disk (partial downloads).

**Reading windows.** A training window is three video seeks plus one slice of the rows array.
Videos are AV1 with a keyframe every second frame, so seeking is cheap. The default decoder is
the `ffmpeg` binary with an accurate input seek; its RGB conversion is the one `prepare-droid`
uses, so streamed windows equal the prepared episodes byte for byte on the same machine. The
in-process `pyav` decoder (`decoder="pyav"`) is faster and needs no subprocess, but its bundled
libswscale may differ from the system binary and move RGB values by up to a few levels. Frames
stay uint8 until the policy scales them on the accelerator.

**Order and resume.** Episodes are shuffled once per epoch from the run seed and split across
ranks and loader workers; the window start and caption paraphrase of an episode visit are drawn
from the run seed, the epoch and the episode index alone, so the windows of an epoch do not depend
on the number of ranks or workers. Each rank records how many batches it consumed; an exact resume
skips them (same world size, worker count and windows per rank), a reseeded resume starts a fresh
epoch order. `examples/droid/bench_decode.py` measures throughput on an indexed dataset.

LeRobot datasets (v2.1 or v3.0) are indexed the same way by `flux-action index-lerobot`, which adds the
normalization statistics needed for standalone training; see the
[optional full fine-tuning section in the SO-101 LoRA guide](so101-lora.md#optional-full-fine-tuning).
