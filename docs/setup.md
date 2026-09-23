# Setup & first inference

Install once for standalone inference and full fine-tuning. The first prediction below uses the
released DROID checkpoint and a recorded observation; no robot is required.
For SO-101 task LoRA, use the [LeRobot workflow](so101-lora.md) and its separate environment.

Inference and training require Linux and an NVIDIA GPU. Data preparation and tests also run on CPU.

## Requirements

| Component | Example environment |
|---|---|
| GPU / driver | NVIDIA H200, CUDA 12.8 |
| Python | 3.12 |
| torch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| transformers | 5.16.1 |
| NATTEN | 0.21.6 (`+torch2100cu128` wheel) |

Match the NATTEN wheel to your architecture, torch and CUDA versions. Install FFmpeg with an AV1
decoder for the recorded-data example and training; check that `ffmpeg -version` runs.

## Install

With `uv` installed, clone the repository and install dependencies:

```sh
git clone https://github.com/black-forest-labs/flux-action.git
cd flux-action
uv sync --locked --extra encoders --extra data
uv pip install --python .venv/bin/python 'natten==0.21.6+torch2100cu128' -f https://whl.natten.org/
```

- `--extra encoders` installs `transformers` and `huggingface-hub` for the Qwen3-VL-4B text encoder
  and for downloading weights from the Hub. Inference requires it.
- `--extra data` installs `pyarrow` and PyAV for reading Cosmos3-DROID parquet and video files. Skip it if
  you only run inference. Preparing episodes and training additionally use FFmpeg with an AV1 decoder; see
  [Data preparation](prepare.md).
- `--extra serve` installs `websockets` and `msgpack` for [serving to RoboLab](#serve-to-robolab).
- NATTEN is not part of `uv.lock` because its wheels are built per torch and CUDA version. Pick the
  wheel from [whl.natten.org](https://whl.natten.org/) whose suffix matches your installed torch
  (`torch2100cu128` for torch 2.10.0 with CUDA 12.8). The video VAE uses neighborhood attention and
  imports NATTEN when it is loaded, so every GPU workflow needs it.

If you run `uv sync` again, reinstall NATTEN afterwards: it is installed separately from the lockfile.
Run the commands below from the repository root. Keep generated files under ignored `outputs/`.

## Choose a checkpoint

Weights are grouped in the [FLUX 3 Action collection](https://huggingface.co/collections/black-forest-labs/flux-3-action).
Authenticate with `uv run hf auth login` if your account needs to grant repository access.

| Repository | Contents |
|---|---|
| [`black-forest-labs/flux-3-action-droid`](https://huggingface.co/black-forest-labs/flux-3-action-droid) | DROID policy at the root; optimized packages under `variants/` |
| [`black-forest-labs/flux-3-action-so101`](https://huggingface.co/black-forest-labs/flux-3-action-so101) | SO-101 history policy, saved processors and normalization at the root |
| [`black-forest-labs/flux-3-action-base`](https://huggingface.co/black-forest-labs/flux-3-action-base) | Action-pretrained trunk, shared video VAE and Qwen text encoder |

Each robot config pins its shared encoders to an immutable base-repository revision.
Standalone inference downloads those components into the Hugging Face cache automatically;
you do not need a separate encoder copy for each robot. The examples use the latest `main`.
Keep each package's config, model and saved processors together, including its encoder references.

| Checkpoint | How to use it |
|---|---|
| Released DROID policy | Follow the first-inference example below; no LeRobot installation needed |
| Released SO-101 policy | Use the [standalone SO-101 example](#so-101-inference); no LeRobot installation needed |
| LeRobot PEFT adapter | Use the [SO-101 LoRA guide](so101-lora.md) and the package’s loader |
| Standalone policy export | Use `flux-action infer` with the export’s saved camera and action conventions; see [existing exports](#inference-from-an-existing-export) |
| Action-pretrained base | Start a full fine-tune; target-embodiment heads are initialized during training |

These formats have different loaders. A resumable `step-N` training checkpoint must first be
converted with `flux-action export-checkpoint`.

## First inference

Download the DROID root package. Frozen encoders resolve automatically from the pinned base repo:

```sh
uv run hf download black-forest-labs/flux-3-action-droid \
  --exclude 'variants/*' --local-dir outputs/droid
```

[Download and prepare the small public DROID sample](prepare.md#download-and-prepare),
then select one recorded observation:

```sh
uv run python examples/droid/make_observation.py outputs/public-droid/episode-000000 \
  --seed 0 --output outputs/public-droid/observation.npz
```

Read its task caption from the JSON sidecar and predict an action chunk:

```sh
task_caption=$(uv run python -c 'import json; print(json.load(open("outputs/public-droid/observation.json"))["task"])')
uv run flux-action infer --checkpoint outputs/droid \
  --observation outputs/public-droid/observation.npz --task "$task_caption" \
  --output outputs/droid/inference-run
```

The output directory contains `actions.npy` with shape `(1, 32, 8)` and `report.json`
with the effective settings and timing. The float32 array contains
32 actions, each with seven absolute joint targets in radians and a gripper closed fraction. This saves predictions;
it does not execute them on a robot. Use a new output directory when repeating the command.

Eager runs like this one are reproducible. `--compile-dit` is faster, but separate compiled runs can
differ by about 0.015 rad, so use eager runs to compare results or record fixtures.

For your own DROID observation, provide synchronized uint8 `(360, 640, 3)` RGB `images.wrist`, `images.left`, `images.right`
and float32 `state` (8 values) in the NPZ. Pass one task instruction literally; for comparisons
with LeRobot’s Cosmos caption handling, use the first ` | `-separated alternative.
See [advanced inference](#advanced-inference) for Python inputs, camera layouts and serving.

The CLI also accepts a repo ID directly:

```sh
uv run flux-action infer \
  --checkpoint black-forest-labs/flux-3-action-droid \
  --observation outputs/public-droid/observation.npz --task "$task_caption" \
  --output outputs/droid/hub-inference-run
```

`FluxActionPolicy.from_pretrained(repo_id, revision=..., subfolder=...)`
and `flux-action infer`, `evaluate`, `serve-robolab`, and `replay-fixtures` select one
package with `--revision` and `--subfolder`. The CLI accepts the repo ID through
`--checkpoint`; local directories also support `--subfolder`.

| Recipe | BF16 subfolder | FP8r subfolder |
|---|---|---|
| Base: 4 steps with guidance | root (omit subfolder) | `variants/fp8r` |
| Guidance-distilled: 4 steps | `variants/gd` | `variants/gd-fp8r` |
| Step-distilled: 1 step | `variants/sd` | `variants/sd-fp8r` |

The selected package supplies its precision and sampler settings. Standalone loading
reads `config.native.json` when present, otherwise `config.json`, and verifies that
config and the weights against the package manifest. BF16 packages retain a separate
LeRobot `config.json`; FP8r packages store their native config in `config.json`.

For example, serve the guidance-distilled FP8r package:

```sh
uv run --extra serve flux-action serve-robolab \
  --checkpoint black-forest-labs/flux-3-action-droid \
  --subfolder variants/gd-fp8r
```

To run the same package on a recorded observation and save actions plus a report:

```sh
uv run flux-action infer \
  --checkpoint black-forest-labs/flux-3-action-droid \
  --subfolder variants/gd-fp8r \
  --observation outputs/public-droid/observation.npz --task "$task_caption" \
  --output outputs/gd-fp8r
```

Use a new output directory for each run. The report records the selected package's
manifest and effective settings. Omit the acceleration flags for eager offline inference.

## SO-101 inference

Download the SO-101 policy and saved processors. Its config resolves the shared encoders:

```sh
uv run hf download black-forest-labs/flux-3-action-so101 \
  --local-dir outputs/so101/weights

uv run --extra encoders python examples/so101/inference.py \
  --weights outputs/so101/weights \
  --observation observation.npz --prompt 'place the box in the container'
```

The NPZ contains `images.scene` and `images.wrist` as HWC uint8 RGB arrays, plus
`state` as six float32 values in the delivered joint order: shoulder pan,
shoulder lift, elbow flex, wrist flex, wrist roll, gripper. Arm joints use degrees;
the gripper uses percentage points. This checkpoint predicts `(1, 42, 6)` absolute
commands in those units, with no simulator-specific clipping.

Without a robot, take the observation from a recorded SO-101 LeRobot dataset. Index it with the
checkpoint's stream names (`scene`, `wrist`) as in the
[SO-101 full fine-tuning section](so101-lora.md#optional-full-fine-tuning), then pick one frame:

```sh
uv run python examples/so101/make_observation.py --source-root outputs/so101/source \
  --index-dir outputs/so101/index --seed 0 --output outputs/so101/observation.npz
```

The script writes the two camera frames and the state at a seeded valid window start, plus a JSON
sidecar with the episode, frame and task caption to pass as `--prompt`. The same seed always
selects the same frame.

The loader reads observation history, camera order, action horizon, sampling settings
and normalization from the package. The released checkpoint uses eight observation ticks,
two independently encoded visual snapshots and past-command/state conditioning. Keep
its config, processor JSONs and quantile files together; inconsistent packages are rejected.

For a control loop, import `load_policy` from `flux_action.inference.so101`, load once,
and call `policy.select_action(batch)` at **every** 30 Hz tick, including ticks served
from the queue. It records each measured observation and the previously issued command,
executes 32 actions, discards the remaining ten predictions, then replans. Call
`policy.reset()` at each episode boundary or intervention.

The SO-101 example also accepts `--weights black-forest-labs/flux-3-action-so101` directly.
The equivalent Python load is:

```python
from flux_action.inference.so101 import load_policy

policy = load_policy("black-forest-labs/flux-3-action-so101").to("cuda")
```

The repo-ID loaders use the latest `main` when `revision` is omitted. Pin a commit for
comparisons and reproducible runs; `revision` applies only to Hub repo IDs, not local paths.

`predict_action_chunk(batch)` is stateless: a single observation represents a new episode,
with history padded from its initial image/state. For an offline history window, supply
`state` and `command_history` as `(B, n_obs_steps, 6)` and each camera as
`(B, n_obs_steps, 3, H, W)`; commands are absolute values issued before each observation.
Repeated stateless predictions do not replace the queued control loop.

Pass uint8 cameras, or float cameras normalized with float32 division by 255 **on CPU**.
GPU normalization can produce different BF16 VAE inputs. The NPZ loader preserves uint8.

The [standalone full-finetuning recipe](so101-lora.md#optional-full-fine-tuning) uses this same
history conditioning. Its exports retain their history settings and dataset normalization,
and load through `FluxActionPolicy.from_pretrained`.
LeRobot is not required for standalone inference.

## Inference from an existing export

An export contains `config.json`, `model.safetensors` and `manifest.json`. Keep its referenced
video VAE and text encoder available and supply an observation matching its camera keys,
state dimensions and units:

```sh
uv run flux-action infer --checkpoint path/to/policy_export \
  --observation observation.npz --task "your task instruction" \
  --output outputs/inference-001 --device cuda
```

The output directory must be new. It receives `actions.npy` and `report.json` (configuration,
checksums, runtime versions, timing and GPU memory); action dimensions
and sampling settings come from the export. Training exports carry no sampling settings: either
apply and save them once with the [Python API](#python-api) (DROID has a preset), or pass them
per run, for example for the SO-101 recipe:

```sh
uv run flux-action infer --checkpoint outputs/so101/export \
  --observation outputs/so101/observation.npz --task "your task instruction" \
  --output outputs/so101/inference-001 --device cuda \
  --setting sampler=euler --setting num_inference_steps=4 \
  --setting sampler_shift=6.93 --setting guidance_scale=3.0
```

The overrides are recorded in `report.json`. Other embodiments must retain their own settings.

## Download for full fine-tuning

For full fine-tuning, download the base and shared encoders:

```sh
uv run hf download black-forest-labs/flux-3-action-base \
  --include 'flux-3-action-base.safetensors' --include 'video_vae.safetensors' \
  --include 'text_encoder/*' \
  --local-dir outputs/weights
```

Both [`configs/droid/train.json`](../configs/droid/train.json) and
[`configs/so101/train.json`](../configs/so101/train.json) use these local paths:

```json
{
  "trunk_weights": "outputs/weights/flux-3-action-base.safetensors",
  "video_vae_id": "outputs/weights/video_vae.safetensors",
  "text_encoder_id": "outputs/weights/text_encoder"
}
```

Keep the downloaded weights fixed for a run. For an existing run, retain the original weight and encoder
paths in its configuration. Local directories avoid downloading unrelated robot packages and
let all components resolve from the same snapshot.

The DROID and SO-101 repositories contain finetuned policies at their roots.
Their configs reference the shared encoders in the base repository. Standalone exports
load with `FluxActionPolicy.from_pretrained`; use `export-checkpoint --dtype bfloat16` for a bf16
inference export (export otherwise preserves the checkpoint dtype).

## Next steps

- [DROID full fine-tuning](droid-finetune.md): reproduce the full training recipe.
- [SO-101 LoRA](so101-lora.md): adapt a prepared policy using LeRobot.
- [Custom tasks and embodiments](embodiments.md): train on your own observations and actions, with a gaming example.

## Advanced inference

### Python API

Load a released DROID package, place it on the GPU, then prepare it for inference.
The package selects precision and sampling settings; changing `subfolder` is sufficient
to switch between the six released variants.

```python
import torch
from flux_action.policy import FluxActionPolicy

policy = FluxActionPolicy.from_pretrained(
    "black-forest-labs/flux-3-action-droid",
    subfolder="variants/gd-fp8r",
    device="cuda",
)
policy.prepare_inference(compile=True)
observation = {
    "images.wrist": wrist_rgb,  # (B, 3, 360, 640), float RGB in [0, 1], on CUDA
    "images.left": left_rgb,
    "images.right": right_rgb,
    "state": joint_state,      # (B, 8): seven joints in radians + gripper closed fraction
    "task": ["task caption"],   # one caption per batch item
}
with torch.no_grad():
    actions = policy.predict_action_chunk(observation)  # float32 (B, 32, 8)
```

Inputs must share a batch, device and observation time. Move the policy before calling
`prepare_inference()`; preparation is an irreversible serving conversion. The first prediction
compiles the DiT and video VAE encoder in `reduce-overhead` mode. Repeated prompts
reuse encoded and prepared text. Every request encodes fresh camera observations.
Inference uses `encode_frame()` instead of repeating the observation into a 45-frame video.
`prepare_inference()` selects this path even for older packages with `single_frame_encode=False`.
Training continues to encode full video windows.
Call `prepare_inference(compile=False)` for eager prepared inference.

Training uses the reference transformer through `flux-action train`; do not prepare a policy
that you intend to train. Reference BF16 evaluation can also omit preparation.

For a BF16 DROID **training export** that needs the base inference preset, apply and save
that preset before preparation. Do not apply it to released GD/SD packages, which already
contain their own sampler settings:

```python
from flux_action.config import DROID_INFERENCE_SETTINGS

policy = FluxActionPolicy.from_pretrained("path/to/policy_export")
for name, value in DROID_INFERENCE_SETTINGS.items():
    setattr(policy.config, name, value)
policy.config.validate_inference()
policy.save_pretrained("outputs/droid/inference-export")
```

The current DROID training recipe uses `action_prediction_droid`, matching the prepared
backend. Existing exports with `action_modality="action"` require reference inference:
use `--no-compile-dit` with `serve-robolab` (and omit `--compile-dit` with `infer`).
Keep existing runs and exports in their saved namespace; changing the sampler preset
does not rename their weights.

Exports contain the DiT and configuration with verified checksums; frozen encoders remain
external references and are not checksummed. They contain no training state.

`predict_action_chunk` produces absolute commands in dataset units. For joint-delta policies,
it integrates predicted deltas onto measured state. `select_action` queues `n_action_steps`
commands before replanning and continues joint deltas from the last returned command.
Call `reset` at episode boundaries, interventions or task/batch changes. The application owns
control timing and execution; RTC arguments are unsupported.

### Camera layouts

Keep the checkpoint’s `camera_keys`, layout and canvas. Keys refer to separate input streams;
do not pre-tile them for the Python API.

| Layout | Input order and composition |
|---|---|
| `droid` | Wrist, left exterior, right exterior; wrist above half-size exterior views, padded to 544 × 736 |
| `single` | One view resized to the canvas |
| `side_by_side` | Scene left, wrist right; each resized to half the canvas width |
| `grid` | Views in row-major order; unused cells black |

See [SO-101 camera mapping](so101-lora.md#so-101-camera-keys-and-tiling) for its physical roles.
`grid` supports custom layouts but requires training for that layout. Use the camera arrangement
expected by your checkpoint.

### Reference DROID inference settings

The DROID root package saves these reference settings. The CLI and RoboLab server preserve
the selected package's settings unless you supply explicit overrides.
For DROID training exports, apply and save them using the Python example above.

| Setting | Value |
|---|---|
| `sampler` | `cosmos_unipc` (order 2) |
| `num_inference_steps` | 4 |
| `sampler_shift` | 5.0 |
| `guidance_scale` / `guidance_scale_action` | 4.0 / 1.0 |
| `n_action_steps` | 32 |

Video tokens are sampled jointly with actions but are not decoded. These are DROID settings;
SO-101 and custom tasks retain their own checkpoint settings.

### Serve to RoboLab

Serve the downloaded DROID policy:

```sh
uv run --extra serve flux-action serve-robolab --checkpoint outputs/droid --port 8000
```

Or serve a standalone DROID export with query logging:

```sh
uv run --extra serve flux-action serve-robolab --checkpoint outputs/droid/inference-export \
  --port 8000 --log-dir outputs/robolab-logs --no-compile-dit
```

`--extra serve` adds WebSocket dependencies while retaining NATTEN. The OpenPI WebSocket
server sends metadata on connection and accepts msgpack requests with NumPy arrays:

| Request key | Content |
|---|---|
| `observation/image` | uint8 `(540, 640, 3)` composite: wrist above two half-size exterior views |
| `observation/joint_position` | 7 joint positions in radians (or a history whose last row is used) |
| `observation/gripper_position` | Gripper closed fraction, as in DROID data; scalar or history |
| `prompt` | Task instruction |

Instead of the composite, pass `observation/wrist_image_left`,
`observation/exterior_image_1_left` and `observation/exterior_image_2_left`.
The response contains `action` (float32 `(32, 8)`) and `server_timing`.
The export-based server uses the package sampler settings; use `--setting key=value` to override them.

`--log-dir` saves observations and JSON containing prompts, seeds, actions and timings.
Replay them with `flux-action replay-fixtures --checkpoint <export> --collection <log-dir>
--output <new-dir> --dtype bfloat16`. Replay is eager; compiled recordings may require a suitable
`--atol`. See `uv run flux-action replay-fixtures --help` for options.

### Attention backends

For the reference transformer used in training and unprepared inference,
`PolicyConfig.attn_mode` selects `torch` (automatic PyTorch SDPA, default), `cudnn` (cuDNN
with PyTorch fallbacks), or `flash` (separately installed flash-attn 4, 3 or 2).
Prepared BF16 and FP8r inference always use PyTorch SDPA, regardless of `attn_mode`.
Kernel changes can produce rounding differences. Training packs windows while keeping their
attention separate; see [training speed](droid-finetune.md#training-speed).
The video VAE automatically selects its NATTEN backend; `F3_NATTEN_BACKEND` overrides it.

### Precision

The selected package's `quantization` field chooses BF16 or prequantized FP8r weights;
there is no runtime quantization flag. FP8r quantizes linear inputs rowwise for GEMMs,
while attention and boundary operations stay BF16. Sampler state and timesteps stay
float32. Use `bfloat16` or `keep` for FP8r serving dtype.
For BF16 exports, the server casts DiT weights on CPU before GPU placement; `--dtype keep`
preserves the export dtype. Offline inference preserves export precision by default,
so export with `--dtype bfloat16` for bf16 inference. Frozen encoders retain their own precision.

### Serving speed and memory

These options apply to `serve-robolab` and `infer`, leaving the export unchanged:

| Option | Effect |
|---|---|
| `--compile-dit` / `--no-compile-dit` | Compile the prepared DiT and video VAE encoder; on by default for serving, off offline. Compiled results are not bit-identical across processes |
| `--offload-text-encoder` | Keep the text encoder on CPU between caption-cache misses |

Serving warms one request before listening (`--warmup 1`); a new prompt length bucket
may compile again. Stable prompt geometry reuses compiled programs across solver
steps and requests. Offline timings with `--compile-dit` include compilation.
Python `prepare_inference()` enables compilation by default; pass `compile=False` to disable it.
