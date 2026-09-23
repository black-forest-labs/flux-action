# DROID full fine-tuning

Train FLUX 3 Action on DROID with the standalone trainer. The complete path includes data indexing,
FSDP2/HSDP, gradient accumulation, power EMA, checkpoint/resume and inference export; no LeRobot
installation is required.

## Run the reproduction

**1. Set up weights and data.** Complete [setup](setup.md), including the base and encoder download, then
[index the DROID dataset](prepare.md#index-the-whole-dataset-for-training).
The defaults in [`configs/droid/train.json`](../configs/droid/train.json) expect
`outputs/droid/source`, `outputs/droid/index` and `outputs/weights`.

**2. Train.** The reference global batch is 2,048 windows.
For one 8-GPU node, accumulation preserves that global batch:

```sh
uv run torchrun --nproc_per_node 8 -m flux_action.cli train --config configs/droid/train.json \
  --override shard_size=8 --override windows_per_rank=16 --override grad_accumulation=16
```

`ranks × windows_per_rank × grad_accumulation` determines the global batch.
Set paths and resource allocations for your machine before launching.

For multiple nodes, launch one `torchrun` process per node with `--nnodes`, a unique
`--node_rank`, and a shared `--rdzv_backend=c10d --rdzv_endpoint=HOST:PORT`.
Under Slurm, `srun` can launch these processes; choose resource requests for your cluster.
Keep code, dependencies and data paths consistent across nodes, and adjust `shard_size`,
`windows_per_rank` and `grad_accumulation` to preserve the intended global batch.

The run writes `metrics.jsonl` (losses, gradient norm, consumed learning rates, EMA betas,
windows and timing) and `step-N` checkpoints under `outputs/droid/run-seed42`.

**3. Resume.** Relaunch the same command to restore the latest complete checkpoint. The DROID
recipe draws a new seed on resume, derived from the run seed and resume count. To restore RNG
and the loader position instead, use `--override reseed_on_resume=false` with the same data,
policy settings, rank count, worker count and windows per rank. Keep the original normalization
and encoder files. Changing the global batch or run seed requires an explicit override.

**4. Export.** Select raw weights (`model`) or a power-EMA profile (`ema_0p10`, `ema_0p05`):

```sh
uv run flux-action export-checkpoint \
  --checkpoint outputs/droid/run-seed42/step-26000 \
  --output outputs/droid/export-26k --profile ema_0p10
```

The export is for [inference](setup.md#inference-from-an-existing-export); retain `step-N` for training resume.
The training defaults leave sampling settings unset; the [Python example](setup.md#python-api)
applies the DROID inference preset, and the RoboLab server applies it automatically.
`policy.save_pretrained()` also writes an inference export, without optimizer or resume state.
The Hub's released robot packages use [LeRobot's format](setup.md#choose-a-checkpoint).

## Recipe

[`configs/droid/train.json`](../configs/droid/train.json) is the runnable configuration.
It is loaded directly by `flux-action train`; omitted fields use `TrainConfig` and `PolicyConfig` defaults.

| Setting | Standalone recipe |
|---|---|
| Initialization | Action-pretrained 724k EMA base; fresh DROID input/output heads |
| Data | 33 frames at 15 Hz; 32 absolute 8-channel actions; three-camera DROID composite |
| Batch | 2,048 windows globally |
| Optimizer | Fused AdamW, betas (0.9, 0.99), epsilon 1e-8, weight decay 0.05 |
| Peak learning rates | Trunk 1.92e-4; heads 9.6e-4 |
| Trunk schedule | Zero LR through index 1,000; warmup through 3,000; constant to 25,000 |
| Head schedule | Warmup over 1,000 updates; constant to 25,000 |
| Cooldown / duration | Linear cooldown from 25,000 to 30,000; 30,000 optimizer updates |
| Training timesteps | Logit-logistic width 0.75, shift 42; one timestep shared by video and action |
| Representation / loss | Action scale 2; joint-token loss with video weight 1 and action weight 50 |
| Caption dropout | 0.1 |
| EMA | Power profiles sigma_rel 0.10 and 0.05 |
| Checkpoints | Every 1,000 updates; model, optimizer, both EMAs, config, loader position and RNG |

The reference selected step 26,000 with power EMA sigma_rel 0.10. This is a reference selection,
not a guarantee that the same step is best for a new run. Inference settings are specified
[separately](setup.md#reference-droid-inference-settings).

The current recipe names its fresh heads `action_prediction_droid` so BF16 exports can
use prepared inference after applying the DROID sampler preset. The pinned base has no
DROID heads, so this retains fresh input/output-head initialization. Existing runs using
`action` must keep that namespace when resuming (`--override policy.action_modality=action`)
and use `--no-compile-dit` when serving their exports. Do not rename saved configurations.

## Training speed

For tuning, set `policy.attn_mode` to `cudnn` to use cuDNN attention with PyTorch fallbacks,
and `policy.compile_model` to `true` to compile the frozen VAE and text encoder. Compilation
makes the first update slower; compare steady-state updates. Reduce `policy.vae_batch_windows`
if encoder batches use too much GPU memory. Monitor `step_time` and `data_wait` in `metrics.jsonl`
to distinguish update time from waiting for data. See [attention backends](setup.md#attention-backends)
for the available kernels.

## Implementation

The [trainer](../src/flux_action/training/trainer.py) packs each micro-batch into one DiT forward,
keeps attention within each window, and overlaps frozen encoder work with backward computation.
[Distributed wrapping](../src/flux_action/training/distributed.py) uses bf16 compute, fp32 masters
and gradient reduction, a separate fp32 timestep embedder, and activation checkpointing.

[Schedules](../src/flux_action/training/schedule.py) apply `factor(g)` to zero-based optimizer
update `g`; the scheduler advances after the optimizer. Zero trunk LR retains AdamW moment updates.
[Power EMA](../src/flux_action/training/ema.py) updates after each optimizer step, with its first
update copying the model. Unused image/audio streams and conditioning output heads are frozen.
[Checkpoints](../src/flux_action/training/checkpoint.py) use PyTorch distributed checkpointing and
are eligible for resume only after their `COMPLETE` marker is written.

The standalone path uses fixed windows per rank rather than token-budget packing, public shards
decoded on demand, accumulation, deterministic reseeding and one-launch cooldown. It retains two
power EMAs; the reference's classic 0.999 EMA and zero-weight teacher forward are omitted.
Fresh head initialization uses the run seed, and attention backends may differ from the reference.
