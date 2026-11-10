# End-effector fine-tuning on DROID (EE-50)

The base checkpoint carries pretrained heads for EE-50, a 50-dimensional end-effector action layout
(`action_prediction_ee50`). This example fine-tunes the base on DROID in that layout instead of joint
positions. It keeps the pretrained heads and does not add fresh ones. It runs through `flux-action train` as a
[dataset plug](../../docs/embodiments.md) on top of the DROID index, and the same files show how to bring
another robot's end-effector poses into the layout.

## The layout

A row holds a left block (dims 0–24) and a right block (dims 25–49):

| Dims of a block | Content |
|---|---|
| 0–2 | Translation relative to the anchor pose, in the anchor frame, metres: `R_anchor^T (p - p_anchor)` |
| 3–8 | First two **rows** of `R_rel = R_anchor^T R`, row-major `[r00 r01 r02 r10 r11 r12]` |
| 9–24 | Hand: 16 absolute channels, not relative to the anchor (see below) |

A single-arm robot uses the left block and leaves the right block zero. To decode rot6d, apply Gram-Schmidt to
the two rows and take their cross product as the third row (`ee50.rot6d_rows_to_matrix`).

### Hand channels

The pretrained heads know two uses of the hand channels; no robot-hand convention is pretrained.

A **parallel gripper** writes its open fraction (1 = open, 0 = closed) to the first channel (dim 9 or 34)
and zeros to the other 15.

A **human hand** uses its wrist as the pose of the block and 16 angles in radians, computed from joint
positions:

| Channel | Content |
|---|---|
| 0–9 | Flexion, negated: 0 = straight, negative = bent. Two per finger, in the order thumb, index, middle, ring, little: the angle between consecutive bone segments at the first joint, then at the second (fingers: MCP, PIP; thumb: CMC, MCP) |
| 10–14 | Abduction of thumb, index, middle, ring, little: the signed angle, in the palm plane, of the finger's first segment (fingers: MCP to PIP; thumb: base to tip), measured from `forward` about `normal` |
| 15 | Thumb opposition: `arcsin(unit(thumb_tip - wrist) · normal)` |

The palm frame is `forward = unit(middle_MCP - wrist)` and
`normal = unit((little_MCP - index_MCP) × forward)`. Both hands use the same formulas without mirroring,
so mirrored left and right poses give the same abduction and opposite opposition.

A window has 33 rows at offsets 0–32 from its anchor. Row 0 is the conditioning `state`, the anchor itself
(the identity pose and the current gripper). Rows 1–32 are the `action` targets. DROID windows are
transition aligned: the anchor is the **observed** pose `state[s]` at the window's first frame, and row
`k` is the **commanded** pose `action[s + k - 1]`, the command that leads to frame `s + k`, relative to that
observed anchor.

Each value is z-scored per (offset, dim), `clip((x - mean[k]) / std[k], -10, 10)`. A `std` below 1e-8
counts as 1, so constant and padded dims stay 0. After normalization, the pose dims of row 0 (0–8 and
25–33) are set to exactly 0. The policy sees these values unchanged: the config leaves
`action_normalization` unset and uses action scale 1.

## Files

- `ee50.py`: the layout in numpy. It covers DROID's Euler convention (`R = Rz Ry Rx`), transition-aligned
  windows, normalization and its inverse, per-offset statistics, the state token and the decoding of a
  predicted chunk back to absolute poses.
- `index.py`: runs after `flux-action index-droid`. It writes the Cartesian state and command of every
  indexed frame, a manifest, and `ee50_stats.json`.
- `dataset.py`: the trainer plug (`TrainConfig.dataset = "examples.droid_ee50.dataset:build"`). It uses the
  DROID loader's sampling, video decoding and resume and converts each window to normalized EE-50.
- `inference.py`: `predict_droid_commands`, which wraps `predict_action_chunk` and returns absolute
  `[x, y, z, rx, ry, rz, gripper_position]` commands.
- `droid_reference_stats.json`: the statistics used for the reference fine-tune, offsets 0–32. They were
  computed over every frame of the `keep_ranges_1_0_1.json` ranges of at least 73 frames
  (55,153 episodes). `index.py --stats compute` reproduces them exactly from the public dataset.
- [`configs/droid/train_ee50.json`](../../configs/droid/train_ee50.json): the training configuration.
- [`tests/test_droid_ee50.py`](../../tests/test_droid_ee50.py): the layout, its inverse, the statistics, the
  index, the plug, and a short training run followed by inference, all on synthetic data.

## Run

Complete [setup](../../docs/setup.md), including the base and encoder download. Then
[index DROID](../../docs/prepare.md#index-the-whole-dataset-for-training) and add the Cartesian rows. Run
these commands from the repository root:

```sh
uv run python -m examples.droid_ee50.index --source-root outputs/droid/source \
  --droid-index outputs/droid/index --output-dir outputs/droid_ee50/index
```

`--stats compute` recomputes the statistics from the indexed rows instead of copying the reference file
(`--min-range-frames` sets the shortest keep range counted, 73 by default).

Training and export work as in the [DROID guide](../../docs/droid-finetune.md). For one 8-GPU node:

```sh
uv run torchrun --nproc_per_node 8 -m flux_action.cli train --config configs/droid/train_ee50.json \
  --override shard_size=8 --override windows_per_rank=16 --override grad_accumulation=16
uv run flux-action export-checkpoint --checkpoint outputs/droid_ee50/run-seed42/step-30000 \
  --output outputs/droid_ee50/export-30k --profile ema_0p10
```

To run inference, set sampler settings on the export as described in the
[setup guide](../../docs/setup.md#python-api). Then pass the observed end-effector pose and gripper with each
request:

```python
from examples.droid_ee50 import ee50
from examples.droid_ee50.inference import predict_droid_commands

stats = ee50.load_stats("outputs/droid_ee50/index/ee50_stats.json")
commands = predict_droid_commands(
    policy,
    {"images.wrist": wrist, "images.left": left, "images.right": right, "task": ["task caption"]},
    cartesian_position=cartesian,  # (B, 6) observed [x, y, z, rx, ry, rz]
    gripper_position=gripper,  # (B,) observed, 0 = open
    stats=stats,
)  # (B, 32, 7) absolute [x, y, z, rx, ry, rz, gripper_position]
```

Each chunk is anchored at the pose observed when it was requested. The robot tracks Cartesian targets
(DROID's `cartesian_position` action space); its controller does the inverse kinematics.

## Recipe

Compared with the [joint-space recipe](../../docs/droid-finetune.md#recipe):

| Setting | EE-50 recipe |
|---|---|
| Initialization | Action-pretrained 724k EMA base, **with** its pretrained `action_prediction_ee50` heads |
| Data | 33 frames at 15 Hz; 32 normalized EE-50 commands; three-camera DROID composite |
| Peak learning rate | 2e-4 for all parameters, heads included |
| Schedule | Warmup over 1,000 updates for all parameters, no frozen phase; constant to 25,000 |
| Representation / loss | Action scale 1; joint-token loss with video weight 1 and action weight 1020/11 |

Batch, optimizer, cooldown, training timesteps, caption dropout, EMA profiles and checkpoints are the same
as in the joint-space recipe.

## Other robots

For your own robot, build windows with `ee50.relative_rows` and the anchor convention above. Add the raw
window of every anchor of your data with `ee50.Moments.add_window`, turn the sums into statistics with
`ee50.finalize`, and yield the normalized rows from a plug like `dataset.py`. Statistics from your own data
replace the DROID file.

A second arm fills the right block, and a hand fills the hand channels. The pretrained heads expect the
conventions on this page.
