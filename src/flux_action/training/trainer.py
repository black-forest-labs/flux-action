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
"""Training loop of the DROID recipe, also driving the SO-101 recipe over an indexed LeRobot dataset.

Fixed ``windows_per_rank`` windows per rank per micro-step, gradient accumulation to the global
batch, fused AdamW (0.9, 0.99) with the reference schedules on the trunk and head groups, optional
global gradient-norm clipping, two power EMAs, checkpoints every ``checkpoint_every`` updates, resume
with the reference's reseeding or an exact continuation. One process on CPU or GPU, or ``torchrun``
with FSDP2/HSDP over the DiT. The index (``index_dir``) decides the data path: a DROID manifest streams
the compact Cosmos3 layout, a LeRobot manifest streams that dataset and its ``statistics.json`` fixes
the policy's action parameterization and normalization bounds.

Update ``g`` (zero-based) consumes ``factor(g)`` and then feeds the EMAs with index ``g``; the
checkpoint after ``N`` updates is ``step-N``. Ranks seed their generators with ``seed + rank`` so
timesteps and noise differ across ranks; the data order uses the shared seed.

The encoders run one micro-batch ahead: while the DiT trains on micro-batch ``i``, ``policy.prepare``
(augmentation, video VAE, text encoder; frozen, no gradients) already runs for micro-batch ``i + 1`` on
a side CUDA stream, so their time hides behind the backward pass instead of adding to the step. The
loop therefore reads one batch ahead of the update it is on. Augmentation and caption dropout draw from
the loader's per-window seeds, so running ahead changes no random stream and an exact resume stays exact.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from ..config import PolicyConfig
from ..data.droid.index import ROWS_FILENAME
from ..data.lerobot import index as lerobot_index
from .checkpoint import (
    TrainState,
    ema_folder,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_rng,
    resume_seed,
    save_checkpoint,
    seed_everything,
)
from .data import DataPosition, WindowDataset, build_dataloader, load_manifest, manifest_dims
from .distributed import (
    all_reduce_means,
    apply_activation_checkpointing,
    build_mesh,
    global_grad_norm,
    init_distributed,
    shard_dit,
)
from .ema import DROID_POWER_EMA_SIGMA_RELS, PowerEMA
from .schedule import (
    DROID_COOLDOWN_START,
    DROID_COOLDOWN_STEPS,
    DROID_FROZEN_STEPS,
    DROID_HEADS_WARMUP_STEPS,
    DROID_TRUNK_WARMUP_STEPS,
    build_lr_scheduler,
    droid_lr_lambdas,
    resume_lr_scheduler,
)

PolicyFactory = Callable[[PolicyConfig], nn.Module]


@dataclass
class TrainConfig:
    source_root: str
    index_dir: str
    output_dir: str
    # Dataset plug. ``"index"`` (default) streams the manifest in ``index_dir`` (DROID or LeRobot). Any other
    # value is ``"pkg.module:callable"``; the callable receives this config and the loader keyword arguments
    # (seed, epoch, rank, world_size, num_workers, windows_per_rank, skip_batches, grad_accumulation) and
    # returns an iterable dataset of training windows with ``batches_per_rank``. ``index_dir`` is then unused.
    dataset: str = "index"
    policy: dict[str, Any] = field(
        default_factory=dict
    )  # PolicyConfig fields: trunk_weights, encoder ids, ...
    steps: int = 30000
    cooldown_start: int | None = DROID_COOLDOWN_START
    cooldown_steps: int = DROID_COOLDOWN_STEPS
    # LR phases of ``droid_lr_lambdas``: trunk frozen for ``frozen_steps``, then warmed over
    # ``trunk_warmup_steps``; heads warmed over ``heads_warmup_steps``. The DROID values (1000 / 2000 / 1000)
    # were sized for a 30k-step recipe; a few-thousand-step finetune never trains the trunk unless these shrink.
    frozen_steps: int = DROID_FROZEN_STEPS
    trunk_warmup_steps: int = DROID_TRUNK_WARMUP_STEPS
    heads_warmup_steps: int = DROID_HEADS_WARMUP_STEPS
    seed: int = 42
    reseed_on_resume: bool = True
    windows_per_rank: int = 32
    grad_accumulation: int = 1
    num_workers: int = 8
    prefetch_factor: int = 2
    in_process_loader: bool = False
    decoder: str = "ffmpeg"
    frame_hw: tuple[int, int] = (360, 640)
    shard_size: int | None = None  # ranks per shard group; None = all ranks
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    activation_checkpointing: bool = True
    reshard_after_forward: bool = True
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    weight_decay: float = 0.05
    fused_optimizer: bool = True
    # Global gradient-norm clipping (torch.nn.utils.clip_grad_norm_ semantics, over all ranks). None = off,
    # the DROID recipe; the SO-101 recipe clips at 1.0. The logged grad_norm is the norm before clipping.
    max_grad_norm: float | None = None
    # Windows drawn per episode per epoch. One (DROID) means an epoch is one window per episode, so a small
    # dataset cannot fill an update: ranks x num_workers x windows_per_rank x grad_accumulation windows are
    # needed per epoch. The SO-101 corpus builder took up to 35 windows per episode; several visits draw
    # independent starts and captions from the run seed, the epoch, the episode and the visit number.
    visits_per_epoch: int = 1
    ema_sigma_rels: tuple[float, ...] = DROID_POWER_EMA_SIGMA_RELS
    # Content streams of the trunk that receive no loss here and are not shipped: frozen, not trained.
    frozen_streams: tuple[str, ...] = ("image", "image_cond", "audio", "audio_cond")
    checkpoint_every: int = 1000
    # Retention: keep the newest ``keep_checkpoints`` complete checkpoints plus every ``keep_every``-th
    # step (None keeps everything, as the source run did). A checkpoint is 4x the fp32 model size.
    keep_checkpoints: int | None = None
    keep_every: int | None = 5000
    log_every: int = 10
    resume: str = "auto"  # auto | none | path of a step directory
    # A resumed run must keep windows_per_rank x ranks x grad_accumulation unless explicitly allowed.
    allow_global_batch_change: bool = False
    # A resumed run must carry the checkpoint's run seed unless explicitly allowed (branching a new run).
    allow_seed_change: bool = False
    max_epochs: int | None = None

    def __post_init__(self):
        self.frame_hw = tuple(self.frame_hw)
        self.betas = tuple(self.betas)
        self.ema_sigma_rels = tuple(self.ema_sigma_rels)
        self.frozen_streams = tuple(self.frozen_streams)
        if self.steps < 1 or self.windows_per_rank < 1 or self.grad_accumulation < 1 or self.num_workers < 1:
            raise ValueError("steps, windows_per_rank, grad_accumulation and num_workers must be positive")
        if self.cooldown_start is not None and self.cooldown_start + self.cooldown_steps < self.steps:
            raise ValueError(
                "steps must not exceed cooldown_start + cooldown_steps (the schedule ends there)"
            )
        if self.checkpoint_every < 1 or self.log_every < 1:
            raise ValueError("checkpoint_every and log_every must be positive")
        if (self.keep_checkpoints is not None and self.keep_checkpoints < 1) or (
            self.keep_every is not None and self.keep_every < 1
        ):
            raise ValueError("keep_checkpoints and keep_every must be positive when set")
        if self.in_process_loader and self.num_workers != 1:
            raise ValueError("in_process_loader needs num_workers == 1")
        if self.max_grad_norm is not None and not (
            math.isfinite(self.max_grad_norm) and self.max_grad_norm > 0
        ):
            raise ValueError("max_grad_norm must be a finite positive number or None")
        if self.visits_per_epoch < 1:
            raise ValueError("visits_per_epoch must be positive")

    @classmethod
    def from_file(cls, path, overrides: tuple[str, ...] | list[str] = ()) -> TrainConfig:
        """JSON file plus ``key=value`` overrides (``policy.key=value`` reaches the policy fields)."""
        data = json.loads(Path(path).read_text())
        for item in overrides:
            key, _, raw = item.partition("=")
            if not _:
                raise ValueError(f"override {item!r} is not key=value")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            if key.startswith("policy."):
                data.setdefault("policy", {})[key.removeprefix("policy.")] = value
            else:
                data[key] = value
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def global_batch(self, world_size: int) -> int:
        return world_size * self.windows_per_rank * self.grad_accumulation


class Trainer:
    def __init__(self, config: TrainConfig, *, policy_factory: PolicyFactory | None = None):
        self.config = config
        self.rank, self.world_size, self.device = init_distributed()
        self.sharded = self.world_size > 1
        self.param_dtype = getattr(torch, config.param_dtype)
        self.compute_dtype = getattr(torch, config.compute_dtype)
        self.policy_factory = policy_factory
        self.output_dir = Path(config.output_dir)
        self.step = 0
        self.epoch_losses: list[float] = []
        self.encode_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self._prepared: Any = None  # PreparedWindows of the next micro-batch, encoded ahead of its update

    # ------------------------------------------------------------------ construction
    def build(self) -> Trainer:
        cfg = self.config
        seed_everything(cfg.seed + self.rank)
        index = Path(cfg.index_dir)
        if cfg.dataset == "index":
            self.manifest = load_manifest(index / "manifest.json")
            self.rows_path = index / ROWS_FILENAME
            if not self.rows_path.is_file():
                raise FileNotFoundError(
                    f"{self.rows_path} missing; run flux-action index-droid or flux-action index-lerobot"
                )
            self._build_dataset = None
        else:
            self.manifest = self.rows_path = None
            self._build_dataset = _import_dataset_builder(cfg.dataset)
        # fresh heads are drawn from the run seed (rank-independent), not from the per-rank RNG seeded above
        fields = {"head_init_seed": cfg.seed, **cfg.policy, "torch_dtype": cfg.param_dtype}
        self._resume_path = (
            latest_checkpoint(self.output_dir)
            if cfg.resume == "auto"
            else (None if cfg.resume == "none" else Path(cfg.resume))
        )
        if self._resume_path is not None and fields.get("content_streams") is None:
            saved = json.loads((self._resume_path / "config.json").read_text())
            fields["content_streams"] = saved.get("content_streams")
        if self.manifest is not None:
            fields = self._reconcile_with_index(fields, index)
        policy_config = PolicyConfig(**fields)
        policy_config.validate_training()
        with torch.random.fork_rng(devices=[]):
            # every rank builds identical tensors on the CPU; only the CPU generator is touched and restored,
            # so the per-rank seeding above keeps governing training randomness on every device
            torch.default_generator.manual_seed(cfg.seed)
            policy = (
                self.policy_factory(policy_config) if self.policy_factory else _default_policy(policy_config)
            )
        frozen = policy.freeze_streams(cfg.frozen_streams) + policy.freeze_conditioning_heads()
        self._log_line(
            {
                "event": "frozen",
                "streams": list(cfg.frozen_streams),
                "tensors": len(frozen),
                "frozen_numel": sum(p.numel() for p in policy.parameters() if not p.requires_grad),
                "trainable_numel": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            }
        )
        if self.sharded:
            mesh = build_mesh(self.world_size, cfg.shard_size, self.device.type)
            shard_dit(
                policy.dit,
                mesh,
                param_dtype=self.compute_dtype,
                reduce_dtype=getattr(torch, cfg.reduce_dtype),
                reshard_after_forward=cfg.reshard_after_forward,
                activation_checkpointing=cfg.activation_checkpointing,
            )
        else:
            if cfg.activation_checkpointing:
                apply_activation_checkpointing(policy.dit)
            policy.dit.to(self.device)
        for component in (getattr(policy.video_vae, "module", policy.video_vae), policy.text_encoder):
            if isinstance(component, nn.Module):
                component.to(self.device)
        policy.set_compute_dtype(self.compute_dtype if self.compute_dtype != self.param_dtype else None)
        policy.train()
        self.policy = policy
        self.optimizer = torch.optim.AdamW(
            policy.get_optim_params(),
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            fused=cfg.fused_optimizer and self.device.type == "cuda",
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            droid_lr_lambdas(
                cooldown_start=cfg.cooldown_start,
                cooldown_steps=cfg.cooldown_steps,
                frozen_steps=cfg.frozen_steps,
                trunk_warmup_steps=cfg.trunk_warmup_steps,
                heads_warmup_steps=cfg.heads_warmup_steps,
            ),
        )
        self.emas = {ema_folder(s): PowerEMA.from_module(policy, s) for s in cfg.ema_sigma_rels}
        self.state = TrainState(run_seed=cfg.seed, world_size=self.world_size)
        self.state.extra["train_config"] = cfg.to_dict()
        self.state.extra["global_batch"] = cfg.global_batch(self.world_size)
        self.state.extra["data_contract"] = self._data_contract()
        self.position = DataPosition(
            world_size=self.world_size, num_workers=cfg.num_workers, windows_per_rank=cfg.windows_per_rank
        )
        self._resume()
        return self

    def _reconcile_with_index(self, fields: dict[str, Any], index: Path) -> dict[str, Any]:
        """The policy's action space must be the indexed dataset's.

        The rows array's action width must equal ``action_dim`` (the state token has the same width). For a
        LeRobot index, its ``statistics.json`` fixes the action parameterization and the normalization
        bounds: empty policy fields take them, set fields must agree with them (a config that carries one
        dataset's bounds is wrong for another dataset, so a mismatch is an error, not a warning).
        """
        chunk = fields.get("chunk_size", PolicyConfig.chunk_size)
        if chunk != self.manifest["chunk_size"]:
            raise ValueError(
                f"policy.chunk_size {chunk} disagrees with index chunk_size {self.manifest['chunk_size']}; re-index with --chunk-size {chunk}"
            )
        state_dim, action_dim = manifest_dims(self.manifest)
        wanted = int(fields.get("action_dim", PolicyConfig.action_dim))
        if wanted != action_dim or state_dim != action_dim:
            raise ValueError(
                f"policy.action_dim {wanted} but the index holds {action_dim}-dimensional actions and "
                f"{state_dim}-dimensional states; the state token needs action_dim values"
            )
        statistics_path = index / lerobot_index.STATISTICS_FILENAME
        if self.manifest.get("kind") != lerobot_index.KIND:
            return fields
        if not statistics_path.is_file():
            raise FileNotFoundError(f"LeRobot training requires normalization statistics: {statistics_path}")
        statistics = lerobot_index.load_statistics(statistics_path)
        fields = dict(fields)
        indexed = statistics["action"]["parameterization"]
        absolute = tuple(statistics["action"]["absolute_dims"])
        parameterization = fields.setdefault("action_parameterization", indexed)
        if "absolute_action_dims" in fields:
            configured = lerobot_index.normalize_dims(fields["absolute_action_dims"], action_dim)
        else:
            configured = absolute if indexed == "joint_delta" else ()
            fields["absolute_action_dims"] = list(configured)
        if parameterization != indexed or (indexed == "joint_delta" and configured != absolute):
            raise ValueError(
                f"the index statistics describe {indexed!r} targets with absolute dims {list(absolute)}, the "
                f"policy config {parameterization!r} with {list(configured)}; re-index with "
                "--action-parameterization / --absolute-action-dims or change the policy fields"
            )
        adopted = []
        for field_name, block in (("action_normalization", "action"), ("state_normalization", "state")):
            bounds = {k: [float(v) for v in statistics[block][k]] for k in ("q01", "q99")}
            current = fields.get(field_name)
            if current is None:
                fields[field_name] = bounds
                adopted.append(field_name)
                continue
            same = all(
                len(current[k]) == len(bounds[k])
                and all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(current[k], bounds[k]))
                for k in ("q01", "q99")
            )
            if not same:
                raise ValueError(
                    f"policy.{field_name} disagrees with {statistics_path}; remove the field to take the "
                    "index's bounds, or index the dataset the checkpoint was normalized against"
                )
        clip = float(statistics["clip"])
        if fields.setdefault("normalization_clip", clip) != clip:
            raise ValueError(
                f"policy.normalization_clip {fields['normalization_clip']} but the index used {clip}"
            )
        self._log_line(
            {
                "event": "normalization",
                "statistics": str(statistics_path),
                "adopted": adopted,
                "action_parameterization": indexed,
                "absolute_action_dims": list(absolute),
            }
        )
        return fields

    def _data_contract(self) -> str:
        # The manifest includes row hashes, dataset identity, cameras and episode ranges.
        # Hash its semantic contents so moving an index or reformatting JSON is harmless.
        data = {"dataset": self.config.dataset, "manifest": self.manifest}
        if self.manifest is None:
            data["source_root"] = self.config.source_root
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _validate_resume_contract(self, path: Path) -> None:
        saved = PolicyConfig(**json.loads((path / "config.json").read_text())).to_dict()
        current = self.policy.config.to_dict()
        fields = (
            "action_dim",
            "action_modality",
            "content_streams",
            "camera_layout",
            "camera_keys",
            "canvas_hw",
            "chunk_size",
            "fps",
            "action_scale",
            "gripper_flip_dims",
            "action_parameterization",
            "absolute_action_dims",
            "action_normalization",
            "state_normalization",
            "normalization_clip",
            "inference_profile",
            "n_obs_steps",
            "history_snapshots",
            "condition_on_past_actions",
            "video_position_fps",
            "text_fixed_length",
            "separate_timesteps",
            "video_logit_mean",
            "video_logit_std",
            "conditioning_noise_max",
            "loss_reduction",
            "action_channel_weights",
            "action_loss_weight",
            "video_loss_weight",
            "train_timestep_width",
            "train_timestep_shift",
        )
        changed = [name for name in fields if saved[name] != current[name]]
        extra = json.loads((path / "state.json").read_text()).get("extra", {})
        if extra.get("data_contract", self._data_contract()) != self._data_contract():
            changed.append("dataset/index")
        previous = extra.get("train_config", {})
        for name in ("dataset", "frame_hw", "decoder"):
            if name in previous and previous[name] != json.loads(json.dumps(self.config.to_dict()[name])):
                changed.append(name)
        if changed:
            raise ValueError(
                f"resume changes the saved policy/data contract: {', '.join(changed)}; "
                "start a separate fine-tuning run instead of restoring optimizer state"
            )

    def _resume(self) -> None:
        cfg = self.config
        path = self._resume_path
        if path is None:
            return
        self._validate_resume_contract(path)
        self.state = load_checkpoint(path, model=self.policy, optimizer=self.optimizer, emas=self.emas)
        self.state.extra["data_contract"] = self._data_contract()
        self.step = self.state.step
        if cfg.seed != self.state.run_seed and not cfg.allow_seed_change:
            raise ValueError(
                f"checkpoint {path} belongs to run seed {self.state.run_seed}, config seed is {cfg.seed}; "
                "use the checkpoint's seed (or a new output_dir), or set allow_seed_change"
            )
        previous, current = self.state.extra.get("global_batch"), cfg.global_batch(self.world_size)
        if (
            previous is None and "train_config" in self.state.extra
        ):  # checkpoints written before the field existed
            saved = self.state.extra["train_config"]
            previous = self.state.world_size * saved["windows_per_rank"] * saved["grad_accumulation"]
        if previous is not None and previous != current and not cfg.allow_global_batch_change:
            raise ValueError(
                f"the checkpoint was trained with {previous} windows per update, this launch gives {current}; "
                "adjust windows_per_rank or grad_accumulation, or set allow_global_batch_change"
            )
        self.state.extra["global_batch"] = current
        resume_lr_scheduler(self.scheduler, self.step)
        recorded = self.state.data.get("position")
        recorded = DataPosition.from_dict(recorded) if recorded else None
        topology = dict(
            world_size=self.world_size, num_workers=cfg.num_workers, windows_per_rank=cfg.windows_per_rank
        )
        exact = (
            not cfg.reseed_on_resume
            and recorded is not None
            and recorded.matches(**topology)
            and restore_rng(path, self.state)
        )
        if exact:
            self.position = recorded
            mode = "exact"
        else:
            if cfg.reseed_on_resume:
                self.state.seed_history.append(resume_seed(self.state.run_seed, len(self.state.seed_history)))
            seed_everything(self.state.seed + self.rank)
            self.position = DataPosition(epoch=(recorded.epoch + 1 if recorded else 0), **topology)
            mode = "reseeded" if cfg.reseed_on_resume else "fresh_epoch"
        self.state.world_size = self.world_size
        self.state.extra.setdefault("resumes", []).append(
            {"from": str(path), "step": self.step, "mode": mode}
        )
        self.state.extra["train_config"] = cfg.to_dict()
        self._log_line(
            {"event": "resume", "from": str(path), "step": self.step, "mode": mode, "seed": self.state.seed}
        )

    # ------------------------------------------------------------------ loop
    def run(self) -> dict[str, Any]:
        self.build()
        return self.train()

    def train(self) -> dict[str, Any]:
        cfg = self.config
        self._written = self.step if latest_checkpoint(self.output_dir) is not None and self.step > 0 else -1
        while self.step < cfg.steps and (cfg.max_epochs is None or self.position.epoch < cfg.max_epochs):
            loader_kwargs = dict(
                seed=self.state.seed,
                epoch=self.position.epoch,
                rank=self.rank,
                world_size=self.world_size,
                num_workers=cfg.num_workers,
                windows_per_rank=cfg.windows_per_rank,
                skip_batches=self.position.batches_consumed,
                grad_accumulation=cfg.grad_accumulation,
            )
            if self._build_dataset is not None:
                dataset = self._build_dataset(cfg, **loader_kwargs)
            else:
                dataset = WindowDataset(
                    self.manifest,
                    cfg.source_root,
                    self.rows_path,
                    **loader_kwargs,
                    decoder=cfg.decoder,
                    frame_hw=cfg.frame_hw,
                    visits_per_epoch=cfg.visits_per_epoch,
                    split="train",
                    n_obs_steps=self.policy.config.n_obs_steps
                    if self.policy.config.inference_profile == "history"
                    else None,
                )
            if self.rank == 0 and self.position.batches_consumed == 0:
                self._log_line(
                    {
                        "event": "epoch",
                        "epoch": self.position.epoch,
                        "batches_per_rank": dataset.batches_per_rank,
                    }
                )
            loader = build_dataloader(
                dataset,
                prefetch_factor=cfg.prefetch_factor,
                pin_memory=self.device.type == "cuda",
                in_process=cfg.in_process_loader,
            )
            accum = cfg.grad_accumulation
            pending: list[dict[str, Any]] = []  # the update's micro-batches plus one batch of lookahead
            exhausted = True
            waited_since = time.perf_counter()
            data_wait = 0.0
            self._prepared = None
            for batch in loader:
                data_wait += time.perf_counter() - waited_since
                pending.append(batch)
                if len(pending) <= accum:  # keep reading until the first batch of the NEXT update is here
                    waited_since = time.perf_counter()
                    continue
                stop = self._train_update(pending[:accum], pending[accum], data_wait)
                pending = pending[accum:]
                waited_since = time.perf_counter()
                data_wait = 0.0
                if stop:
                    exhausted = False
                    break
            if exhausted:
                # the epoch's last update has no lookahead batch
                if len(pending) == accum and self._train_update(pending, None, data_wait):
                    break
                self.position.epoch += 1
                self.position.batches_consumed = 0
        if self._written != self.step:
            self._checkpoint()
        return {
            "step": self.step,
            "epoch": self.position.epoch,
            "seed": self.state.seed,
            "output_dir": str(self.output_dir),
        }

    def _train_update(
        self, micro: list[dict[str, Any]], next_batch: dict[str, Any] | None, data_wait: float
    ) -> bool:
        """One update on ``micro`` (``next_batch`` is the following update's first batch, encoded ahead);
        logs, checkpoints, advances the position. Returns True when the run has reached ``steps``."""
        cfg = self.config
        metrics = self._update(micro, next_batch)
        metrics["data_wait"] = data_wait
        self.position.batches_consumed += cfg.grad_accumulation
        self.step += 1
        metrics.update(step=self.step, epoch=self.position.epoch)
        self.epoch_losses.append(metrics["loss"])
        if self.rank == 0 and (self.step % cfg.log_every == 0 or self.step == cfg.steps):
            self._log_line(metrics)
        if self.step % cfg.checkpoint_every == 0 or self.step == cfg.steps:
            self._checkpoint()
            self._written = self.step
        return self.step >= cfg.steps

    def _prepare(self, batch: dict[str, Any]) -> Any:
        """``policy.prepare`` on the side stream, with an event the consumer waits for."""
        if self.encode_stream is None:
            return self.policy.prepare(batch)
        with torch.cuda.stream(self.encode_stream):
            prepared = self.policy.prepare(batch)
            prepared.event = torch.cuda.Event()
            prepared.event.record(self.encode_stream)
        return prepared

    def _update(
        self, micro: list[dict[str, Any]], next_batch: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        accum = len(micro)
        # FP64 matches the previous Python-float accumulation without per-microbatch host waits.
        metrics = torch.zeros(3, device=self.device, dtype=torch.float64)
        windows = 0
        autocast = torch.autocast(
            self.device.type,
            dtype=self.compute_dtype,
            enabled=not self.sharded
            and self.compute_dtype != self.param_dtype
            and self.device.type == "cuda",
        )
        for i, batch in enumerate(micro):
            if self.sharded:
                self.policy.dit.set_requires_gradient_sync(i == accum - 1)
            prepared = self._prepared if self._prepared is not None else self._prepare(batch)
            self._prepared = None
            with autocast:
                loss, info = self.policy(batch, prepared=prepared)
            following = micro[i + 1] if i + 1 < accum else next_batch
            if following is not None:  # its encoders overlap this micro-batch's backward pass
                self._prepared = self._prepare(following)
            (loss / accum).backward()
            metrics += (
                torch.stack(
                    [
                        loss.detach().double(),
                        torch.as_tensor(info["video_mse"], device=self.device, dtype=torch.float64),
                        torch.as_tensor(info["action_mse"], device=self.device, dtype=torch.float64),
                    ]
                )
                / accum
            )
            windows += int(info["n_valid_windows"])
        if self.sharded:
            self.policy.dit.set_requires_gradient_sync(True)
        if self.step == 0:
            self._check_gradients()
        # Abort on all ranks before anything is mutated: a non-finite update would poison the parameters,
        # the optimizer moments and both EMAs, then be written as a complete checkpoint.
        loss_mean, video, action = all_reduce_means(metrics)
        replicas = self.world_size // (self.config.shard_size or self.world_size) if self.sharded else 1
        grad_norm = global_grad_norm(self.policy, self.device, replicas)
        if not (math.isfinite(loss_mean) and math.isfinite(grad_norm)):
            raise RuntimeError(
                f"non-finite update at step {self.step + 1}: loss {loss_mean}, grad_norm {grad_norm}; "
                "nothing was applied, resume from the last complete checkpoint"
            )
        clipped = {}
        if self.config.max_grad_norm is not None:
            # clip_grad_norm_ semantics on the global norm: every rank scales its shards by the same factor
            coefficient = min(1.0, self.config.max_grad_norm / (grad_norm + 1e-6))
            if coefficient < 1.0:
                for param in self.policy.parameters():
                    if param.grad is not None:
                        param.grad.mul_(coefficient)
            clipped = {"grad_clip": coefficient}
        consumed = [group["lr"] for group in self.optimizer.param_groups]  # factor(step) for this update
        self.optimizer.step()
        self.scheduler.step()
        betas = {name: ema.update_from_module(self.policy, self.step) for name, ema in self.emas.items()}
        self.optimizer.zero_grad(set_to_none=True)
        peak = {}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = {"peak_mem_gb": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 1)}
        return {
            "loss": loss_mean,
            "video_mse": video,
            "action_mse": action,
            "grad_norm": grad_norm,
            **clipped,
            "windows": windows * self.world_size,
            "lr_trunk": consumed[0],
            "lr_heads": consumed[1] if len(consumed) > 1 else None,
            **{f"beta_{name}": beta for name, beta in betas.items()},
            "step_time": time.perf_counter() - started,
            **peak,
        }

    def _check_gradients(self) -> None:
        """Every trainable parameter must take part in the loss; otherwise freeze it explicitly.

        Parameters without gradients would hold no optimizer state and make the checkpoint
        incomplete; the trunk's image and audio streams are frozen through ``frozen_streams`` and
        the conditioning final layers through ``freeze_conditioning_heads``.
        """
        missing = [n for n, p in self.policy.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError(
                f"{len(missing)} trainable parameters received no gradient, e.g. {missing[:4]}; "
                "add their streams to frozen_streams"
            )

    # ------------------------------------------------------------------ side effects
    def _checkpoint(self) -> None:
        started = time.perf_counter()
        self.state.data["position"] = self.position.to_dict()
        path = save_checkpoint(
            self.output_dir,
            step=self.step,
            model=self.policy,
            optimizer=self.optimizer,
            emas=self.emas,
            state=self.state,
            config=self.policy.config,
        )
        if self.rank == 0:
            self._log_line(
                {
                    "event": "checkpoint",
                    "step": self.step,
                    "path": str(path),
                    "seconds": round(time.perf_counter() - started, 1),
                }
            )
            if self.config.keep_checkpoints is not None:
                removed = prune_checkpoints(
                    self.output_dir,
                    keep_last=self.config.keep_checkpoints,
                    keep_every=self.config.keep_every,
                    protect=self.step,
                )
                if removed:
                    self._log_line({"event": "prune", "removed_steps": removed})

    def _log_line(self, record: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"time": time.time(), **record})
        with (self.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


def _default_policy(config: PolicyConfig) -> nn.Module:
    from ..policy import FluxActionPolicy

    return FluxActionPolicy(config)


def _import_dataset_builder(spec: str) -> Callable[..., Any]:
    """Resolve "pkg.module:callable" (TrainConfig.dataset) to the callable."""
    if ":" not in spec:
        raise ValueError(f"dataset must be 'index' or 'pkg.module:callable', got {spec!r}")
    module_name, attr = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), attr)
