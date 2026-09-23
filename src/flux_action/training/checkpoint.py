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
"""Resumable training checkpoints on ``torch.distributed.checkpoint``.

``<root>/step-<N>/`` is written after ``N`` optimizer updates, the reference's naming, and holds

* ``model/``      DCP: the policy's DiT parameters (dense or FSDP shards, resharded on load)
* ``optimizer/``  DCP: AdamW state keyed by parameter name
* ``ema_0p10/``, ``ema_0p05/``  DCP: the power-EMA shadow tensors of each profile
* ``state.json``  rank 0: :class:`TrainState` (step, seeds, world size, EMA steps, loader position)
* ``config.json`` rank 0: the ``PolicyConfig``, so :func:`export_policy` rebuilds the model without the run
* ``rng/rank-<r>.pt``  per-rank torch / CUDA / ``random`` / numpy generator states
* ``COMPLETE``    rank 0, after every rank finished: :func:`latest_checkpoint` ignores directories without it

The learning-rate scheduler is not stored. Rebuild it from the configuration and call
``resume_lr_scheduler(scheduler, state.step)``. Generator states are restored only when
reseeding is off and the world size is unchanged; the reference drew a fresh seed on every
resume, which :func:`resume_seed` reproduces deterministically from the run seed.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from ..config import PolicyConfig
from .distributed import barrier, rank_world
from .ema import PowerEMA

STATE_FORMAT = 1
COMPLETE_MARKER = "COMPLETE"
_SINGLE_PROCESS_WARNING = "torch.distributed is disabled, unavailable or uninitialized"


def ema_folder(sigma_rel: float) -> str:
    """``0.10 -> ema_0p10``; the profile is part of the folder name."""
    return "ema_" + f"{sigma_rel:.2f}".replace(".", "p")


@dataclass
class TrainState:
    step: int = 0  # completed optimizer updates
    run_seed: int = 42
    seed_history: list[int] = field(default_factory=list)  # seeds in effect, in order; [0] is run_seed
    world_size: int = 1
    emas: dict[str, dict[str, Any]] = field(default_factory=dict)  # folder -> {sigma_rel, last_step}
    data: dict[str, Any] = field(default_factory=dict)  # loader position, owned by the data path
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.seed_history:
            self.seed_history = [self.run_seed]

    @property
    def seed(self) -> int:
        """The seed in effect for the current segment of the run."""
        return self.seed_history[-1]

    def to_json(self) -> str:
        return json.dumps({"format": STATE_FORMAT, **asdict(self)}, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> TrainState:
        data = json.loads(text)
        if data.pop("format", None) != STATE_FORMAT:
            raise ValueError("unsupported training state format")
        return cls(**data)


def resume_seed(run_seed: int, resume_index: int) -> int:
    """Seed of the ``resume_index``-th resume (1-based): distinct from the run seed, reproducible."""
    if resume_index < 1:
        raise ValueError("resume_index counts resumes from one")
    digest = hashlib.sha256(f"{run_seed}:{resume_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1) + 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def rng_state() -> dict[str, Any]:
    """Generator states of this process; everything is a tensor, int or list so it loads with weights_only."""
    version, keys, gauss = random.getstate()
    np_name, np_keys, np_pos, np_has_gauss, np_gauss = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": {"version": version, "keys": list(keys), "gauss": gauss},
        "numpy": {
            "name": np_name,
            "keys": torch.from_numpy(np.asarray(np_keys).copy()),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "gauss": float(np_gauss),
        },
    }


def set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    py = state["python"]
    random.setstate((py["version"], tuple(py["keys"]), py["gauss"]))
    npy = state["numpy"]
    np.random.set_state((npy["name"], npy["keys"].numpy(), npy["pos"], npy["has_gauss"], npy["gauss"]))


def _dcp_save(state_dict: dict[str, Any], path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SINGLE_PROCESS_WARNING)
        dcp.save(state_dict, checkpoint_id=str(path))


def _dcp_load(state_dict: dict[str, Any], path: Path, *, partial: bool = False) -> None:
    """Load in place; ``partial`` leaves template entries the checkpoint lacks untouched."""
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SINGLE_PROCESS_WARNING)
        planner = DefaultLoadPlanner(allow_partial_load=True) if partial else None
        dcp.load(state_dict, checkpoint_id=str(path), planner=planner)


def checkpoint_dir(root: str | Path, step: int) -> Path:
    return Path(root) / f"step-{step}"


def complete_checkpoints(root: str | Path) -> list[tuple[int, Path]]:
    """All complete ``step-<N>`` directories under ``root``, sorted by step."""
    root = Path(root)
    if not root.is_dir():
        return []
    steps = []
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.startswith("step-") and (entry / COMPLETE_MARKER).exists():
            try:
                steps.append((int(entry.name.removeprefix("step-")), entry))
            except ValueError:
                continue
    return sorted(steps)


def latest_checkpoint(root: str | Path) -> Path | None:
    """The complete checkpoint with the highest step, or ``None``."""
    steps = complete_checkpoints(root)
    return steps[-1][1] if steps else None


def prune_checkpoints(root: str | Path, *, keep_last: int, keep_every: int | None, protect: int) -> list[int]:
    """Delete complete checkpoints except the newest ``keep_last``, every ``keep_every``-th step and
    ``protect`` (the step just written). Call on one rank only. Returns the removed steps."""
    if keep_last < 1:
        raise ValueError("keep_last must be positive")
    steps = complete_checkpoints(root)
    keep = {step for step, _ in steps[-keep_last:]} | {protect}
    if keep_every:
        keep |= {step for step, _ in steps if step % keep_every == 0}
    removed = []
    for step, path in steps:
        if step not in keep:
            shutil.rmtree(path)
            removed.append(step)
    return removed


def save_checkpoint(
    root: str | Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    emas: dict[str, PowerEMA],
    state: TrainState,
    config: PolicyConfig,
) -> Path:
    """Write ``<root>/step-<step>``; every rank calls this. Existing directories are replaced."""
    rank, world_size = rank_world()
    path = checkpoint_dir(root, step)
    if rank == 0 and path.exists():
        shutil.rmtree(path)
    barrier()
    path.mkdir(parents=True, exist_ok=True)
    state.step = step
    state.world_size = world_size
    state.emas = {
        name: {"sigma_rel": ema.sigma_rel, "last_step": ema.last_step, "n_tensors": len(ema.names)}
        for name, ema in emas.items()
    }
    _dcp_save({"model": get_model_state_dict(model)}, path / "model")
    _dcp_save({"optimizer": get_optimizer_state_dict(model, optimizer)}, path / "optimizer")
    for name, ema in emas.items():
        _dcp_save({"tensors": ema.state_dict()["tensors"]}, path / name)
    (path / "rng").mkdir(exist_ok=True)
    torch.save(rng_state(), path / "rng" / f"rank-{rank:04d}.pt")
    if rank == 0:
        (path / "state.json").write_text(state.to_json())
        (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    barrier()
    if rank == 0:
        (path / COMPLETE_MARKER).write_text(f"{step}\n")
    barrier()
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    emas: dict[str, PowerEMA] | None = None,
) -> TrainState:
    """Restore model, optimizer and EMA tensors in place (resharding as needed) and return the state."""
    path = Path(path)
    if not (path / COMPLETE_MARKER).exists():
        raise FileNotFoundError(f"{path} is not a complete checkpoint")
    state = TrainState.from_json((path / "state.json").read_text())
    model_sd = {"model": get_model_state_dict(model)}
    _dcp_load(model_sd, path / "model")
    set_model_state_dict(model, model_sd["model"])
    if optimizer is not None:
        optim_sd = {"optimizer": get_optimizer_state_dict(model, optimizer)}
        _dcp_load(optim_sd, path / "optimizer")
        set_optimizer_state_dict(model, optimizer, optim_sd["optimizer"])
    for name, ema in (emas or {}).items():
        meta = state.emas.get(name)
        if meta is None:
            raise ValueError(f"checkpoint has no EMA profile {name!r}; found {sorted(state.emas)}")
        if float(meta["sigma_rel"]) != ema.sigma_rel:
            raise ValueError(f"{name}: checkpoint sigma_rel {meta['sigma_rel']} differs from {ema.sigma_rel}")
        _dcp_load({"tensors": dict(zip(ema.names, ema.shadow, strict=True))}, path / name)
        ema.last_step = meta["last_step"]
    return state


def restore_rng(path: str | Path, state: TrainState) -> bool:
    """Restore this rank's generator states when the world size is unchanged; ``False`` otherwise."""
    rank, world_size = rank_world()
    if state.world_size != world_size:
        return False
    file = Path(path) / "rng" / f"rank-{rank:04d}.pt"
    if not file.exists():
        return False
    set_rng_state(torch.load(file, weights_only=True))
    return True


def export_policy(
    path: str | Path, output: str | Path, *, profile: str = "model", dtype: str | None = None
) -> dict[str, Any]:
    """Write a standalone inference export from a checkpoint's ``model`` or an EMA profile folder.

    Runs in one process on the CPU regardless of how many ranks wrote the checkpoint. Tensors that a
    profile does not average (frozen streams and conditioning heads) come from ``model/``.

    ``dtype`` casts the exported weights; ``None`` keeps the checkpoint's dtype. The checkpoint holds fp32
    masters and an export is served in the dtype it was written in: ``"bfloat16"`` for serving, ``None`` to keep
    the masters for further training or diffing.
    """
    from ..models.transformer import JointSingleSeqParams
    from ..models.wiring import action_dit_params, build_action_dit, restrict_content_streams
    from ..policy import write_policy_export

    path = Path(path)
    if not (path / COMPLETE_MARKER).exists():
        raise FileNotFoundError(f"{path} is not a complete checkpoint")
    state = TrainState.from_json((path / "state.json").read_text())
    if profile != "model" and profile not in state.emas:
        raise ValueError(f"unknown profile {profile!r}; choose 'model' or one of {sorted(state.emas)}")
    config = PolicyConfig(**json.loads((path / "config.json").read_text()))
    dit = build_action_dit(
        action_dit_params(
            restrict_content_streams(JointSingleSeqParams(**config.dit_config), config.content_streams),
            config.action_modality,
            config.action_dim,
            attn_mode=config.attn_mode,
            conditioning_channels=config.conditioning_channels,
        ),
        None,
        modality=config.action_modality,
        dtype=getattr(torch, config.torch_dtype),
    )
    tensors: dict[str, Tensor] = {f"dit.{k}": v for k, v in dit.state_dict().items()}
    _dcp_load({"model": tensors}, path / "model")
    if profile != "model":
        averaged = {f"dit.{k}": v for k, v in dit.named_parameters()}
        _dcp_load({"tensors": averaged}, path / profile, partial=True)
    if dtype is not None:
        target = getattr(torch, dtype)
        tensors = {k: (v.to(target) if v.is_floating_point() else v) for k, v in tensors.items()}
    write_policy_export(output, config, tensors, weight_profile=profile)
    return {
        "checkpoint": str(path),
        "step": state.step,
        "profile": profile,
        "output": str(output),
        "tensors": len(tensors),
    }
