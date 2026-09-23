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
"""Process group, device mesh and FSDP2 sharding of the DiT, following the reference layout.

The reference sharded every transformer block and the root with ``bfloat16`` parameters for
compute, ``float32`` gradient reduction, ``reshard_after_forward=True`` and full activation
checkpointing per block (``preserve_rng_state=False``). Master weights stay in float32 inside
FSDP; the optimizer and the EMAs see float32 DTensor shards.

Group layout as in the reference: block groups cast their forward inputs to the compute dtype (the
rotary tables enter attention in bfloat16), ``time_in`` is its own group in the master precision, and
the root casts nothing (the policy feeds streams and context in the compute dtype and fp32 timesteps
that already sit on the bf16 grid).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DeviceMesh, init_device_mesh

CHECKPOINT_WRAPPED = "_checkpoint_wrapped_module."


def init_distributed(backend: str | None = None) -> tuple[int, int, torch.device]:
    """``(rank, world_size, device)`` from the torchrun environment; single process without it."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = backend or ("nccl" if device.type == "cuda" else "gloo")
        kwargs = {"device_id": device} if device.type == "cuda" else {}
        # Collectives wait this long before failing (default 10 min); checkpoint writes from many ranks to
        # shared storage and slow container starts need more. FLUX_ACTION_PG_TIMEOUT_MINUTES overrides.
        timeout = timedelta(minutes=float(os.environ.get("FLUX_ACTION_PG_TIMEOUT_MINUTES", "30")))
        dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout, **kwargs)
    return rank, world_size, device


def rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor(float(value), device=device if device.type == "cuda" else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / dist.get_world_size())


def all_reduce_means(values: torch.Tensor) -> list[float]:
    """Reduce detached metrics together, synchronizing with the host once per update."""
    if dist.is_available() and dist.is_initialized():
        values = values.float()
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return values.tolist()


def global_grad_norm(module: nn.Module, device: torch.device, replicas: int = 1) -> float:
    """L2 norm of all gradients across ranks, ``inf``/``nan`` if any shard is non-finite.

    Sharded gradients (DTensors) contribute their local shard; ``replicas`` is the number of HSDP replica
    groups holding identical shards, so their sum is divided out. One collective per call.
    """
    from torch.distributed.tensor import DTensor

    grads = []
    for param in module.parameters():
        grad = param.grad
        if grad is None:
            continue
        grads.append((grad.to_local() if isinstance(grad, DTensor) else grad).detach())
    if grads:
        total = torch.stack([n.float() for n in torch._foreach_norm(grads)]).pow(2).sum()
    else:
        total = torch.zeros((), dtype=torch.float32)
    total = total.to(device if device.type == "cuda" else "cpu")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    value = float(total.item())
    return math.sqrt(value / max(replicas, 1)) if math.isfinite(value) else value


def build_mesh(world_size: int, shard_size: int | None, device_type: str) -> DeviceMesh:
    """``shard_size`` ranks shard each parameter (FSDP); ``world_size // shard_size`` replicas (HSDP).

    Reference: replicate 16 x shard 4 on GB200 (one node per shard group).
    """
    shard = shard_size or world_size
    if shard < 1 or world_size % shard:
        raise ValueError(f"shard_size {shard} must divide the world size {world_size}")
    replicate = world_size // shard
    if replicate == 1:
        return init_device_mesh(device_type, (shard,), mesh_dim_names=("shard",))
    return init_device_mesh(device_type, (replicate, shard), mesh_dim_names=("replicate", "shard"))


def dit_blocks(dit: nn.Module) -> Iterable[tuple[nn.ModuleList, int]]:
    """Every transformer block of the DiT as ``(container, index)`` so it can be replaced in place."""
    for kind in ("content_mode_blocks", "late_content_mode_blocks"):
        for blocks in getattr(dit, kind, {}).values():
            for i in range(len(blocks)):
                yield blocks, i
    for kind in ("txt_mode_blocks", "single_blocks", "late_txt_mode_blocks"):
        blocks = getattr(dit, kind, None)
        if blocks is not None:
            for i in range(len(blocks)):
                yield blocks, i


def apply_activation_checkpointing(dit: nn.Module) -> int:
    """Non-reentrant checkpointing of every block (the reference's ``AC_MODE=full``)."""
    n = 0
    for blocks, i in list(dit_blocks(dit)):
        blocks[i] = checkpoint_wrapper(
            blocks[i], checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False
        )
        n += 1
    return n


def shard_dit(
    dit: nn.Module,
    mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
    activation_checkpointing: bool = True,
) -> nn.Module:
    """Wrap the DiT in place the way the reference trainer does: blocks, ``time_in``, then the root.

    Blocks compute in ``param_dtype`` and cast their inputs to it (so the fp32 rotary tables enter the
    attention in ``param_dtype``, as in the reference). ``time_in`` keeps the precision of its stored
    parameters, the reference's "high precision for t_embedder" group; ``JointSingleSeq.embed_timesteps``
    feeds it in that dtype and casts the result for the modulations. The root casts nothing: the policy
    hands it streams in the compute dtype and fp32 timesteps (already on the bf16 grid, see
    ``packing.add_noise``).
    """
    if activation_checkpointing:
        apply_activation_checkpointing(dit)
    block_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
    )
    for blocks, i in dit_blocks(dit):
        fully_shard(blocks[i], mesh=mesh, mp_policy=block_policy, reshard_after_forward=reshard_after_forward)
    time_policy = MixedPrecisionPolicy(
        param_dtype=next(dit.time_in.parameters()).dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False
    )
    fully_shard(dit.time_in, mesh=mesh, mp_policy=time_policy, reshard_after_forward=reshard_after_forward)
    root_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False
    )
    fully_shard(dit, mesh=mesh, mp_policy=root_policy, reshard_after_forward=reshard_after_forward)
    return dit


def clean_name(name: str) -> str:
    """Parameter name without activation-checkpoint wrapper segments."""
    return name.replace(CHECKPOINT_WRAPPED, "")
