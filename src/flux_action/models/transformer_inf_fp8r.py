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
"""Native rowwise-FP8 inference for the released DROID DiT.

Every transformer-block linear is loaded from an E4M3 weight and an FP32
per-row scale. The published checkpoint also quantizes video/text boundaries,
time/vector embedders, and non-action modulations; action boundaries and
modulations plus RMSNorm weights remain BF16. Runtime activation quantization,
Q/K RMSNorm plus RoPE, and SwiGLU are ordinary Torch expressions; matrix
multiplication uses :func:`torch._scaled_mm`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

VIDEO = "video"
VIDEO_COND = "video_cond"
ACTION = "action_prediction_droid"
ACTION_COND = "action_prediction_droid_cond"
_STREAMS = (VIDEO, VIDEO_COND, ACTION, ACTION_COND)
_JOINT_STREAMS = ("txt", VIDEO, VIDEO_COND, ACTION, ACTION_COND)
_CHANNELS = {VIDEO: 96, VIDEO_COND: 96, ACTION: 8, ACTION_COND: 8}
CONTEXT_DIM = 20480
VECTOR_DIM = 768
HIDDEN = 3072
HEADS = 24
DEPTH = 5
JOINT_DEPTH = 28
AXES = (32, 32, 32, 32)
THETA = 10000
MLP = 9216
GEMM_ALIGNMENT = 256
ROW_ALIGNMENT = 128

_BF16_LINEAR_PREFIXES = (
    f"emb_in.{ACTION}",
    f"emb_in.{ACTION_COND}",
    f"early_stream_modulations.{ACTION}",
    f"early_stream_modulations.{ACTION_COND}",
    f"single_stream_modulations.{ACTION}",
    f"single_stream_modulations.{ACTION_COND}",
    f"final_layer.{ACTION}",
    f"final_layer.{ACTION_COND}",
)

ModulationTuple = tuple[Tensor, Tensor, Tensor]


def _is_native_fp8r_weight(name: str) -> bool:
    return name.endswith(".weight") and not name.startswith(_BF16_LINEAR_PREFIXES)


def _quantize_fp8r(value: Tensor) -> tuple[Tensor, Tensor]:
    assert value.dtype == torch.bfloat16 and value.is_contiguous() and value.ndim >= 1, (
        "expected contiguous BF16 weight"
    )
    rows = value.reshape(-1, value.shape[-1]).float()
    scale = (rows.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    quantized = (rows / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.reshape_as(value), scale


def _scaled_mm(
    activation_q: Tensor,
    weight_q_nk: Tensor,
    activation_scale: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    return torch._scaled_mm(
        activation_q,
        weight_q_nk.T,
        activation_scale[:, None].contiguous(),
        weight_scale[None, :].contiguous(),
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )


def _rmsnorm_rope(value: Tensor, weight: Tensor, rope: Tensor, eps: float) -> Tensor:
    normalized = value.float() * torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized.to(torch.bfloat16) * weight
    pairs = normalized.float().reshape(*normalized.shape[:-1], -1, 2)
    cosine = rope[:, :, None, :, 0]
    sine = rope[:, :, None, :, 1]
    even = cosine * pairs[..., 0] - sine * pairs[..., 1]
    odd = sine * pairs[..., 0] + cosine * pairs[..., 1]
    return torch.stack((even, odd), dim=-1).reshape_as(value).to(torch.bfloat16)


def _qk_norm_rope(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    q_rope: Tensor,
    k_rope: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    return (
        _rmsnorm_rope(q, q_weight, q_rope, eps),
        _rmsnorm_rope(k, k_weight, k_rope, eps),
    )


def _swiglu(value: Tensor) -> Tensor:
    gate, up = value.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


@dataclass(frozen=True)
class _PreparedText:
    batch: int
    dtype: torch.dtype
    device: torch.device
    text_len: int
    text: Tensor
    rope: Tensor
    condition_early: tuple[ModulationTuple, ModulationTuple]
    static_joint: tuple[ModulationTuple, ModulationTuple, ModulationTuple]
    vector: Tensor


@dataclass(frozen=True)
class _PreparedRequest:
    batch: int
    dtype: torch.dtype
    device: torch.device
    lengths: tuple[int, int, int, int, int]
    text: Tensor
    video_cond: Tensor
    action_cond: Tensor
    rope: Tensor
    static_joint: tuple[ModulationTuple, ModulationTuple, ModulationTuple]
    vector: Tensor


@dataclass(frozen=True)
class _PreparedStep:
    video_early: ModulationTuple
    action_early: ModulationTuple
    dynamic_joint: tuple[ModulationTuple, ModulationTuple]
    final_video: tuple[Tensor, Tensor]
    final_action: tuple[Tensor, Tensor]


@dataclass(frozen=True)
class DroidPrediction:
    video: Tensor
    action: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        return {"x_video": self.video, "x_action_prediction_droid": self.action}


def _ceil(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _pad_sequence(value: Tensor, alignment: int) -> Tensor:
    padded = _ceil(value.shape[1], alignment)
    if padded == value.shape[1]:
        return value.contiguous()
    return F.pad(value, (0, 0, 0, padded - value.shape[1]))


def _pad_to_tokens(value: Tensor, tokens: int) -> Tensor:
    if value.shape[1] == tokens:
        return value
    return F.pad(value, (0, 0, 0, tokens - value.shape[1]))


def _rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0, "RoPE dimension must be even"
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    phase = torch.einsum("...n,d->...nd", pos, 1.0 / (theta**scale))
    return torch.stack((phase.cos(), phase.sin()), dim=-1).float()


def _timestep_embedding(t: Tensor, dim: int = 256) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = (1000.0 * t.float())[:, None] * frequencies[None]
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class _EmbedND(nn.Module):
    def __init__(self):
        super().__init__()
        self.axes = AXES
        self.theta = THETA

    def forward(self, ids: Tensor) -> Tensor:
        pieces = [_rope(ids[..., axis], dim, self.theta) for axis, dim in enumerate(self.axes)]
        return torch.cat(pieces, dim=-2).contiguous()


class _FP8RLinear(nn.Module):
    """One packed rowwise-FP8 weight with explicit GEMM padding."""

    def __init__(
        self,
        weight_q: Tensor,
        weight_scale: Tensor,
        in_features: int,
        out_features: int,
        row_alignment: int,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.row_alignment = row_alignment
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("weight_scale", weight_scale)

    @classmethod
    @torch.no_grad()
    def pack(
        cls,
        linear: nn.Linear,
        native_scale: Tensor,
    ) -> _FP8RLinear:
        weight = linear.weight
        padded_n = _ceil(linear.out_features, GEMM_ALIGNMENT)
        padded_k = _ceil(linear.in_features, GEMM_ALIGNMENT)
        padding = (0, padded_k - linear.in_features, 0, padded_n - linear.out_features)
        assert weight.dtype == torch.float8_e4m3fn, "expected E4M3 weight"
        weight_q = F.pad(weight, padding).contiguous()
        weight_scale = F.pad(native_scale, (0, padded_n - linear.out_features), value=1.0).contiguous()
        assert linear.bias is None, "FP8r linear cannot have bias"
        return cls(
            weight_q,
            weight_scale,
            linear.in_features,
            linear.out_features,
            ROW_ALIGNMENT,
        )

    def quantize_input(self, x: Tensor) -> tuple[Tensor, Tensor, int, tuple[int, ...]]:
        assert x.dtype == torch.bfloat16, "expected BF16 activations"
        assert x.shape[-1] == self.in_features, "invalid linear input width"
        leading = x.shape[:-1]
        rows = math.prod(leading)
        assert rows >= 1, "empty linear input"
        padded_rows = _ceil(rows, self.row_alignment)
        flat = x.reshape(rows, self.in_features)
        pad_k = self.weight_q.shape[1] - self.in_features
        pad_m = padded_rows - rows
        flat = F.pad(flat, (0, pad_k, 0, pad_m)) if pad_k or pad_m else flat.contiguous()
        activation_q, activation_scale = _quantize_fp8r(flat)
        return activation_q, activation_scale, rows, leading

    def forward_quantized(
        self,
        activation_q: Tensor,
        activation_scale: Tensor,
        rows: int,
        leading: tuple[int, ...],
    ) -> Tensor:
        result = _scaled_mm(
            activation_q,
            self.weight_q,
            activation_scale,
            self.weight_scale,
        )[:rows, : self.out_features]
        return result.reshape(*leading, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_quantized(*self.quantize_input(x))


def _standalone_qkv_epilogue(
    q_value: Tensor,
    k_value: Tensor,
    v_value: Tensor,
    *,
    batch: int,
    tokens: int,
    valid_length: int,
    heads: int,
    norm: _QKNorm,
    rope: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    # Independent scaled-mm outputs become exact contiguous BLHD views before
    # Torch RMSNorm and RoPE.
    q_blhd = q_value.reshape(batch, tokens, heads, -1).contiguous()
    k_blhd = k_value.reshape(batch, tokens, heads, -1).contiguous()
    valid_rope = rope[:, :valid_length].contiguous()
    q, k = _qk_norm_rope(
        q_blhd[:, :valid_length].contiguous(),
        k_blhd[:, :valid_length].contiguous(),
        norm.query_norm.scale,
        norm.key_norm.scale,
        valid_rope,
        valid_rope,
        1e-6,
    )
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v_value.reshape(batch, tokens, heads, -1)[:, :valid_length].transpose(1, 2)
    return q, k, v


class _BlockProjections(nn.Module):
    """One canonical quantization of QKV and W13 for an FP8 block."""

    def __init__(
        self,
        qkv_q: Tensor,
        qkv_scale: Tensor,
        mlp_q: Tensor,
        mlp_scale: Tensor,
        *,
        hidden: int,
        mlp: int,
        row_alignment: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.mlp = mlp
        self.row_alignment = row_alignment
        self.register_buffer("qkv_q", qkv_q)
        self.register_buffer("qkv_scale", qkv_scale)
        self.register_buffer("mlp_q", mlp_q)
        self.register_buffer("mlp_scale", mlp_scale)

    @classmethod
    @torch.no_grad()
    def pack(
        cls,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        mlp_in: nn.Linear,
        *,
        native_scales: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> _BlockProjections:
        qkv_weights = (q_proj.weight, k_proj.weight, v_proj.weight)
        qkv_scales = native_scales[:3]
        assert all(weight.dtype == torch.float8_e4m3fn for weight in qkv_weights), "expected E4M3 QKV"
        qkv_q = torch.cat(qkv_weights, dim=0).contiguous()
        qkv_scale = torch.cat(qkv_scales, dim=0).contiguous()

        mlp_scale = native_scales[3]
        assert mlp_in.weight.dtype == torch.float8_e4m3fn, "expected E4M3 MLP weight"
        mlp_q = mlp_in.weight.contiguous()
        return cls(
            qkv_q,
            qkv_scale,
            mlp_q,
            mlp_scale,
            hidden=q_proj.out_features,
            mlp=mlp_in.out_features // 2,
            row_alignment=ROW_ALIGNMENT,
        )

    def quantize_input(self, active: Tensor) -> tuple[Tensor, Tensor]:
        flat = active.reshape(-1, active.shape[-1]).contiguous()
        assert flat.shape[0] % self.row_alignment == 0, "invalid packed row alignment"
        return _quantize_fp8r(flat)

    def qkv(
        self,
        active_q: Tensor,
        active_scale: Tensor,
        *,
        batch: int,
        tokens: int,
        valid_length: int,
        heads: int,
        norm: _QKNorm,
        rope: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        weights = self.qkv_q.split(self.hidden, dim=0)
        scales = self.qkv_scale.split(self.hidden, dim=0)
        q_value, k_value, v_value = (
            _scaled_mm(active_q, weight, active_scale, scale)
            for weight, scale in zip(weights, scales, strict=True)
        )
        return _standalone_qkv_epilogue(
            q_value,
            k_value,
            v_value,
            batch=batch,
            tokens=tokens,
            valid_length=valid_length,
            heads=heads,
            norm=norm,
            rope=rope,
        )

    def mlp_value(self, active_q: Tensor, active_scale: Tensor, *, batch: int, tokens: int) -> Tensor:
        packed = _scaled_mm(active_q, self.mlp_q, active_scale, self.mlp_scale)
        return _swiglu(packed.reshape(batch, tokens, 2 * self.mlp).contiguous())


class _MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden, bias=False)
        self.out_layer = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


class _RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class _QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = _RMSNorm(dim)
        self.key_norm = _RMSNorm(dim)


class _Modulation(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.lin = nn.Linear(hidden, 3 * hidden, bias=False)

    def forward(self, vector: Tensor) -> ModulationTuple:
        values = self.lin(F.silu(vector))[:, None].chunk(3, dim=-1)
        return values  # type: ignore[return-value]


class _ModeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = HIDDEN
        mlp = MLP
        self.heads = HEADS
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, 2 * mlp, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp, hidden, bias=False)
        self.norm = _QKNorm(hidden // HEADS)
        self.pre_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.projections: _BlockProjections | None = None

    @torch.no_grad()
    def pack_fp8r(
        self,
        native_scales: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> None:
        self.projections = _BlockProjections.pack(
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.mlp_in,
            native_scales=native_scales,
        )
        del self.q_proj, self.k_proj, self.v_proj, self.mlp_in

    def forward(self, x: Tensor, pe: Tensor, modulation: ModulationTuple, valid_length: int) -> Tensor:
        shift, scale, gate = modulation
        active = (1 + scale) * self.pre_norm(x) + shift
        batch, tokens = active.shape[:2]
        assert self.projections is not None, "mode projections are not packed"
        active_q, active_scale = self.projections.quantize_input(active)
        q, k, v = self.projections.qkv(
            active_q,
            active_scale,
            batch=batch,
            tokens=tokens,
            valid_length=valid_length,
            heads=self.heads,
            norm=self.norm,
            rope=pe,
        )
        mlp = self.projections.mlp_value(active_q, active_scale, batch=batch, tokens=tokens)
        attention = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        attention = _pad_to_tokens(attention, x.shape[1])
        branch = self.attn_out(attention) + self.mlp_out(mlp)
        updated = x[:, :valid_length] + gate * branch[:, :valid_length]
        return _pad_to_tokens(updated, x.shape[1])


class _JointBlock(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = HIDDEN
        mlp = MLP
        self.heads = HEADS
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, 2 * mlp, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp, hidden, bias=False)
        self.norm = _QKNorm(hidden // HEADS)
        self.pre_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.projections: _BlockProjections | None = None

    @torch.no_grad()
    def pack_fp8r(
        self,
        native_scales: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> None:
        self.projections = _BlockProjections.pack(
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.mlp_in,
            native_scales=native_scales,
        )
        del self.q_proj, self.k_proj, self.v_proj, self.mlp_in

    def forward(
        self,
        sequence: Tensor,
        lengths: tuple[int, int, int, int, int],
        pe: Tensor,
        modulations: tuple[
            ModulationTuple, ModulationTuple, ModulationTuple, ModulationTuple, ModulationTuple
        ],
    ) -> Tensor:
        valid_length = sum(lengths)
        normalized = self.pre_norm(sequence[:, :valid_length])
        pieces = normalized.split(lengths, dim=1)
        active_valid = torch.cat(
            [(1 + modulation[1]) * piece + modulation[0] for piece, modulation in zip(pieces, modulations)],
            dim=1,
        )
        active = _pad_to_tokens(active_valid, sequence.shape[1])
        assert self.projections is not None, "joint projections are not packed"
        batch, tokens = active.shape[:2]
        active_q, active_scale = self.projections.quantize_input(active)
        q, k, v = self.projections.qkv(
            active_q,
            active_scale,
            batch=batch,
            tokens=tokens,
            valid_length=valid_length,
            heads=self.heads,
            norm=self.norm,
            rope=pe,
        )
        attention = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        attention = _pad_to_tokens(attention, sequence.shape[1])
        mlp = self.projections.mlp_value(active_q, active_scale, batch=batch, tokens=tokens)
        branch = self.attn_out(attention) + self.mlp_out(mlp)
        branch_pieces = branch[:, :valid_length].split(lengths, dim=1)
        sequence_pieces = sequence[:, :valid_length].split(lengths, dim=1)
        updated = torch.cat(
            [
                value + modulation[2] * update
                for value, modulation, update in zip(sequence_pieces, modulations, branch_pieces)
            ],
            dim=1,
        )
        return _pad_to_tokens(updated, sequence.shape[1])


class _LastLayer(nn.Module):
    def __init__(self, hidden: int, channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))


class FP8RInferenceDiT(nn.Module):
    """Released DROID architecture with owned packed FP8 weights and prepared state."""

    def __init__(self):
        super().__init__()
        self._packed = False
        self._compiled_text = None
        self._compiled_observation = None
        self._compiled_step = None
        self._compiled_hot = None
        self.pe_embedder = _EmbedND()
        self.emb_in = nn.ModuleDict(
            {name: nn.Linear(width, HIDDEN, bias=False) for name, width in _CHANNELS.items()}
        )
        self.txt_in = nn.Linear(CONTEXT_DIM, HIDDEN, bias=False)
        self.time_in = _MLPEmbedder(256, HIDDEN)
        self.vector_in = _MLPEmbedder(VECTOR_DIM, HIDDEN)
        self.early_stream_modulations = nn.ModuleDict(
            {name: _Modulation(HIDDEN) for name in (*_STREAMS, "txt")}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {name: _Modulation(HIDDEN) for name in (*_STREAMS, "txt")}
        )
        self.content_mode_blocks = nn.ModuleDict(
            {name: nn.ModuleList([_ModeBlock() for _ in range(DEPTH)]) for name in _STREAMS}
        )
        self.txt_mode_blocks = nn.ModuleList([_ModeBlock() for _ in range(DEPTH)])
        self.single_blocks = nn.ModuleList([_JointBlock() for _ in range(JOINT_DEPTH)])
        self.final_layer = nn.ModuleDict(
            {name: _LastLayer(HIDDEN, width) for name, width in _CHANNELS.items()}
        )

    @staticmethod
    def _check_runtime_device(device: torch.device) -> None:
        assert device.type == "cuda", "native FP8r inference requires CUDA"
        capability = torch.cuda.get_device_capability(device)
        supported = {(8, 9), (9, 0), (10, 0), (10, 3), (12, 0)}
        # TODO: REMOVE ME after SM89 rowwise FP8 runtime validation.
        if capability not in supported:
            raise RuntimeError(
                "FP8r inference requires SM89/SM90/SM100/SM103/SM120 native FP8 support, "
                f"got SM{capability[0]}{capability[1]}"
            )

    @classmethod
    def from_quantized_state_dict(cls, state_dict: Mapping[str, Tensor]) -> FP8RInferenceDiT:
        """Strictly load the released native-FP8r checkpoint without changing its payloads."""
        devices = {value.device for value in state_dict.values()}
        assert len(devices) == 1, "weights need one device"
        base = {key: value for key, value in state_dict.items() if not key.endswith(".weight_scale")}
        scales = {key: value for key, value in state_dict.items() if key.endswith(".weight_scale")}
        expected_scales: set[str] = set()
        for key, value in base.items():
            if not _is_native_fp8r_weight(key):
                assert value.dtype == torch.bfloat16, "expected BF16 boundary weight"
            else:
                assert value.dtype == torch.float8_e4m3fn, "expected E4M3 weight"
                scale_key = f"{key.removesuffix('.weight')}.weight_scale"
                expected_scales.add(scale_key)
                scale = scales[scale_key]
                assert scale.dtype == torch.float32 and scale.shape == (value.shape[0],), "invalid FP8r scale"
                assert value.is_contiguous() and scale.is_contiguous(), "FP8r payload must be contiguous"
        assert set(scales) == expected_scales, "FP8r scale set mismatch"

        with torch.device("meta"):
            model = cls()
        model.load_state_dict(base, strict=True, assign=True)
        del model.final_layer[VIDEO_COND]
        del model.final_layer[ACTION_COND]
        model._pack_linears(scales)
        model.eval()
        return model

    def _pack_linears(self, native_scales: Mapping[str, Tensor]) -> None:
        assert not self._packed, "model already packed"

        def scale_for(path: str) -> Tensor:
            scale_path = f"{path.removesuffix('.weight')}.weight_scale"
            return native_scales[scale_path]

        def convert(module: nn.Module, path: str) -> None:
            for name, child in list(module.named_children()):
                child_path = f"{path}.{name}" if path else name
                weight_path = f"{child_path}.weight"
                if isinstance(child, nn.Linear) and _is_native_fp8r_weight(weight_path):
                    setattr(
                        module,
                        name,
                        _FP8RLinear.pack(child, scale_for(weight_path)),
                    )
                else:
                    convert(child, child_path)

        # Each block shares one activation quantization across Q/K/V and W13.
        for stream, blocks in self.content_mode_blocks.items():
            for index, block in enumerate(blocks):
                prefix = f"content_mode_blocks.{stream}.{index}"
                block.pack_fp8r(
                    tuple(
                        scale_for(f"{prefix}.{name}.weight")
                        for name in ("q_proj", "k_proj", "v_proj", "mlp_in")
                    ),
                )
        for index, block in enumerate(self.txt_mode_blocks):
            prefix = f"txt_mode_blocks.{index}"
            block.pack_fp8r(
                tuple(
                    scale_for(f"{prefix}.{name}.weight") for name in ("q_proj", "k_proj", "v_proj", "mlp_in")
                ),
            )
        for index, block in enumerate(self.single_blocks):
            prefix = f"single_blocks.{index}"
            block.pack_fp8r(
                tuple(
                    scale_for(f"{prefix}.{name}.weight") for name in ("q_proj", "k_proj", "v_proj", "mlp_in")
                ),
            )

        for name, child in list(self.named_children()):
            if name in ("content_mode_blocks", "txt_mode_blocks", "single_blocks"):
                convert(child, name)
            elif isinstance(child, nn.Linear):
                weight_path = f"{name}.weight"
                if _is_native_fp8r_weight(weight_path):
                    setattr(self, name, _FP8RLinear.pack(child, scale_for(weight_path)))
            else:
                convert(child, name)
        self._packed = True

    def train(self, mode: bool = True) -> FP8RInferenceDiT:
        assert not mode, "FP8RInferenceDiT is inference-only"
        super().train(False)
        return self

    def _weight_spec(self) -> tuple[torch.device, torch.dtype]:
        action_input = self.emb_in[ACTION]
        assert self._packed and isinstance(action_input, nn.Linear), "native model is not packed"
        assert action_input.weight.dtype == torch.bfloat16, "expected BF16 action input"
        return action_input.weight.device, torch.bfloat16

    def validate_runtime_device(self) -> None:
        """Require native-FP8 hardware only when the restored model is used for inference."""
        device, _ = self._weight_spec()
        self._check_runtime_device(device)
        for module in self.modules():
            if isinstance(module, _FP8RLinear):
                assert module.weight_q.dtype == torch.float8_e4m3fn, "expected E4M3 weight"
                assert module.weight_scale.dtype == torch.float32, "expected FP32 scale"
            elif isinstance(module, _BlockProjections):
                assert module.qkv_q.dtype == module.mlp_q.dtype == torch.float8_e4m3fn, (
                    "expected E4M3 block weights"
                )
                assert module.qkv_scale.dtype == module.mlp_scale.dtype == torch.float32, (
                    "expected FP32 block scales"
                )

    def compile_static(self) -> None:
        assert not any(
            value is not None
            for value in (self._compiled_text, self._compiled_observation, self._compiled_step)
        ), "static path already compiled"
        options = dict(dynamic=False, fullgraph=True, mode="reduce-overhead")
        self._compiled_text = torch.compile(self._prepare_text_math, **options)
        self._compiled_observation = torch.compile(self._prepare_observation_math, **options)
        self._compiled_step = torch.compile(self._prepare_step_math, **options)

    def compile_hot(self) -> None:
        assert self._compiled_hot is None, "hot path already compiled"
        self._compiled_hot = torch.compile(
            self._forward_math, dynamic=False, fullgraph=True, mode="reduce-overhead"
        )

    def _embed_time(self, timestep: Tensor) -> Tensor:
        return self.time_in(_timestep_embedding(timestep).to(torch.bfloat16))

    def _zero_vector(self, vector: Tensor) -> Tensor:
        zero = torch.zeros(vector.shape[0], device=vector.device, dtype=torch.float32)
        return self._embed_time(zero) + vector

    def _prepare_text_math(self, ctx: Tensor, ctx_ids: Tensor, vector: Tensor) -> tuple[Tensor, ...]:
        vector_embedding = self.vector_in(vector)
        static = self._zero_vector(vector_embedding)
        text_early = self.early_stream_modulations["txt"](static)
        condition_early = (
            self.early_stream_modulations[VIDEO_COND](static),
            self.early_stream_modulations[ACTION_COND](static),
        )
        static_joint = tuple(
            self.single_stream_modulations[name](static) for name in ("txt", VIDEO_COND, ACTION_COND)
        )
        pe = self.pe_embedder(ctx_ids)
        text = _pad_sequence(self.txt_in(ctx), ROW_ALIGNMENT)
        for block in self.txt_mode_blocks:
            text = block(text, pe, text_early, ctx.shape[1])
        return (
            text,
            pe,
            *condition_early[0],
            *condition_early[1],
            *static_joint[0],
            *static_joint[1],
            *static_joint[2],
            vector_embedding,
        )

    @torch.no_grad()
    def prepare_text(self, ctx: Tensor, ctx_ids: Tensor, vector: Tensor) -> _PreparedText:
        device, dtype = self._weight_spec()
        batch = ctx.shape[0]
        core = self._compiled_text or self._prepare_text_math
        text, pe, *values = core(ctx, ctx_ids, vector)
        return _PreparedText(
            batch,
            dtype,
            device,
            ctx.shape[1],
            text.detach().clone(),
            pe.detach().clone(),
            (tuple(x.detach().clone() for x in values[0:3]), tuple(x.detach().clone() for x in values[3:6])),  # type: ignore[arg-type]
            tuple(tuple(x.detach().clone() for x in values[start : start + 3]) for start in (6, 9, 12)),  # type: ignore[arg-type]
            values[15].detach().clone(),
        )

    def _prepare_observation_math(
        self,
        text_rope: Tensor,
        video_ids: Tensor,
        video_cond: Tensor,
        video_cond_ids: Tensor,
        action_ids: Tensor,
        action_cond: Tensor,
        action_cond_ids: Tensor,
        video_shift: Tensor,
        video_scale: Tensor,
        video_gate: Tensor,
        action_shift: Tensor,
        action_scale: Tensor,
        action_gate: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ropes = tuple(
            self.pe_embedder(ids) for ids in (video_ids, video_cond_ids, action_ids, action_cond_ids)
        )
        video_value = _pad_sequence(self.emb_in[VIDEO_COND](video_cond), ROW_ALIGNMENT)
        action_value = _pad_sequence(self.emb_in[ACTION_COND](action_cond), ROW_ALIGNMENT)
        for block in self.content_mode_blocks[VIDEO_COND]:
            video_value = block(
                video_value,
                ropes[1],
                (video_shift, video_scale, video_gate),
                video_cond.shape[1],
            )
        for block in self.content_mode_blocks[ACTION_COND]:
            action_value = block(
                action_value,
                ropes[3],
                (action_shift, action_scale, action_gate),
                action_cond.shape[1],
            )
        return video_value, action_value, torch.cat((text_rope, *ropes), dim=1)

    @torch.no_grad()
    def prepare_observation(
        self,
        text: _PreparedText,
        *,
        video_ids: Tensor,
        video_cond: Tensor,
        video_cond_ids: Tensor,
        action_ids: Tensor,
        action_cond: Tensor,
        action_cond_ids: Tensor,
    ) -> _PreparedRequest:
        core = self._compiled_observation or self._prepare_observation_math
        video_value, action_value, pe = core(
            text.rope,
            video_ids,
            video_cond,
            video_cond_ids,
            action_ids,
            action_cond,
            action_cond_ids,
            *text.condition_early[0],
            *text.condition_early[1],
        )
        return _PreparedRequest(
            text.batch,
            text.dtype,
            text.device,
            (
                text.text_len,
                video_ids.shape[1],
                video_cond.shape[1],
                action_ids.shape[1],
                action_cond.shape[1],
            ),
            text.text,
            video_value.detach().clone(),
            action_value.detach().clone(),
            pe.detach().clone(),
            text.static_joint,
            text.vector,
        )

    def _prepare_step_math(
        self, vector: Tensor, video_timestep: Tensor, action_timestep: Tensor
    ) -> tuple[Tensor, ...]:
        video_vector = self._embed_time(video_timestep) + vector
        action_vector = self._embed_time(action_timestep) + vector
        video_early = self.early_stream_modulations[VIDEO](video_vector)
        action_early = self.early_stream_modulations[ACTION](action_vector)
        dynamic_joint = (
            self.single_stream_modulations[VIDEO](video_vector),
            self.single_stream_modulations[ACTION](action_vector),
        )
        video_final = self.final_layer[VIDEO].adaLN_modulation(video_vector)[:, None].chunk(2, dim=-1)
        action_final = self.final_layer[ACTION].adaLN_modulation(action_vector)[:, None].chunk(2, dim=-1)
        return (
            *video_early,
            *action_early,
            *dynamic_joint[0],
            *dynamic_joint[1],
            video_final[0],
            video_final[1] + 1,
            action_final[0],
            action_final[1] + 1,
        )

    def _scalar_timestep(self, value: Tensor, request: _PreparedRequest) -> Tensor:
        if value.ndim == 0:
            value = value.expand(request.batch)
        elif value.shape == (request.batch, 1):
            value = value[:, 0]
        else:
            assert value.shape == (request.batch,), "invalid timestep shape"
        assert value.dtype == torch.float32 and value.device == request.device, "invalid timestep placement"
        return value

    def _prepare_step(
        self, request: _PreparedRequest, video_timestep: Tensor, action_timestep: Tensor
    ) -> _PreparedStep:
        video_timestep = self._scalar_timestep(video_timestep, request)
        action_timestep = self._scalar_timestep(action_timestep, request)
        core = self._compiled_step or self._prepare_step_math
        values = core(request.vector, video_timestep, action_timestep)
        values = tuple(value.detach().clone() for value in values)
        return _PreparedStep(
            tuple(values[0:3]),  # type: ignore[arg-type]
            tuple(values[3:6]),  # type: ignore[arg-type]
            (tuple(values[6:9]), tuple(values[9:12])),  # type: ignore[arg-type]
            (values[12], values[13]),
            (values[14], values[15]),
        )

    @torch.no_grad()
    def prepare_steps(
        self, request: _PreparedRequest, video_timesteps: Tensor, action_timesteps: Tensor
    ) -> tuple[_PreparedStep, ...]:
        assert video_timesteps.ndim == action_timesteps.ndim == 1, "invalid timestep rank"
        assert video_timesteps.shape == action_timesteps.shape, "timestep shape mismatch"
        return tuple(
            self._prepare_step(request, video_timesteps[index], action_timesteps[index])
            for index in range(video_timesteps.numel())
        )

    def _forward_math(
        self, request: _PreparedRequest, step: _PreparedStep, video: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor]:
        video_value = _pad_sequence(self.emb_in[VIDEO](video), ROW_ALIGNMENT)
        action_value = _pad_sequence(self.emb_in[ACTION](action), ROW_ALIGNMENT)
        text_len, video_len, video_cond_len, action_len, _ = request.lengths
        video_pe = request.rope[:, text_len : text_len + video_len]
        action_start = text_len + video_len + video_cond_len
        action_pe = request.rope[:, action_start : action_start + action_len]
        for block in self.content_mode_blocks[VIDEO]:
            video_value = block(video_value, video_pe, step.video_early, video_len)
        for block in self.content_mode_blocks[ACTION]:
            action_value = block(action_value, action_pe, step.action_early, action_len)
        valid_streams = (
            request.text[:, :text_len],
            video_value[:, :video_len],
            request.video_cond[:, :video_cond_len],
            action_value[:, :action_len],
            request.action_cond[:, : request.lengths[4]],
        )
        sequence = _pad_sequence(torch.cat(valid_streams, dim=1), ROW_ALIGNMENT)
        mods = (
            request.static_joint[0],
            step.dynamic_joint[0],
            request.static_joint[1],
            step.dynamic_joint[1],
            request.static_joint[2],
        )
        for block in self.single_blocks:
            sequence = block(sequence, request.lengths, request.rope, mods)
        offsets = (text_len, text_len + video_len + video_cond_len)
        video_hidden = self.final_layer[VIDEO].norm_final(sequence[:, offsets[0] : offsets[0] + video_len])
        video_hidden = video_hidden * step.final_video[1] + step.final_video[0]
        action_hidden = self.final_layer[ACTION].norm_final(sequence[:, offsets[1] : offsets[1] + action_len])
        action_hidden = action_hidden * step.final_action[1] + step.final_action[0]
        return self.final_layer[VIDEO].linear(video_hidden), self.final_layer[ACTION].linear(action_hidden)

    @torch.no_grad()
    def forward_prepared(
        self,
        request: _PreparedRequest,
        step: _PreparedStep,
        *,
        video: Tensor,
        action: Tensor,
    ) -> DroidPrediction:
        core = self._compiled_hot or self._forward_math
        video_output, action_output = core(request, step, video, action)
        return DroidPrediction(video_output, action_output)

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("use prepare_text/prepare_observation/prepare_steps/forward_prepared")


__all__ = ["DroidPrediction", "FP8RInferenceDiT"]
