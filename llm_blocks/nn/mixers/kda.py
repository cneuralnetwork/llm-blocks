"""Kimi Delta Attention mixer (Kimi Team, "Kimi Linear", arXiv:2510.26692).

Layer structure:

* projections for q, k, v and an output gate; short causal depthwise convolutions + SiLU on
  q/k/v; per-head L2-normalised q and k;
* ``beta = sigmoid(W_b x + b_b)`` (one write strength per head);
* per-channel decay from a low-rank projection with a per-head temperature
  ``g = g_min * sigmoid(exp(A_log) * (W_up W_down x + b_up))``, so ``g`` lies in ``(g_min, 0)``,
  which bounds the per-step decay and keeps the chunked kernel's tiles in range;
* KDA recurrence (:mod:`llm_blocks.kernels.kda`);
* per-head RMSNorm with a separate weight per head, ``sigmoid(W_o x)`` output gate, output
  projection.

Training / prefill use the chunked algorithm in pure PyTorch; decoding uses the O(1) recurrent
update. q is not scaled by ``1/sqrt(d_k)``; the per-head RMSNorm on the output removes any
per-head scale on q anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ...configs.spec import BuildContext, SpecError
from ...kernels._dtype import acc_dtype
from ...kernels.kda import kda_chunked, kda_safe_tile, kda_step
from ..base import MixerSpec, MixerState, SequenceMixer, mark_residual_output
from ..norm import rms_norm
from ..position import PositionSpec, check_position
from .conv import CausalDepthwiseConv1d

__all__ = ["KDAState", "KimiDeltaAttention", "KDA"]


class KDAState(MixerState):
    """Conv window ``(B, 2*Dk + Dv, d_conv-1)`` and fast-weight state ``(B, H, dk, dv)`` (fp32)."""

    def __init__(self, conv: Tensor, S: Tensor):
        self.conv, self.S = conv, S
        self.offset = 0


class KimiDeltaAttention(SequenceMixer):
    def __init__(
        self,
        dim: int,
        *,
        heads: int,
        head_dim: int,
        expand_v: float = 1.0,
        d_conv: int = 4,
        conv_bias: bool = True,
        decay_rank: int = 32,
        g_min: float = -5.0,
        chunk_size: int = 64,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if g_min >= 0:
            raise SpecError(f"KDA: g_min must be negative (lower bound of the per-step log-decay), got {g_min}")
        self.dim, self.heads, self.head_k = dim, heads, head_dim
        self.head_v = int(head_dim * expand_v)
        if self.head_v <= 0:
            raise SpecError("KDA: expand_v * head_dim must be positive")
        self.key_dim, self.value_dim = heads * head_dim, heads * self.head_v
        self.g_min, self.chunk_size, self.d_conv, self.norm_eps = g_min, chunk_size, d_conv, norm_eps
        self.tile = kda_safe_tile(chunk_size, g_min)
        self.conv_dim = 2 * self.key_dim + self.value_dim
        # fused projection: [q | k | v | output gate]
        self.in_proj = nn.Linear(dim, self.conv_dim + self.value_dim, bias=False)
        self.conv1d = CausalDepthwiseConv1d(self.conv_dim, d_conv, bias=conv_bias)
        self.b_proj = nn.Linear(dim, heads, bias=True)
        self.f_down = nn.Linear(dim, decay_rank, bias=False)
        self.f_up = nn.Linear(decay_rank, self.key_dim, bias=True)
        self.A_log = nn.Parameter(torch.zeros(heads))  # per-head temperature exp(A_log), 1 at init
        self.o_norm_weight = nn.Parameter(torch.ones(heads, self.head_v))
        for p in (self.A_log, self.o_norm_weight):
            p._no_weight_decay = True  # type: ignore[attr-defined]
        self.o_proj = mark_residual_output(nn.Linear(self.value_dim, dim, bias=False))

    def forward(self, x: Tensor, *, state: Optional[KDAState] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x.shape
        H, dk, dv = self.heads, self.head_k, self.head_v
        acc = acc_dtype(x)
        qkv, gate = self.in_proj(x).split([self.conv_dim, self.value_dim], dim=-1)
        if padding_mask is not None:
            qkv = qkv * padding_mask[..., None].to(qkv.dtype)
        qkv, new_conv = self.conv1d(qkv, state.conv if state is not None else None)
        qkv = F.silu(qkv)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = F.normalize(q.to(acc).view(B, T, H, dk), dim=-1)
        k = F.normalize(k.to(acc).view(B, T, H, dk), dim=-1)
        v = v.to(acc).view(B, T, H, dv)
        beta = torch.sigmoid(self.b_proj(x).to(acc))
        z = self.f_up(self.f_down(x)).to(acc).view(B, T, H, dk)
        g = self.g_min * torch.sigmoid(self.A_log.to(acc).exp()[:, None] * z)
        if padding_mask is not None:
            m = padding_mask.to(acc)
            beta, g = beta * m[..., None], g * m[..., None, None]  # padded steps leave the state unchanged

        if state is not None and T == 1:
            o, S = kda_step(state.S, q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0])
            o = o.unsqueeze(1)
        else:
            o, S = kda_chunked(q, k, v, g, beta, self.chunk_size, state.S if state is not None else None,
                               g_min=self.g_min, tile=self.tile)
        if state is not None:
            if new_conv is not None:
                state.conv = new_conv
            state.S = S
            state.offset += T
        o = (rms_norm(o, None, self.norm_eps) * self.o_norm_weight.to(acc)).to(x.dtype)
        o = o.reshape(B, T, self.value_dim) * torch.sigmoid(gate)
        return self.o_proj(o)

    def init_state(self, batch_size, max_seq_len, *, device=None, dtype=None) -> KDAState:
        p = self.in_proj.weight
        device, dtype = device or p.device, dtype or p.dtype
        conv = torch.zeros(batch_size, self.conv_dim, self.d_conv - 1, device=device, dtype=dtype)
        S = torch.zeros(batch_size, self.heads, self.head_k, self.head_v, device=device,
                        dtype=torch.float64 if dtype == torch.float64 else torch.float32)
        return KDAState(conv, S)

    def flops_per_token(self, seq_len: int) -> float:
        linears = (self.in_proj, self.b_proj, self.f_down, self.f_up, self.o_proj)
        proj = 2.0 * sum(m.weight.numel() for m in linears)
        conv = 2.0 * self.conv_dim * self.d_conv
        # recurrent form: decay, k^T S, rank-1 update, q^T S  ≈ 4 MACs per state entry
        return proj + conv + 2.0 * 4 * self.heads * self.head_k * self.head_v

    def extra_repr(self) -> str:
        return (f"heads={self.heads}, head_k={self.head_k}, head_v={self.head_v}, g_min={self.g_min}, "
                f"chunk_size={self.chunk_size}, tile={self.tile}")


@dataclass(frozen=True, kw_only=True)
class KDA(MixerSpec):
    """Kimi Delta Attention mixer spec.

    Args:
        heads: number of heads (default: ``dim // head_dim``).
        head_dim: key/query head dimension.
        expand_v: value head dim = ``expand_v * head_dim``.
        decay_rank: rank of the low-rank per-channel decay projection.
        g_min: lower bound of the per-step, per-channel log-decay (negative).
        conv_bias: bias on the short q/k/v convolutions.
        chunk_size: chunk length of the parallel training/prefill algorithm.
    """

    heads: Optional[int] = None
    head_dim: int = 64
    expand_v: float = 1.0
    d_conv: int = 4
    conv_bias: bool = True
    decay_rank: int = 32
    g_min: float = -5.0
    chunk_size: int = 64
    position: Optional[PositionSpec] = None

    def build(self, ctx: BuildContext) -> KimiDeltaAttention:
        check_position(self.position, ("none",), "KDA")
        heads = self.heads or max(1, ctx.dim // self.head_dim)
        return KimiDeltaAttention(ctx.dim, heads=heads, head_dim=self.head_dim, expand_v=self.expand_v,
                                  d_conv=self.d_conv, conv_bias=self.conv_bias, decay_rank=self.decay_rank,
                                  g_min=self.g_min, chunk_size=self.chunk_size)
