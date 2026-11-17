"""Pure-PyTorch Kimi Delta Attention kernel (KDA, Kimi Team, "Kimi Linear", arXiv:2510.26692).

KDA is the gated delta rule with a *per-channel* decay: one decay per key dimension instead
of one scalar per head. Per head, with fast-weight state ``S`` of shape ``(d_k, d_v)``:

    S~_t = Diag(exp(g_t)) S_{t-1}                       # g_t in R^{d_k}, g_t <= 0
    S_t  = S~_t + beta_t k_t (v_t - S~_t^T k_t)^T        # delta rule correction
    o_t  = S_t^T q_t

i.e. ``S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T``.

Chunked algorithm (written from scratch in PyTorch to run on MPS, without CUDA/Triton kernels):

* With ``G`` the within-chunk cumulative log-decay (per channel), the WY system for the
  effective writes is ``(I + Diag(beta) M) U = Diag(beta) (V - K^ S_0)`` with
  ``M_ij = sum_d k_id k_jd exp(G_id - G_jd)`` strictly lower triangular. Unlike the scalar
  gate, the decay cannot be factored out of the inner product, so the decayed inner products
  are computed as ``(X * e^{G - ref}) (Y * e^{ref - G})^T``.
* **Secondary tiles.** ``e^{-G}`` overflows float32 once the decay accumulated across a chunk
  exceeds ~88 nats. Each column tile of width ``tile`` is therefore re-referenced to its own
  first row ``ref``: the X factor ``exp(min(G_i - ref, 0)) <= 1`` and the Y factor
  ``exp(ref - G_j) <= exp((tile - 1) * |g_min|)``. The ``min(., 0)`` matters: rows above the
  diagonal are masked out afterwards, but ``inf * 0 = nan`` would survive the mask.
* **Triangular inverse by doubling.** ``(I - A)^{-1} = (I + A)(I + A^2)(I + A^4)...`` for the
  strictly lower (hence nilpotent) ``A``: ``ceil(log2 C)`` dense matmuls, parallel and
  autograd-safe, with no in-place forward substitution and no ``solve_triangular``. It runs
  on any backend, including MPS.

Inputs are expected L2-normalised (q, k); math runs in float32 (float64 for float64 inputs).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ._dtype import acc_dtype

__all__ = ["kda_chunked", "kda_recurrent", "kda_step", "kda_safe_tile"]


def kda_step(state: Tensor, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> Tuple[Tensor, Tensor]:
    """One token. ``state (b,h,dk,dv)``, ``q,k,g (b,h,dk)``, ``v (b,h,dv)``, ``beta (b,h)``."""
    acc = acc_dtype(v)
    state = state * torch.exp(g.to(acc))[..., None]  # Diag(alpha_t) S_{t-1}
    k = k.to(acc)
    kv_mem = torch.einsum("bhkv,bhk->bhv", state, k)
    delta = (v.to(acc) - kv_mem) * beta.to(acc)[..., None]
    state = state + k[..., :, None] * delta[..., None, :]
    o = torch.einsum("bhkv,bhk->bhv", state, q.to(acc))
    return o, state


def kda_recurrent(q, k, v, g, beta, initial_state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Token-by-token reference. ``q,k,g (b,l,h,dk)``, ``v (b,l,h,dv)``, ``beta (b,l,h)``."""
    b, L, h, dk = k.shape
    dv = v.shape[-1]
    acc = acc_dtype(v)
    state = (torch.zeros(b, h, dk, dv, device=q.device, dtype=acc)
             if initial_state is None else initial_state.to(acc))
    outs = []
    for t in range(L):
        o, state = kda_step(state, q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t])
        outs.append(o)
    return torch.stack(outs, dim=1).to(v.dtype), state


def kda_safe_tile(chunk_size: int, g_min: float, dtype: torch.dtype = torch.float32) -> int:
    """Largest divisor of ``chunk_size`` with ``(tile - 1) * |g_min|`` safely inside ``dtype``'s exponent range."""
    limit = math.log(torch.finfo(dtype).max) - 8.0  # float32: ~80.7, float64: ~701
    tile = chunk_size if g_min == 0 else min(chunk_size, int(limit / abs(g_min)) + 1)
    tile = max(1, tile)
    while chunk_size % tile:
        tile -= 1
    return tile


def _inv_unit_lower(A: Tensor) -> Tensor:
    """``(I - A)^{-1}`` for strictly lower-triangular ``A`` (..., C, C) by repeated squaring."""
    C = A.shape[-1]
    eye = torch.eye(C, device=A.device, dtype=A.dtype)
    X, P = eye + A, A
    for _ in range(max(0, (C - 1).bit_length() - 1)):
        P = P @ P  # A^(2^s)
        X = X @ (eye + P)
    return X


def _decayed_inner(k: Tensor, q: Tensor, G: Tensor, tile: int) -> Tuple[Tensor, Tensor]:
    """``M_ij = sum_d k_id k_jd e^{G_id - G_jd}`` and ``Aqk_ij = sum_d q_id k_jd e^{G_id - G_jd}``.

    ``k, q, G``: ``(..., C, dk)``. Returned *unmasked* ``(..., C, C)``; entries with ``i < j``
    are finite but meaningless and must be masked by the caller.
    """
    C = k.shape[-2]
    kk_cols, qk_cols = [], []
    for J in range(0, C, tile):
        ref = G[..., J:J + 1, :]
        fx = (G - ref).clamp(max=0.0).exp()  # <= 1; rows above the tile are masked later
        yk = k[..., J:J + tile, :] * (ref - G[..., J:J + tile, :]).exp()  # <= e^{(tile-1)|g_min|}
        ykT = yk.transpose(-1, -2)
        kk_cols.append((k * fx) @ ykT)
        qk_cols.append((q * fx) @ ykT)
    return torch.cat(kk_cols, dim=-1), torch.cat(qk_cols, dim=-1)


def kda_chunked(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    chunk_size: int = 64, initial_state: Optional[Tensor] = None,
    *, g_min: float = -5.0, tile: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Chunk-parallel KDA; same contract as :func:`kda_recurrent`.

    ``g_min`` is a lower bound on every entry of ``g`` (per-step log-decay). It is used only to
    size the secondary tile and is **not checked** (a check would cost a host sync per call).
    If ``g`` goes below it, the Y factors may overflow and the result becomes inf/NaN.
    """
    dtype = v.dtype
    acc = acc_dtype(v)
    b, L, h, dk = k.shape
    dv = v.shape[-1]
    C = chunk_size
    if tile is None:
        tile = kda_safe_tile(C, g_min, acc)
    if C % tile:
        raise ValueError(f"tile={tile} must divide chunk_size={C}")

    # -> (b, h, l, d) in the accumulation dtype
    q, k, v, g = (x.transpose(1, 2).to(acc) for x in (q, k, v, g))
    beta = beta.transpose(1, 2).to(acc)
    pad = (C - L % C) % C
    if pad:  # zero k/v/beta and zero log-decay: padding leaves the state untouched
        q, k, v, g = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, pad))
    n = q.shape[2] // C
    q, k, v, g = (x.reshape(b, h, n, C, x.shape[-1]) for x in (q, k, v, g))
    beta = beta.reshape(b, h, n, C, 1)

    G = g.cumsum(-2)  # (b,h,n,C,dk): per-channel log-decay accumulated within the chunk
    M, Aqk = _decayed_inner(k, q, G, tile)
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    M = M * strict  # token i's own write is on the LHS of the WY system, so strictly lower
    Aqk = Aqk * incl  # o_i reads S_i, i.e. after token i's own write: diagonal kept

    Tm = _inv_unit_lower(-(beta * M))  # (I + Diag(beta) M)^{-1}
    U = Tm @ (beta * v)
    W = Tm @ (beta * k * G.exp())
    q_dec = q * G.exp()
    G_last = G[..., -1:, :]  # (b,h,n,1,dk)
    k_dec = k * (G_last - G).exp()
    chunk_decay = G_last.exp().transpose(-1, -2)  # (b,h,n,dk,1)

    S = (torch.zeros(b, h, dk, dv, device=q.device, dtype=acc)
         if initial_state is None else initial_state.to(acc))
    outs = []
    for i in range(n):
        u = U[:, :, i] - W[:, :, i] @ S  # what each token writes, given the incoming state
        outs.append(q_dec[:, :, i] @ S + Aqk[:, :, i] @ u)
        S = chunk_decay[:, :, i] * S + k_dec[:, :, i].transpose(-1, -2) @ u
    o = torch.stack(outs, dim=2).reshape(b, h, n * C, dv)[:, :, :L]
    return o.transpose(1, 2).to(dtype), S
