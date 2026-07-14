"""Shared coarse block selector for the PyTorch and Triton paths."""

import math
from typing import Optional

import torch
from torch import Tensor


def validate_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_size: int,
    topk: int,
    causal: bool,
) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, tokens, dim]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError("q, k, and v must share batch/heads, and k must match v")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head dimension")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must be on the same device")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, and v must have the same dtype")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if q.shape[-2] % block_size or k.shape[-2] % block_size:
        raise ValueError("query and key token counts must be divisible by block_size")

    key_blocks = k.shape[-2] // block_size
    if not 1 <= topk <= key_blocks:
        raise ValueError(f"topk must be in [1, {key_blocks}]")
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError("the simple causal path supports self-attention only")


def select_topk_blocks(
    q: Tensor,
    k: Tensor,
    *,
    block_size: int,
    topk: int,
    causal: bool,
    sm_scale: Optional[float] = None,
) -> Tensor:
    """Select key blocks from mean-pooled query/key block similarity."""
    batch, heads, query_tokens, head_dim = q.shape
    key_tokens = k.shape[-2]
    query_blocks = query_tokens // block_size
    key_blocks = key_tokens // block_size
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)

    # Pooling and scoring in fp32 keeps the discrete Top-K choice stable.
    pooled_q = q.reshape(batch, heads, query_blocks, block_size, head_dim).float().mean(-2)
    pooled_k = k.reshape(batch, heads, key_blocks, block_size, head_dim).float().mean(-2)
    coarse_scores = torch.einsum("bhqd,bhkd->bhqk", pooled_q, pooled_k) * scale

    if causal:
        query_ids = torch.arange(query_blocks, device=q.device)[:, None]
        key_ids = torch.arange(key_blocks, device=q.device)[None, :]
        coarse_scores = coarse_scores.masked_fill(key_ids > query_ids, -torch.inf)

    return coarse_scores.topk(topk, dim=-1, sorted=True).indices
