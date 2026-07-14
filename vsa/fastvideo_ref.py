"""Portable PyTorch reference for FastVideo's Video Sparse Attention (VSA).

This mirrors the math of ``fastvideo_kernel.ops.video_sparse_attn`` (the
``hao-ai-lab/FastVideo`` project) without any of its compiled CUDA / Triton
kernels, so it runs on CPU and CUDA and is fully differentiable. It is a
*reference*: correctness and readability over speed.

FastVideo VSA has two branches that are summed per token:

1. **Compression (coarse) branch** — every tile/block is mean-pooled to a
   single token; those coarse tokens do dense attention; each query block's
   coarse output is broadcast back to all of its tokens.
2. **Sparse (fine) branch** — the coarse block-vs-block scores pick the Top-K
   key blocks per query block; fine token attention runs over only the valid
   tokens of the selected blocks.

The branches combine as ``out = out_c * gate + out_s`` (``gate`` optional).

Blocks may hold fewer than ``block_elements`` valid tokens; ``variable_block_sizes``
gives the real token count per block and padding tokens are excluded from both
the mean pooling and the fine attention.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def _as_block_elements(block_size: Union[int, Tuple[int, int, int]]) -> int:
    if isinstance(block_size, int):
        return block_size * block_size * block_size
    return int(math.prod(block_size))


def _valid_token_mask(variable_block_sizes: Tensor, block_elements: int) -> Tensor:
    """Boolean [num_blocks, block_elements]: True where a slot is a real token."""
    positions = torch.arange(block_elements, device=variable_block_sizes.device)
    return positions[None, :] < variable_block_sizes[:, None]


def block_mean(
    x: Tensor,
    variable_block_sizes: Tensor,
    block_elements: int,
) -> Tensor:
    """Mean-pool each block over its valid tokens, computed in fp32.

    Mirrors ``fused_block_mean``: sum the block's tokens and divide by the
    block's valid-token count. Padding tokens are masked so the result does
    not depend on the (conventionally zero) padding values.

    Args:
        x: ``[batch, heads, seq_len, dim]`` with ``seq_len`` a multiple of
            ``block_elements``.
        variable_block_sizes: ``[num_blocks]`` valid token count per block.
        block_elements: padded tokens per block.

    Returns:
        ``[batch, heads, num_blocks, dim]`` block means in ``x``'s dtype.
    """
    batch, heads, seq_len, dim = x.shape
    num_blocks = seq_len // block_elements
    blocks = x.reshape(batch, heads, num_blocks, block_elements, dim).float()
    valid = _valid_token_mask(variable_block_sizes, block_elements)  # [nb, be]
    valid = valid.view(1, 1, num_blocks, block_elements, 1)
    summed = (blocks * valid).sum(dim=3)
    counts = variable_block_sizes.clamp(min=1).to(summed.dtype).view(1, 1, num_blocks, 1)
    return (summed / counts).to(x.dtype)


def topk_block_mask(scores: Tensor, topk: int) -> Tensor:
    """Boolean Top-K mask over the last dim, exactly ``topk`` True per row.

    Matches ``fused_topk_mask``'s PyTorch fallback (``torch.topk`` + scatter).
    """
    kv_blocks = scores.shape[-1]
    topk = min(topk, kv_blocks)
    topk_idx = torch.topk(scores, topk, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)


def video_sparse_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    topk: int,
    *,
    block_size: Union[int, Tuple[int, int, int]] = 64,
    compress_attn_weight: Optional[Tensor] = None,
    sm_scale: Optional[float] = None,
    return_aux: bool = False,
) -> Union[Tensor, Tuple[Tensor, dict]]:
    """Portable PyTorch reference of FastVideo's ``video_sparse_attn``.

    Layout is ``[batch, heads, seq_len, dim]`` (BHSD), matching the default
    64-element-tile path in FastVideo. ``seq_len`` for q and kv must each be a
    multiple of ``block_elements = prod(block_size)``; blocks are padded to
    that size and ``*_variable_block_sizes`` gives the real token count each.

    Args:
        q, k, v: ``[batch, heads, seq_len, dim]``. k and v share the kv length.
        variable_block_sizes: ``[kv_num_blocks]`` valid tokens per kv block.
        q_variable_block_sizes: ``[q_num_blocks]`` valid tokens per query block.
        topk: number of key blocks each query block attends to (fine branch).
        block_size: tile shape or its integer volume; only the volume matters.
        compress_attn_weight: optional gate ``[batch, heads, seq_len, dim]``
            multiplying the compression branch before it is added.
        sm_scale: attention scale; defaults to ``1/sqrt(dim)``.
        return_aux: also return a dict with ``coarse_scores``, ``block_mask``,
            ``out_compress`` and ``out_sparse`` for inspection/testing.

    Returns:
        Output ``[batch, heads, seq_len, dim]``, or ``(output, aux)`` when
        ``return_aux`` is set.
    """
    block_elements = _as_block_elements(block_size)
    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]

    if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != heads or v.shape[1] != heads:
        raise ValueError("q, k, and v must share batch and head dimensions")
    if v.shape[2] != kv_seq_len:
        raise ValueError("k and v must have the same sequence length")
    if q_seq_len % block_elements or kv_seq_len % block_elements:
        raise ValueError(
            f"q/kv seq lengths must be divisible by block_elements={block_elements}"
        )
    q_num_blocks = q_seq_len // block_elements
    kv_num_blocks = kv_seq_len // block_elements
    if variable_block_sizes.numel() != kv_num_blocks:
        raise ValueError(
            f"variable_block_sizes must have length {kv_num_blocks}, "
            f"got {variable_block_sizes.numel()}"
        )
    if q_variable_block_sizes.numel() != q_num_blocks:
        raise ValueError(
            f"q_variable_block_sizes must have length {q_num_blocks}, "
            f"got {q_variable_block_sizes.numel()}"
        )
    if not 1 <= topk <= kv_num_blocks:
        raise ValueError(f"topk must be in [1, {kv_num_blocks}]")

    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(dim)

    # --- Compression (coarse) branch ---------------------------------------
    q_c = block_mean(q, q_variable_block_sizes, block_elements)  # [B,H,Qb,D]
    k_c = block_mean(k, variable_block_sizes, block_elements)  # [B,H,Kb,D]
    v_c = block_mean(v, variable_block_sizes, block_elements)

    coarse_scores = torch.matmul(q_c.float(), k_c.float().transpose(-2, -1)) * scale
    coarse_attn = torch.softmax(coarse_scores, dim=-1)
    out_c = torch.matmul(coarse_attn, v_c.float())  # [B,H,Qb,D]
    # Broadcast each query block's coarse output to all of its tokens.
    out_c = (
        out_c.view(batch, heads, q_num_blocks, 1, dim)
        .expand(batch, heads, q_num_blocks, block_elements, dim)
        .reshape(batch, heads, q_seq_len, dim)
    )

    # --- Sparse (fine) branch ----------------------------------------------
    block_mask = topk_block_mask(coarse_scores, topk)  # [B,H,Qb,Kb] bool

    # Expand the block-level selection to token level and intersect with the
    # kv padding mask so only real, selected tokens are attended.
    token_sel = block_mask.repeat_interleave(block_elements, dim=2).repeat_interleave(
        block_elements, dim=3
    )  # [B,H,q_seq,kv_seq]
    kv_valid = _valid_token_mask(variable_block_sizes, block_elements).reshape(kv_seq_len)
    allowed = token_sel & kv_valid.view(1, 1, 1, kv_seq_len)

    fine_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    fine_scores = fine_scores.masked_fill(~allowed, float("-inf"))
    fine_attn = torch.softmax(fine_scores, dim=-1)
    out_s = torch.matmul(fine_attn, v.float())  # [B,H,q_seq,D]

    # --- Combine ------------------------------------------------------------
    if compress_attn_weight is not None:
        out = out_c * compress_attn_weight.float() + out_s
    else:
        out = out_c + out_s
    out = out.to(q.dtype)

    if return_aux:
        aux = {
            "coarse_scores": coarse_scores,
            "block_mask": block_mask,
            "out_compress": out_c.to(q.dtype),
            "out_sparse": out_s.to(q.dtype),
        }
        return out, aux
    return out
