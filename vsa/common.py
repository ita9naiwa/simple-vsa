"""Shared VSA building blocks and the FastVideo-faithful reference.

Everything the two fast paths (``torch_vsa``, ``triton_vsa``) share lives here:
block pooling, input validation, the coarse/compression branch, the compression
gate layer, and ``video_sparse_attn`` -- the portable masked-fill reference that
doubles as the dense baseline and the correctness oracle.

``video_sparse_attn`` mirrors ``fastvideo_kernel.ops.video_sparse_attn``: two
branches summed per token, ``out = out_c * compress_attn_weight + out_s``.

1. Compression (coarse) branch -- every block is mean-pooled to one token, those
   coarse tokens do dense attention, each query block's output is broadcast back
   to all of its tokens. (Shared: ``coarse_branch``.)
2. Sparse (fine) branch -- the coarse block-vs-block scores pick the Top-K key
   blocks; token attention runs over the valid tokens of the selected blocks.
   The reference does this by masking the full dense score matrix; ``torch_vsa``
   gathers only the selected blocks instead.

Blocks may hold fewer than ``block_elements`` valid tokens; ``variable_block_sizes``
gives the real token count per block and padding is excluded everywhere.

Shape symbols used in the inline annotations below:
    B   batch                      H    heads          D   head dim
    Sq  query seq_len              Skv  kv seq_len
    Qb  query blocks (Sq / be)     Kb   kv blocks (Skv / be)
    be  block_elements (padded tokens per block)       tk  topk
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


def _as_block_elements(block_size: Union[int, Tuple[int, int, int]]) -> int:
    if isinstance(block_size, int):
        return block_size * block_size * block_size
    return int(math.prod(block_size))


def _valid_token_mask(variable_block_sizes: Tensor, block_elements: int) -> Tensor:
    """Boolean [num_blocks, block_elements]: True where a slot is a real token.

    In: variable_block_sizes [nb]. Out: [nb, be] bool.
    """
    positions = torch.arange(block_elements, device=variable_block_sizes.device)  # [be]
    # [1, be] < [nb, 1] -> [nb, be]
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
        block_elements: Total token slots per block after padding.

    Returns:
        ``[batch, heads, num_blocks, dim]`` block means in ``x``'s dtype.
    """
    batch, heads, seq_len, dim = x.shape                       # x: [B, H, S, D]
    num_blocks = seq_len // block_elements                     # nb = S / be
    blocks = x.reshape(batch, heads, num_blocks, block_elements, dim).float()  # [B, H, nb, be, D]
    valid = _valid_token_mask(variable_block_sizes, block_elements)  # [nb, be] bool
    valid = valid.view(1, 1, num_blocks, block_elements, 1)     # [1, 1, nb, be, 1]
    summed = (blocks * valid).sum(dim=3)                       # [B, H, nb, be, D] -> [B, H, nb, D]
    counts = variable_block_sizes.clamp(min=1).to(summed.dtype).view(1, 1, num_blocks, 1)  # [1, 1, nb, 1]
    return (summed / counts).to(x.dtype)                       # [B, H, nb, D]


def topk_block_mask(scores: Tensor, topk: int) -> Tensor:
    """Boolean Top-K mask over the last dim, exactly ``topk`` True per row.

    Matches ``fused_topk_mask``'s PyTorch fallback (``torch.topk`` + scatter).
    """
    kv_blocks = scores.shape[-1]                               # scores: [B, H, Qb, Kb]
    topk = min(topk, kv_blocks)
    topk_idx = torch.topk(scores, topk, dim=-1).indices        # [B, H, Qb, topk]
    # scatter the topk indices into an all-False grid -> [B, H, Qb, Kb] bool
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)


def validate_vsa_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    topk: int,
    block_elements: int,
) -> None:
    """Match the shape/length contract of ``video_sparse_attn``."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, seq_len, dim]")
    batch, heads, q_seq_len, dim = q.shape
    kv_seq_len = k.shape[2]
    if k.shape[0] != batch or v.shape[0] != batch or k.shape[1] != heads or v.shape[1] != heads:
        raise ValueError("q, k, and v must share batch and head dimensions")
    if v.shape[2] != kv_seq_len:
        raise ValueError("k and v must have the same sequence length")
    if k.shape[-1] != dim or v.shape[-1] != dim:
        raise ValueError("q, k, and v must have the same head dimension")
    if q_seq_len % block_elements or kv_seq_len % block_elements:
        raise ValueError(
            f"q/kv sequence lengths must be divisible by block_elements={block_elements}"
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


def coarse_branch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    block_elements: int,
    scale: float,
) -> Tuple[Tensor, Tensor]:
    """Pooled block-vs-block scores and the dense compression output.

    Shared by ``video_sparse_attn``, ``torch_vsa`` and ``triton_vsa`` so the
    compression branch is defined exactly once. Returns ``(scores, out_c)`` where
    ``scores`` is ``[B, H, Qb, Kb]`` in the input dtype and feeds both the compression output
    and the Top-K selection, and ``out_c`` is the compression attention output
    broadcast back to every token, ``[B, H, Sq, D]`` in the input dtype.
    """
    batch, heads, q_seq_len, dim = q.shape                     # q: [B, H, Sq, D]
    q_num_blocks = q_seq_len // block_elements                 # Qb = Sq / be

    q_c = block_mean(q, q_variable_block_sizes, block_elements)  # [B, H, Qb, D]
    k_c = block_mean(k, variable_block_sizes, block_elements)    # [B, H, Kb, D]
    v_c = block_mean(v, variable_block_sizes, block_elements)    # [B, H, Kb, D]

    # [B,H,Qb,D] @ [B,H,D,Kb] -> [B, H, Qb, Kb]
    # Match FastVideo's production path: block means are accumulated in fp32
    # and written in the input dtype, then coarse attention stays in that dtype.
    # In particular, BF16 routing must not silently become FP32 routing because
    # that can change Top-K block selection near score ties.
    scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)                       # [B, H, Qb, Kb]
    out_c = torch.matmul(attn, v_c)                            # [B,H,Qb,Kb] @ [B,H,Kb,D] -> [B, H, Qb, D]
    out_c = (
        out_c.view(batch, heads, q_num_blocks, 1, dim)         # [B, H, Qb, 1, D]
        .expand(batch, heads, q_num_blocks, block_elements, dim)  # [B, H, Qb, be, D]
        .reshape(batch, heads, q_seq_len, dim)                 # [B, H, Sq, D]
    )
    return scores, out_c                                       # [B,H,Qb,Kb], [B,H,Sq,D]


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
    """Portable masked-fill reference of FastVideo's ``video_sparse_attn``.

    The dense baseline and correctness oracle: the sparse branch computes the
    full ``q @ k^T`` and masks out non-selected / padding tokens, so it is slow
    but obviously correct. ``torch_vsa`` gathers only the selected blocks and must
    match this. Layout is ``[batch, heads, seq_len, dim]`` (BHSD).

    Args:
        q, k, v: ``[batch, heads, seq_len, dim]``. k and v share the kv length.
        variable_block_sizes: ``[Kb]`` valid tokens per kv block.
        q_variable_block_sizes: ``[Qb]`` valid tokens per query block.
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
    block_elements = _as_block_elements(block_size)            # be = prod(block_size)
    validate_vsa_inputs(q, k, v, variable_block_sizes, q_variable_block_sizes, topk, block_elements)

    batch, heads, q_seq_len, dim = q.shape                     # q: [B, H, Sq, D]
    kv_seq_len = k.shape[2]                                     # k, v: [B, H, Skv, D]
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(dim)
    variable_block_sizes = variable_block_sizes.to(q.device)   # [Kb]
    q_variable_block_sizes = q_variable_block_sizes.to(q.device)  # [Qb]

    # --- Compression (coarse) branch (shared with the fast paths) ----------
    # coarse_scores: [B, H, Qb, Kb]   out_c: [B, H, Sq, D]
    coarse_scores, out_c = coarse_branch(
        q, k, v, variable_block_sizes, q_variable_block_sizes, block_elements, scale
    )

    # --- Sparse (fine) branch: masked-fill over the full dense scores ------
    block_mask = topk_block_mask(coarse_scores, topk)          # [B, H, Qb, Kb] bool
    # Expand the block-level selection to token level and intersect with the
    # kv padding mask so only real, selected tokens are attended.
    token_sel = block_mask.repeat_interleave(block_elements, dim=2).repeat_interleave(
        block_elements, dim=3
    )  # [B,H,Qb,Kb] -> [B, H, Sq, Skv] bool
    # [Kb, be] -> [Skv]
    kv_valid = _valid_token_mask(variable_block_sizes, block_elements).reshape(kv_seq_len)
    allowed = token_sel & kv_valid.view(1, 1, 1, kv_seq_len)   # [B, H, Sq, Skv] bool

    # [B,H,Sq,D] @ [B,H,D,Skv] -> [B, H, Sq, Skv]
    fine_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    fine_scores = fine_scores.masked_fill(~allowed, float("-inf"))  # [B, H, Sq, Skv]
    fine_attn = torch.softmax(fine_scores, dim=-1)             # [B, H, Sq, Skv]
    # [B,H,Sq,Skv] @ [B,H,Skv,D] -> [B, H, Sq, D]
    out_s = torch.matmul(fine_attn, v.float())

    # --- Combine -----------------------------------------------------------
    if compress_attn_weight is not None:
        # [B,H,Sq,D] * [B,H,Sq,D] + [B,H,Sq,D] -> [B, H, Sq, D]
        out = out_c * compress_attn_weight.float() + out_s
    else:
        out = out_c + out_s                                    # [B, H, Sq, D]
    out = out.to(q.dtype)                                      # [B, H, Sq, D]

    if return_aux:
        aux = {
            "coarse_scores": coarse_scores,
            "block_mask": block_mask,
            "out_compress": out_c.to(q.dtype),
            "out_sparse": out_s.to(q.dtype),
        }
        return out, aux
    return out


class CompressionGate(nn.Module):
    """FastVideo's ``to_gate_compress``: the layer that *produces* the gate.

    The VSA attention functions only *apply* the compression gate (they multiply
    ``out_c`` by ``compress_attn_weight``). The gate itself is generated here, by
    the layer that in FastVideo lives inside the DiT block right next to
    ``to_q``/``to_k``/``to_v`` and runs on the same normalized hidden states
    (``fastvideo/models/dits/wanvideo.py``: ``self.to_gate_compress``).

    It is a plain ``Linear(dim, dim)`` with **no activation**: the raw output
    scales the compression branch per token and per channel, so the model learns
    how much low-resolution global context to blend into each token. The result
    is reshaped to the ``[batch, heads, seq_len, head_dim]`` layout the attention
    functions expect for ``compress_attn_weight``.

    Example:
        >>> gate_layer = CompressionGate(dim=heads * head_dim, num_heads=heads)
        >>> gate = gate_layer(hidden_states)            # [B, S, dim] -> [B, H, S, D]
        >>> out = video_sparse_attn(q, k, v, vbs, vbs, topk,
        ...                         compress_attn_weight=gate)
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.to_gate_compress = nn.Linear(dim, dim, bias=bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Map ``[batch, seq_len, dim]`` hidden states to a per-head gate.

        Returns ``[batch, heads, seq_len, head_dim]`` matching the ``q/k/v``
        layout of the attention functions.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, seq_len, dim]")
        batch, seq_len, _ = hidden_states.shape                # hidden_states: [B, S, dim]
        gate = self.to_gate_compress(hidden_states)            # [B, S, dim] -> [B, S, dim]
        # [B, S, dim] -> [B, S, H, D] -> [B, H, S, D]
        return gate.view(batch, seq_len, self.num_heads, -1).transpose(1, 2)
