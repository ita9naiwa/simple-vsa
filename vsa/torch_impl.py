"""Readable PyTorch VSA, matching ``fastvideo_kernel.ops.video_sparse_attn``.

Same two-branch math and signature as FastVideo: a dense compression branch over
mean-pooled blocks, and a sparse branch over the Top-K selected blocks, summed as
``out = out_c * compress_attn_weight + out_s``. Unlike the masked-fill
``video_sparse_attn`` reference in ``common``, the sparse branch here *gathers* only
the selected blocks (so it does real sparse work) and excludes padding tokens via
``variable_block_sizes``.
Fully differentiable; the discrete Top-K choice itself carries no gradient, but the
shared compression scores route gradient into the pooled query/key blocks.

Shape symbols used in the inline annotations below:
    B   batch                      H    heads          D   head dim
    Sq  query seq_len              Skv  kv seq_len
    Qb  query blocks (Sq / be)     Kb   kv blocks (Skv / be)
    be  block_elements             tk   topk
"""

import math
from typing import Optional

import torch
from torch import Tensor

from .common import (
    _as_block_elements,
    _valid_token_mask,
    coarse_branch,
    validate_vsa_inputs,
)


def _gather_blocks(x: Tensor, block_ids: Tensor, block_elements: int) -> Tensor:
    """Gather the ``block_ids`` blocks of ``x`` per query block.

    ``x`` is ``[B, H, seq, dim]`` and ``block_ids`` is ``[B, H, q_blocks, topk]``;
    the result is ``[B, H, q_blocks, topk, block_elements, dim]``.
    """
    batch, heads, tokens, head_dim = x.shape                   # x: [B, H, Skv, D]
    kv_blocks = tokens // block_elements                       # Kb = Skv / be
    q_blocks, topk = block_ids.shape[-2:]                      # block_ids: [B, H, Qb, tk]
    blocks = x.reshape(batch, heads, kv_blocks, block_elements, head_dim)  # [B, H, Kb, be, D]
    source = blocks[:, :, None].expand(-1, -1, q_blocks, -1, -1, -1)  # [B, H, Qb, Kb, be, D]
    index = block_ids[..., None, None].expand(-1, -1, -1, -1, block_elements, head_dim)  # [B, H, Qb, tk, be, D]
    # gather along the Kb axis -> [B, H, Qb, tk, be, D]
    return torch.gather(source, dim=3, index=index)


def torch_vsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: Optional[Tensor] = None,
) -> Tensor:
    """Video Sparse Attention over ``[batch, heads, seq_len, dim]`` tensors.

    Args mirror ``fastvideo_kernel.ops.video_sparse_attn``:

    - ``variable_block_sizes`` / ``q_variable_block_sizes``: valid token count per
      kv / query block (padding-aware; ``seq_len`` is padded to a block multiple).
    - ``topk``: key blocks each query block attends to in the sparse branch.
    - ``block_size``: tile shape or its integer volume; only the volume matters.
    - ``compress_attn_weight``: optional gate on the compression branch.
    """
    block_elements = _as_block_elements(block_size)            # be = prod(block_size)
    validate_vsa_inputs(q, k, v, variable_block_sizes, q_variable_block_sizes, topk, block_elements)

    batch, heads, q_seq_len, dim = q.shape                     # q: [B, H, Sq, D]
    q_num_blocks = q_seq_len // block_elements                 # Qb = Sq / be
    scale = 1.0 / math.sqrt(dim)
    variable_block_sizes = variable_block_sizes.to(q.device)   # [Kb]
    q_variable_block_sizes = q_variable_block_sizes.to(q.device)  # [Qb]

    # Coarse scores + compression branch (the scores are shared with selection).
    # scores: [B, H, Qb, Kb]   out_c: [B, H, Sq, D]
    scores, out_c = coarse_branch(
        q, k, v, variable_block_sizes, q_variable_block_sizes, block_elements, scale
    )

    # Sparse branch: gather the Top-K selected blocks and attend, dropping padding.
    selected = scores.topk(topk, dim=-1).indices               # [B, H, Qb, tk]
    selected_k = _gather_blocks(k, selected, block_elements).float()  # [B, H, Qb, tk, be, D]
    selected_v = _gather_blocks(v, selected, block_elements).float()  # [B, H, Qb, tk, be, D]
    valid = _valid_token_mask(variable_block_sizes, block_elements)   # [Kb, be] bool
    valid_selected = valid[selected]                           # [B, H, Qb, tk, be] bool

    query = q.reshape(batch, heads, q_num_blocks, block_elements, dim).float()  # [B, H, Qb, be, D]
    # per query token, score against every token of its tk selected blocks:
    # [B,H,Qb,be,D] x [B,H,Qb,tk,be,D] -> [B, H, Qb, be(q_token), tk, be(key_token)]
    fine_scores = torch.einsum("bhqtd,bhqksd->bhqtks", query, selected_k) * scale
    # mask padded key tokens (broadcast over the q_token axis) -> same shape
    fine_scores = fine_scores.masked_fill(~valid_selected[:, :, :, None, :, :], float("-inf"))
    # flatten the (tk, key_token) candidates and softmax over them -> [B, H, Qb, be, tk*be]
    fine_attn = torch.softmax(fine_scores.flatten(-2), dim=-1)
    # [B,H,Qb,be,tk*be] @ [B,H,Qb,tk*be,D] -> [B, H, Qb, be, D]
    selected_values = selected_v.flatten(3, 4)  # [B,H,Qb,N,D], N=tk*be
    out_s = torch.einsum(
        "bhqtn,bhqnd->bhqtd",
        fine_attn,
        selected_values,
    )
    out_s = out_s.reshape(batch, heads, q_seq_len, dim)        # [B, H, Sq, D]

    if compress_attn_weight is not None:
        # [B,H,Sq,D] * [B,H,Sq,D] + [B,H,Sq,D] -> [B, H, Sq, D]
        out = out_c * compress_attn_weight.float() + out_s
    else:
        out = out_c + out_s                                    # [B, H, Sq, D]
    return out.to(q.dtype)                                     # [B, H, Sq, D]
