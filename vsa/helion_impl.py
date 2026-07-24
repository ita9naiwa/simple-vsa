"""Forward-only Helion VSA.

The compression branch and Top-K routing stay in the shared PyTorch
implementation.  Helion handles the sparse fine-attention branch: one GPU
program owns one ``(batch, head, query_block)`` and streams its selected KV
blocks with an online softmax.

This is intentionally a forward-only implementation.  It is useful for
experimenting with Helion's higher-level tile programming model without hiding
the VSA routing or materializing gathered ``[B, H, Qb, topk, block, D]``
tensors.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

import helion
import helion.language as hl

from .common import _as_block_elements, coarse_branch, validate_vsa_inputs


@helion.kernel(static_shapes=True, autotune_effort="none")
def _helion_sparse_attention_forward(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    selected_in: Tensor,
    variable_block_sizes: Tensor,
    block_elements_in: int,
) -> Tensor:
    """Apply fine attention over preselected KV blocks.

    ``selected_in`` is ``[B, H, Qb, topk]``.  The kernel reshapes the leading
    dimensions to ``B*H`` and launches one program per query block.  Selected
    K/V blocks are loaded indirectly, so the large gathered representation used
    by the readable PyTorch implementation is never materialized.
    """

    query_tokens = q_in.size(-2)
    key_tokens = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    block_elements = hl.specialize(block_elements_in)
    query_blocks = selected_in.size(-2)
    topk = hl.specialize(selected_in.size(-1))

    assert key_tokens == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    assert query_tokens == query_blocks * block_elements

    q = q_in.reshape([-1, query_tokens, head_dim])
    k = k_in.reshape([-1, key_tokens, head_dim])
    v = v_in.reshape([-1, key_tokens, head_dim])
    selected = selected_in.reshape([-1, query_blocks, topk])
    out = torch.empty_like(q)

    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.4426950408889634

    for batch_head, query_block in hl.grid([q.size(0), query_blocks]):
        offsets = hl.arange(block_elements)
        query_positions = query_block * block_elements + offsets
        query = q[batch_head, query_positions, :]

        row_max = hl.full(
            [block_elements],
            float("-inf"),
            dtype=torch.float32,
        )
        row_sum = hl.zeros([block_elements], dtype=torch.float32)
        accumulator = hl.zeros(
            [block_elements, head_dim],
            dtype=torch.float32,
        )

        for selected_slot in hl.static_range(topk):
            kv_block = selected[batch_head, query_block, selected_slot]
            kv_positions = kv_block * block_elements + offsets

            key = k[batch_head, kv_positions, :]
            scores = hl.dot(
                query * qk_scale,
                key.T,
                out_dtype=torch.float32,
            )
            valid_kv = offsets < variable_block_sizes[kv_block]
            scores = torch.where(valid_kv[None, :], scores, float("-inf"))

            new_max = torch.maximum(row_max, torch.amax(scores, dim=-1))
            probabilities = torch.exp2(scores - new_max[:, None])
            alpha = torch.exp2(row_max - new_max)

            row_sum = row_sum * alpha + torch.sum(probabilities, dim=-1)
            accumulator = accumulator * alpha[:, None]
            value = v[batch_head, kv_positions, :]
            accumulator = hl.dot(
                probabilities.to(value.dtype),
                value,
                acc=accumulator,
            )
            row_max = new_max

        out[batch_head, query_positions, :] = (
            accumulator / row_sum[:, None]
        ).to(out.dtype)

    return out.reshape(q_in.size())


def helion_vsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: Optional[Tensor] = None,
) -> Tensor:
    """Run PyTorch coarse attention plus Helion sparse fine attention.

    The signature and two-branch math match :func:`torch_vsa`.  Only forward
    execution is supported; inputs requiring gradients are rejected instead of
    silently returning incomplete gradients.
    """

    block_elements = _as_block_elements(block_size)
    validate_vsa_inputs(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_elements,
    )
    if compress_attn_weight is not None and compress_attn_weight.shape != q.shape:
        raise ValueError("compress_attn_weight must have the same shape as q")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("the Helion implementation requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("the Helion path supports float16, bfloat16, and float32")
    if block_elements > 128:
        raise ValueError("the Helion path currently supports block volume <= 128")
    if q.shape[-1] > 256:
        raise ValueError("the Helion path currently supports head_dim <= 256")

    differentiable_inputs = (q, k, v)
    if compress_attn_weight is not None:
        differentiable_inputs += (compress_attn_weight,)
    if torch.is_grad_enabled() and any(x.requires_grad for x in differentiable_inputs):
        raise RuntimeError(
            "helion_vsa is forward-only; call it under torch.no_grad() or use "
            "torch_vsa/triton_vsa when gradients are required"
        )

    scale = 1.0 / math.sqrt(q.shape[-1])
    variable_block_sizes = variable_block_sizes.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()
    q_variable_block_sizes = q_variable_block_sizes.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()

    scores, out_c = coarse_branch(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        block_elements,
        scale,
    )
    selected = scores.topk(topk, dim=-1, sorted=False).indices.to(
        dtype=torch.int32
    ).contiguous()
    out_s = _helion_sparse_attention_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        selected,
        variable_block_sizes,
        block_elements,
    )

    if compress_attn_weight is None:
        output = out_c + out_s.float()
    else:
        output = out_c * compress_attn_weight.float() + out_s.float()
    return output.to(q.dtype)
