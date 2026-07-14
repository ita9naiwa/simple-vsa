"""Simple forward-only Triton implementation of VSA fine attention."""

import math
from typing import Optional, Tuple, Union, overload

import torch
from torch import Tensor
import triton
import triton.language as tl

from .selector import select_topk_blocks, validate_inputs


@triton.jit
def _vsa_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    query_blocks: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    block_dim: tl.constexpr,
    sm_scale,
    causal: tl.constexpr,
):
    block_program = tl.program_id(0)
    token_in_block = tl.program_id(1)
    query_block = block_program % query_blocks
    batch_head = block_program // query_blocks
    query_token = query_block * block_size + token_in_block

    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    q_offset = (batch_head * query_tokens + query_token) * head_dim
    q = tl.load(q_ptr + q_offset + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)

    accumulator = tl.zeros([block_dim], dtype=tl.float32)
    normalizer = 0.0
    row_max = -1.0e6

    for selected_offset in tl.static_range(0, topk):
        selected_index = (block_program * topk) + selected_offset
        key_block = tl.load(selected_ptr + selected_index)

        for token_offset in tl.static_range(0, block_size):
            key_token = key_block * block_size + token_offset
            kv_offset = (batch_head * key_tokens + key_token) * head_dim
            key = tl.load(k_ptr + kv_offset + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)
            value = tl.load(v_ptr + kv_offset + dim_offsets, mask=dim_mask, other=0.0).to(tl.float32)
            score = tl.sum(q * key, axis=0) * sm_scale

            valid = True
            if causal:
                valid = key_token <= query_token

            candidate = tl.where(valid, score, -1.0e6)
            new_max = tl.maximum(row_max, candidate)
            alpha = tl.exp(row_max - new_max)
            probability = tl.where(valid, tl.exp(score - new_max), 0.0)
            accumulator = accumulator * alpha + probability * value
            normalizer = normalizer * alpha + probability
            row_max = new_max

    output = accumulator / normalizer
    tl.store(out_ptr + q_offset + dim_offsets, output, mask=dim_mask)


@overload
def triton_vsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    block_size: int = 16,
    topk: int = 2,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    return_indices: bool = False,
) -> Tensor: ...


@overload
def triton_vsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    block_size: int = 16,
    topk: int = 2,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    return_indices: bool = True,
) -> Tuple[Tensor, Tensor]: ...


def triton_vsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    block_size: int = 16,
    topk: int = 2,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    return_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Run the simplified VSA forward pass on CUDA.

    Top-K selection uses PyTorch. Triton owns the sparse token attention and its
    online softmax. Backward is deliberately left to the readable PyTorch path.
    """
    validate_inputs(q, k, v, block_size, topk, causal)
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("the Triton implementation requires CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("the Triton path supports float16, bfloat16, and float32")
    if q.shape[-1] > 256:
        raise ValueError("the simple Triton kernel supports head_dim <= 256")
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise RuntimeError("triton_vsa is forward-only; use torch_vsa when gradients are needed")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, heads, query_tokens, head_dim = q.shape
    key_tokens = k.shape[-2]
    query_blocks = query_tokens // block_size
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim)

    selected = select_topk_blocks(
        q,
        k,
        block_size=block_size,
        topk=topk,
        causal=causal,
        sm_scale=scale,
    )
    selected_for_kernel = selected.to(torch.int32).contiguous()
    output = torch.empty_like(q)
    block_dim = triton.next_power_of_2(head_dim)
    num_warps = 4 if block_dim >= 64 else 2
    grid = (batch * heads * query_blocks, block_size)
    _vsa_forward_kernel[grid](
        q,
        k,
        v,
        selected_for_kernel,
        output,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        head_dim=head_dim,
        query_blocks=query_blocks,
        block_size=block_size,
        topk=topk,
        block_dim=block_dim,
        sm_scale=scale,
        causal=causal,
        num_warps=num_warps,
    )
    return (output, selected) if return_indices else output
