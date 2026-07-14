"""Readable PyTorch reference implementation of simplified VSA."""

import math
from typing import Optional, Tuple, Union, overload

import torch
from torch import Tensor

from .selector import select_topk_blocks, validate_inputs


def _gather_blocks(x: Tensor, block_ids: Tensor, block_size: int) -> Tensor:
    batch, heads, tokens, head_dim = x.shape
    key_blocks = tokens // block_size
    query_blocks, topk = block_ids.shape[-2:]
    blocks = x.reshape(batch, heads, key_blocks, block_size, head_dim)
    source = blocks[:, :, None].expand(-1, -1, query_blocks, -1, -1, -1)
    index = block_ids[..., None, None].expand(-1, -1, -1, -1, block_size, head_dim)
    return torch.gather(source, dim=3, index=index)


@overload
def torch_vsa(
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
def torch_vsa(
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


def torch_vsa(
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
    """Run block-selected token attention.

    This is intentionally smaller than the full VSA paper implementation: mean
    pooling supplies the coarse selector, and only the selected blocks enter the
    fine token attention. It is differentiable through the fine attention; the
    discrete Top-K indices are not differentiable.
    """
    validate_inputs(q, k, v, block_size, topk, causal)
    batch, heads, query_tokens, head_dim = q.shape
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
    selected_k = _gather_blocks(k, selected, block_size)
    selected_v = _gather_blocks(v, selected, block_size)
    query = q.reshape(batch, heads, query_blocks, block_size, head_dim)

    # [B, H, query_block, query_token, selected_block, key_token]
    scores = torch.einsum("bhqtd,bhqksd->bhqtks", query, selected_k) * scale

    if causal:
        offsets = torch.arange(block_size, device=q.device)
        query_positions = (
            torch.arange(query_blocks, device=q.device)[:, None] * block_size + offsets[None]
        )
        key_positions = selected[..., None] * block_size + offsets
        allowed = key_positions[..., None, :, :] <= query_positions[None, None, :, :, None, None]
        scores = scores.masked_fill(~allowed, -torch.inf)

    flat_scores = scores.flatten(-2).float()
    probabilities = torch.softmax(flat_scores, dim=-1).to(v.dtype)
    flat_values = selected_v.flatten(3, 4)
    output = torch.einsum("bhqtk,bhqkd->bhqtd", probabilities, flat_values)
    output = output.reshape(batch, heads, query_tokens, head_dim)
    return (output, selected) if return_indices else output
