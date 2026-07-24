"""Optional FA4 CuTe forward with the owned Triton backward."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .common import _as_block_elements
from .triton_impl import (
    _triton_sparse_attention_backward,
    _vsa_with_sparse_executor,
)


def _cute_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
) -> tuple[Tensor, Tensor]:
    batch, heads, query_tokens, _ = q.shape
    selected_128 = (
        selected.to(torch.int64)[..., None] * 2
        + torch.arange(2, device=q.device)
    ).flatten(-2)
    mask_128 = torch.zeros(
        (
            batch,
            heads,
            selected.shape[2],
            variable_block_sizes.numel() * 2,
        ),
        device=q.device,
        dtype=torch.bool,
    )
    mask_128.scatter_(-1, selected_128, True)
    sizes = variable_block_sizes.to(torch.int32)
    sizes_128 = torch.stack(
        (sizes.clamp(0, 128), (sizes - 128).clamp(0, 128)),
        dim=1,
    ).reshape(-1)

    from fastvideo_kernel.block_sparse_attn_cute_fwd import (
        block_sparse_attn_cute_fwd,
    )

    output, lse = block_sparse_attn_cute_fwd(
        q, k, v, mask_128, sizes_128
    )
    expected = (batch, heads, query_tokens)
    if lse.shape == (batch, query_tokens, heads):
        lse = lse.transpose(1, 2).contiguous()
    elif lse.shape != expected:
        raise RuntimeError(
            f"unexpected CuTe LSE shape {tuple(lse.shape)}; expected {expected}"
        )
    return output, lse * 1.4426950408889634


class _CuteTritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, selected, variable_block_sizes, sm_scale):
        output, lse = _cute_forward(
            q, k, v, selected, variable_block_sizes
        )
        ctx.save_for_backward(
            q, k, v, output, lse, selected, variable_block_sizes
        )
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse, selected, variable_block_sizes = ctx.saved_tensors
        dq, dk, dv = _triton_sparse_attention_backward(
            grad_output,
            q,
            k,
            v,
            output,
            lse,
            selected,
            variable_block_sizes,
            block_size=256,
            sm_scale=ctx.sm_scale,
        )
        return dq, dk, dv, None, None, None


def _execute(q, k, v, selected, variable_block_sizes, *, sm_scale):
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    selected = selected.to(q.device, torch.int32).contiguous()
    variable_block_sizes = variable_block_sizes.to(
        q.device, torch.int32
    ).contiguous()
    if torch.is_grad_enabled() and any(
        x.requires_grad for x in (q, k, v)
    ):
        return _CuteTritonAttention.apply(
            q, k, v, selected, variable_block_sizes, sm_scale
        )
    return _cute_forward(q, k, v, selected, variable_block_sizes)[0]


def cute_triton_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    *,
    block_size: int | tuple = (1, 1, 256),
    sm_scale: Optional[float] = None,
) -> Tensor:
    """Run the hybrid sparse executor on an externally supplied route."""

    if _as_block_elements(block_size) != 256:
        raise ValueError("cute_triton_sparse_attention requires 256-token blocks")
    scale = 1.0 / math.sqrt(q.shape[-1])
    if sm_scale is not None and not math.isclose(sm_scale, scale):
        raise ValueError("the CuTe forward requires sm_scale=1/sqrt(dim)")
    return _execute(
        q, k, v, selected, variable_block_sizes, sm_scale=scale
    )


def cute_triton_vsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    topk: int,
    block_size: int | tuple = (1, 1, 256),
    compress_attn_weight: Optional[Tensor] = None,
) -> Tensor:
    """Run shared VSA routing with CuTe forward and Triton backward."""

    if _as_block_elements(block_size) != 256:
        raise ValueError("cute_triton_vsa requires 256-token blocks")
    return _vsa_with_sparse_executor(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
        sparse_executor=_execute,
    )
