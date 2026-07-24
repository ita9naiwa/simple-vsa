"""Training-safe Triton helpers shared by the fast VSA backends.

The readable reference intentionally stays in PyTorch. These kernels remove two
large eager-mode intermediates from the CUDA training path:

* block means accumulate directly from BF16/FP16 inputs into FP32 registers;
* compact coarse vectors are broadcast and combined with the sparse output in
  one pass, with custom backward reductions for the coarse branch and gate.
"""

from __future__ import annotations

import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _block_mean_forward_kernel(
    X,
    SIZES,
    OUT,
    SEQ_LEN: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_row = tl.program_id(0)
    dim_tile = tl.program_id(1)
    block_id = block_row % NUM_BLOCKS
    batch_head = block_row // NUM_BLOCKS
    token_offsets = tl.arange(0, BLOCK_T)
    dim_offsets = dim_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_tokens = tl.load(SIZES + block_id).to(tl.int32)
    token_positions = block_id * BLOCK_ELEMENTS + token_offsets
    ptrs = (
        X
        + batch_head * SEQ_LEN * HEAD_DIM
        + token_positions[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    mask = (
        (token_offsets[:, None] < valid_tokens)
        & (token_offsets[:, None] < BLOCK_ELEMENTS)
        & (dim_offsets[None, :] < HEAD_DIM)
    )
    values = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    summed = tl.sum(values, axis=0)
    divisor = tl.maximum(valid_tokens, 1).to(tl.float32)
    out_ptrs = OUT + block_row * HEAD_DIM + dim_offsets
    tl.store(
        out_ptrs,
        summed / divisor,
        mask=dim_offsets < HEAD_DIM,
    )


@triton.jit
def _block_mean_backward_kernel(
    DOUT,
    SIZES,
    DX,
    SEQ_LEN: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_row = tl.program_id(0)
    dim_tile = tl.program_id(1)
    block_id = block_row % NUM_BLOCKS
    batch_head = block_row // NUM_BLOCKS
    token_offsets = tl.arange(0, BLOCK_T)
    dim_offsets = dim_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_tokens = tl.load(SIZES + block_id).to(tl.int32)
    divisor = tl.maximum(valid_tokens, 1).to(tl.float32)
    grad = tl.load(
        DOUT + block_row * HEAD_DIM + dim_offsets,
        mask=dim_offsets < HEAD_DIM,
        other=0.0,
    ).to(tl.float32) / divisor
    token_positions = block_id * BLOCK_ELEMENTS + token_offsets
    dx_ptrs = (
        DX
        + batch_head * SEQ_LEN * HEAD_DIM
        + token_positions[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    store_mask = (
        (token_offsets[:, None] < BLOCK_ELEMENTS)
        & (dim_offsets[None, :] < HEAD_DIM)
    )
    values = tl.where(
        token_offsets[:, None] < valid_tokens,
        grad[None, :],
        0.0,
    )
    tl.store(dx_ptrs, values, mask=store_mask)


class _FusedBlockMean(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        variable_block_sizes: Tensor,
        block_elements: int,
    ) -> Tensor:
        x = x.contiguous()
        sizes = variable_block_sizes.to(
            device=x.device,
            dtype=torch.int32,
        ).contiguous()
        batch, heads, seq_len, head_dim = x.shape
        num_blocks = seq_len // block_elements
        out = torch.empty(
            (batch, heads, num_blocks, head_dim),
            device=x.device,
            dtype=x.dtype,
        )
        block_t = triton.next_power_of_2(block_elements)
        block_d = min(32, triton.next_power_of_2(head_dim))
        grid = (batch * heads * num_blocks, triton.cdiv(head_dim, block_d))
        _block_mean_forward_kernel[grid](
            x,
            sizes,
            out,
            SEQ_LEN=seq_len,
            NUM_BLOCKS=num_blocks,
            HEAD_DIM=head_dim,
            BLOCK_ELEMENTS=block_elements,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            num_warps=4,
        )
        ctx.save_for_backward(sizes)
        ctx.input_shape = x.shape
        ctx.block_elements = block_elements
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (sizes,) = ctx.saved_tensors
        batch, heads, seq_len, head_dim = ctx.input_shape
        block_elements = ctx.block_elements
        num_blocks = seq_len // block_elements
        dx = torch.empty(
            ctx.input_shape,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        block_t = triton.next_power_of_2(block_elements)
        block_d = min(32, triton.next_power_of_2(head_dim))
        grid = (batch * heads * num_blocks, triton.cdiv(head_dim, block_d))
        _block_mean_backward_kernel[grid](
            grad_output.contiguous(),
            sizes,
            dx,
            SEQ_LEN=seq_len,
            NUM_BLOCKS=num_blocks,
            HEAD_DIM=head_dim,
            BLOCK_ELEMENTS=block_elements,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            num_warps=4,
        )
        return dx, None, None


@triton.jit
def _combine_forward_kernel(
    SPARSE,
    COARSE,
    GATE,
    OUT,
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    HAS_GATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = NUM_TOKENS * HEAD_DIM
    mask = offsets < total
    token_linear = offsets // HEAD_DIM
    dim = offsets - token_linear * HEAD_DIM
    block_linear = token_linear // BLOCK_ELEMENTS
    coarse_offset = block_linear * HEAD_DIM + dim
    sparse = tl.load(SPARSE + offsets, mask=mask, other=0.0).to(tl.float32)
    coarse = tl.load(COARSE + coarse_offset, mask=mask, other=0.0).to(tl.float32)
    if HAS_GATE:
        gate = tl.load(GATE + offsets, mask=mask, other=0.0).to(tl.float32)
        coarse *= gate
    tl.store(OUT + offsets, sparse + coarse, mask=mask)


@triton.jit
def _combine_gate_backward_kernel(
    DOUT,
    COARSE,
    DGATE,
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = NUM_TOKENS * HEAD_DIM
    mask = offsets < total
    token_linear = offsets // HEAD_DIM
    dim = offsets - token_linear * HEAD_DIM
    block_linear = token_linear // BLOCK_ELEMENTS
    coarse_offset = block_linear * HEAD_DIM + dim
    dout = tl.load(DOUT + offsets, mask=mask, other=0.0).to(tl.float32)
    coarse = tl.load(COARSE + coarse_offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(DGATE + offsets, dout * coarse, mask=mask)


@triton.jit
def _combine_coarse_backward_kernel(
    DOUT,
    GATE,
    DCOARSE,
    NUM_BLOCK_ROWS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    block_row = tl.program_id(0)
    dim_tile = tl.program_id(1)
    token_offsets = tl.arange(0, BLOCK_T)
    dim_offsets = dim_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    token_linear = block_row * BLOCK_ELEMENTS + token_offsets
    ptr_offsets = token_linear[:, None] * HEAD_DIM + dim_offsets[None, :]
    mask = (
        (block_row < NUM_BLOCK_ROWS)
        & (token_offsets[:, None] < BLOCK_ELEMENTS)
        & (dim_offsets[None, :] < HEAD_DIM)
    )
    values = tl.load(DOUT + ptr_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_GATE:
        gate = tl.load(GATE + ptr_offsets, mask=mask, other=0.0).to(tl.float32)
        values *= gate
    reduced = tl.sum(values, axis=0)
    tl.store(
        DCOARSE + block_row * HEAD_DIM + dim_offsets,
        reduced,
        mask=dim_offsets < HEAD_DIM,
    )


class _FusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sparse: Tensor,
        coarse_blocks: Tensor,
        gate: Tensor | None,
        block_elements: int,
    ) -> Tensor:
        sparse = sparse.contiguous()
        coarse_blocks = coarse_blocks.contiguous()
        if gate is not None:
            gate = gate.contiguous()
        out = torch.empty_like(sparse)
        num_tokens = sparse.numel() // sparse.shape[-1]
        block = 1024
        grid = (triton.cdiv(sparse.numel(), block),)
        _combine_forward_kernel[grid](
            sparse,
            coarse_blocks,
            gate if gate is not None else sparse,
            out,
            NUM_TOKENS=num_tokens,
            HEAD_DIM=sparse.shape[-1],
            BLOCK_ELEMENTS=block_elements,
            HAS_GATE=gate is not None,
            BLOCK=block,
            num_warps=4,
        )
        if gate is None:
            ctx.save_for_backward(coarse_blocks)
        else:
            ctx.save_for_backward(coarse_blocks, gate)
        ctx.block_elements = block_elements
        ctx.sparse_shape = sparse.shape
        ctx.has_gate = gate is not None
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved = ctx.saved_tensors
        coarse_blocks = saved[0]
        gate = saved[1] if ctx.has_gate else None
        grad_output = grad_output.contiguous()
        head_dim = grad_output.shape[-1]
        block_elements = ctx.block_elements
        num_block_rows = coarse_blocks.numel() // head_dim
        dcoarse = torch.empty_like(coarse_blocks)
        block_t = triton.next_power_of_2(block_elements)
        block_d = min(32, triton.next_power_of_2(head_dim))
        grid = (num_block_rows, triton.cdiv(head_dim, block_d))
        _combine_coarse_backward_kernel[grid](
            grad_output,
            gate if gate is not None else grad_output,
            dcoarse,
            NUM_BLOCK_ROWS=num_block_rows,
            HEAD_DIM=head_dim,
            BLOCK_ELEMENTS=block_elements,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            HAS_GATE=ctx.has_gate,
            num_warps=4,
        )
        dgate = None
        if gate is not None:
            dgate = torch.empty_like(gate)
            block = 256
            gate_grid = (triton.cdiv(gate.numel(), block),)
            _combine_gate_backward_kernel[gate_grid](
                grad_output,
                coarse_blocks,
                dgate,
                NUM_TOKENS=grad_output.numel() // head_dim,
                HEAD_DIM=head_dim,
                BLOCK_ELEMENTS=block_elements,
                BLOCK=block,
                num_warps=4,
            )
        return grad_output, dcoarse, dgate, None


def fused_block_mean(
    x: Tensor,
    variable_block_sizes: Tensor,
    block_elements: int,
) -> Tensor:
    """Block mean without a full-size FP32 temporary."""

    return _FusedBlockMean.apply(x, variable_block_sizes, block_elements)


def fused_combine(
    sparse: Tensor,
    coarse_blocks: Tensor,
    gate: Tensor | None,
    block_elements: int,
) -> Tensor:
    """Broadcast compact coarse vectors and combine in one CUDA pass."""

    return _FusedCombine.apply(sparse, coarse_blocks, gate, block_elements)
