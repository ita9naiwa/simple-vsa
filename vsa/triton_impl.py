"""Tiled Triton VSA matching ``fastvideo_kernel.ops.video_sparse_attn``.

The sparse branch follows the same work decomposition as FastVideo's H100
ThunderKittens kernel:

* one Triton program owns one ``(batch, head, query_block)``;
* the complete Q block is loaded once and reused for every selected KV block;
* ``tl.dot`` computes both ``Q @ K.T`` and ``P @ V`` on Tensor Cores;
* row-wise online softmax combines Top-K tiles without materializing all scores.

The compression branch and Top-K routing remain in readable PyTorch. The fine
branch provides custom Triton forward and backward kernels for Q, K, and V.
"""

import math
import os
from typing import Optional

import torch
from torch import Tensor
import triton
import triton.language as tl

from .common import (
    _as_block_elements,
    coarse_branch,
    coarse_branch_compact,
    validate_vsa_inputs,
)
from .fused_common import fused_block_mean, fused_combine


_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=warps, num_stages=stages)
    for warps in (4, 8)
    for stages in (2, 3, 4)
]

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["Q_TOKENS", "HEAD_DIM", "BLOCK_ELEMENTS", "TOPK", "STORE_LSE"],
)
@triton.jit
def _vsa_tiled_forward_kernel(
    Q,
    K,
    V,
    SELECTED,
    KV_BLOCK_SIZES,
    OUT,
    LSE,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    Q_TOKENS,
    Q_BLOCKS,
    HEADS,
    SM_SCALE,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    """Compute one complete fine-attention query block per Triton program."""

    # grid = (Q_BLOCKS, B*H)
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS

    # BLOCK_M/N/D are padded powers of two. BLOCK_ELEMENTS and HEAD_DIM are the
    # real logical extents.
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    query_start = query_block * BLOCK_ELEMENTS
    query_positions = query_start + rows

    # Q: [BLOCK_M, BLOCK_D]. This tile is loaded once and reused Top-K times.
    q_ptrs = (
        Q
        + batch.to(tl.int64) * stride_qb
        + head.to(tl.int64) * stride_qh
        + query_positions[:, None] * stride_qs
        + dims[None, :] * stride_qd
    )
    q_mask = (rows[:, None] < BLOCK_ELEMENTS) & (dims[None, :] < HEAD_DIM)
    query = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online-softmax state, one value/vector per fine query token.
    row_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
    row_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    selected_base = (batch_head * Q_BLOCKS + query_block) * TOPK
    log2e: tl.constexpr = 1.4426950408889634
    scale = SM_SCALE.to(tl.float32)
    qk_scale = scale * log2e

    # Keep the indirect Top-K loop compact. In particular, do not expand
    # TOPK*BLOCK_ELEMENTS copies as the old nested static_range kernel did.
    for selected_slot in tl.range(0, TOPK, loop_unroll_factor=1):
        kv_block = tl.load(SELECTED + selected_base + selected_slot).to(tl.int32)
        valid_kv_tokens = tl.load(KV_BLOCK_SIZES + kv_block).to(tl.int32)
        kv_start = kv_block * BLOCK_ELEMENTS
        kv_positions = kv_start + cols

        # K: [BLOCK_D, BLOCK_N].
        k_ptrs = (
            K
            + batch.to(tl.int64) * stride_kb
            + head.to(tl.int64) * stride_kh
            + kv_positions[None, :] * stride_ks
            + dims[:, None] * stride_kd
        )
        k_mask = (dims[:, None] < HEAD_DIM) & (cols[None, :] < BLOCK_ELEMENTS)
        key = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # [BLOCK_M,D] @ [D,BLOCK_N] -> [BLOCK_M,BLOCK_N].
        scores = tl.dot(query, key).to(tl.float32) * qk_scale
        valid_scores = cols[None, :] < valid_kv_tokens
        scores = tl.where(valid_scores, scores, -float("inf"))

        # Stable online softmax across all selected KV blocks.
        tile_max = tl.max(scores, axis=1)
        tile_has_value = tile_max != -float("inf")
        new_max = tl.where(
            tile_has_value,
            tl.maximum(row_max, tile_max),
            row_max,
        )
        safe_old_max = tl.where(tile_has_value, row_max, 0.0)
        safe_new_max = tl.where(tile_has_value, new_max, 0.0)
        alpha = tl.where(
            tile_has_value,
            tl.exp2(safe_old_max - safe_new_max),
            1.0,
        )
        probabilities = tl.where(
            valid_scores,
            tl.exp2(scores - safe_new_max[:, None]),
            0.0,
        )

        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        accumulator = accumulator * alpha[:, None]

        # V: [BLOCK_N, BLOCK_D]. P @ V -> [BLOCK_M, BLOCK_D].
        v_ptrs = (
            V
            + batch.to(tl.int64) * stride_vb
            + head.to(tl.int64) * stride_vh
            + kv_positions[:, None] * stride_vs
            + dims[None, :] * stride_vd
        )
        v_mask = (cols[:, None] < valid_kv_tokens) & (dims[None, :] < HEAD_DIM)
        value = tl.load(v_ptrs, mask=v_mask, other=0.0)
        accumulator += tl.dot(probabilities.to(value.dtype), value)
        row_max = new_max

    output = accumulator / row_sum[:, None]
    out_ptrs = (
        OUT
        + batch.to(tl.int64) * stride_ob
        + head.to(tl.int64) * stride_oh
        + query_positions[:, None] * stride_os
        + dims[None, :] * stride_od
    )
    out_mask = (rows[:, None] < BLOCK_ELEMENTS) & (dims[None, :] < HEAD_DIM)
    tl.store(out_ptrs, output, mask=out_mask)

    if STORE_LSE:
        lse = row_max + tl.log2(row_sum)
        lse_ptrs = LSE + batch_head * Q_TOKENS + query_positions
        tl.store(lse_ptrs, lse, mask=rows < BLOCK_ELEMENTS)


@triton.jit
def _vsa_tiled_forward_256_kernel(
    Q,
    K,
    V,
    SELECTED,
    KV_BLOCK_SIZES,
    OUT,
    LSE,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    Q_TOKENS,
    Q_BLOCKS,
    HEADS,
    SM_SCALE,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    STORE_LSE: tl.constexpr,
    Q_TILE: tl.constexpr,
):
    """Logical-Q256 forward with physical Q128 and KV64 tiles."""

    query_subtile_pid = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    q_subtiles = 256 // Q_TILE
    query_block = query_subtile_pid // q_subtiles
    query_subtile = query_subtile_pid - query_block * q_subtiles

    rows = tl.arange(0, Q_TILE)
    cols = tl.arange(0, 64)
    dims = tl.arange(0, BLOCK_D)
    query_positions = query_block * 256 + query_subtile * Q_TILE + rows

    q_ptrs = (
        Q
        + batch.to(tl.int64) * stride_qb
        + head.to(tl.int64) * stride_qh
        + query_positions[:, None] * stride_qs
        + dims[None, :] * stride_qd
    )
    q_mask = (rows[:, None] < Q_TILE) & (dims[None, :] < HEAD_DIM)
    query = tl.load(q_ptrs, mask=q_mask, other=0.0)

    row_max = tl.full([Q_TILE], -float("inf"), tl.float32)
    row_sum = tl.zeros([Q_TILE], dtype=tl.float32)
    accumulator = tl.zeros([Q_TILE, BLOCK_D], dtype=tl.float32)

    selected_base = (batch_head * Q_BLOCKS + query_block) * TOPK
    log2e: tl.constexpr = 1.4426950408889634
    scale = SM_SCALE.to(tl.float32)
    qk_scale = scale * log2e

    # Hoist route metadata once, then consume the logical KV256 block as four
    # physical KV64 tiles.
    for selected_slot in tl.range(0, TOPK, loop_unroll_factor=1):
        kv_block = tl.load(
            SELECTED + selected_base + selected_slot
        ).to(tl.int32)
        valid_kv_tokens = tl.load(
            KV_BLOCK_SIZES + kv_block
        ).to(tl.int32)
        for kv_subtile in tl.range(0, 4, loop_unroll_factor=1):
            kv_offsets = kv_subtile * 64 + cols
            kv_positions = kv_block * 256 + kv_offsets

            k_ptrs = (
                K
                + batch.to(tl.int64) * stride_kb
                + head.to(tl.int64) * stride_kh
                + kv_positions[None, :] * stride_ks
                + dims[:, None] * stride_kd
            )
            key = tl.load(
                k_ptrs,
                mask=(dims[:, None] < HEAD_DIM)
                & (cols[None, :] < 64),
                other=0.0,
            )

            scores = tl.dot(query, key).to(tl.float32) * qk_scale
            valid_scores = kv_offsets[None, :] < valid_kv_tokens
            scores = tl.where(valid_scores, scores, -float("inf"))

            tile_max = tl.max(scores, axis=1)
            tile_has_value = tile_max != -float("inf")
            new_max = tl.where(
                tile_has_value,
                tl.maximum(row_max, tile_max),
                row_max,
            )
            safe_old_max = tl.where(tile_has_value, row_max, 0.0)
            safe_new_max = tl.where(tile_has_value, new_max, 0.0)
            alpha = tl.where(
                tile_has_value,
                tl.exp2(safe_old_max - safe_new_max),
                1.0,
            )
            probabilities = tl.where(
                valid_scores,
                tl.exp2(scores - safe_new_max[:, None]),
                0.0,
            )

            row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
            accumulator = accumulator * alpha[:, None]

            v_ptrs = (
                V
                + batch.to(tl.int64) * stride_vb
                + head.to(tl.int64) * stride_vh
                + kv_positions[:, None] * stride_vs
                + dims[None, :] * stride_vd
            )
            value = tl.load(
                v_ptrs,
                mask=(kv_offsets[:, None] < valid_kv_tokens)
                & (dims[None, :] < HEAD_DIM),
                other=0.0,
            )
            accumulator += tl.dot(probabilities.to(value.dtype), value)
            row_max = new_max

    output = accumulator / row_sum[:, None]
    out_ptrs = (
        OUT
        + batch.to(tl.int64) * stride_ob
        + head.to(tl.int64) * stride_oh
        + query_positions[:, None] * stride_os
        + dims[None, :] * stride_od
    )
    tl.store(
        out_ptrs,
        output,
        mask=(rows[:, None] < Q_TILE)
        & (dims[None, :] < HEAD_DIM),
    )

    if STORE_LSE:
        lse = row_max + tl.log2(row_sum)
        lse_ptrs = LSE + batch_head * Q_TOKENS + query_positions
        tl.store(lse_ptrs, lse)


def _triton_sparse_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    *,
    block_size: int,
    sm_scale: float,
    save_lse: bool,
) -> tuple[Tensor, Tensor]:
    """Launch tiled fine attention from precomputed ``[B,H,Qb,topk]`` ids."""

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("the Triton implementation requires CUDA tensors")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("the Triton path supports float16, bfloat16, and float32")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, and v must have the same dtype")
    if not (1 <= block_size <= 128 or block_size == 256):
        raise ValueError(
            "the tiled Triton kernel supports block_size in [1, 128] or 256"
        )
    if q.shape[-1] > 256:
        raise ValueError("the tiled Triton kernel supports head_dim <= 256")

    batch, heads, query_tokens, head_dim = q.shape
    key_tokens = k.shape[-2]
    query_blocks = query_tokens // block_size
    key_blocks = key_tokens // block_size
    if selected.shape[:3] != (batch, heads, query_blocks):
        raise ValueError(
            "selected must have shape [batch, heads, query_blocks, topk]"
        )
    topk = selected.shape[-1]
    if not 1 <= topk <= key_blocks:
        raise ValueError(f"selected topk must be in [1, {key_blocks}]")
    if variable_block_sizes.numel() != key_blocks:
        raise ValueError(
            f"variable_block_sizes must have {key_blocks} entries, "
            f"got {variable_block_sizes.numel()}"
        )

    selected = selected.to(device=q.device, dtype=torch.int32).contiguous()
    variable_block_sizes = variable_block_sizes.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()
    output = torch.empty_like(q)
    if save_lse:
        lse = torch.empty(
            (batch, heads, query_tokens),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        lse = torch.empty((1,), device=q.device, dtype=torch.float32)

    block_m = max(16, triton.next_power_of_2(block_size))
    block_n = block_m
    block_d = max(16, triton.next_power_of_2(head_dim))
    forward_kernel = (
        _vsa_tiled_forward_256_kernel
        if block_size == 256
        else _vsa_tiled_forward_kernel
    )
    if block_size == 256:
        grid = (query_blocks * 2, batch * heads)
        launch_options = {
            "Q_TILE": 128,
            "num_warps": 4,
            "num_stages": 2,
        }
    else:
        grid = (query_blocks, batch * heads)
        launch_options = {}
    forward_kernel[grid](
        q,
        k,
        v,
        selected,
        variable_block_sizes,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        Q_TOKENS=query_tokens,
        Q_BLOCKS=query_blocks,
        HEADS=heads,
        SM_SCALE=sm_scale,
        HEAD_DIM=head_dim,
        BLOCK_ELEMENTS=block_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        TOPK=topk,
        STORE_LSE=save_lse,
        **launch_options,
    )
    return output, lse


@triton.jit
def _count_inverse_indices_kernel(
    Q2K,
    K2Q_COUNT,
    Q_BLOCKS,
    KV_BLOCKS,
    HEADS,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Count selected query blocks for every KV block."""

    batch = tl.program_id(0)
    head = tl.program_id(1)
    query_block = tl.program_id(2)
    batch_head = batch * HEADS + head
    query_row = (batch_head * Q_BLOCKS + query_block) * TOPK

    selected_slots = tl.arange(0, BLOCK_TOPK)
    selected_mask = selected_slots < TOPK
    kv_blocks = tl.load(
        Q2K + query_row + selected_slots,
        mask=selected_mask,
        other=0,
    ).to(tl.int32)
    tl.atomic_add(
        K2Q_COUNT + batch_head * KV_BLOCKS + kv_blocks,
        1,
        mask=selected_mask,
    )


@triton.jit
def _scatter_inverse_indices_kernel(
    Q2K,
    K2Q,
    K2Q_CURSOR,
    Q_BLOCKS,
    KV_BLOCKS,
    HEADS,
    EDGES_PER_HEAD,
    TOPK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Scatter query ids into compact CSR storage after prefix-summing counts."""

    batch = tl.program_id(0)
    head = tl.program_id(1)
    query_block = tl.program_id(2)
    batch_head = batch * HEADS + head
    query_row = (batch_head * Q_BLOCKS + query_block) * TOPK

    selected_slots = tl.arange(0, BLOCK_TOPK)
    selected_mask = selected_slots < TOPK
    kv_blocks = tl.load(
        Q2K + query_row + selected_slots,
        mask=selected_mask,
        other=0,
    ).to(tl.int32)
    cursor_ptrs = K2Q_CURSOR + batch_head * KV_BLOCKS + kv_blocks
    edge_offsets = tl.atomic_add(cursor_ptrs, 1, mask=selected_mask)
    tl.store(
        K2Q + batch_head * EDGES_PER_HEAD + edge_offsets,
        query_block,
        mask=selected_mask,
    )


def _invert_indices(
    selected: Tensor,
    key_blocks: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build compact KV->query-block CSR metadata entirely on the GPU.

    The edge buffer stores exactly ``query_blocks * topk`` entries per
    batch/head instead of reserving ``key_blocks * query_blocks`` entries.
    """

    batch, heads, query_blocks, topk = selected.shape
    selected = selected.to(torch.int32).contiguous()
    block_topk = triton.next_power_of_2(topk)
    counts = torch.zeros(
        (batch, heads, key_blocks),
        device=selected.device,
        dtype=torch.int32,
    )
    grid = (batch, heads, query_blocks)
    _count_inverse_indices_kernel[grid](
        selected,
        counts,
        Q_BLOCKS=query_blocks,
        KV_BLOCKS=key_blocks,
        HEADS=heads,
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )

    # Exclusive prefix offsets for each KV block, relative to its batch/head.
    offsets = torch.cumsum(counts, dim=-1, dtype=torch.int32) - counts
    cursor = offsets.clone()
    edges_per_head = query_blocks * topk
    inverse = torch.empty(
        (batch, heads, edges_per_head),
        device=selected.device,
        dtype=torch.int32,
    )
    _scatter_inverse_indices_kernel[grid](
        selected,
        inverse,
        cursor,
        Q_BLOCKS=query_blocks,
        KV_BLOCKS=key_blocks,
        HEADS=heads,
        EDGES_PER_HEAD=edges_per_head,
        TOPK=topk,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )
    return inverse, offsets, counts


@triton.jit
def _delta_kernel(
    OUT,
    DOUT,
    DELTA,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    Q_TOKENS,
    HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    positions = query_block * BLOCK_ELEMENTS + rows
    mask = (rows[:, None] < BLOCK_ELEMENTS) & (dims[None, :] < HEAD_DIM)

    out_ptrs = (
        OUT
        + batch.to(tl.int64) * stride_ob
        + head.to(tl.int64) * stride_oh
        + positions[:, None] * stride_os
        + dims[None, :] * stride_od
    )
    dout_ptrs = (
        DOUT
        + batch.to(tl.int64) * stride_dob
        + head.to(tl.int64) * stride_doh
        + positions[:, None] * stride_dos
        + dims[None, :] * stride_dod
    )
    out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(out * dout, axis=1)
    tl.store(
        DELTA + batch_head * Q_TOKENS + positions,
        delta,
        mask=rows < BLOCK_ELEMENTS,
    )


@triton.jit
def _delta_256_kernel(
    OUT,
    DOUT,
    DELTA,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    Q_TOKENS,
    HEADS,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute delta for one physical Q64 quarter of a logical Q256 block."""

    query_subtile_pid = tl.program_id(0)
    batch_head = tl.program_id(1)
    query_block = query_subtile_pid // 4
    query_subtile = query_subtile_pid - query_block * 4
    batch = batch_head // HEADS
    head = batch_head % HEADS

    rows = tl.arange(0, 64)
    dims = tl.arange(0, BLOCK_D)
    positions = query_block * 256 + query_subtile * 64 + rows
    mask = (rows[:, None] < 64) & (dims[None, :] < HEAD_DIM)

    out_ptrs = (
        OUT
        + batch.to(tl.int64) * stride_ob
        + head.to(tl.int64) * stride_oh
        + positions[:, None] * stride_os
        + dims[None, :] * stride_od
    )
    dout_ptrs = (
        DOUT
        + batch.to(tl.int64) * stride_dob
        + head.to(tl.int64) * stride_doh
        + positions[:, None] * stride_dos
        + dims[None, :] * stride_dod
    )
    out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptrs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(out * dout, axis=1)
    tl.store(DELTA + batch_head * Q_TOKENS + positions, delta)


@triton.jit
def _vsa_dq_256_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    DELTA,
    SELECTED,
    KV_BLOCK_SIZES,
    DQ,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dqd,
    Q_TOKENS,
    Q_BLOCKS,
    HEADS,
    SM_SCALE,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
    Q_TILE: tl.constexpr,
):
    """Compute dQ for a physical Q128 tile against logical KV256."""

    query_subtile_pid = tl.program_id(0)
    batch_head = tl.program_id(1)
    q_subtiles = 256 // Q_TILE
    query_block = query_subtile_pid // q_subtiles
    query_subtile = query_subtile_pid - query_block * q_subtiles
    batch = batch_head // HEADS
    head = batch_head % HEADS

    rows = tl.arange(0, Q_TILE)
    cols = tl.arange(0, 64)
    dims = tl.arange(0, BLOCK_D)
    query_positions = query_block * 256 + query_subtile * Q_TILE + rows
    q_mask = (rows[:, None] < Q_TILE) & (dims[None, :] < HEAD_DIM)

    q_ptrs = (
        Q
        + batch.to(tl.int64) * stride_qb
        + head.to(tl.int64) * stride_qh
        + query_positions[:, None] * stride_qs
        + dims[None, :] * stride_qd
    )
    do_ptrs = (
        DOUT
        + batch.to(tl.int64) * stride_dob
        + head.to(tl.int64) * stride_doh
        + query_positions[:, None] * stride_dos
        + dims[None, :] * stride_dod
    )
    query = tl.load(q_ptrs, mask=q_mask, other=0.0)
    dout = tl.load(do_ptrs, mask=q_mask, other=0.0)
    lse = tl.load(LSE + batch_head * Q_TOKENS + query_positions)
    delta = tl.load(DELTA + batch_head * Q_TOKENS + query_positions)

    dq = tl.zeros([Q_TILE, BLOCK_D], dtype=tl.float32)
    selected_base = (batch_head * Q_BLOCKS + query_block) * TOPK
    log2e: tl.constexpr = 1.4426950408889634
    scale = SM_SCALE.to(tl.float32)
    qk_scale = scale * log2e

    for selected_slot in tl.range(0, TOPK, loop_unroll_factor=1):
        kv_block = tl.load(
            SELECTED + selected_base + selected_slot
        ).to(tl.int32)
        valid_kv_tokens = tl.load(
            KV_BLOCK_SIZES + kv_block
        ).to(tl.int32)
        for kv_subtile in tl.range(0, 4, loop_unroll_factor=1):
            kv_offsets = kv_subtile * 64 + cols
            kv_positions = kv_block * 256 + kv_offsets

            k_ptrs = (
                K
                + batch.to(tl.int64) * stride_kb
                + head.to(tl.int64) * stride_kh
                + kv_positions[None, :] * stride_ks
                + dims[:, None] * stride_kd
            )
            vt_ptrs = (
                V
                + batch.to(tl.int64) * stride_vb
                + head.to(tl.int64) * stride_vh
                + kv_positions[None, :] * stride_vs
                + dims[:, None] * stride_vd
            )
            tile_mask = (dims[:, None] < HEAD_DIM) & (
                cols[None, :] < 64
            )
            key_t = tl.load(k_ptrs, mask=tile_mask, other=0.0)
            value_t = tl.load(vt_ptrs, mask=tile_mask, other=0.0)

            scores = tl.dot(query, key_t).to(tl.float32) * qk_scale
            valid = kv_offsets[None, :] < valid_kv_tokens
            probability = tl.where(
                valid,
                tl.exp2(scores - lse[:, None]),
                0.0,
            )
            dp = tl.dot(dout, value_t).to(tl.float32)
            ds = probability * (dp - delta[:, None]) * scale
            dq += tl.dot(ds.to(query.dtype), tl.trans(key_t))

    dq_ptrs = (
        DQ
        + batch.to(tl.int64) * stride_dqb
        + head.to(tl.int64) * stride_dqh
        + query_positions[:, None] * stride_dqs
        + dims[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=q_mask)


@triton.jit
def _vsa_dq_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    DELTA,
    SELECTED,
    KV_BLOCK_SIZES,
    DQ,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqs,
    stride_dqd,
    Q_TOKENS,
    Q_BLOCKS,
    HEADS,
    SM_SCALE,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOPK: tl.constexpr,
):
    """One program computes dQ for one complete query block."""

    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    query_positions = query_block * BLOCK_ELEMENTS + rows
    q_mask = (rows[:, None] < BLOCK_ELEMENTS) & (dims[None, :] < HEAD_DIM)

    q_ptrs = (
        Q
        + batch.to(tl.int64) * stride_qb
        + head.to(tl.int64) * stride_qh
        + query_positions[:, None] * stride_qs
        + dims[None, :] * stride_qd
    )
    do_ptrs = (
        DOUT
        + batch.to(tl.int64) * stride_dob
        + head.to(tl.int64) * stride_doh
        + query_positions[:, None] * stride_dos
        + dims[None, :] * stride_dod
    )
    query = tl.load(q_ptrs, mask=q_mask, other=0.0)
    dout = tl.load(do_ptrs, mask=q_mask, other=0.0)
    lse = tl.load(
        LSE + batch_head * Q_TOKENS + query_positions,
        mask=rows < BLOCK_ELEMENTS,
        other=0.0,
    )
    delta = tl.load(
        DELTA + batch_head * Q_TOKENS + query_positions,
        mask=rows < BLOCK_ELEMENTS,
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    selected_base = (batch_head * Q_BLOCKS + query_block) * TOPK
    log2e: tl.constexpr = 1.4426950408889634
    scale = SM_SCALE.to(tl.float32)
    qk_scale = scale * log2e

    for selected_slot in tl.range(0, TOPK, loop_unroll_factor=1):
        kv_block = tl.load(SELECTED + selected_base + selected_slot).to(tl.int32)
        valid_kv_tokens = tl.load(KV_BLOCK_SIZES + kv_block).to(tl.int32)
        kv_positions = kv_block * BLOCK_ELEMENTS + cols

        k_ptrs = (
            K
            + batch.to(tl.int64) * stride_kb
            + head.to(tl.int64) * stride_kh
            + kv_positions[None, :] * stride_ks
            + dims[:, None] * stride_kd
        )
        vt_ptrs = (
            V
            + batch.to(tl.int64) * stride_vb
            + head.to(tl.int64) * stride_vh
            + kv_positions[None, :] * stride_vs
            + dims[:, None] * stride_vd
        )
        tile_mask = (dims[:, None] < HEAD_DIM) & (
            cols[None, :] < BLOCK_ELEMENTS
        )
        key_t = tl.load(k_ptrs, mask=tile_mask, other=0.0)
        value_t = tl.load(vt_ptrs, mask=tile_mask, other=0.0)

        scores = tl.dot(query, key_t).to(tl.float32) * qk_scale
        valid = (rows[:, None] < BLOCK_ELEMENTS) & (
            cols[None, :] < valid_kv_tokens
        )
        probability = tl.where(
            valid,
            tl.exp2(scores - lse[:, None]),
            0.0,
        )
        dp = tl.dot(dout, value_t).to(tl.float32)
        ds = probability * (dp - delta[:, None]) * scale
        dq += tl.dot(ds.to(query.dtype), tl.trans(key_t))

    dq_ptrs = (
        DQ
        + batch.to(tl.int64) * stride_dqb
        + head.to(tl.int64) * stride_dqh
        + query_positions[:, None] * stride_dqs
        + dims[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq, mask=q_mask)


@triton.jit
def _vsa_dkdv_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    DELTA,
    K2Q,
    K2Q_OFFSETS,
    K2Q_COUNT,
    KV_BLOCK_SIZES,
    DK,
    DV,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dkb,
    stride_dkh,
    stride_dks,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvs,
    stride_dvd,
    Q_TOKENS,
    KV_BLOCKS,
    HEADS,
    EDGES_PER_HEAD,
    SM_SCALE,
    HEAD_DIM: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program computes dK and dV for one complete KV block."""

    kv_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    kv_positions = kv_block * BLOCK_ELEMENTS + cols
    valid_kv_tokens = tl.load(KV_BLOCK_SIZES + kv_block).to(tl.int32)

    kv_mask = (cols[:, None] < BLOCK_ELEMENTS) & (dims[None, :] < HEAD_DIM)
    k_ptrs = (
        K
        + batch.to(tl.int64) * stride_kb
        + head.to(tl.int64) * stride_kh
        + kv_positions[:, None] * stride_ks
        + dims[None, :] * stride_kd
    )
    v_ptrs = (
        V
        + batch.to(tl.int64) * stride_vb
        + head.to(tl.int64) * stride_vh
        + kv_positions[:, None] * stride_vs
        + dims[None, :] * stride_vd
    )
    key = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    value = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    metadata_offset = batch_head * KV_BLOCKS + kv_block
    inverse_row = (
        batch_head * EDGES_PER_HEAD
        + tl.load(K2Q_OFFSETS + metadata_offset)
    )
    query_count = tl.load(K2Q_COUNT + metadata_offset)
    log2e: tl.constexpr = 1.4426950408889634
    scale = SM_SCALE.to(tl.float32)
    qk_scale = scale * log2e

    for query_slot in tl.range(0, query_count, loop_unroll_factor=1):
        query_block = tl.load(K2Q + inverse_row + query_slot).to(tl.int32)
        query_positions = query_block * BLOCK_ELEMENTS + rows
        q_mask = (rows[:, None] < BLOCK_ELEMENTS) & (dims[None, :] < HEAD_DIM)
        q_ptrs = (
            Q
            + batch.to(tl.int64) * stride_qb
            + head.to(tl.int64) * stride_qh
            + query_positions[:, None] * stride_qs
            + dims[None, :] * stride_qd
        )
        do_ptrs = (
            DOUT
            + batch.to(tl.int64) * stride_dob
            + head.to(tl.int64) * stride_doh
            + query_positions[:, None] * stride_dos
            + dims[None, :] * stride_dod
        )
        query = tl.load(q_ptrs, mask=q_mask, other=0.0)
        dout = tl.load(do_ptrs, mask=q_mask, other=0.0)
        lse = tl.load(
            LSE + batch_head * Q_TOKENS + query_positions,
            mask=rows < BLOCK_ELEMENTS,
            other=0.0,
        )
        delta = tl.load(
            DELTA + batch_head * Q_TOKENS + query_positions,
            mask=rows < BLOCK_ELEMENTS,
            other=0.0,
        )

        scores = tl.dot(query, tl.trans(key)).to(tl.float32) * qk_scale
        valid = (rows[:, None] < BLOCK_ELEMENTS) & (
            cols[None, :] < valid_kv_tokens
        )
        probability = tl.where(
            valid,
            tl.exp2(scores - lse[:, None]),
            0.0,
        )
        dp = tl.dot(dout, tl.trans(value)).to(tl.float32)
        ds = probability * (dp - delta[:, None]) * scale

        dk += tl.dot(tl.trans(ds.to(query.dtype)), query)
        dv += tl.dot(tl.trans(probability.to(dout.dtype)), dout)

    dk_ptrs = (
        DK
        + batch.to(tl.int64) * stride_dkb
        + head.to(tl.int64) * stride_dkh
        + kv_positions[:, None] * stride_dks
        + dims[None, :] * stride_dkd
    )
    dv_ptrs = (
        DV
        + batch.to(tl.int64) * stride_dvb
        + head.to(tl.int64) * stride_dvh
        + kv_positions[:, None] * stride_dvs
        + dims[None, :] * stride_dvd
    )
    tl.store(dk_ptrs, dk, mask=kv_mask)
    tl.store(dv_ptrs, dv, mask=kv_mask)


@triton.jit
def _vsa_dkdv_256_kernel(
    Q,
    K,
    V,
    DOUT,
    LSE,
    DELTA,
    K2Q,
    K2Q_OFFSETS,
    K2Q_COUNT,
    KV_BLOCK_SIZES,
    DK,
    DV,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dos,
    stride_dod,
    stride_dkb,
    stride_dkh,
    stride_dks,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvs,
    stride_dvd,
    Q_TOKENS,
    KV_BLOCKS,
    HEADS,
    EDGES_PER_HEAD,
    SM_SCALE,
    TOPK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Q_TILE: tl.constexpr,
):
    """Own one KV64 quarter and consume physical Q128 tiles."""

    kv_subtile_pid = tl.program_id(0)
    batch_head = tl.program_id(1)
    kv_block = kv_subtile_pid // 4
    kv_subtile = kv_subtile_pid - kv_block * 4
    batch = batch_head // HEADS
    head = batch_head % HEADS

    rows = tl.arange(0, Q_TILE)
    cols = tl.arange(0, 64)
    dims = tl.arange(0, BLOCK_D)
    kv_offsets = kv_subtile * 64 + cols
    kv_positions = kv_block * 256 + kv_offsets
    valid_kv_tokens = tl.load(
        KV_BLOCK_SIZES + kv_block
    ).to(tl.int32)

    kv_mask = (cols[:, None] < 64) & (
        dims[None, :] < HEAD_DIM
    )
    k_ptrs = (
        K
        + batch.to(tl.int64) * stride_kb
        + head.to(tl.int64) * stride_kh
        + kv_positions[:, None] * stride_ks
        + dims[None, :] * stride_kd
    )
    v_ptrs = (
        V
        + batch.to(tl.int64) * stride_vb
        + head.to(tl.int64) * stride_vh
        + kv_positions[:, None] * stride_vs
        + dims[None, :] * stride_vd
    )
    key = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    value = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    dk = tl.zeros([64, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([64, BLOCK_D], dtype=tl.float32)

    metadata_offset = batch_head * KV_BLOCKS + kv_block
    inverse_row = (
        batch_head * EDGES_PER_HEAD
        + tl.load(K2Q_OFFSETS + metadata_offset)
    )
    query_count = tl.load(K2Q_COUNT + metadata_offset)
    log2e: tl.constexpr = 1.4426950408889634
    scale = SM_SCALE.to(tl.float32)
    qk_scale = scale * log2e

    for query_slot in tl.range(
        0,
        query_count,
        loop_unroll_factor=1,
    ):
        query_block = tl.load(
            K2Q + inverse_row + query_slot
        ).to(tl.int32)

        # Keep the physical Q tiles sequential. Unrolling duplicates
        # score/probability/gradient tiles and can exceed Blackwell SMEM.
        for query_subtile in tl.range(
            0,
            256 // Q_TILE,
            loop_unroll_factor=1,
        ):
            query_positions = (
                query_block * 256
                + query_subtile * Q_TILE
                + rows
            )
            q_mask = (rows[:, None] < Q_TILE) & (
                dims[None, :] < HEAD_DIM
            )
            q_ptrs = (
                Q
                + batch.to(tl.int64) * stride_qb
                + head.to(tl.int64) * stride_qh
                + query_positions[:, None] * stride_qs
                + dims[None, :] * stride_qd
            )
            do_ptrs = (
                DOUT
                + batch.to(tl.int64) * stride_dob
                + head.to(tl.int64) * stride_doh
                + query_positions[:, None] * stride_dos
                + dims[None, :] * stride_dod
            )
            query = tl.load(q_ptrs, mask=q_mask, other=0.0)
            dout = tl.load(do_ptrs, mask=q_mask, other=0.0)
            lse = tl.load(
                LSE + batch_head * Q_TOKENS + query_positions
            )
            delta = tl.load(
                DELTA + batch_head * Q_TOKENS + query_positions
            )

            scores = (
                tl.dot(query, tl.trans(key)).to(tl.float32)
                * qk_scale
            )
            valid = kv_offsets[None, :] < valid_kv_tokens
            probability = tl.where(
                valid,
                tl.exp2(scores - lse[:, None]),
                0.0,
            )
            dp = tl.dot(
                dout,
                tl.trans(value),
            ).to(tl.float32)
            ds = probability * (
                dp - delta[:, None]
            ) * scale

            dk += tl.dot(
                tl.trans(ds.to(query.dtype)),
                query,
            )
            dv += tl.dot(
                tl.trans(probability.to(dout.dtype)),
                dout,
            )

    dk_ptrs = (
        DK
        + batch.to(tl.int64) * stride_dkb
        + head.to(tl.int64) * stride_dkh
        + kv_positions[:, None] * stride_dks
        + dims[None, :] * stride_dkd
    )
    dv_ptrs = (
        DV
        + batch.to(tl.int64) * stride_dvb
        + head.to(tl.int64) * stride_dvh
        + kv_positions[:, None] * stride_dvs
        + dims[None, :] * stride_dvd
    )
    tl.store(dk_ptrs, dk, mask=kv_mask)
    tl.store(dv_ptrs, dv, mask=kv_mask)


def _triton_sparse_attention_backward(
    grad_output: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    output: Tensor,
    lse: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    *,
    block_size: int,
    sm_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Launch preprocess, dQ, and dK/dV kernels."""

    grad_output = grad_output.contiguous()
    batch, heads, query_tokens, head_dim = q.shape
    key_tokens = k.shape[-2]
    query_blocks = query_tokens // block_size
    key_blocks = key_tokens // block_size
    topk = selected.shape[-1]
    block_m = max(16, triton.next_power_of_2(block_size))
    block_n = block_m
    block_d = max(16, triton.next_power_of_2(head_dim))

    delta = torch.empty_like(lse)
    if block_size == 256:
        delta_grid = (query_blocks * 4, batch * heads)
        _delta_256_kernel[delta_grid](
            output,
            grad_output,
            delta,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            Q_TOKENS=query_tokens,
            HEADS=heads,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            num_warps=4,
        )
    else:
        delta_grid = (query_blocks, batch * heads)
        _delta_kernel[delta_grid](
            output,
            grad_output,
            delta,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            Q_TOKENS=query_tokens,
            HEADS=heads,
            HEAD_DIM=head_dim,
            BLOCK_ELEMENTS=block_size,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            num_warps=4,
        )

    inverse, inverse_offsets, inverse_counts = _invert_indices(
        selected,
        key_blocks,
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if block_size == 256:
        dq_grid = (query_blocks * 2, batch * heads)
        _vsa_dq_256_kernel[dq_grid](
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            selected,
            variable_block_sizes,
            dq,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            Q_TOKENS=query_tokens,
            Q_BLOCKS=query_blocks,
            HEADS=heads,
            SM_SCALE=sm_scale,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            TOPK=topk,
            Q_TILE=128,
            num_warps=4,
            num_stages=3,
        )

        dkdv_grid = (key_blocks * 4, batch * heads)
        _vsa_dkdv_256_kernel[dkdv_grid](
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            inverse,
            inverse_offsets,
            inverse_counts,
            variable_block_sizes,
            dk,
            dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            Q_TOKENS=query_tokens,
            KV_BLOCKS=key_blocks,
            HEADS=heads,
            EDGES_PER_HEAD=query_blocks * topk,
            SM_SCALE=sm_scale,
            TOPK=topk,
            HEAD_DIM=head_dim,
            BLOCK_D=block_d,
            Q_TILE=128,
            num_warps=4,
            num_stages=1,
        )
    else:
        dq_grid = (query_blocks, batch * heads)
        _vsa_dq_kernel[dq_grid](
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            selected,
            variable_block_sizes,
            dq,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            Q_TOKENS=query_tokens,
            Q_BLOCKS=query_blocks,
            HEADS=heads,
            SM_SCALE=sm_scale,
            HEAD_DIM=head_dim,
            BLOCK_ELEMENTS=block_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            TOPK=topk,
            num_warps=8,
            num_stages=3,
        )

        dkdv_grid = (key_blocks, batch * heads)
        _vsa_dkdv_kernel[dkdv_grid](
            q,
            k,
            v,
            grad_output,
            lse,
            delta,
            inverse,
            inverse_offsets,
            inverse_counts,
            variable_block_sizes,
            dk,
            dv,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            grad_output.stride(0), grad_output.stride(1),
            grad_output.stride(2), grad_output.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            Q_TOKENS=query_tokens,
            KV_BLOCKS=key_blocks,
            HEADS=heads,
            EDGES_PER_HEAD=query_blocks * topk,
            SM_SCALE=sm_scale,
            HEAD_DIM=head_dim,
            BLOCK_ELEMENTS=block_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
    return dq, dk, dv


class _TritonSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        selected: Tensor,
        variable_block_sizes: Tensor,
        block_size: int,
        sm_scale: float,
    ) -> Tensor:
        output, lse = _triton_sparse_attention_forward(
            q,
            k,
            v,
            selected,
            variable_block_sizes,
            block_size=block_size,
            sm_scale=sm_scale,
            save_lse=True,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            output,
            lse,
            selected,
            variable_block_sizes,
        )
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
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
            block_size=ctx.block_size,
            sm_scale=ctx.sm_scale,
        )
        return dq, dk, dv, None, None, None, None


def _triton_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    *,
    block_size: int,
    sm_scale: float,
) -> Tensor:
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    selected = selected.to(device=q.device, dtype=torch.int32).contiguous()
    variable_block_sizes = variable_block_sizes.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()

    if torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad
    ):
        return _TritonSparseAttention.apply(
            q,
            k,
            v,
            selected,
            variable_block_sizes,
            block_size,
            sm_scale,
        )
    return _triton_sparse_attention_forward(
        q,
        k,
        v,
        selected,
        variable_block_sizes,
        block_size=block_size,
        sm_scale=sm_scale,
        save_lse=False,
    )[0]


def triton_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    *,
    block_size: int | tuple,
    sm_scale: Optional[float] = None,
) -> Tensor:
    """Run only the Triton sparse executor on an externally supplied route.

    This is the routing/fusion experimentation boundary. ``selected`` has shape
    ``[batch, heads, query_blocks, topk]`` and may come from any policy.
    """

    block_elements = _as_block_elements(block_size)
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    return _triton_sparse_attention(
        q,
        k,
        v,
        selected,
        variable_block_sizes,
        block_size=block_elements,
        sm_scale=scale,
    )


def _vsa_with_sparse_executor(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: Optional[Tensor] = None,
    *,
    sparse_executor,
) -> Tensor:
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

    dim = q.shape[-1]
    scale = 1.0 / math.sqrt(dim)
    variable_block_sizes = variable_block_sizes.to(q.device)
    q_variable_block_sizes = q_variable_block_sizes.to(q.device)

    use_fused_common = os.environ.get(
        "SIMPLE_VSA_FUSED_COMMON", "1"
    ) != "0"
    if use_fused_common:
        # Keep compression output compact until the final fused broadcast/add.
        scores, out_c_blocks = coarse_branch_compact(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            variable_block_sizes,
            q_variable_block_sizes,
            block_elements,
            scale,
            block_mean_fn=fused_block_mean,
        )
    else:
        scores, out_c = coarse_branch(
            q,
            k,
            v,
            variable_block_sizes,
            q_variable_block_sizes,
            block_elements,
            scale,
        )
    selected = scores.topk(topk, dim=-1, sorted=False).indices
    out_s = sparse_executor(
        q,
        k,
        v,
        selected,
        variable_block_sizes,
        sm_scale=scale,
    )

    if use_fused_common:
        return fused_combine(
            out_s,
            out_c_blocks,
            compress_attn_weight,
            block_elements,
        )
    if compress_attn_weight is None:
        output = out_c + out_s.float()
    else:
        output = out_c * compress_attn_weight.float() + out_s.float()
    return output.to(q.dtype)


def triton_vsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    variable_block_sizes: Tensor,
    q_variable_block_sizes: Tensor,
    topk: int,
    block_size: int | tuple = 64,
    compress_attn_weight: Optional[Tensor] = None,
) -> Tensor:
    """Run the shared VSA path with owned Triton forward and backward."""

    block_elements = _as_block_elements(block_size)

    def execute(*args, **kwargs):
        return _triton_sparse_attention(
            *args,
            block_size=block_elements,
            **kwargs,
        )

    return _vsa_with_sparse_executor(
        q,
        k,
        v,
        variable_block_sizes,
        q_variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
        sparse_executor=execute,
    )
