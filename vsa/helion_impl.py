"""Helion VSA forward and backward.

The compression branch and Top-K routing stay in the shared PyTorch
implementation.  Helion handles the sparse fine-attention branch: one GPU
program owns one ``(batch, head, query_block)`` and streams its selected KV
blocks with an online softmax.

The custom backward recomputes fine-attention probabilities from the base-2 LSE
saved by forward.  The logical-256 path splits into a Q-owned ``dQ`` kernel and
a KV-owned ``dK``/``dV`` kernel, using compact inverse-routing metadata to avoid
large gradient atomics.  Neither direction materializes gathered
``[B, H, Qb, topk, block, D]`` tensors.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import torch
from torch import Tensor

import helion
import helion.language as hl

from .common import (
    _as_block_elements,
    coarse_branch,
    coarse_branch_compact,
    validate_vsa_inputs,
)
from .fused_common import fused_block_mean, fused_combine


def _helion_tile_specs(
    env_name: str,
    specs: tuple[tuple[Any, ...], ...],
) -> tuple[tuple[Any, ...], ...]:
    value = os.environ.get(env_name)
    if value is None or value == "auto":
        return specs
    try:
        q_tile, kv_tile = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"{env_name} must use QxKV form, e.g. 128x64") from exc
    selected = tuple(
        spec for spec in specs if spec[0] == q_tile and spec[1] == kv_tile
    )
    if not selected:
        choices = "/".join(f"{spec[0]}x{spec[1]}" for spec in specs)
        raise ValueError(f"{env_name} must be one of auto/{choices}")
    return selected


_HELION_FWD_256_SPECS = _helion_tile_specs(
    "SIMPLE_VSA_HELION_FWD_TILE",
    (
        (128, 64, None, "flat", 4, 1),
        (128, 128, True, "persistent_interleaved", 4, 2),
        (256, 64, True, "persistent_interleaved", 4, 3),
        (256, 128, True, "persistent_interleaved", 8, 3),
    ),
)
_HELION_DQ_256_SPECS = _helion_tile_specs(
    "SIMPLE_VSA_HELION_DQ_TILE",
    (
        (64, 64, 4, 1),
        (128, 64, 4, 1),
        (128, 64, 8, 1),
        (128, 128, 8, 2),
    ),
)


@helion.kernel(
    config=helion.Config(
        loop_orders=[[0, 1]],
        l2_groupings=[2],
        range_unroll_factors=[0, 1],
        range_warp_specializes=[None, True],
        pid_type="flat",
        indexing="pointer",
        num_warps=4,
        num_stages=1,
    ),
    static_shapes=True,
)
def _helion_sparse_attention_forward(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    selected_in: Tensor,
    variable_block_sizes: Tensor,
    block_elements_in: int,
) -> tuple[Tensor, Tensor]:
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
    lse = torch.empty(
        [q.size(0), query_tokens],
        device=q_in.device,
        dtype=torch.float32,
    )

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

        for selected_slot in hl.grid(topk):
            kv_block = selected[batch_head, query_block, selected_slot]
            kv_positions = kv_block * block_elements + offsets

            key = k[batch_head, kv_positions, :]
            key_t = key.transpose(-2, -1)
            scores = hl.dot(
                query * qk_scale,
                key_t,
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
        lse[batch_head, query_positions] = row_max + torch.log2(row_sum)

    return out.reshape(q_in.size()), lse.reshape(q_in.size()[:-1])


@helion.kernel(
    configs=[
        helion.Config(
            block_sizes=[q_tile, kv_tile],
            range_warp_specializes=[warp_specialize, None],
            range_multi_buffers=[None, False],
            pid_type=pid_type,
            indexing="pointer",
            num_warps=warps,
            num_stages=stages,
        )
        for q_tile, kv_tile, warp_specialize, pid_type, warps, stages in
        _HELION_FWD_256_SPECS
    ],
    static_shapes=True,
)
def _helion_sparse_attention_forward_256(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    selected_in: Tensor,
    variable_block_sizes: Tensor,
    block_elements_in: int,
) -> tuple[Tensor, Tensor]:
    """Apply logical-256 fine attention with autotuned physical Q/KV tiles.

    Q128/Q256 and KV64/KV128 candidates trade K/V reuse against occupancy.
    Subtiling is used only while rescaling the accumulator, avoiding a full
    TMEM-to-register materialization.
    """

    query_tokens = q_in.size(-2)
    key_tokens = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    block_elements = hl.specialize(block_elements_in)
    query_blocks = selected_in.size(-2)
    topk = hl.specialize(selected_in.size(-1))

    assert block_elements == 256
    assert key_tokens == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    assert query_tokens == query_blocks * block_elements

    q = q_in.reshape([-1, query_tokens, head_dim])
    k = k_in.reshape([-1, key_tokens, head_dim])
    v = v_in.reshape([-1, key_tokens, head_dim])
    selected = selected_in.reshape([-1, query_blocks, topk])
    q_flat = q.reshape([-1, head_dim])
    k_flat = k.reshape([-1, head_dim])
    v_flat = v.reshape([-1, head_dim])
    out = torch.empty_like(q_flat)
    lse = torch.empty(
        [q_flat.size(0)],
        device=q_in.device,
        dtype=torch.float32,
    )

    block_m = hl.register_block_size(128, block_elements)
    block_n = hl.register_block_size(64, 128)
    sparse_kv_tokens = topk * block_elements
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.4426950408889634

    for query_rows in hl.tile(q_flat.size(0), block_size=block_m):
        batch_head = query_rows.begin // query_tokens
        query_block = (
            query_rows.begin - batch_head * query_tokens
        ) // block_elements
        query = q_flat[query_rows, :]

        row_max = hl.full(
            [query_rows],
            float("-inf"),
            dtype=torch.float32,
        )
        row_sum = hl.zeros([query_rows], dtype=torch.float32)
        accumulator = hl.zeros(
            [query_rows, head_dim],
            dtype=torch.float32,
        )

        for sparse_kv_tile in hl.tile(
            sparse_kv_tokens,
            block_size=block_n,
        ):
            selected_slot = sparse_kv_tile.begin // block_elements
            kv_block = selected[
                batch_head,
                query_block,
                selected_slot,
            ]
            valid_kv_tokens = variable_block_sizes[kv_block]
            kv_offsets = (
                sparse_kv_tile.index
                - selected_slot * block_elements
            )
            kv_positions = kv_block * block_elements + kv_offsets
            kv_rows = batch_head * key_tokens + kv_positions
            key = k_flat[kv_rows, :]
            key_t = key.transpose(-2, -1)
            scores = hl.dot(
                query,
                key_t,
                out_dtype=torch.float32,
            )
            valid_kv = kv_offsets < valid_kv_tokens
            scores = torch.where(
                valid_kv[None, :],
                scores * qk_scale,
                float("-inf"),
            )

            new_max = torch.maximum(
                row_max,
                torch.amax(scores, dim=-1),
            )
            probabilities = torch.exp2(
                scores - new_max[:, None]
            )
            alpha = torch.exp2(row_max - new_max)

            row_sum = (
                row_sum * alpha
                + torch.sum(probabilities, dim=-1)
            )
            acc0, acc1 = hl.split(
                accumulator.reshape(
                    [query_rows, 2, head_dim // 2]
                ).permute(0, 2, 1)
            )
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            accumulator = (
                hl.join(acc0, acc1)
                .permute(0, 2, 1)
                .reshape([query_rows, head_dim])
            )
            value = v_flat[kv_rows, :]
            accumulator = hl.dot(
                probabilities.to(value.dtype),
                value,
                acc=accumulator,
            )
            row_max = new_max

        out[query_rows, :] = (
            accumulator / row_sum[:, None]
        ).to(out.dtype)
        lse[query_rows] = row_max + torch.log2(row_sum)

    return out.reshape(q_in.size()), lse.reshape(q_in.size()[:-1])


@helion.kernel(
    config=helion.Config(
        loop_orders=[[0, 1]],
        l2_groupings=[1],
        range_unroll_factors=[0, 1],
        range_warp_specializes=[None, True],
        pid_type="flat",
        indexing="pointer",
        num_warps=4,
        num_stages=1,
    ),
    static_shapes=True,
)
def _helion_sparse_attention_backward(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    selected_in: Tensor,
    variable_block_sizes: Tensor,
    out_in: Tensor,
    lse_in: Tensor,
    grad_out_in: Tensor,
    block_elements_in: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Differentiate fine attention using saved base-2 LSE.

    The launch grid matches forward: one program per ``(batch_head,
    query_block)``.  That program exclusively owns its ``dQ`` rows.  Multiple
    query blocks may select the same KV block, so their ``dK``/``dV``
    contributions are accumulated atomically in FP32.
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
    assert out_in.size(-2) == query_tokens and out_in.size(-1) == head_dim
    assert (
        grad_out_in.size(-2) == query_tokens
        and grad_out_in.size(-1) == head_dim
    )

    q = q_in.reshape([-1, query_tokens, head_dim])
    k = k_in.reshape([-1, key_tokens, head_dim])
    v = v_in.reshape([-1, key_tokens, head_dim])
    selected = selected_in.reshape([-1, query_blocks, topk])
    out = out_in.reshape([-1, query_tokens, head_dim])
    lse = lse_in.reshape([-1, query_tokens])
    grad_out = grad_out_in.reshape([-1, query_tokens, head_dim])

    dq = torch.empty(
        q.size(),
        device=q_in.device,
        dtype=torch.float32,
    )
    dk = torch.zeros(
        k.numel(),
        device=k_in.device,
        dtype=torch.float32,
    )
    dv = torch.zeros(
        v.numel(),
        device=v_in.device,
        dtype=torch.float32,
    )

    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.4426950408889634

    for batch_head, query_block in hl.grid([q.size(0), query_blocks]):
        offsets = hl.arange(block_elements)
        dimensions = hl.arange(head_dim)
        query_positions = query_block * block_elements + offsets

        query = q[batch_head, query_positions, :]
        grad_output = grad_out[batch_head, query_positions, :]
        output = out[batch_head, query_positions, :]
        row_lse = lse[batch_head, query_positions]
        delta = torch.sum(
            output.to(torch.float32) * grad_output.to(torch.float32),
            dim=-1,
        )
        dq_accumulator = hl.zeros(
            [block_elements, head_dim],
            dtype=torch.float32,
        )

        for selected_slot in hl.grid(topk):
            kv_block = selected[batch_head, query_block, selected_slot]
            kv_positions = kv_block * block_elements + offsets
            valid_kv = offsets < variable_block_sizes[kv_block]

            key = k[batch_head, kv_positions, :]
            value = v[batch_head, kv_positions, :]
            key_t = key.transpose(-2, -1)
            value_t = value.transpose(-2, -1)
            scores = hl.dot(
                query * qk_scale,
                key_t,
                out_dtype=torch.float32,
            )
            scores = torch.where(valid_kv[None, :], scores, float("-inf"))
            probabilities = torch.exp2(scores - row_lse[:, None])

            grad_probabilities = hl.dot(
                grad_output,
                value_t,
                out_dtype=torch.float32,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )
            grad_scores_input = grad_scores.to(query.dtype)

            dq_accumulator = hl.dot(
                grad_scores_input,
                key,
                acc=dq_accumulator,
            )
            grad_scores_input_t = grad_scores_input.transpose(-2, -1)
            probabilities_t = probabilities.transpose(-2, -1)
            dk_contribution = hl.dot(
                grad_scores_input_t,
                query,
                out_dtype=torch.float32,
            )
            dv_contribution = hl.dot(
                probabilities_t.to(grad_output.dtype),
                grad_output,
                out_dtype=torch.float32,
            )
            flat_kv_positions = (
                (batch_head * key_tokens + kv_positions[:, None]) * head_dim
                + dimensions[None, :]
            )

            hl.atomic_add(
                dk,
                [flat_kv_positions],
                dk_contribution * sm_scale,
            )
            hl.atomic_add(
                dv,
                [flat_kv_positions],
                dv_contribution,
            )

        dq[batch_head, query_positions, :] = dq_accumulator * sm_scale

    return (
        dq.reshape(q_in.size()),
        dk.reshape(k_in.size()),
        dv.reshape(v_in.size()),
    )


@helion.kernel(
    configs=[
        helion.Config(
            block_sizes=[q_tile, kv_tile],
            range_warp_specializes=[None, None],
            range_multi_buffers=[None, None],
            pid_type="flat",
            indexing="pointer",
            num_warps=warps,
            num_stages=stages,
        )
        for q_tile, kv_tile, warps, stages in _HELION_DQ_256_SPECS
    ],
    static_shapes=True,
)
def _helion_sparse_attention_dq_256(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    selected_in: Tensor,
    variable_block_sizes: Tensor,
    out_in: Tensor,
    lse_in: Tensor,
    grad_out_in: Tensor,
    block_elements_in: int,
) -> tuple[Tensor, Tensor]:
    """Compute delta and dQ with bounded Q64/Q128 and KV64/KV128 tuning.

    Each physical Q tile exclusively owns its dQ and delta rows, so no atomics
    are needed.
    """

    query_tokens = q_in.size(-2)
    key_tokens = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    block_elements = hl.specialize(block_elements_in)
    query_blocks = selected_in.size(-2)
    topk = hl.specialize(selected_in.size(-1))

    assert block_elements == 256
    assert key_tokens == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    assert query_tokens == query_blocks * block_elements
    assert out_in.size(-2) == query_tokens
    assert out_in.size(-1) == head_dim
    assert grad_out_in.size(-2) == query_tokens
    assert grad_out_in.size(-1) == head_dim

    q = q_in.reshape([-1, query_tokens, head_dim])
    k = k_in.reshape([-1, key_tokens, head_dim])
    v = v_in.reshape([-1, key_tokens, head_dim])
    selected = selected_in.reshape([-1, query_blocks, topk])

    q_flat = q.reshape([-1, head_dim])
    k_flat = k.reshape([-1, head_dim])
    v_flat = v.reshape([-1, head_dim])
    out_flat = out_in.reshape([-1, head_dim])
    lse_flat = lse_in.reshape([-1])
    grad_out_flat = grad_out_in.reshape([-1, head_dim])

    dq = torch.empty_like(q_flat)
    delta_out = torch.empty(
        [q_flat.size(0)],
        device=q_in.device,
        dtype=torch.float32,
    )

    block_m = hl.register_block_size(64, 128)
    block_n = hl.register_block_size(64, 128)
    sparse_kv_tokens = topk * block_elements

    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.4426950408889634

    for query_rows in hl.tile(
        q_flat.size(0),
        block_size=block_m,
    ):
        batch_head = query_rows.begin // query_tokens
        query_block = (
            query_rows.begin
            - batch_head * query_tokens
        ) // block_elements

        query = q_flat[query_rows, :]
        grad_output = grad_out_flat[query_rows, :]
        output = out_flat[query_rows, :]
        row_lse = lse_flat[query_rows]

        delta = torch.sum(
            output.to(torch.float32)
            * grad_output.to(torch.float32),
            dim=-1,
        )
        delta_out[query_rows] = delta
        dq_accumulator = hl.zeros(
            [query_rows, head_dim],
            dtype=torch.float32,
        )

        for sparse_kv_tile in hl.tile(
            sparse_kv_tokens,
            block_size=block_n,
        ):
            selected_slot = (
                sparse_kv_tile.begin // block_elements
            )
            kv_block = selected[
                batch_head,
                query_block,
                selected_slot,
            ]
            valid_kv_tokens = variable_block_sizes[kv_block]

            kv_offsets = (
                sparse_kv_tile.index
                - selected_slot * block_elements
            )
            kv_positions = (
                kv_block * block_elements + kv_offsets
            )
            kv_rows = (
                batch_head * key_tokens + kv_positions
            )

            key = k_flat[kv_rows, :]
            value = v_flat[kv_rows, :]
            valid_kv = kv_offsets < valid_kv_tokens
            key = torch.where(valid_kv[:, None], key, 0.0)
            value = torch.where(valid_kv[:, None], value, 0.0)
            key_t = key.transpose(-2, -1)
            value_t = value.transpose(-2, -1)

            scores = hl.dot(
                query,
                key_t,
                out_dtype=torch.float32,
            )
            scores = torch.where(
                valid_kv[None, :],
                scores * qk_scale,
                float("-inf"),
            )
            probabilities = torch.exp2(
                scores - row_lse[:, None]
            )

            grad_probabilities = hl.dot(
                grad_output,
                value_t,
                out_dtype=torch.float32,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )
            grad_scores_input = grad_scores.to(query.dtype)

            dq_accumulator = hl.dot(
                grad_scores_input,
                key,
                acc=dq_accumulator,
            )

        dq[query_rows, :] = dq_accumulator * sm_scale

    return dq.reshape(q_in.size()), delta_out.reshape(
        lse_in.size()
    )


@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 64],
        range_warp_specializes=[None, True],
        range_multi_buffers=[None, False],
        pid_type="flat",
        indexing="pointer",
        num_warps=4,
        num_stages=1,
    ),
    static_shapes=True,
)
def _helion_sparse_attention_dq_64(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    selected_in: Tensor,
    variable_block_sizes: Tensor,
    out_in: Tensor,
    lse_in: Tensor,
    grad_out_in: Tensor,
    block_elements_in: int,
) -> tuple[Tensor, Tensor]:
    """Compute delta and dQ with one owner program per logical Q64 block."""

    query_tokens = q_in.size(-2)
    key_tokens = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    block_elements = hl.specialize(block_elements_in)
    query_blocks = selected_in.size(-2)
    topk = hl.specialize(selected_in.size(-1))

    assert block_elements == 64
    assert key_tokens == v_in.size(-2)
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    assert query_tokens == query_blocks * block_elements

    q = q_in.reshape([-1, query_tokens, head_dim])
    k = k_in.reshape([-1, key_tokens, head_dim])
    v = v_in.reshape([-1, key_tokens, head_dim])
    selected = selected_in.reshape([-1, query_blocks, topk])

    q_flat = q.reshape([-1, head_dim])
    k_flat = k.reshape([-1, head_dim])
    v_flat = v.reshape([-1, head_dim])
    out_flat = out_in.reshape([-1, head_dim])
    lse_flat = lse_in.reshape([-1])
    grad_out_flat = grad_out_in.reshape([-1, head_dim])

    dq = torch.empty_like(q_flat)
    delta_out = torch.empty(
        [q_flat.size(0)],
        device=q_in.device,
        dtype=torch.float32,
    )

    block_m = hl.register_block_size(64)
    block_n = hl.register_block_size(64)
    sparse_kv_tokens = topk * block_elements
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.4426950408889634

    for query_rows in hl.tile(
        q_flat.size(0),
        block_size=block_m,
    ):
        batch_head = query_rows.begin // query_tokens
        query_block = (
            query_rows.begin
            - batch_head * query_tokens
        ) // block_elements

        query = q_flat[query_rows, :]
        grad_output = grad_out_flat[query_rows, :]
        output = out_flat[query_rows, :]
        row_lse = lse_flat[query_rows]

        delta = torch.sum(
            output.to(torch.float32)
            * grad_output.to(torch.float32),
            dim=-1,
        )
        delta_out[query_rows] = delta
        dq_accumulator = hl.zeros(
            [query_rows, head_dim],
            dtype=torch.float32,
        )

        for sparse_kv_tile in hl.tile(
            sparse_kv_tokens,
            block_size=block_n,
        ):
            selected_slot = (
                sparse_kv_tile.begin // block_elements
            )
            kv_block = selected[
                batch_head,
                query_block,
                selected_slot,
            ]
            valid_kv_tokens = variable_block_sizes[kv_block]
            kv_offsets = (
                sparse_kv_tile.index
                - selected_slot * block_elements
            )
            kv_positions = (
                kv_block * block_elements + kv_offsets
            )
            kv_rows = (
                batch_head * key_tokens + kv_positions
            )

            key = k_flat[kv_rows, :]
            value = v_flat[kv_rows, :]
            valid_kv = kv_offsets < valid_kv_tokens
            key = torch.where(valid_kv[:, None], key, 0.0)
            value = torch.where(valid_kv[:, None], value, 0.0)

            scores = hl.dot(
                query,
                key.transpose(-2, -1),
                out_dtype=torch.float32,
            )
            scores = torch.where(
                valid_kv[None, :],
                scores * qk_scale,
                float("-inf"),
            )
            probabilities = torch.exp2(
                scores - row_lse[:, None]
            )
            grad_probabilities = hl.dot(
                grad_output,
                value.transpose(-2, -1),
                out_dtype=torch.float32,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )
            dq_accumulator = hl.dot(
                grad_scores.to(query.dtype),
                key,
                acc=dq_accumulator,
            )

        dq[query_rows, :] = dq_accumulator * sm_scale

    return dq.reshape(q_in.size()), delta_out.reshape(
        lse_in.size()
    )


@helion.kernel(
    config=helion.Config(
        block_sizes=[256],
        pid_type="flat",
        indexing="pointer",
        atomic_indexing="pointer",
        num_warps=4,
        num_stages=1,
    ),
    static_shapes=True,
)
def _helion_inverse_counts_256(
    selected_in: Tensor,
    key_blocks_in: int,
) -> Tensor:
    """Count incoming query-block edges for every batch/head/KV block."""

    query_blocks = hl.specialize(selected_in.size(-2))
    topk = hl.specialize(selected_in.size(-1))
    key_blocks = hl.specialize(key_blocks_in)
    edges_per_head = query_blocks * topk
    batch_heads = selected_in.numel() // edges_per_head

    selected_flat = selected_in.reshape([-1])
    counts = torch.zeros(
        [batch_heads * key_blocks],
        device=selected_in.device,
        dtype=torch.int32,
    )
    block_e = hl.register_block_size(256)

    for edges in hl.tile(
        selected_flat.size(0),
        block_size=block_e,
    ):
        batch_head = edges.index // edges_per_head
        kv_block = selected_flat[edges]
        metadata_index = batch_head * key_blocks + kv_block
        hl.atomic_add(counts, [metadata_index], 1)

    return counts.reshape([batch_heads, key_blocks])


@helion.kernel(
    config=helion.Config(
        block_sizes=[256],
        pid_type="flat",
        indexing="pointer",
        atomic_indexing="pointer",
        num_warps=4,
        num_stages=1,
    ),
    static_shapes=True,
)
def _helion_inverse_scatter_256(
    selected_in: Tensor,
    cursor_in: Tensor,
    key_blocks_in: int,
) -> Tensor:
    """Scatter query-block ids into compact KV-to-query CSR storage."""

    query_blocks = hl.specialize(selected_in.size(-2))
    topk = hl.specialize(selected_in.size(-1))
    key_blocks = hl.specialize(key_blocks_in)
    edges_per_head = query_blocks * topk
    batch_heads = selected_in.numel() // edges_per_head

    selected_flat = selected_in.reshape([-1])
    cursor_flat = cursor_in.reshape([-1])
    inverse = torch.empty(
        [batch_heads * edges_per_head],
        device=selected_in.device,
        dtype=torch.int32,
    )
    block_e = hl.register_block_size(256)

    for edges in hl.tile(
        selected_flat.size(0),
        block_size=block_e,
    ):
        batch_head = edges.index // edges_per_head
        edge_in_head = edges.index - batch_head * edges_per_head
        query_block = edge_in_head // topk
        kv_block = selected_flat[edges]
        metadata_index = batch_head * key_blocks + kv_block
        inverse_offset = hl.atomic_add(
            cursor_flat,
            [metadata_index],
            1,
        )
        inverse[
            batch_head * edges_per_head + inverse_offset
        ] = query_block

    return inverse.reshape([batch_heads, edges_per_head])


@helion.kernel(
    config=helion.Config(
        block_sizes=[64],
        pid_type="flat",
        indexing="pointer",
        num_warps=4,
        num_stages=1,
    ),
    static_shapes=True,
)
def _helion_sparse_attention_dkdv_256(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    inverse_in: Tensor,
    inverse_offsets_in: Tensor,
    inverse_counts_in: Tensor,
    variable_block_sizes: Tensor,
    lse_in: Tensor,
    delta_in: Tensor,
    grad_out_in: Tensor,
    block_elements_in: int,
) -> tuple[Tensor, Tensor]:
    """Compute dK/dV with one owner program per physical KV64 tile."""

    query_tokens = q_in.size(-2)
    key_tokens = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    block_elements = hl.specialize(block_elements_in)
    key_blocks = inverse_counts_in.size(-1)

    assert block_elements == 256
    assert key_tokens == v_in.size(-2)
    assert key_tokens == key_blocks * block_elements
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    assert grad_out_in.size(-2) == query_tokens
    assert grad_out_in.size(-1) == head_dim

    q = q_in.reshape([-1, query_tokens, head_dim])
    k = k_in.reshape([-1, key_tokens, head_dim])
    v = v_in.reshape([-1, key_tokens, head_dim])
    q_flat = q.reshape([-1, head_dim])
    k_flat = k.reshape([-1, head_dim])
    v_flat = v.reshape([-1, head_dim])
    lse_flat = lse_in.reshape([-1])
    delta_flat = delta_in.reshape([-1])
    grad_out_flat = grad_out_in.reshape([-1, head_dim])

    dk = torch.empty_like(k_flat)
    dv = torch.empty_like(v_flat)

    block_n = hl.register_block_size(64, 64)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.4426950408889634

    for kv_rows in hl.tile(
        k_flat.size(0),
        block_size=block_n,
    ):
        batch_head = kv_rows.begin // key_tokens
        kv_begin = kv_rows.begin - batch_head * key_tokens
        kv_block = kv_begin // block_elements
        kv_offsets = (
            kv_rows.index
            - batch_head * key_tokens
            - kv_block * block_elements
        )
        valid_kv_tokens = variable_block_sizes[kv_block]
        valid_kv = kv_offsets < valid_kv_tokens
        query_offsets = hl.arange(128)

        key = k_flat[kv_rows, :]
        value = v_flat[kv_rows, :]
        key = torch.where(valid_kv[:, None], key, 0.0)
        value = torch.where(valid_kv[:, None], value, 0.0)
        key_t = key.transpose(-2, -1)
        value_t = value.transpose(-2, -1)

        dk_accumulator = hl.zeros(
            [kv_rows, head_dim],
            dtype=torch.float32,
        )
        dv_accumulator = hl.zeros(
            [kv_rows, head_dim],
            dtype=torch.float32,
        )

        inverse_offset = inverse_offsets_in[
            batch_head,
            kv_block,
        ]
        query_count = inverse_counts_in[
            batch_head,
            kv_block,
        ]

        for incoming in hl.tile(query_count, block_size=1):
            query_block = inverse_in[
                batch_head,
                inverse_offset + incoming.begin,
            ]

            for query_subtile in hl.static_range(2):
                query_positions = (
                    query_block * block_elements
                    + query_subtile * 128
                    + query_offsets
                )
                query_rows = (
                    batch_head * query_tokens + query_positions
                )
                query = q_flat[query_rows, :]
                grad_output = grad_out_flat[query_rows, :]
                row_lse = lse_flat[query_rows]
                delta = delta_flat[query_rows]

                scores = hl.dot(
                    query,
                    key_t,
                    out_dtype=torch.float32,
                )
                scores = torch.where(
                    valid_kv[None, :],
                    scores * qk_scale,
                    float("-inf"),
                )
                probabilities = torch.exp2(
                    scores - row_lse[:, None]
                )
                grad_probabilities = hl.dot(
                    grad_output,
                    value_t,
                    out_dtype=torch.float32,
                )
                grad_scores = probabilities * (
                    grad_probabilities - delta[:, None]
                )
                grad_scores_input = grad_scores.to(query.dtype)

                dk_accumulator = hl.dot(
                    grad_scores_input.transpose(-2, -1),
                    query,
                    acc=dk_accumulator,
                )
                dv_accumulator = hl.dot(
                    probabilities.transpose(-2, -1).to(
                        grad_output.dtype
                    ),
                    grad_output,
                    acc=dv_accumulator,
                )

        dk[kv_rows, :] = dk_accumulator * sm_scale
        dv[kv_rows, :] = dv_accumulator

    return dk.reshape(k_in.size()), dv.reshape(v_in.size())


@helion.kernel(
    config=helion.Config(
        block_sizes=[64],
        pid_type="flat",
        indexing="pointer",
        num_warps=4,
        num_stages=1,
    ),
    static_shapes=True,
)
def _helion_sparse_attention_dkdv_64(
    q_in: Tensor,
    k_in: Tensor,
    v_in: Tensor,
    inverse_in: Tensor,
    inverse_offsets_in: Tensor,
    inverse_counts_in: Tensor,
    variable_block_sizes: Tensor,
    lse_in: Tensor,
    delta_in: Tensor,
    grad_out_in: Tensor,
    block_elements_in: int,
) -> tuple[Tensor, Tensor]:
    """Compute dK/dV with one owner program per logical KV64 block."""

    query_tokens = q_in.size(-2)
    key_tokens = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    block_elements = hl.specialize(block_elements_in)
    key_blocks = inverse_counts_in.size(-1)

    assert block_elements == 64
    assert key_tokens == v_in.size(-2)
    assert key_tokens == key_blocks * block_elements
    assert head_dim == k_in.size(-1) == v_in.size(-1)

    q = q_in.reshape([-1, query_tokens, head_dim])
    k = k_in.reshape([-1, key_tokens, head_dim])
    v = v_in.reshape([-1, key_tokens, head_dim])
    q_flat = q.reshape([-1, head_dim])
    k_flat = k.reshape([-1, head_dim])
    v_flat = v.reshape([-1, head_dim])
    lse_flat = lse_in.reshape([-1])
    delta_flat = delta_in.reshape([-1])
    grad_out_flat = grad_out_in.reshape([-1, head_dim])

    dk = torch.empty_like(k_flat)
    dv = torch.empty_like(v_flat)

    block_n = hl.register_block_size(64)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.4426950408889634

    for kv_rows in hl.tile(
        k_flat.size(0),
        block_size=block_n,
    ):
        batch_head = kv_rows.begin // key_tokens
        kv_begin = kv_rows.begin - batch_head * key_tokens
        kv_block = kv_begin // block_elements
        kv_offsets = (
            kv_rows.index
            - batch_head * key_tokens
            - kv_block * block_elements
        )
        valid_kv_tokens = variable_block_sizes[kv_block]
        valid_kv = kv_offsets < valid_kv_tokens
        query_offsets = hl.arange(64)

        key = k_flat[kv_rows, :]
        value = v_flat[kv_rows, :]
        key = torch.where(valid_kv[:, None], key, 0.0)
        value = torch.where(valid_kv[:, None], value, 0.0)
        key_t = key.transpose(-2, -1)
        value_t = value.transpose(-2, -1)

        dk_accumulator = hl.zeros(
            [kv_rows, head_dim],
            dtype=torch.float32,
        )
        dv_accumulator = hl.zeros(
            [kv_rows, head_dim],
            dtype=torch.float32,
        )

        inverse_offset = inverse_offsets_in[
            batch_head,
            kv_block,
        ]
        query_count = inverse_counts_in[
            batch_head,
            kv_block,
        ]

        for incoming in hl.tile(query_count, block_size=1):
            query_block = inverse_in[
                batch_head,
                inverse_offset + incoming.begin,
            ]
            query_positions = (
                query_block * block_elements + query_offsets
            )
            query_rows = (
                batch_head * query_tokens + query_positions
            )
            query = q_flat[query_rows, :]
            grad_output = grad_out_flat[query_rows, :]
            row_lse = lse_flat[query_rows]
            delta = delta_flat[query_rows]

            scores = hl.dot(
                query,
                key_t,
                out_dtype=torch.float32,
            )
            scores = torch.where(
                valid_kv[None, :],
                scores * qk_scale,
                float("-inf"),
            )
            probabilities = torch.exp2(
                scores - row_lse[:, None]
            )
            grad_probabilities = hl.dot(
                grad_output,
                value_t,
                out_dtype=torch.float32,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )

            dk_accumulator = hl.dot(
                grad_scores.to(query.dtype).transpose(-2, -1),
                query,
                acc=dk_accumulator,
            )
            dv_accumulator = hl.dot(
                probabilities.transpose(-2, -1).to(
                    grad_output.dtype
                ),
                grad_output,
                acc=dv_accumulator,
            )

        dk[kv_rows, :] = dk_accumulator * sm_scale
        dv[kv_rows, :] = dv_accumulator

    return dk.reshape(k_in.size()), dv.reshape(v_in.size())


def _helion_sparse_attention_backward_64(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    output: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    block_elements: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Launch split dQ and KV-owned dK/dV for logical block size 64."""

    dq, delta = _helion_sparse_attention_dq_64(
        q,
        k,
        v,
        selected,
        variable_block_sizes,
        output,
        lse,
        grad_output,
        block_elements,
    )

    key_blocks = variable_block_sizes.numel()
    inverse_counts = _helion_inverse_counts_256(
        selected,
        key_blocks,
    )
    inverse_offsets = (
        torch.cumsum(
            inverse_counts,
            dim=-1,
            dtype=torch.int32,
        )
        - inverse_counts
    )
    inverse = _helion_inverse_scatter_256(
        selected,
        inverse_offsets.clone(),
        key_blocks,
    )

    dk, dv = _helion_sparse_attention_dkdv_64(
        q,
        k,
        v,
        inverse,
        inverse_offsets,
        inverse_counts,
        variable_block_sizes,
        lse,
        delta,
        grad_output,
        block_elements,
    )
    return dq, dk, dv


def _helion_sparse_attention_backward_256(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    output: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    block_elements: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Launch split dQ and KV-owned dK/dV Helion backward kernels."""

    dq, delta = _helion_sparse_attention_dq_256(
        q,
        k,
        v,
        selected,
        variable_block_sizes,
        output,
        lse,
        grad_output,
        block_elements,
    )

    key_blocks = variable_block_sizes.numel()
    inverse_counts = _helion_inverse_counts_256(
        selected,
        key_blocks,
    )
    inverse_offsets = (
        torch.cumsum(
            inverse_counts,
            dim=-1,
            dtype=torch.int32,
        )
        - inverse_counts
    )
    inverse = _helion_inverse_scatter_256(
        selected,
        inverse_offsets.clone(),
        key_blocks,
    )

    dk, dv = _helion_sparse_attention_dkdv_256(
        q,
        k,
        v,
        inverse,
        inverse_offsets,
        inverse_counts,
        variable_block_sizes,
        lse,
        delta,
        grad_output,
        block_elements,
    )
    return dq, dk, dv


class _HelionSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        selected: Tensor,
        variable_block_sizes: Tensor,
        block_elements: int,
    ) -> Tensor:
        forward_kernel = (
            _helion_sparse_attention_forward_256
            if block_elements == 256
            else _helion_sparse_attention_forward
        )
        output, lse = forward_kernel(
            q,
            k,
            v,
            selected,
            variable_block_sizes,
            block_elements,
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            selected,
            variable_block_sizes,
            output,
            lse,
        )
        ctx.block_elements = block_elements
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        None,
        None,
        None,
    ]:
        (
            q,
            k,
            v,
            selected,
            variable_block_sizes,
            output,
            lse,
        ) = ctx.saved_tensors
        if ctx.block_elements == 256:
            backward_kernel = _helion_sparse_attention_backward_256
        elif ctx.block_elements == 64:
            backward_kernel = _helion_sparse_attention_backward_64
        else:
            backward_kernel = _helion_sparse_attention_backward
        dq, dk, dv = backward_kernel(
            q,
            k,
            v,
            selected,
            variable_block_sizes,
            output,
            lse,
            grad_output.contiguous(),
            ctx.block_elements,
        )
        return (
            dq.to(q.dtype) if ctx.needs_input_grad[0] else None,
            dk.to(k.dtype) if ctx.needs_input_grad[1] else None,
            dv.to(v.dtype) if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
        )


def _helion_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    block_elements: int,
) -> Tensor:
    if torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad
    ):
        return _HelionSparseAttention.apply(
            q,
            k,
            v,
            selected,
            variable_block_sizes,
            block_elements,
        )
    forward_kernel = (
        _helion_sparse_attention_forward_256
        if block_elements == 256
        else _helion_sparse_attention_forward
    )
    return forward_kernel(
        q,
        k,
        v,
        selected,
        variable_block_sizes,
        block_elements,
    )[0]


def helion_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    selected: Tensor,
    variable_block_sizes: Tensor,
    *,
    block_size: int | tuple,
) -> Tensor:
    """Run only the Helion sparse executor on an externally supplied route."""

    block_elements = _as_block_elements(block_size)
    return _helion_sparse_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        selected.to(device=q.device, dtype=torch.int32).contiguous(),
        variable_block_sizes.to(
            device=q.device,
            dtype=torch.int32,
        ).contiguous(),
        block_elements,
    )


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

    The signature and two-branch math match :func:`torch_vsa`.  Q/K/V gradients
    use the custom Helion backward; the shared PyTorch coarse branch also keeps
    routing and compression-gate gradients in the ordinary autograd graph.
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
    if block_elements > 256:
        raise ValueError("the Helion path currently supports block volume <= 256")
    if q.shape[-1] > 256:
        raise ValueError("the Helion path currently supports head_dim <= 256")

    scale = 1.0 / math.sqrt(q.shape[-1])
    variable_block_sizes = variable_block_sizes.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()
    q_variable_block_sizes = q_variable_block_sizes.to(
        device=q.device,
        dtype=torch.int32,
    ).contiguous()

    use_fused_common = os.environ.get(
        "SIMPLE_VSA_FUSED_COMMON", "1"
    ) != "0"
    if use_fused_common:
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
    selected = scores.topk(topk, dim=-1, sorted=False).indices.to(
        dtype=torch.int32
    ).contiguous()
    out_s = _helion_sparse_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        selected,
        variable_block_sizes,
        block_elements,
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
