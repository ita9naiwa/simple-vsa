"""Tests for the portable FastVideo VSA reference (vsa.fastvideo_ref)."""

import math

import pytest
import torch
import torch.nn.functional as F

from vsa import block_mean, build_vsa_metadata, tile, untile, video_sparse_attn


def _full_size_blocks(num_blocks, block_elements, device="cpu"):
    return torch.full((num_blocks,), block_elements, dtype=torch.long, device=device)


def test_block_mean_matches_masked_mean():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 12, 8)
    vbs = torch.tensor([4, 2, 4])  # 3 blocks of 4, middle one half-padded
    out = block_mean(x, vbs, block_elements=4)

    blocks = x.reshape(2, 3, 3, 4, 8)
    expected = torch.stack(
        [
            blocks[:, :, 0].mean(2),
            blocks[:, :, 1, :2].mean(2),
            blocks[:, :, 2].mean(2),
        ],
        dim=2,
    )
    torch.testing.assert_close(out, expected)


def test_block_mean_ignores_padding_values():
    # With vbs < block_elements, padding token values must not affect the mean.
    x = torch.zeros(1, 1, 4, 2)
    x[0, 0, 0] = torch.tensor([1.0, 3.0])
    x[0, 0, 1] = torch.tensor([3.0, 1.0])
    x[0, 0, 2:] = 999.0  # padding
    out = block_mean(x, torch.tensor([2]), block_elements=4)
    torch.testing.assert_close(out, torch.tensor([[[[2.0, 2.0]]]]))


@pytest.mark.parametrize("be", [4, 8])
def test_selecting_all_blocks_sparse_branch_equals_dense(be):
    # topk == kv_num_blocks and full blocks => sparse branch is dense attention.
    torch.manual_seed(1)
    nb = 3
    seq = nb * be
    q = torch.randn(2, 2, seq, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_size_blocks(nb, be)

    _, aux = video_sparse_attn(
        q, k, v, vbs, vbs, topk=nb, block_size=(1, 1, be), return_aux=True
    )
    dense = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(aux["out_sparse"], dense, atol=1e-5, rtol=1e-5)


def test_output_is_compress_plus_sparse():
    torch.manual_seed(2)
    q = torch.randn(1, 2, 24, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_size_blocks(4, 6)
    out, aux = video_sparse_attn(
        q, k, v, vbs, vbs, topk=2, block_size=(1, 1, 6), return_aux=True
    )
    torch.testing.assert_close(out, aux["out_compress"] + aux["out_sparse"])


def test_gate_scales_compression_branch():
    torch.manual_seed(3)
    q = torch.randn(1, 1, 12, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_size_blocks(3, 4)

    base, aux = video_sparse_attn(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, 4), return_aux=True)
    zero_gate = video_sparse_attn(
        q, k, v, vbs, vbs, topk=2, block_size=(1, 1, 4), compress_attn_weight=torch.zeros_like(q)
    )
    torch.testing.assert_close(zero_gate, aux["out_sparse"])

    two_gate = video_sparse_attn(
        q, k, v, vbs, vbs, topk=2, block_size=(1, 1, 4),
        compress_attn_weight=torch.full_like(q, 2.0),
    )
    torch.testing.assert_close(two_gate, 2.0 * aux["out_compress"] + aux["out_sparse"])


def test_topk_mask_selects_exactly_topk_blocks():
    torch.manual_seed(4)
    q = torch.randn(1, 1, 20, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_size_blocks(5, 4)
    _, aux = video_sparse_attn(q, k, v, vbs, vbs, topk=3, block_size=(1, 1, 4), return_aux=True)
    assert aux["block_mask"].sum(-1).unique().tolist() == [3]


def test_differentiable():
    torch.manual_seed(5)
    q = torch.randn(1, 1, 12, 8, requires_grad=True)
    k = torch.randn(1, 1, 12, 8, requires_grad=True)
    v = torch.randn(1, 1, 12, 8, requires_grad=True)
    vbs = _full_size_blocks(3, 4)
    video_sparse_attn(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, 4)).square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert k.grad is not None and torch.isfinite(k.grad).all()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_rejects_bad_shapes():
    q = k = v = torch.randn(1, 1, 10, 8)
    vbs = _full_size_blocks(2, 4)  # implies seq 8, but seq is 10
    with pytest.raises(ValueError, match="divisible"):
        video_sparse_attn(q, k, v, vbs, vbs, topk=1, block_size=(1, 1, 4))


def test_tiling_roundtrip_is_identity():
    meta = build_vsa_metadata((2, 6, 6), tile_size=(2, 3, 3), device="cpu")
    seq = 2 * 6 * 6
    x = torch.arange(seq, dtype=torch.float32).reshape(1, seq, 1)
    tiled = tile(
        x,
        meta["tile_partition_indices"],
        meta["non_pad_index"],
        num_blocks=math.prod(meta["num_tiles"]),
        max_block_size=meta["max_block_size"],
    )
    restored = untile(
        tiled, meta["non_pad_index"], meta["reverse_tile_partition_indices"]
    )
    torch.testing.assert_close(restored, x)


def test_variable_block_sizes_from_tiling():
    # A 3x5x5 latent tiled by (2,4,4): boundary tiles are partially filled.
    meta = build_vsa_metadata((3, 5, 5), tile_size=(2, 4, 4), device="cpu")
    vbs = meta["variable_block_sizes"]
    # Total valid tokens must equal the real token count.
    assert int(vbs.sum()) == 3 * 5 * 5
    # non_pad_index length matches too.
    assert meta["non_pad_index"].numel() == 3 * 5 * 5
