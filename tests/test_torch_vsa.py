import pytest
import torch

from vsa import torch_vsa, video_sparse_attn
from vsa.common import coarse_branch


def _full_blocks(num_blocks, block_elements, device="cpu"):
    return torch.full((num_blocks,), block_elements, dtype=torch.long, device=device)


def test_matches_fastvideo_reference_full_blocks():
    # torch_vsa (gather sparse branch) must equal the masked-fill reference.
    torch.manual_seed(1)
    be, nb = 4, 4
    seq = be * nb
    q = torch.randn(2, 2, seq, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be)

    actual = torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))
    expected = video_sparse_attn(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_matches_fastvideo_reference_padded_blocks():
    # Partially-filled (padded) blocks: padding must be excluded from both branches.
    torch.manual_seed(2)
    be, nb = 4, 3
    seq = be * nb
    q = torch.randn(1, 2, seq, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = torch.tensor([4, 2, 3])  # last two blocks are padded

    actual = torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))
    expected = video_sparse_attn(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_compress_attn_weight_matches_reference_gate():
    torch.manual_seed(3)
    be, nb = 4, 4
    seq = be * nb
    q = torch.randn(1, 2, seq, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be)
    gate = torch.randn(1, 2, seq, 16)

    actual = torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be), compress_attn_weight=gate)
    expected = video_sparse_attn(
        q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be), compress_attn_weight=gate
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_int_block_size_volume():
    # block_size as int is treated as a cube, matching _as_block_elements.
    torch.manual_seed(4)
    be, nb = 8, 3  # int block_size=2 -> 2**3 = 8 tokens per block
    seq = be * nb
    q = torch.randn(1, 1, seq, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be)

    actual = torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=2)
    expected = video_sparse_attn(q, k, v, vbs, vbs, topk=2, block_size=2)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_differentiable():
    torch.manual_seed(5)
    be, nb = 4, 3
    seq = be * nb
    q = torch.randn(1, 1, seq, 8, requires_grad=True)
    k = torch.randn(1, 1, seq, 8, requires_grad=True)
    v = torch.randn(1, 1, seq, 8, requires_grad=True)
    vbs = _full_blocks(nb, be)

    torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be)).square().mean().backward()

    for grad in (q.grad, k.grad, v.grad):
        assert grad is not None and torch.isfinite(grad).all()


def test_compression_branch_gives_gradient_to_unselected_blocks():
    # The compression branch's softmax spans all blocks, so even a key block that
    # no query block selects (topk=1) still receives gradient.
    torch.manual_seed(6)
    be, nb = 4, 4
    seq = be * nb
    q = torch.randn(1, 1, seq, 8)
    k = torch.randn(1, 1, seq, 8, requires_grad=True)
    v = torch.randn(1, 1, seq, 8, requires_grad=True)
    vbs = _full_blocks(nb, be)

    scores, _ = coarse_branch(q, k.detach(), v.detach(), vbs, vbs, be, 1.0 / (8 ** 0.5))
    selected = scores.topk(1, dim=-1).indices
    unselected = sorted(set(range(nb)) - set(selected.flatten().tolist()))
    assert unselected, "expected at least one never-selected block for this seed"

    torch_vsa(q, k, v, vbs, vbs, topk=1, block_size=(1, 1, be)).square().mean().backward()

    for block in unselected:
        grad_block = k.grad[0, 0, block * be : (block + 1) * be]
        assert grad_block.abs().sum() > 0


def test_rejects_non_divisible_sequence_length():
    q = k = v = torch.randn(1, 1, 10, 8)
    vbs = _full_blocks(2, 4)  # implies seq 8, but seq is 10
    with pytest.raises(ValueError, match="divisible"):
        torch_vsa(q, k, v, vbs, vbs, topk=1, block_size=(1, 1, 4))


def test_rejects_wrong_block_sizes_length():
    q = k = v = torch.randn(1, 1, 12, 8)
    vbs = _full_blocks(2, 4)  # 12 / 4 = 3 blocks, but vbs has length 2
    with pytest.raises(ValueError, match="variable_block_sizes"):
        torch_vsa(q, k, v, vbs, vbs, topk=1, block_size=(1, 1, 4))


def test_rejects_topk_out_of_range():
    q = k = v = torch.randn(1, 1, 12, 8)
    vbs = _full_blocks(3, 4)
    with pytest.raises(ValueError, match="topk"):
        torch_vsa(q, k, v, vbs, vbs, topk=4, block_size=(1, 1, 4))
