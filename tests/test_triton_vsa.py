import importlib.util

import pytest
import torch

from vsa import torch_vsa, triton_vsa


HAS_TRITON = importlib.util.find_spec("triton") is not None


def _full_blocks(num_blocks, block_elements, device):
    return torch.full((num_blocks,), block_elements, dtype=torch.long, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_inverse_indices_use_compact_edge_storage():
    from vsa.triton_impl import _invert_indices

    selected = torch.tensor(
        [
            [
                [[0, 2], [1, 2], [0, 3]],
                [[3, 1], [1, 0], [2, 3]],
            ]
        ],
        device="cuda",
        dtype=torch.int32,
    )
    key_blocks = 4
    inverse, offsets, counts = _invert_indices(selected, key_blocks)

    # Storage is one int per selected edge, not a dense [Kb, Qb] table.
    assert inverse.shape == (*selected.shape[:2], selected.shape[2] * selected.shape[3])
    torch.testing.assert_close(
        counts.sum(dim=-1, dtype=torch.int32),
        torch.full_like(counts[..., 0], selected.shape[2] * selected.shape[3]),
    )

    selected_cpu = selected.cpu()
    for batch in range(selected.shape[0]):
        for head in range(selected.shape[1]):
            for key_block in range(key_blocks):
                start = int(offsets[batch, head, key_block])
                count = int(counts[batch, head, key_block])
                actual = sorted(
                    inverse[batch, head, start : start + count].cpu().tolist()
                )
                expected = sorted(
                    selected_cpu[batch, head].eq(key_block).nonzero()[:, 0].tolist()
                )
                assert actual == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_triton_matches_torch_full_blocks():
    torch.manual_seed(17)
    be, nb = 8, 4
    seq = be * nb
    q = torch.randn(1, 2, seq, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be, "cuda")

    with torch.no_grad():
        expected = torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))
        actual = triton_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_triton_matches_torch_padded_blocks():
    torch.manual_seed(19)
    be, nb = 8, 3
    seq = be * nb
    q = torch.randn(1, 2, seq, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = torch.tensor([8, 5, 3], device="cuda")  # padded kv/query blocks

    with torch.no_grad():
        expected = torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))
        actual = triton_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be))

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_triton_matches_torch_with_gate():
    torch.manual_seed(23)
    be, nb = 8, 4
    seq = be * nb
    q = torch.randn(1, 2, seq, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be, "cuda")
    gate = torch.randn(1, 2, seq, 64, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        expected = torch_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be), compress_attn_weight=gate)
        actual = triton_vsa(q, k, v, vbs, vbs, topk=2, block_size=(1, 1, be), compress_attn_weight=gate)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_triton_backward_matches_torch_with_padding_and_gate():
    torch.manual_seed(31)
    be, nb, dim = 64, 4, 128
    shape = (1, 2, be * nb, dim)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    vbs = torch.tensor([64, 41, 64, 19], device="cuda")

    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    triton_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    reference_gate = gate.detach().clone().requires_grad_()
    triton_gate = gate.detach().clone().requires_grad_()

    expected = torch_vsa(
        *reference_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
        compress_attn_weight=reference_gate,
    )
    actual = triton_vsa(
        *triton_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
        compress_attn_weight=triton_gate,
    )

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    for actual_input, expected_input in zip(
        triton_inputs, reference_inputs, strict=True
    ):
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            atol=5e-2,
            rtol=5e-2,
        )
    torch.testing.assert_close(
        triton_gate.grad,
        reference_gate.grad,
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_triton_256_forward_matches_torch_with_padding():
    torch.manual_seed(33)
    be, nb, dim = 256, 3, 128
    shape = (1, 1, be * nb, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = torch.tensor([256, 191, 73], device="cuda")

    with torch.no_grad():
        expected = torch_vsa(
            q,
            k,
            v,
            vbs,
            vbs,
            topk=2,
            block_size=(1, 1, be),
        )
        actual = triton_vsa(
            q,
            k,
            v,
            vbs,
            vbs,
            topk=2,
            block_size=(1, 1, be),
        )

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_triton_256_backward_matches_torch_with_padding():
    torch.manual_seed(35)
    be, nb, dim = 256, 2, 128
    shape = (1, 1, be * nb, dim)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    grad_output = torch.randn(
        shape,
        device="cuda",
        dtype=torch.bfloat16,
    )
    vbs = torch.tensor([256, 173], device="cuda")

    reference_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]
    triton_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]

    expected = torch_vsa(
        *reference_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )
    actual = triton_vsa(
        *triton_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    for actual_input, expected_input in zip(
        triton_inputs,
        reference_inputs,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            atol=8e-2,
            rtol=8e-2,
        )
