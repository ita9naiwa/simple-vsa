import importlib.util

import pytest
import torch

from vsa import (
    cute_triton_vsa,
    torch_vsa,
    triton_sparse_attention,
    triton_vsa,
)


HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_CUTE = (
    importlib.util.find_spec("flash_attn") is not None
    and importlib.util.find_spec("flash_attn.cute") is not None
)


def _full_blocks(num_blocks, block_elements, device):
    return torch.full((num_blocks,), block_elements, dtype=torch.long, device=device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_fused_block_mean_forward_and_backward_match_torch():
    from vsa.common import block_mean
    from vsa.fused_common import fused_block_mean

    torch.manual_seed(11)
    x = torch.randn(
        1,
        2,
        3 * 64,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    vbs = torch.tensor([64, 37, 11], device="cuda")
    grad = torch.randn(
        1,
        2,
        3,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference = x.detach().clone().requires_grad_()
    actual = x.detach().clone().requires_grad_()
    expected_out = block_mean(reference, vbs, 64)
    actual_out = fused_block_mean(actual, vbs, 64)
    expected_out.backward(grad)
    actual_out.backward(grad)
    torch.testing.assert_close(actual_out, expected_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        actual.grad,
        reference.grad,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
@pytest.mark.parametrize("with_gate", [False, True])
def test_fused_combine_forward_and_backward_match_torch(with_gate):
    from vsa.fused_common import fused_combine

    torch.manual_seed(13)
    be, nb, dim = 64, 3, 128
    sparse_data = torch.randn(
        1, 2, be * nb, dim, device="cuda", dtype=torch.bfloat16
    )
    coarse_data = torch.randn(
        1, 2, nb, dim, device="cuda", dtype=torch.bfloat16
    )
    gate_data = (
        torch.randn_like(sparse_data)
        if with_gate
        else None
    )
    grad = torch.randn_like(sparse_data)

    sparse_ref = sparse_data.detach().clone().requires_grad_()
    coarse_ref = coarse_data.detach().clone().requires_grad_()
    gate_ref = (
        gate_data.detach().clone().requires_grad_()
        if gate_data is not None
        else None
    )
    sparse_actual = sparse_data.detach().clone().requires_grad_()
    coarse_actual = coarse_data.detach().clone().requires_grad_()
    gate_actual = (
        gate_data.detach().clone().requires_grad_()
        if gate_data is not None
        else None
    )

    # The fused kernel multiplies and reduces in FP32 before storing BF16.
    # Mirror that accumulation order instead of using a BF16 autograd reduce.
    coarse_expanded = coarse_ref.float()[:, :, :, None, :].expand(
        -1, -1, -1, be, -1
    ).reshape_as(sparse_ref)
    expected = sparse_ref.float() + (
        coarse_expanded * gate_ref.float()
        if gate_ref is not None
        else coarse_expanded
    )
    actual = fused_combine(
        sparse_actual,
        coarse_actual,
        gate_actual,
        be,
    )
    expected.backward(grad.float())
    actual.backward(grad)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        sparse_actual.grad, sparse_ref.grad, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        coarse_actual.grad, coarse_ref.grad, atol=3e-2, rtol=3e-2
    )
    if gate_ref is not None:
        torch.testing.assert_close(
            gate_actual.grad, gate_ref.grad, atol=3e-2, rtol=3e-2
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
def test_route_explicit_triton_executor_matches_full_sparse_branch():
    torch.manual_seed(15)
    be, nb, dim = 64, 3, 128
    shape = (1, 2, be * nb, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be, "cuda")
    selected = torch.tensor(
        [[[[0, 2], [1, 2], [0, 1]]] * 2],
        device="cuda",
        dtype=torch.int32,
    ).reshape(1, 2, nb, 2)

    actual = triton_sparse_attention(
        q,
        k,
        v,
        selected,
        vbs,
        block_size=(1, 1, be),
    )
    assert actual.shape == q.shape
    assert torch.isfinite(actual).all()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
@pytest.mark.skipif(not HAS_CUTE, reason="FA4 CuTe is not installed")
def test_cute_triton_256_matches_owned_triton_forward_and_backward():
    torch.manual_seed(37)
    be, nb, dim = 256, 2, 128
    shape = (1, 1, be * nb, dim)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    grad_output = torch.randn_like(inputs[0])
    vbs = torch.tensor([256, 173], device="cuda")
    triton_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]
    hybrid_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]

    expected = triton_vsa(
        *triton_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )
    actual = cute_triton_vsa(
        *hybrid_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )
    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for actual_input, expected_input in zip(
        hybrid_inputs,
        triton_inputs,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            atol=2e-2,
            rtol=2e-2,
        )
