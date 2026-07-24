import importlib.util

import pytest
import torch

from vsa import helion_vsa, torch_vsa


HAS_HELION = importlib.util.find_spec("helion") is not None

if HAS_HELION:
    from vsa.helion_impl import _helion_sparse_attention


def _full_blocks(num_blocks, block_elements, device):
    return torch.full(
        (num_blocks,),
        block_elements,
        dtype=torch.long,
        device=device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_HELION, reason="Helion is not installed")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_helion_matches_torch_full_blocks(dtype):
    torch.manual_seed(41)
    be, nb = 8, 4
    shape = (1, 2, be * nb, 64)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be, "cuda")

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
        actual = helion_vsa(
            q,
            k,
            v,
            vbs,
            vbs,
            topk=2,
            block_size=(1, 1, be),
        )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_HELION, reason="Helion is not installed")
def test_helion_matches_torch_with_padding_and_gate():
    torch.manual_seed(43)
    be, nb = 8, 3
    shape = (1, 2, be * nb, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    gate = torch.randn_like(q)
    vbs = torch.tensor([8, 5, 3], device="cuda")

    with torch.no_grad():
        expected = torch_vsa(
            q,
            k,
            v,
            vbs,
            vbs,
            topk=2,
            block_size=(1, 1, be),
            compress_attn_weight=gate,
        )
        actual = helion_vsa(
            q,
            k,
            v,
            vbs,
            vbs,
            topk=2,
            block_size=(1, 1, be),
            compress_attn_weight=gate,
        )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_HELION, reason="Helion is not installed")
def test_helion_256_forward_matches_torch_with_padding():
    torch.manual_seed(45)
    be, nb, dim = 256, 3, 128
    shape = (1, 2, be * nb, dim)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = torch.tensor([256, 193, 71], device="cuda")

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
        actual = helion_vsa(
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
@pytest.mark.skipif(not HAS_HELION, reason="Helion is not installed")
def test_helion_backward_matches_torch_with_padding_and_gate():
    torch.manual_seed(47)
    be, nb, dim = 8, 3, 64
    shape = (1, 2, be * nb, dim)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    vbs = torch.tensor([8, 5, 3], device="cuda")

    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    helion_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    reference_gate = gate.detach().clone().requires_grad_()
    helion_gate = gate.detach().clone().requires_grad_()

    expected = torch_vsa(
        *reference_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
        compress_attn_weight=reference_gate,
    )
    actual = helion_vsa(
        *helion_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
        compress_attn_weight=helion_gate,
    )

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    for actual_input, expected_input in zip(
        helion_inputs,
        reference_inputs,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            atol=6e-2,
            rtol=6e-2,
        )
    torch.testing.assert_close(
        helion_gate.grad,
        reference_gate.grad,
        atol=4e-2,
        rtol=4e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_HELION, reason="Helion is not installed")
def test_helion_64_split_backward_matches_torch_with_padding():
    torch.manual_seed(48)
    be, nb, dim = 64, 3, 128
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
    vbs = torch.tensor([64, 43, 19], device="cuda")

    reference_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]
    helion_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]

    expected = torch_vsa(
        *reference_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )
    actual = helion_vsa(
        *helion_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    for actual_input, expected_input in zip(
        helion_inputs,
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
@pytest.mark.skipif(not HAS_HELION, reason="Helion is not installed")
def test_helion_256_backward_matches_torch_with_padding():
    torch.manual_seed(49)
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
    helion_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]

    expected = torch_vsa(
        *reference_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )
    actual = helion_vsa(
        *helion_inputs,
        vbs,
        vbs,
        topk=2,
        block_size=(1, 1, be),
    )

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    for actual_input, expected_input in zip(
        helion_inputs,
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
@pytest.mark.skipif(not HAS_HELION, reason="Helion is not installed")
def test_helion_256_backward_accumulates_shared_kv_without_atomics():
    """Every query block routes to KV block zero, stressing KV ownership."""

    torch.manual_seed(51)
    be, nb, heads, dim = 256, 3, 2, 128
    shape = (1, heads, be * nb, dim)
    inputs = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    grad_output = torch.randn(
        shape,
        device="cuda",
        dtype=torch.bfloat16,
    )
    selected = torch.zeros(
        [1, heads, nb, 1],
        device="cuda",
        dtype=torch.int32,
    )
    vbs = torch.tensor([256, 173, 91], device="cuda")

    reference_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]
    helion_inputs = [
        x.detach().clone().requires_grad_() for x in inputs
    ]

    q_ref, k_ref, v_ref = reference_inputs
    key = k_ref[:, :, :be, :].float()
    value = v_ref[:, :, :be, :].float()
    reference_blocks = []
    scale = dim**-0.5
    for query_block in range(nb):
        query = q_ref[
            :,
            :,
            query_block * be : (query_block + 1) * be,
            :,
        ].float()
        probabilities = torch.softmax(
            torch.matmul(query, key.transpose(-2, -1)) * scale,
            dim=-1,
        )
        reference_blocks.append(torch.matmul(probabilities, value))
    expected = torch.cat(reference_blocks, dim=-2).to(torch.bfloat16)

    actual = _helion_sparse_attention(
        *helion_inputs,
        selected,
        vbs,
        be,
    )

    expected.backward(grad_output)
    actual.backward(grad_output)

    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)
    for actual_input, expected_input in zip(
        helion_inputs,
        reference_inputs,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            atol=8e-2,
            rtol=8e-2,
        )
