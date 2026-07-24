import importlib.util

import pytest
import torch

from vsa import helion_vsa, torch_vsa


HAS_HELION = importlib.util.find_spec("helion") is not None


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
def test_helion_rejects_autograd_inputs():
    be, nb = 8, 2
    q = torch.randn(
        (1, 1, be * nb, 32),
        device="cuda",
        dtype=torch.float16,
        requires_grad=True,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = _full_blocks(nb, be, "cuda")

    with pytest.raises(RuntimeError, match="forward-only"):
        helion_vsa(
            q,
            k,
            v,
            vbs,
            vbs,
            topk=1,
            block_size=(1, 1, be),
        )
