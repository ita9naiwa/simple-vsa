import importlib.util

import pytest
import torch

from vsa import torch_vsa, triton_vsa


HAS_TRITON = importlib.util.find_spec("triton") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is not installed")
@pytest.mark.parametrize("causal", [False, True])
def test_triton_matches_torch(causal):
    torch.manual_seed(17)
    q = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with torch.no_grad():
        expected, torch_indices = torch_vsa(
            q, k, v, block_size=8, topk=2, causal=causal, return_indices=True
        )
        actual, triton_indices = triton_vsa(
            q, k, v, block_size=8, topk=2, causal=causal, return_indices=True
        )

    torch.testing.assert_close(triton_indices, torch_indices)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

