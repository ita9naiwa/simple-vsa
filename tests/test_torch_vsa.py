import math

import pytest
import torch
import torch.nn.functional as F

from vsa import torch_vsa


def _naive_selected_attention(q, k, v, selected, block_size, causal=False):
    batch, heads, query_tokens, head_dim = q.shape
    output = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_dim)

    for b in range(batch):
        for h in range(heads):
            for query_token in range(query_tokens):
                query_block = query_token // block_size
                key_ids = []
                for key_block in selected[b, h, query_block].tolist():
                    key_ids.extend(range(key_block * block_size, (key_block + 1) * block_size))
                if causal:
                    key_ids = [index for index in key_ids if index <= query_token]
                key_index = torch.tensor(key_ids, device=q.device)
                scores = (q[b, h, query_token].float() @ k[b, h, key_index].float().T) * scale
                probabilities = torch.softmax(scores, dim=-1).to(v.dtype)
                output[b, h, query_token] = probabilities @ v[b, h, key_index]
    return output


@pytest.mark.parametrize("causal", [False, True])
def test_torch_vsa_matches_naive_selected_attention(causal):
    torch.manual_seed(7)
    q = torch.randn(2, 2, 16, 8)
    k = torch.randn(2, 2, 16, 8)
    v = torch.randn(2, 2, 16, 8)

    actual, selected = torch_vsa(
        q, k, v, block_size=4, topk=2, causal=causal, return_indices=True
    )
    expected = _naive_selected_attention(q, k, v, selected, 4, causal)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_torch_vsa_backpropagates_through_fine_attention():
    torch.manual_seed(11)
    q = torch.randn(1, 1, 8, 4, requires_grad=True)
    k = torch.randn(1, 1, 8, 4, requires_grad=True)
    v = torch.randn(1, 1, 8, 4, requires_grad=True)

    torch_vsa(q, k, v, block_size=4, topk=1).square().mean().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.parametrize("causal", [False, True])
def test_selecting_every_block_matches_dense_attention(causal):
    torch.manual_seed(19)
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    sparse = torch_vsa(q, k, v, block_size=4, topk=4, causal=causal)
    dense = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(sparse, dense, atol=1e-5, rtol=1e-5)


def test_rejects_non_divisible_sequence_length():
    q = k = v = torch.randn(1, 1, 7, 8)
    with pytest.raises(ValueError, match="divisible"):
        torch_vsa(q, k, v, block_size=4, topk=1)


def test_rejects_mixed_dtypes():
    q = torch.randn(1, 1, 8, 8, dtype=torch.float32)
    k = v = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="same dtype"):
        torch_vsa(q, k, v, block_size=4, topk=1)
