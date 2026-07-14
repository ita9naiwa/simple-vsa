import torch

from vsa import torch_vsa


torch.manual_seed(0)
q = torch.randn(1, 2, 32, 64)
k = torch.randn_like(q)
v = torch.randn_like(q)

output, selected = torch_vsa(
    q,
    k,
    v,
    block_size=8,
    topk=2,
    return_indices=True,
)

print("output:", tuple(output.shape))
print("selected key blocks for head 0:", selected[0, 0])

