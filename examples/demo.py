import torch

from vsa import torch_vsa

torch.manual_seed(0)

# VSA works on padded, tile-contiguous tokens. Here we use 4 full blocks of 8
# tokens (a 1D stand-in for a 3D tile); variable_block_sizes records the real
# token count per block (all full = 8 here).
block_elements = 8
num_blocks = 4
seq_len = block_elements * num_blocks

q = torch.randn(1, 2, seq_len, 64)
k = torch.randn_like(q)
v = torch.randn_like(q)
variable_block_sizes = torch.full((num_blocks,), block_elements, dtype=torch.long)

output = torch_vsa(
    q,
    k,
    v,
    variable_block_sizes,
    variable_block_sizes,
    topk=2,
    block_size=(1, 1, block_elements),
)

print("output:", tuple(output.shape))
