# Simple VSA: PyTorch + Triton

This repository is a small, readable implementation of Video Sparse Attention
(VSA). It follows the exact math and API of
[`fastvideo_kernel.ops.video_sparse_attn`](https://github.com/hao-ai-lab/FastVideo),
in three forms:

- `video_sparse_attn`: faithful PyTorch reference (masked-fill sparse branch).
- `torch_vsa`: same math, but the sparse branch *gathers* only the selected
  blocks (real sparse work); CPU/CUDA, fully differentiable.
- `triton_vsa`: CUDA forward kernel for the sparse branch with online softmax.

Each token stream is split into padded blocks (a 1D block stands in for a 3D
spatiotemporal tile). The two branches are summed per token:

1. **Compression branch** — mean-pool every block, run dense block-vs-block
   attention, broadcast each query block's output back to its tokens.
2. **Sparse branch** — the same block-vs-block `scores` pick the Top-K key blocks
   per query block; full token attention runs only inside them.

`out = out_c * compress_attn_weight + out_s`. The Top-K choice is discrete (no
gradient), but the shared compression `scores` route gradient into the pooled
blocks, so the selector trains end-to-end — exactly as VSA/NSA do.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
python examples/demo.py
```

On a Linux CUDA machine, install the optional Triton dependency:

```bash
pip install -e '.[gpu,test]'
pytest -q tests/test_triton_vsa.py
```

## API

All three functions share the FastVideo signature and accept
`[batch, heads, seq_len, head_dim]` tensors:

```python
from vsa import torch_vsa, triton_vsa

# valid token count per kv / query block (blocks are padded to block_size volume)
variable_block_sizes = torch.full((num_blocks,), block_elements, dtype=torch.long)

out = torch_vsa(
    q, k, v,
    variable_block_sizes,      # kv blocks
    variable_block_sizes,      # query blocks
    topk=2,
    block_size=(1, 1, block_elements),   # tile shape; only its volume matters
    compress_attn_weight=None,           # optional gate on the compression branch
)

with torch.no_grad():
    out_gpu = triton_vsa(q.cuda(), k.cuda(), v.cuda(),
                         variable_block_sizes.cuda(), variable_block_sizes.cuda(),
                         topk=2, block_size=(1, 1, block_elements))
```

`seq_len` must be divisible by `block_elements = prod(block_size)` and
`*_variable_block_sizes` gives the real (non-padding) token count per block. The
scale is fixed at `1/sqrt(head_dim)`. The Triton path supports autograd for
`q`, `k`, and `v` through custom backward kernels (Top-K routing remains
discrete), and supports head dimensions up to 256.

## References

- Paper: [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)
- Production code: [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)
