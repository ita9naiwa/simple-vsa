# Simple VSA: PyTorch + Triton

This repository is a small, readable implementation of the core Video Sparse
Attention (VSA) idea. It contains the same algorithm in two forms:

- `torch_vsa`: reference implementation, CPU/CUDA, autograd through fine attention.
- `triton_vsa`: CUDA forward kernel with online softmax.

The data flow is:

1. Split Q and K into contiguous token blocks.
2. Mean-pool each block.
3. Score every query block against every key block at low resolution.
4. Keep the Top-K key blocks for each query block.
5. Run full token attention only inside those selected blocks.

This is intentionally a teaching implementation, not a drop-in replacement for
the complete VSA training stack. The original method adds a trainable selector,
coarse/fine fusion, optimized backward kernels, and video-specific 3D cube
layout. Here, a 1D token block stands in for a spatiotemporal cube so the sparse
attention mechanism stays visible.

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

Both implementations accept `[batch, heads, tokens, head_dim]` tensors:

```python
from vsa import torch_vsa, triton_vsa

out = torch_vsa(q, k, v, block_size=16, topk=2, causal=False)

with torch.no_grad():
    out_gpu = triton_vsa(q.cuda(), k.cuda(), v.cuda(), block_size=16, topk=2)
```

Sequence lengths must be divisible by `block_size`. The Triton path is
forward-only and supports head dimensions up to 256.

## References

- Paper: [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389)
- Production code: [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)
