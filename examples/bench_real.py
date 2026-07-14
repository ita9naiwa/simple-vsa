"""Speed comparison: real FastVideo VSA kernel vs this repo's torch_vsa.

Real op: ``fastvideo_kernel.ops.video_sparse_attn`` (fused Triton compression +
TK/CuTe block-sparse kernel). Compared against the readable ``torch_vsa``.
Forward-only, bf16 (the sm90 TK kernel requires it), CUDA. Run with:

    python examples/bench_real.py
"""

import time

import torch

from fastvideo_kernel.ops import video_sparse_attn as real_vsa
from vsa import torch_vsa


def bench(fn, *args, iters=50, warmup=10, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def run(be, nb, heads, dim, topk, iters=50):
    seq = be * nb
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = torch.full((nb,), be, dtype=torch.long, device="cuda")
    kw = dict(topk=topk, block_size=(1, 1, be))

    print(f"\n=== seq={seq} (nb={nb}, be={be}) heads={heads} dim={dim} topk={topk} ===", flush=True)
    with torch.no_grad():
        ref = real_vsa(q, k, v, vbs, vbs, **kw)
        tor = torch_vsa(q, k, v, vbs, vbs, **kw)
        print(f"  max|torch-real| = {(tor - ref).abs().max().item():.4f}", flush=True)

        t_real = bench(real_vsa, q, k, v, vbs, vbs, iters=iters, **kw)
        t_tor = bench(torch_vsa, q, k, v, vbs, vbs, iters=iters, **kw)

    print(f"  fastvideo_kernel : {t_real:8.3f} ms", flush=True)
    print(f"  torch_vsa        : {t_tor:8.3f} ms   ({t_tor / t_real:5.1f}x slower than kernel)", flush=True)


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    # be=64 mirrors FastVideo's 4x4x4 tile volume (the TK/Triton 64-block path).
    run(be=64, nb=16, heads=8, dim=64, topk=4)
    run(be=64, nb=64, heads=8, dim=64, topk=8)
    run(be=64, nb=128, heads=8, dim=64, topk=8)
    run(be=64, nb=256, heads=8, dim=64, topk=16)
