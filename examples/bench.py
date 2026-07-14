"""Speed comparison: video_sparse_attn (fastvideo ref) vs torch_vsa.

Forward-only, fp16, CUDA. Run with:

    python examples/bench.py
"""

import time

import torch

from vsa import torch_vsa, video_sparse_attn


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
    q = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = torch.full((nb,), be, dtype=torch.long, device="cuda")
    kw = dict(topk=topk, block_size=(1, 1, be))

    print(f"\n=== seq={seq} (nb={nb}, be={be}) heads={heads} dim={dim} topk={topk} ===", flush=True)
    with torch.no_grad():
        ref = video_sparse_attn(q, k, v, vbs, vbs, **kw)
        tor = torch_vsa(q, k, v, vbs, vbs, **kw)
        print(f"  max|torch-ref| = {(tor - ref).abs().max().item():.4f}", flush=True)

        try:
            t_ref = bench(video_sparse_attn, q, k, v, vbs, vbs, iters=iters, **kw)
        except torch.cuda.OutOfMemoryError:
            t_ref = float("nan")
            torch.cuda.empty_cache()
        t_tor = bench(torch_vsa, q, k, v, vbs, vbs, iters=iters, **kw)

    def fmt(t):
        return f"{t:8.3f} ms" if t == t else "     OOM"

    print(f"  fastvideo_ref : {fmt(t_ref)}", flush=True)
    line = f"  torch_vsa     : {fmt(t_tor)}"
    if t_ref == t_ref:
        line += f"   ({t_ref / t_tor:5.1f}x faster than ref)"
    print(line, flush=True)


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    # be=64 mirrors FastVideo's 4x4x4 tile volume.
    run(be=64, nb=16, heads=8, dim=64, topk=4)
    run(be=64, nb=64, heads=8, dim=64, topk=8)
    run(be=64, nb=128, heads=8, dim=64, topk=8)
    run(be=64, nb=256, heads=8, dim=64, topk=16)
