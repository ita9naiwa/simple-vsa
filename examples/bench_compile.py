"""Speed comparison: real FastVideo VSA kernel vs torch_vsa (eager) vs
torch_vsa compiled with torch.compile(mode="max-autotune").

Forward-only, bf16, CUDA. Run with:

    python examples/bench_compile.py
"""

import time

import torch

from fastvideo_kernel.ops import video_sparse_attn as real_vsa
from vsa import torch_vsa

# One compiled callable, reused across shapes (dynamic=False -> recompiles per shape,
# which is what we want for steady-state timing on each config).
compiled_vsa = torch.compile(torch_vsa, mode="max-autotune", dynamic=False)


def bench(fn, *args, iters=50, warmup=15, **kwargs):
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
        eager = torch_vsa(q, k, v, vbs, vbs, **kw)
        comp = compiled_vsa(q, k, v, vbs, vbs, **kw)
        print(f"  max|eager-real| = {(eager - ref).abs().max().item():.4f}   "
              f"max|compiled-eager| = {(comp - eager).abs().max().item():.4f}", flush=True)

        t_real = bench(real_vsa, q, k, v, vbs, vbs, iters=iters, **kw)
        t_eager = bench(torch_vsa, q, k, v, vbs, vbs, iters=iters, **kw)
        t_comp = bench(compiled_vsa, q, k, v, vbs, vbs, iters=iters, **kw)

    print(f"  fastvideo_kernel   : {t_real:8.3f} ms", flush=True)
    print(f"  torch_vsa (eager)  : {t_eager:8.3f} ms   ({t_eager / t_real:5.1f}x vs kernel)", flush=True)
    print(f"  torch_vsa (compile): {t_comp:8.3f} ms   ({t_comp / t_real:5.1f}x vs kernel, "
          f"{t_eager / t_comp:4.2f}x vs eager)", flush=True)


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    run(be=64, nb=16, heads=8, dim=64, topk=4)
    run(be=64, nb=64, heads=8, dim=64, topk=8)
    run(be=64, nb=128, heads=8, dim=64, topk=8)
    run(be=64, nb=256, heads=8, dim=64, topk=16)
