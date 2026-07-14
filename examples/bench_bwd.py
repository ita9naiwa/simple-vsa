"""Forward + backward speed: triton_vsa vs FastVideo ThunderKittens, eager and
torch.compile, at Wan2.1-scale shapes.

Backward is reported as (fwd+bwd - fwd). torch.compile fuses the PyTorch coarse
branch (fwd and bwd); the sparse Triton/TK kernels stay as their own optimized
kernels. bf16, CUDA.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u examples/bench_bwd.py
"""

import math
import time

import torch

from vsa import triton_vsa

# Let the fp32 coarse-branch matmuls use TF32 tensor cores (the "optimal" path).
torch.set_float32_matmul_precision("high")

try:
    from fastvideo_kernel.ops import video_sparse_attn as real_vsa
except Exception:  # noqa: BLE001
    real_vsa = None


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _time(thunk, iters, warmup):
    try:
        for _ in range(warmup):
            thunk()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            thunk()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3, ""
    except Exception as exc:  # noqa: BLE001
        torch.cuda.empty_cache()
        return float("nan"), ("OOM" if _is_oom(exc) else type(exc).__name__)


def bench_fwd(fn, q, k, v, vbs, kw, iters, warmup):
    def thunk():
        with torch.no_grad():
            fn(q, k, v, vbs, vbs, **kw)
    return _time(thunk, iters, warmup)


def bench_fwd_bwd(fn, q, k, v, vbs, kw, do, iters, warmup):
    def thunk():
        for t in (q, k, v):
            t.grad = None
        fn(q, k, v, vbs, vbs, **kw).backward(do)
    return _time(thunk, iters, warmup)


def run(name, nb, heads, dim, sparsity, be=64, iters=30, warmup=15):
    seq = be * nb
    topk = max(1, min(nb, math.ceil((1 - sparsity) * nb)))
    torch.manual_seed(0)
    try:
        q = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        do = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
    except torch.cuda.OutOfMemoryError:
        print(f"\n### {name}: OOM allocating inputs", flush=True)
        return
    vbs = torch.full((nb,), be, dtype=torch.long, device="cuda")
    kw = dict(topk=topk, block_size=(1, 1, be))

    impls = [("triton_vsa", triton_vsa)]
    if real_vsa is not None:
        impls.append(("thunderkittens", real_vsa))

    print(f"\n### {name}", flush=True)
    print(
        f"    sequence length   = {seq:,} tokens ({nb} blocks x {be} tokens per block)\n"
        f"    attention heads   = {heads}   |   head dimension = {dim}   |   dtype = bfloat16\n"
        f"    selected top-k    = {topk} of {nb} key blocks   |   "
        f"sparsity = {sparsity:.2%} (each query block attends {1 - sparsity:.2%} of key blocks)",
        flush=True,
    )
    print(
        f"    {'implementation / mode':<28}{'forward':>16}"
        f"{'forward+backward':>20}{'backward':>16}",
        flush=True,
    )

    def cell(t, n):
        return f"{t:.3f} ms" if t == t else n

    for label, fn in impls:
        variants = [("eager", fn), ("compiled", torch.compile(fn))]
        for mode, f in variants:
            t_f, n_f = bench_fwd(f, q, k, v, vbs, kw, iters, warmup)
            t_fb, n_fb = bench_fwd_bwd(f, q, k, v, vbs, kw, do, iters, warmup)
            t_b = (t_fb - t_f) if (t_f == t_f and t_fb == t_fb) else float("nan")
            print(f"    {label + ' (' + mode + ')':<28}"
                  f"{cell(t_f, n_f):>16}{cell(t_fb, n_fb):>20}{cell(t_b, n_fb):>16}",
                  flush=True)

    for t in (q, k, v, do):
        del t
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    run("Wan2.1  ~256px  (16k, 87.5% sparse)", nb=256, heads=12, dim=128, sparsity=0.875)
    run("Wan2.1  480p 81f  (40k, 87.5% sparse)", nb=624, heads=12, dim=128, sparsity=0.875)
    run("Wan2.1  720p 81f  (92k, 87.5% sparse)", nb=1440, heads=12, dim=128, sparsity=0.875)
