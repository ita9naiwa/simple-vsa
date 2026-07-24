"""VSA-256 fwd / fwd+bwd / bwd table across backends.

Same full VSA math (compression + sparse branch) at 256-token blocks, so all
rows are apples-to-apples. Backends for the sparse branch:
  - triton_vsa              : the readable simple-vsa Triton impl (fwd+bwd)
  - fastvideo(triton)       : FastVideo route-A 256->64 Triton fallback (fwd+bwd)
  - fastvideo CuTe FA4      : FA4 CuTe 256x128, forward-only -> backward is X

Backward is reported as (fwd+bwd - fwd). CuTe has no block-sparse backward, so
its fwd+bwd / backward cells are marked "X". bf16, CUDA, batch=1, eager.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore \
        python -u examples/bench_256_full.py
"""

import importlib.util
import math
import os
import time

import torch

from vsa import cute_triton_vsa, triton_vsa

torch.set_float32_matmul_precision("high")

try:
    from fastvideo_kernel.ops import video_sparse_attn as fvk_vsa
except Exception:  # noqa: BLE001
    fvk_vsa = None

BE = 256  # 256-token logical block


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
        return float("nan"), type(exc).__name__


def _set_fvk_backend(name):
    for k in ("FASTVIDEO_VSA_TRITON", "FASTVIDEO_VSA_CUTEDSL",
              "FASTVIDEO_KERNEL_VSA_FORCE_TRITON"):
        os.environ.pop(k, None)
    if name == "triton":
        os.environ["FASTVIDEO_VSA_TRITON"] = "1"
    elif name == "cutedsl":
        os.environ["FASTVIDEO_VSA_CUTEDSL"] = "1"


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


def run(name, nb, heads, dim, sparsity, iters=30, warmup=15):
    seq = BE * nb
    topk = max(1, min(nb, math.ceil((1 - sparsity) * nb)))
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    do = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
    vbs = torch.full((nb,), BE, dtype=torch.long, device="cuda")
    kw = dict(topk=topk, block_size=(1, 1, BE))

    # (label, callable, backend-env-or-None, has_backward)
    rows = [("triton_vsa", triton_vsa, None, True)]
    if (
        importlib.util.find_spec("flash_attn") is not None
        and importlib.util.find_spec("flash_attn.cute") is not None
    ):
        rows.append(
            ("cute+triton_vsa", cute_triton_vsa, None, True)
        )
    if fvk_vsa is not None:
        rows.append(("fastvideo(triton) route-A 256->64", fvk_vsa, "triton", True))
        rows.append(("fastvideo CuTe FA4 256x128", fvk_vsa, "cutedsl", False))

    print(f"\n### {name}", flush=True)
    print(
        f"    sequence length   = {seq:,} tokens ({nb} blocks x {BE} tokens per block)\n"
        f"    attention heads   = {heads}   |   head dimension = {dim}   |   dtype = bfloat16\n"
        f"    selected top-k    = {topk} of {nb} key blocks   |   sparsity = {sparsity:.2%}",
        flush=True,
    )
    print(f"    {'implementation / backend':<38}{'forward':>14}"
          f"{'forward+backward':>20}{'backward':>14}", flush=True)

    def cell(t):
        return f"{t:.3f} ms" if t == t else "err"

    for label, fn, backend, has_bwd in rows:
        if backend is not None:
            _set_fvk_backend(backend)
        t_f, _ = bench_fwd(fn, q, k, v, vbs, kw, iters, warmup)
        if has_bwd:
            t_fb, _ = bench_fwd_bwd(fn, q, k, v, vbs, kw, do, iters, warmup)
            t_b = (t_fb - t_f) if (t_f == t_f and t_fb == t_fb) else float("nan")
            fb_s, b_s = cell(t_fb), cell(t_b)
        else:
            fb_s, b_s = "N/A", "X (fwd-only)"
        print(f"    {label:<38}{cell(t_f):>14}{fb_s:>20}{b_s:>14}", flush=True)

    for t in (q, k, v, do):
        del t
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    run("Wan2.1  ~256px  (16k, 87.5% sparse)", nb=64, heads=12, dim=128, sparsity=0.875)
    run("Wan2.1  480p 81f  (40k, 87.5% sparse)", nb=156, heads=12, dim=128, sparsity=0.875)
    run("Wan2.1  720p 81f  (92k, 87.5% sparse)", nb=360, heads=12, dim=128, sparsity=0.875)
