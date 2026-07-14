"""Video-realistic VSA speed comparison.

Real FastVideo VSA kernel vs this repo's torch_vsa and triton_vsa
(each eager + a light torch.compile), at Wan2.1-scale sequence lengths and
paper-realistic sparsity.

VSA config (from fastvideo/attention/backends/video_sparse_attn.py):
- tile = (4,4,4) -> block_elements = 64
- topk = ceil((1 - sparsity) * num_blocks); paper uses ~0.875..0.9375 sparsity
- head_dim 64 or 128 (Wan2.1 uses 128)

Shapes are derived from Wan2.1 latents (VAE 8x spatial / 4x temporal, patch (1,2,2)),
padded up to whole (4,4,4) tiles. Forward-only, bf16, CUDA.

    python examples/bench_video.py
"""

import math
import time

import torch

from fastvideo_kernel.ops import video_sparse_attn as real_vsa
from vsa import torch_vsa, triton_vsa

# Light compile: default inductor mode (fuses, no autotune-scratch OOM).
torch_compiled = torch.compile(torch_vsa, dynamic=False)
triton_compiled = torch.compile(triton_vsa, dynamic=False)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def bench(fn, *args, iters=10, warmup=5, **kwargs):
    """Return (milliseconds, note). note is "" on success, else a short reason."""
    try:
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3, ""
    except Exception as exc:  # noqa: BLE001 - keep the sweep going on any single failure
        torch.cuda.empty_cache()
        if _is_oom(exc):
            return float("nan"), "OOM"
        return float("nan"), f"{type(exc).__name__}"


def run(name, nb, heads, dim, sparsity, be=64, iters=10):
    seq = be * nb
    topk = max(1, min(nb, math.ceil((1 - sparsity) * nb)))
    torch.manual_seed(0)
    try:
        q = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
    except torch.cuda.OutOfMemoryError:
        print(f"\n### {name}: OOM allocating q/k/v", flush=True)
        return
    vbs = torch.full((nb,), be, dtype=torch.long, device="cuda")
    kw = dict(topk=topk, block_size=(1, 1, be))

    print(f"\n### {name}", flush=True)
    print(f"    seq={seq}  blocks={nb}  heads={heads}  dim={dim}  "
          f"sparsity={sparsity:.4f}  topk={topk}", flush=True)

    with torch.no_grad():
        rows = [
            ("fastvideo_kernel  ", *bench(real_vsa, q, k, v, vbs, vbs, iters=iters, **kw)),
            ("torch_vsa (eager) ", *bench(torch_vsa, q, k, v, vbs, vbs, iters=iters, **kw)),
            ("torch_vsa (comp)  ", *bench(torch_compiled, q, k, v, vbs, vbs, iters=iters, **kw)),
            ("triton_vsa (eager)", *bench(triton_vsa, q, k, v, vbs, vbs, iters=iters, **kw)),
            ("triton_vsa (comp) ", *bench(triton_compiled, q, k, v, vbs, vbs, iters=iters, **kw)),
        ]
    t_real = rows[0][1]
    for label, t, note in rows:
        if t == t:
            s = f"{t:9.3f} ms"
            rel = f" ({t / t_real:5.1f}x)" if t_real == t_real else ""
        else:
            s = f"  {note:>8}  "
            rel = ""
        print(f"    {label} : {s}{rel}", flush=True)

    del q, k, v, vbs
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    # heads/dim = Wan2.1-1.3B attention (12 heads x 128).
    run("Wan2.1  ~256px  (warmup / small)", nb=256, heads=12, dim=128, sparsity=0.875)
    run("Wan2.1  480p 81f  (1.3B, 87.5% sparse)", nb=624, heads=12, dim=128, sparsity=0.875)
    run("Wan2.1  480p 81f  (1.3B, 93.75% sparse)", nb=624, heads=12, dim=128, sparsity=0.9375)
    run("Wan2.1  720p 81f  (720p grid, 87.5% sparse)", nb=1440, heads=12, dim=128, sparsity=0.875)
