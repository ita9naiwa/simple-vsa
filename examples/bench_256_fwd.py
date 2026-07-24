"""Forward-only speed of the VSA-256 sparse branch: CuTe 256x128 vs route-A
Triton 256->64.

Both backends consume the SAME logical 256-block top-k map; only the sparse
kernel differs:
  - FASTVIDEO_VSA_TRITON=1   -> route-A: each logical 256x256 edge expands to a
                               dense 4x4 = 16 physical 64x64 tiles, run on the
                               64-token Triton block-sparse kernel.
  - FASTVIDEO_VSA_CUTEDSL=1  -> FA4 CuTe: logical Q256 routing with physical
                               KV128 blocks (forward only, Blackwell sm_100+).
  - repo Triton              -> autotunes physical Q64/Q128/Q256 x KV64.

The backend is resolved from the env var at call time, so we flip it in-process
between timing loops. bf16, CUDA, batch=1. Times are ms/iter (fwd only).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u examples/bench_256_fwd.py
"""

import argparse
import math
import os
import time

import torch

from fastvideo_kernel.block_sparse_attn_256 import block_sparse_attn_256
from vsa.triton_impl import _triton_sparse_attention_forward

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
        print(exc)
        msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else ""
        return float("nan"), (type(exc).__name__ + (f": {msg}" if msg else ""))


def _set_backend(name):
    for k in ("FASTVIDEO_VSA_TRITON", "FASTVIDEO_VSA_CUTEDSL",
              "FASTVIDEO_KERNEL_VSA_FORCE_TRITON"):
        os.environ.pop(k, None)
    if name == "triton":
        os.environ["FASTVIDEO_VSA_TRITON"] = "1"
    elif name == "cutedsl":
        os.environ["FASTVIDEO_VSA_CUTEDSL"] = "1"


def _topk_map(heads, nb, topk, device):
    """Bool [1, heads, nb, nb]: each query block selects `topk` key blocks."""
    scores = torch.randn(1, heads, nb, nb, device=device)
    idx = scores.topk(topk, dim=-1).indices
    m = torch.zeros(1, heads, nb, nb, dtype=torch.bool, device=device)
    m.scatter_(-1, idx, True)
    return idx.sort(dim=-1).values.to(torch.int32).contiguous(), m


def run(name, nb, heads, dim, sparsity, iters=30, warmup=15):
    seq = BE * nb
    topk = max(1, min(nb, math.ceil((1 - sparsity) * nb)))
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    vbs = torch.full((nb,), BE, dtype=torch.int32, device="cuda")
    selected, block_map = _topk_map(heads, nb, topk, "cuda")

    print(f"\n### {name}", flush=True)
    print(
        f"    sequence length   = {seq:,} tokens ({nb} blocks x {BE} tokens per block)\n"
        f"    attention heads   = {heads}   |   head dimension = {dim}   |   dtype = bfloat16\n"
        f"    selected top-k    = {topk} of {nb} key blocks   |   "
        f"sparsity = {sparsity:.2%}",
        flush=True,
    )
    print(f"    {'backend / tiling':<34}{'forward':>16}", flush=True)

    outs = {}
    for label, backend in [
        ("route-A Triton 256->64", "triton"),
        #("CuTe FA4 logical256/KV128", "cutedsl"),
    ]:
        _set_backend(backend)

        def thunk():
            with torch.no_grad():
                outs[backend] = block_sparse_attn_256(q, k, v, block_map, vbs)[0]

        t, note = _time(thunk, iters, warmup)
        cell = f"{t:.3f} ms" if t == t else (note or "n/a")
        print(f"    {label:<34}{cell:>16}", flush=True)

    def triton_thunk():
        with torch.no_grad():
            outs["repo_triton"] = _triton_sparse_attention_forward(
                q,
                k,
                v,
                selected,
                vbs,
                block_size=BE,
                sm_scale=dim**-0.5,
                save_lse=False,
            )[0]

    t, note = _time(triton_thunk, iters, warmup)
    cell = f"{t:.3f} ms" if t == t else (note or "n/a")
    print(f"    {'repo Triton autotuned':<34}{cell:>16}", flush=True)

    if "cutedsl" in outs and outs["cutedsl"] is not None:
        for backend in ("triton",):
            if backend not in outs or outs[backend] is None:
                continue
            d = (
                outs[backend].float()
                - outs["cutedsl"].float()
            ).abs()
            print(
                f"    max|{backend}-cute| = {d.max().item():.3e}   "
                f"mean = {d.mean().item():.3e}",
                flush=True,
            )

    del q, k, v, selected, block_map, vbs, outs
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        choices=("all", "256px", "480p", "720p"),
        default="all",
        help="Run one shape to bound Triton autotuning time.",
    )
    args = parser.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    cases = {
        "256px": ("Wan2.1  ~256px  (16k, 87.5% sparse)", 64),
        "480p": ("Wan2.1  480p 81f  (40k, 87.5% sparse)", 156),
        "720p": ("Wan2.1  720p 81f  (92k, 87.5% sparse)", 360),
    }
    selected_cases = (
        cases.items()
        if args.shape == "all"
        else [(args.shape, cases[args.shape])]
    )
    for _, (name, nb) in selected_cases:
        run(
            name,
            nb=nb,
            heads=12,
            dim=128,
            sparsity=0.875,
        )
