"""Profile VSA components and routing skew on CUDA.

This benchmark separates the shared coarse path, route inversion, sparse
executor forward/backward, and full VSA. It also compares three route-degree
distributions with identical tensor shapes:

* uniform: independent random Top-K per query block;
* local: contiguous cyclic neighbors;
* hotset: every query selects the same Top-K KV blocks.

Example:

    PYTHONWARNINGS=ignore python -u examples/bench_components.py \
        --shape 720p --block 256 --routes all
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import time

import torch

from vsa import (
    helion_sparse_attention,
    helion_vsa,
    triton_sparse_attention,
    triton_vsa,
)
from vsa.common import coarse_branch, coarse_branch_compact
from vsa.fused_common import fused_block_mean
from vsa.triton_impl import _invert_indices


def _time(thunk, iters, warmup):
    for _ in range(warmup):
        thunk()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        thunk()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def _peak_mib(thunk):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    thunk()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 2**20


def _routes(kind, heads, blocks, topk, device):
    if kind == "uniform":
        scores = torch.randn(1, heads, blocks, blocks, device=device)
        return scores.topk(topk, dim=-1, sorted=False).indices.to(torch.int32)
    query = torch.arange(blocks, device=device)[:, None]
    slots = torch.arange(topk, device=device)[None, :]
    if kind == "local":
        selected = (query + slots) % blocks
    elif kind == "hotset":
        selected = slots.expand(blocks, -1)
    else:
        raise ValueError(f"unknown route kind: {kind}")
    return selected[None, None].expand(1, heads, -1, -1).to(torch.int32).contiguous()


def run(shape_name, blocks, block_elements, route_kinds, iters, warmup):
    heads, dim = 12, 128
    seq = blocks * block_elements
    topk = max(1, math.ceil(blocks * 0.125))
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn(
        1, heads, seq, dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    grad = torch.randn_like(q)
    vbs = torch.full(
        (blocks,),
        block_elements,
        device=device,
        dtype=torch.int32,
    )
    scale = dim**-0.5

    print(
        f"\n### {shape_name}: seq={seq:,}, block={block_elements}, "
        f"blocks={blocks}, topk={topk}",
        flush=True,
    )
    print(f"{'component':<38}{'time':>13}{'peak alloc':>15}", flush=True)

    def eager_coarse():
        with torch.no_grad():
            scores, _ = coarse_branch(
                q,
                k,
                v,
                vbs,
                vbs,
                block_elements,
                scale,
            )
            scores.topk(topk, dim=-1, sorted=False)

    print(
        f"{'shared eager coarse + topk':<38}"
        f"{_time(eager_coarse, iters, warmup):>10.3f} ms"
        f"{_peak_mib(eager_coarse):>12.1f} MiB",
        flush=True,
    )

    def fused_coarse():
        with torch.no_grad():
            scores, _ = coarse_branch_compact(
                q,
                k,
                v,
                vbs,
                vbs,
                block_elements,
                scale,
                block_mean_fn=fused_block_mean,
            )
            scores.topk(topk, dim=-1, sorted=False)

    print(
        f"{'shared fused coarse + topk':<38}"
        f"{_time(fused_coarse, iters, warmup):>10.3f} ms"
        f"{_peak_mib(fused_coarse):>12.1f} MiB",
        flush=True,
    )

    has_helion = importlib.util.find_spec("helion") is not None
    for route_kind in route_kinds:
        selected = _routes(
            route_kind,
            heads,
            blocks,
            topk,
            device,
        )
        degree = torch.bincount(
            selected[0, 0].reshape(-1).to(torch.int64),
            minlength=blocks,
        ).float()
        print(
            f"    route={route_kind:<8} degree mean={degree.mean().item():.1f} "
            f"max={degree.max().item():.0f} "
            f"max/mean={degree.max().item() / degree.mean().item():.2f}",
            flush=True,
        )

        def inverse():
            _invert_indices(selected, blocks)

        print(
            f"{('inverse metadata [' + route_kind + ']'):<38}"
            f"{_time(inverse, iters, warmup):>10.3f} ms"
            f"{_peak_mib(inverse):>12.1f} MiB",
            flush=True,
        )

        backends = [
            ("triton", triton_sparse_attention),
        ]
        if has_helion:
            backends.append(("helion", helion_sparse_attention))

        for backend_name, executor in backends:
            def fine_fwd():
                with torch.no_grad():
                    executor(
                        q,
                        k,
                        v,
                        selected,
                        vbs,
                        block_size=(1, 1, block_elements),
                    )

            fq = q.detach().requires_grad_()
            fk = k.detach().requires_grad_()
            fv = v.detach().requires_grad_()

            def fine_fb():
                fq.grad = fk.grad = fv.grad = None
                executor(
                    fq,
                    fk,
                    fv,
                    selected,
                    vbs,
                    block_size=(1, 1, block_elements),
                ).backward(grad)

            fwd_ms = _time(fine_fwd, iters, warmup)
            fb_ms = _time(fine_fb, iters, warmup)
            label = f"{backend_name} fine fwd [{route_kind}]"
            print(
                f"{label:<38}{fwd_ms:>10.3f} ms"
                f"{_peak_mib(fine_fwd):>12.1f} MiB",
                flush=True,
            )
            label = f"{backend_name} fine fwd+bwd [{route_kind}]"
            print(
                f"{label:<38}{fb_ms:>10.3f} ms"
                f"{_peak_mib(fine_fb):>12.1f} MiB",
                flush=True,
            )

    full_backends = [("triton full", triton_vsa)]
    if has_helion:
        full_backends.append(("helion full", helion_vsa))
    for label, fn in full_backends:
        fq = q.detach().requires_grad_()
        fk = k.detach().requires_grad_()
        fv = v.detach().requires_grad_()

        def full_fb():
            fq.grad = fk.grad = fv.grad = None
            fn(
                fq,
                fk,
                fv,
                vbs,
                vbs,
                topk=topk,
                block_size=(1, 1, block_elements),
            ).backward(grad)

        print(
            f"{label:<38}{_time(full_fb, iters, warmup):>10.3f} ms"
            f"{_peak_mib(full_fb):>12.1f} MiB",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=("256px", "480p", "720p"), default="720p")
    parser.add_argument("--block", choices=(64, 256), type=int, default=256)
    parser.add_argument(
        "--routes",
        choices=("all", "uniform", "local", "hotset"),
        default="all",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    seqs = {"256px": 16_384, "480p": 39_936, "720p": 92_160}
    seq = seqs[args.shape]
    if seq % args.block:
        raise ValueError(f"sequence length {seq} is not divisible by block {args.block}")
    route_kinds = (
        ("uniform", "local", "hotset")
        if args.routes == "all"
        else (args.routes,)
    )
    print(
        f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}",
        flush=True,
    )
    run(
        args.shape,
        seq // args.block,
        args.block,
        route_kinds,
        args.iters,
        args.warmup,
    )
