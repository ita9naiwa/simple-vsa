"""Simple Video Sparse Attention implementations."""

from collections.abc import Callable
from typing import Any

from .common import CompressionGate, block_mean, topk_block_mask, video_sparse_attn
from .fastvideo_tiling import build_vsa_metadata, tile, untile
from .torch_impl import torch_vsa

__all__ = [
    "torch_vsa",
    "triton_vsa",
    "video_sparse_attn",
    "CompressionGate",
    "block_mean",
    "topk_block_mask",
    "build_vsa_metadata",
    "tile",
    "untile",
]


def triton_vsa(*args: Any, **kwargs: Any) -> Any:
    """Load the optional Triton implementation only when it is requested."""
    from .triton_impl import triton_vsa as _triton_vsa

    implementation: Callable[..., Any] = _triton_vsa
    return implementation(*args, **kwargs)

