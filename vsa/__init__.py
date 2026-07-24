"""Simple Video Sparse Attention implementations."""

from collections.abc import Callable
from typing import Any

from .common import CompressionGate, block_mean, topk_block_mask, video_sparse_attn
from .fastvideo_tiling import build_vsa_metadata, tile, untile
from .torch_impl import torch_vsa

__all__ = [
    "torch_vsa",
    "triton_vsa",
    "triton_sparse_attention",
    "cute_triton_vsa",
    "cute_triton_sparse_attention",
    "helion_vsa",
    "helion_sparse_attention",
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


def triton_sparse_attention(*args: Any, **kwargs: Any) -> Any:
    """Load the route-explicit Triton sparse executor on demand."""
    from .triton_impl import triton_sparse_attention as _implementation

    return _implementation(*args, **kwargs)


def cute_triton_vsa(*args: Any, **kwargs: Any) -> Any:
    """Load the optional CuTe-forward/Triton-backward implementation."""
    from .cute_triton import cute_triton_vsa as _implementation

    return _implementation(*args, **kwargs)


def cute_triton_sparse_attention(*args: Any, **kwargs: Any) -> Any:
    """Load the route-explicit CuTe/Triton hybrid executor."""
    from .cute_triton import cute_triton_sparse_attention as _implementation

    return _implementation(*args, **kwargs)


def helion_vsa(*args: Any, **kwargs: Any) -> Any:
    """Load the optional Helion implementation on demand."""
    from .helion_impl import helion_vsa as _helion_vsa

    implementation: Callable[..., Any] = _helion_vsa
    return implementation(*args, **kwargs)


def helion_sparse_attention(*args: Any, **kwargs: Any) -> Any:
    """Load the route-explicit Helion sparse executor on demand."""
    from .helion_impl import helion_sparse_attention as _implementation

    return _implementation(*args, **kwargs)
