"""VSA 3D tiling metadata helpers, ported from FastVideo's ``vsa_utils.py``.

Video latents arrive in raster (t, h, w) order. VSA reorders tokens so each
``(ts_t x ts_h x ts_w)`` spatial tile is contiguous, pads every tile up to the
tile volume, and records how many real tokens each tile holds. These helpers
produce the index tensors that move tokens between raster and tiled layouts and
the ``variable_block_sizes`` consumed by ``video_sparse_attn``.
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Tuple

import torch
from torch import Tensor

VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=16)
def get_tile_partition_indices(
    dit_seq_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    """Raster-order token ids grouped into tile-contiguous order."""
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    chunks = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                chunks.append(
                    indices[
                        t * ts : min(t * ts + ts, T),
                        h * hs : min(h * hs + hs, H),
                        w * ws : min(w * ws + ws, W),
                    ].flatten()
                )
    return torch.cat(chunks, dim=0)


@functools.lru_cache(maxsize=16)
def get_reverse_tile_partition_indices(
    dit_seq_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int, int],
    device: torch.device,
) -> Tensor:
    """Inverse permutation of :func:`get_tile_partition_indices`."""
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=16)
def construct_variable_block_sizes(
    dit_seq_shape: Tuple[int, int, int],
    num_tiles: Tuple[int, int, int],
    device: torch.device,
    tile_size: Tuple[int, int, int] = VSA_TILE_SIZE,
) -> Tensor:
    """Valid token count per tile, flattened in (t, h, w) tile order."""
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n: int) -> Tensor:
        sizes = torch.full((n,), tile, dtype=torch.long, device=device)
        remainder = dim_len - (n - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)
    h_sizes = _sizes(h, ts_h, n_h)
    w_sizes = _sizes(w, ts_w, n_w)
    return (
        t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]
    ).reshape(-1)


def get_non_pad_index(variable_block_sizes: Tensor, max_block_size: int) -> Tensor:
    """Flat indices of real (non-padding) slots in a block-padded layout."""
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


def build_vsa_metadata(
    dit_seq_shape: Tuple[int, int, int],
    tile_size: Tuple[int, int, int] = VSA_TILE_SIZE,
    device: torch.device | str = "cpu",
) -> Dict[str, object]:
    """Build every VSA tiling tensor from a latent shape in one call.

    Args:
        dit_seq_shape: ``(T, H, W)`` latent grid (after patchification).
        tile_size: tokens per tile in each dimension.
        device: device for the index tensors.

    Returns dict with ``tile_partition_indices``,
    ``reverse_tile_partition_indices``, ``variable_block_sizes``,
    ``non_pad_index``, ``num_tiles`` and ``max_block_size``.
    """
    device = torch.device(device)
    T, H, W = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    max_block_size = math.prod(tile_size)
    num_tiles = (math.ceil(T / ts_t), math.ceil(H / ts_h), math.ceil(W / ts_w))

    tile_idx = get_tile_partition_indices(dit_seq_shape, tile_size, device)
    reverse_idx = get_reverse_tile_partition_indices(dit_seq_shape, tile_size, device)
    vbs = construct_variable_block_sizes(dit_seq_shape, num_tiles, device, tile_size)
    npi = get_non_pad_index(vbs, max_block_size)
    return {
        "tile_partition_indices": tile_idx,
        "reverse_tile_partition_indices": reverse_idx,
        "variable_block_sizes": vbs,
        "non_pad_index": npi,
        "num_tiles": num_tiles,
        "max_block_size": max_block_size,
    }


def tile(x: Tensor, tile_partition_indices: Tensor, non_pad_index: Tensor, num_blocks: int, max_block_size: int) -> Tensor:
    """Reorder raster-order tokens into the zero-padded, tile-contiguous layout.

    Args:
        x: ``[batch, seq_len, ...]`` in raster order.
        tile_partition_indices, non_pad_index: from :func:`build_vsa_metadata`.
        num_blocks: total tiles (``prod(num_tiles)``).
        max_block_size: padded tokens per tile (``prod(tile_size)``).
    """
    target = (x.shape[0], num_blocks * max_block_size, *x.shape[2:])
    buf = torch.zeros(target, device=x.device, dtype=x.dtype)
    buf[:, non_pad_index] = x[:, tile_partition_indices]
    return buf


def untile(x: Tensor, non_pad_index: Tensor, reverse_tile_partition_indices: Tensor) -> Tensor:
    """Inverse of :func:`tile`: padded tile layout back to raster order."""
    return x[:, non_pad_index][:, reverse_tile_partition_indices]
