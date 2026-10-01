"""
slm.model.rope — Rotary Position Embeddings (RoPE).

Reference
---------
Su et al. (2021): "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864

Rotary Position Embedding encodes absolute position with a rotation matrix and
naturally incorporates explicit relative position dependency in self-attention.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta_base: float = 10000.0,
) -> torch.Tensor:
    """
    Precompute complex exponential frequencies cis = cos + i*sin for RoPE.

    Parameters
    ----------
    dim : int
        Head dimension (must be even).
    max_seq_len : int
        Maximum sequence length supported.
    theta_base : float
        Base frequency constant (default 10,000.0).

    Returns
    -------
    torch.Tensor
        Complex tensor of shape (max_seq_len, dim // 2).
    """
    assert dim % 2 == 0, f"Head dimension must be even, got {dim}"
    # Frequencies theta_i = 1.0 / (theta_base ^ (2i / dim))
    freqs = 1.0 / (theta_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # Outer product: (max_seq_len, dim // 2)
    freqs = torch.outer(t, freqs)
    # Polar form cis = exp(i * freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape precomputed complex frequencies to broadcast against x.

    freqs_cis: (seq_len, dim // 2)
    x: (batch_size, seq_len, n_heads, dim // 2) as complex
    Returns: (1, seq_len, 1, dim // 2)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        f"freqs_cis shape {freqs_cis.shape} does not match x seq_len and dim {x.shape[1], x.shape[-1]}"
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to Query and Key tensors.

    Parameters
    ----------
    xq : torch.Tensor
        Query tensor of shape (batch, seq_len, n_q_heads, head_dim).
    xk : torch.Tensor
        Key tensor of shape (batch, seq_len, n_kv_heads, head_dim).
    freqs_cis : torch.Tensor
        Precomputed complex frequencies for current sequence slice (seq_len, head_dim // 2).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated (xq_out, xk_out) with the same shapes and dtypes as inputs.
    """
    # View as 2D complex numbers: (..., head_dim) -> (..., head_dim // 2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    # Complex multiplication rotates coordinates: (a + ib) * (cos + i sin)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
