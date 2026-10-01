"""
slm.model.attention — Multi-Head & Grouped-Query Attention (GQA) with RoPE and KV-Cache.

References
----------
Ainslie et al. (2023): "GQA: Training Generalized Multi-Query Transformer Models
from Multi-Head Checkpoints" https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm.model.rope import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand Key/Value heads to match Query heads when using GQA.

    Input shape:  (batch, n_kv_heads, seq_len, head_dim)
    Output shape: (batch, n_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) / Multi-Head Attention (MHA) module.

    Parameters
    ----------
    d_model : int
        Hidden model dimension.
    n_heads : int
        Number of Query attention heads.
    n_kv_heads : Optional[int]
        Number of Key/Value attention heads. Defaults to n_heads (standard MHA).
        When n_kv_heads < n_heads, GQA is active.
    dropout : float
        Attention dropout probability.
    bias : bool
        Whether Q, K, V, and Output projections use bias (default False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        self.head_dim = d_model // n_heads
        self.num_kv_groups = self.n_heads // self.n_kv_heads

        # Projections
        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=bias)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input activations of shape (batch, seq_len, d_model).
        freqs_cis : torch.Tensor
            Precomputed complex rotary frequencies for positions [start_pos : start_pos + seq_len].
        mask : Optional[torch.Tensor]
            Optional attention mask (if None, causal masking is applied via SDPA).
        kv_cache : Optional[Tuple[torch.Tensor, torch.Tensor]]
            Cached (k_cache, v_cache) from prior autoregressive decoding steps.
        start_pos : int
            Starting sequence index (for incremental inference with KV cache).

        Returns
        -------
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]
            (output, (k_cache, v_cache))
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project inputs to Q, K, V
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # 2. Reshape to multi-head format: (batch, seq_len, n_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 3. Apply Rotary Position Embeddings to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 4. Handle KV-Cache (inference mode)
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            # Concatenate along sequence length
            xk = torch.cat([prev_k, xk], dim=1)
            xv = torch.cat([prev_v, xv], dim=1)
            new_kv_cache = (xk, xv)
        else:
            new_kv_cache = None

        # 5. Transpose to PyTorch SDPA format: (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. Expand KV heads for GQA to match Q heads
        xk = repeat_kv(xk, self.num_kv_groups)
        xv = repeat_kv(xv, self.num_kv_groups)

        # 7. Scaled Dot-Product Attention with FlashAttention / Math fallback
        # If no explicit mask and no KV cache, SDPA handles causal masking natively
        is_causal = (mask is None) and (kv_cache is None) and (seq_len > 1)

        attn_output = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # 8. Recombine heads: (batch, seq_len, n_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 9. Final linear projection
        output = self.out_proj(attn_output)

        return output, new_kv_cache
