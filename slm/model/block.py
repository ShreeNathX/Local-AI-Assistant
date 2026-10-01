"""
slm.model.block — Decoder Transformer Block with Pre-RMSNorm and Residual Connections.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from slm.model.attention import GroupedQueryAttention
from slm.model.mlp import SwiGLUMLP
from slm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    A single decoder-only Transformer block with Pre-RMSNorm.

    Structure:
      x -> RMSNorm -> GroupedQueryAttention -> + x (residual)
        -> RMSNorm -> SwiGLUMLP             -> + x (residual)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: Optional[int] = None,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            bias=bias,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLUMLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the Transformer block.
        """
        # Pre-norm Self-Attention with residual connection
        normed_attn_in = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(
            normed_attn_in,
            freqs_cis=freqs_cis,
            mask=mask,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        h = x + attn_out

        # Pre-norm MLP with residual connection
        normed_mlp_in = self.ffn_norm(h)
        mlp_out = self.mlp(normed_mlp_in)
        out = h + mlp_out

        return out, new_kv_cache
