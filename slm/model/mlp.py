"""
slm.model.mlp — SwiGLU Feed-Forward Network.

Reference
---------
Shazeer (2020): "GLU Variants Improve Transformer"
https://arxiv.org/abs/2002.05202

SwiGLU applies Swish/SiLU non-linearity in a gated linear unit:
    SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """
    SwiGLU Gated Feed-Forward Network.

    Parameters
    ----------
    d_model : int
        Input and output embedding dimension.
    d_ff : int
        Intermediate hidden expansion dimension (typically ~8/3 * d_model).
    dropout : float
        Dropout probability.
    bias : bool
        Whether linear projections include bias (False for modern Llama-style).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Gated activation: SiLU(W_gate x) * W_up x
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        # Down projection back to d_model
        out = self.down_proj(hidden)
        return self.dropout(out)
