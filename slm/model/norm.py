"""
slm.model.norm — Root Mean Square Layer Normalization (RMSNorm).

Reference
---------
Zhang & Sennrich (2019): "Root Mean Square Layer Normalization"
https://arxiv.org/abs/1910.07467

RMSNorm regularizes the summed inputs to a neuron by the root mean square (RMS),
providing similar training stability to LayerNorm without computing or centering
the mean across features, yielding a 10%–50% speedup.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Mathematical definition:
        RMS(x) = sqrt( mean(x^2) + eps )
        y = (x / RMS(x)) * weight

    Parameters
    ----------
    dim : int
        Feature dimension (d_model).
    eps : float
        Small positive constant added to variance for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Learnable affine scale parameter gamma
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization in float32 for numerical stability."""
        # x: (..., dim)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, dim).

        Returns
        -------
        torch.Tensor
            Normalized tensor of shape (batch_size, seq_len, dim).
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
