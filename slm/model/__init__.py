"""
slm.model — Neural network architecture, attention mechanisms, and layers for SLM.

Modules
-------
norm        RMSNorm (Root Mean Square Layer Normalization).
rope        Rotary Position Embedding (RoPE) precomputations and operations.
mlp         SwiGLU feed-forward network.
attention   Grouped-Query Attention (GQA) and Multi-Head Attention.
block       TransformerBlock (Pre-RMSNorm decoder layer).
transformer Full SLM Transformer language model assembly.
"""

from slm.model.norm import RMSNorm
from slm.model.rope import (
    apply_rotary_emb,
    precompute_freqs_cis,
    reshape_for_broadcast,
)
from slm.model.mlp import SwiGLUMLP
from slm.model.attention import GroupedQueryAttention, repeat_kv
from slm.model.block import TransformerBlock

__all__ = [
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "reshape_for_broadcast",
    "SwiGLUMLP",
    "GroupedQueryAttention",
    "repeat_kv",
    "TransformerBlock",
]
