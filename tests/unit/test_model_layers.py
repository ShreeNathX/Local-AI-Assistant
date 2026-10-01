"""
tests/unit/test_model_layers.py
===============================
Unit tests for SLM neural network layers: RMSNorm, RoPE, SwiGLU, GQA, TransformerBlock.
"""

import pytest
import torch

from slm.model.attention import GroupedQueryAttention, repeat_kv
from slm.model.block import TransformerBlock
from slm.model.mlp import SwiGLUMLP
from slm.model.norm import RMSNorm
from slm.model.rope import apply_rotary_emb, precompute_freqs_cis


@pytest.mark.unit
class TestRMSNorm:
    """Test Root Mean Square Layer Normalization."""

    def test_output_shape(self):
        norm = RMSNorm(dim=128)
        x = torch.randn(2, 16, 128)
        out = norm(x)
        assert out.shape == x.shape

    def test_rms_variance_normalized(self):
        norm = RMSNorm(dim=256, eps=1e-8)
        x = torch.randn(4, 32, 256) * 15.0 + 5.0
        out = norm(x)
        # RMS of normalized vector with weight=1.0 should be ~1.0
        rms = torch.sqrt(out.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

    def test_gradient_flow(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None


@pytest.mark.unit
class TestRoPE:
    """Test Rotary Position Embeddings (RoPE)."""

    def test_precompute_freqs_shape(self):
        head_dim = 64
        max_seq_len = 512
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
        assert freqs_cis.shape == (max_seq_len, head_dim // 2)
        assert freqs_cis.is_complex()

    def test_norm_preservation(self):
        head_dim = 32
        seq_len = 16
        freqs_cis = precompute_freqs_cis(head_dim, seq_len)

        xq = torch.randn(2, seq_len, 4, head_dim)
        xk = torch.randn(2, seq_len, 4, head_dim)

        xq_norm_before = xq.norm(dim=-1)
        xq_rot, _ = apply_rotary_emb(xq, xk, freqs_cis)
        xq_norm_after = xq_rot.norm(dim=-1)

        # 2D coordinate rotation is an orthogonal transformation -> preserves L2 norm
        assert torch.allclose(xq_norm_before, xq_norm_after, atol=1e-5)


@pytest.mark.unit
class TestSwiGLU:
    """Test SwiGLU Gated Feed-Forward Network."""

    def test_forward_shape(self):
        mlp = SwiGLUMLP(d_model=128, d_ff=344)
        x = torch.randn(2, 10, 128)
        out = mlp(x)
        assert out.shape == (2, 10, 128)

    def test_gradient_propagation(self):
        mlp = SwiGLUMLP(d_model=64, d_ff=172)
        x = torch.randn(2, 5, 64, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None


@pytest.mark.unit
class TestAttention:
    """Test Grouped-Query Attention (GQA) and Multi-Head Attention (MHA)."""

    def test_mha_standard(self):
        d_model = 128
        n_heads = 4
        seq_len = 16
        freqs_cis = precompute_freqs_cis(d_model // n_heads, seq_len)

        attn = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_heads)
        x = torch.randn(2, seq_len, d_model)
        out, cache = attn(x, freqs_cis=freqs_cis)

        assert out.shape == (2, seq_len, d_model)
        assert cache is None

    def test_gqa_grouped(self):
        d_model = 128
        n_heads = 8
        n_kv_heads = 2  # 4 query heads per KV group
        seq_len = 16
        freqs_cis = precompute_freqs_cis(d_model // n_heads, seq_len)

        attn = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads)
        x = torch.randn(2, seq_len, d_model)
        out, _ = attn(x, freqs_cis=freqs_cis)

        assert out.shape == (2, seq_len, d_model)

    def test_repeat_kv(self):
        # (batch=2, n_kv_heads=2, seq_len=4, head_dim=8)
        kv = torch.randn(2, 2, 4, 8)
        expanded = repeat_kv(kv, n_rep=3)
        assert expanded.shape == (2, 6, 4, 8)


@pytest.mark.unit
class TestTransformerBlock:
    """Test complete TransformerBlock layer."""

    def test_block_forward_and_backward(self):
        d_model = 128
        n_heads = 4
        n_kv_heads = 2
        d_ff = 344
        seq_len = 8

        block = TransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
        )

        freqs_cis = precompute_freqs_cis(d_model // n_heads, seq_len)
        x = torch.randn(2, seq_len, d_model, requires_grad=True)

        out, kv_cache = block(x, freqs_cis=freqs_cis)

        assert out.shape == (2, seq_len, d_model)
        assert kv_cache is None

        # Test backward pass
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
