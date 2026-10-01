"""
tests/unit/test_dataset.py
==========================
Unit tests for binary token dataset packing and memory-mapped PyTorch dataset.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from slm.core.exceptions import DataCorruptedError, DataFileNotFoundError
from slm.data.dataset import (
    BinaryTokenWriter,
    PretrainingDataset,
    pack_corpus_to_binary,
)
from slm.data.schema import Document, Domain
from slm.tokenizer.bpe import ByteLevelBPETokenizer


@pytest.mark.unit
class TestBinaryTokenWriter:
    """Test buffered writing of uint16 binary tokens."""

    def test_writer_creates_valid_uint16_binary(self, tmp_path: Path):
        bin_path = tmp_path / "test.bin"
        tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        with BinaryTokenWriter(bin_path, buffer_size=4) as writer:
            writer.write_tokens(tokens)

        assert bin_path.exists()
        # 10 tokens * 2 bytes each = 20 bytes
        assert bin_path.stat().st_size == 20

        # Read back with numpy
        loaded = np.fromfile(bin_path, dtype=np.uint16)
        assert np.array_equal(loaded, np.array(tokens, dtype=np.uint16))


@pytest.mark.unit
class TestPackCorpusToBinary:
    """Test encoding and packing documents into binary tokens."""

    def test_pack_documents_with_eos(self, tmp_path: Path):
        tok = ByteLevelBPETokenizer()
        docs = [
            Document(text="Doc one.", source="test", domain=Domain.STORIES_LITERATURE),
            Document(text="Doc two.", source="test", domain=Domain.STORIES_LITERATURE),
        ]
        out_bin = tmp_path / "packed.bin"
        total = pack_corpus_to_binary(docs, tok, out_bin, add_eos=True)

        assert total > 0
        assert out_bin.stat().st_size == total * 2

        loaded = np.fromfile(out_bin, dtype=np.uint16)
        # Both documents end with EOS token
        eos_id = tok.eos_id
        assert loaded[-1] == eos_id


@pytest.mark.unit
class TestPretrainingDataset:
    """Test PyTorch PretrainingDataset backed by memory-mapped tokens."""

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(DataFileNotFoundError):
            PretrainingDataset(tmp_path / "missing.bin", seq_len=16)

    def test_corrupted_file_size_raises(self, tmp_path: Path):
        corrupted = tmp_path / "corrupted.bin"
        corrupted.write_bytes(b"\x00\x01\x02")  # 3 bytes, odd length
        with pytest.raises(DataCorruptedError):
            PretrainingDataset(corrupted, seq_len=16)

    def test_insufficient_tokens_raises(self, tmp_path: Path):
        tiny = tmp_path / "tiny.bin"
        # Only 5 tokens written, need 16 + 1 = 17
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
        arr.tofile(tiny)
        with pytest.raises(DataCorruptedError):
            PretrainingDataset(tiny, seq_len=16)

    def test_dataset_slices_and_targets(self, tmp_path: Path):
        bin_path = tmp_path / "tokens.bin"
        seq_len = 8
        # Create 25 tokens: 0..24
        raw_tokens = np.arange(25, dtype=np.uint16)
        raw_tokens.tofile(bin_path)

        ds = PretrainingDataset(bin_path, seq_len=seq_len)
        # num_samples = (25 - 1) // 8 = 3
        assert len(ds) == 3

        # Sample 0: x = [0..7], y = [1..8]
        x0, y0 = ds[0]
        assert isinstance(x0, torch.Tensor)
        assert isinstance(y0, torch.Tensor)
        assert x0.shape == (8,)
        assert y0.shape == (8,)
        assert x0.dtype == torch.int64
        assert y0.dtype == torch.int64

        assert torch.equal(x0, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
        assert torch.equal(y0, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))

        # Target y is shifted by exactly 1 token
        assert torch.equal(y0[:-1], x0[1:])

    def test_dataloader_integration(self, tmp_path: Path):
        bin_path = tmp_path / "tokens.bin"
        seq_len = 4
        # 41 tokens -> 10 samples
        raw_tokens = np.arange(41, dtype=np.uint16)
        raw_tokens.tofile(bin_path)

        ds = PretrainingDataset(bin_path, seq_len=seq_len)
        loader = DataLoader(ds, batch_size=2, shuffle=False)

        batch_count = 0
        for x, y in loader:
            assert x.shape == (2, 4)
            assert y.shape == (2, 4)
            batch_count += 1

        assert batch_count == 5
