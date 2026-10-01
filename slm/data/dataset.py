"""
slm.data.dataset — Memory-mapped binary token dataset and packing engine for SLM.

Features
--------
- uint16 binary packing (2 bytes/token, 75% memory/IO savings over int64).
- Sequence packing with document boundary EOS markers.
- PyTorch Dataset backed by np.memmap for zero-RAM overhead high-throughput data loading.
- Input (x) and Autoregressive Target (y) generation for language modeling.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from slm.core.exceptions import DataCorruptedError, DataFileNotFoundError
from slm.core.logging import get_logger
from slm.data.schema import Document
from slm.tokenizer.bpe import ByteLevelBPETokenizer

logger = get_logger("data.dataset")


class BinaryTokenWriter:
    """
    Buffered streaming writer for uint16 binary token arrays.
    """

    def __init__(self, output_path: Path | str, buffer_size: int = 100_000) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self._buffer: list[int] = []
        self._total_tokens = 0
        self._file = open(self.output_path, "wb")

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def write_token(self, token_id: int) -> None:
        """Append a single token ID to buffer."""
        self._buffer.append(token_id)
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def write_tokens(self, token_ids: Iterable[int]) -> None:
        """Append multiple token IDs to buffer."""
        for tid in token_ids:
            self._buffer.append(tid)
            if len(self._buffer) >= self.buffer_size:
                self.flush()

    def flush(self) -> None:
        """Flush in-memory buffer to disk as uint16 binary array."""
        if not self._buffer:
            return
        arr = np.array(self._buffer, dtype=np.uint16)
        self._file.write(arr.tobytes())
        self._total_tokens += len(self._buffer)
        self._buffer.clear()

    def close(self) -> int:
        """Flush remaining buffer, close file, and return total tokens written."""
        self.flush()
        self._file.close()
        logger.info(f"Finished writing {self._total_tokens:,} tokens to {self.output_path}")
        return self._total_tokens

    def __enter__(self) -> BinaryTokenWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def pack_corpus_to_binary(
    documents: Iterable[Document],
    tokenizer: ByteLevelBPETokenizer,
    output_bin_path: Path | str,
    add_eos: bool = True,
    max_total_tokens: Optional[int] = None,
) -> int:
    """
    Encode an iterable of documents and write them packed as uint16 binary tokens.

    Parameters
    ----------
    documents : Iterable[Document]
        Stream of cleaned text documents.
    tokenizer : ByteLevelBPETokenizer
        Tokenizer instance to convert text to IDs.
    output_bin_path : Path | str
        Target .bin file path.
    add_eos : bool
        Whether to append EOS token after each document to demarcate boundaries.
    max_total_tokens : Optional[int]
        Early stop when token limit is reached.

    Returns
    -------
    int
        Total number of tokens written.
    """
    writer = BinaryTokenWriter(output_bin_path)
    total_written = 0

    with writer:
        for doc in documents:
            token_ids = tokenizer.encode(doc.text, add_bos=False, add_eos=add_eos)
            if not token_ids:
                continue

            if max_total_tokens is not None:
                remaining = max_total_tokens - total_written
                if remaining <= 0:
                    break
                if len(token_ids) > remaining:
                    token_ids = token_ids[:remaining]

            writer.write_tokens(token_ids)
            total_written += len(token_ids)

            if max_total_tokens is not None and total_written >= max_total_tokens:
                break

    return total_written


class PretrainingDataset(Dataset):
    """
    PyTorch Dataset backed by a memory-mapped binary array of uint16 tokens.

    Provides constant-memory random access and high-throughput batching.
    Each sample contains:
      - x: input sequence of length `seq_len` (tokens[t : t + seq_len])
      - y: next-token target sequence of length `seq_len` (tokens[t + 1 : t + seq_len + 1])
    """

    def __init__(
        self,
        bin_path: Path | str,
        seq_len: int = 512,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len

        if not self.bin_path.is_file():
            raise DataFileNotFoundError(
                f"Binary token file not found at: {self.bin_path}",
                suggestion="Run dataset packing script first to generate tokenized binary files."
            )

        file_size_bytes = os.path.getsize(self.bin_path)
        if file_size_bytes % 2 != 0:
            raise DataCorruptedError(
                f"Binary file size ({file_size_bytes} bytes) is not a multiple of 2 (uint16)."
            )

        self.num_tokens = file_size_bytes // 2
        # We need at least seq_len + 1 tokens for 1 sample (input x and target y)
        if self.num_tokens < self.seq_len + 1:
            raise DataCorruptedError(
                f"Dataset contains {self.num_tokens} tokens, but requires at least {self.seq_len + 1} "
                f"tokens for a sequence length of {self.seq_len}."
            )

        # Memory map file with copy-on-write read mode
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.num_samples = (self.num_tokens - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")

        start = idx * self.seq_len
        end = start + self.seq_len

        # Extract uint16 window, convert to int64 for PyTorch embedding lookup
        x_np = self.data[start:end].astype(np.int64)
        y_np = self.data[start + 1 : end + 1].astype(np.int64)

        return torch.from_numpy(x_np), torch.from_numpy(y_np)
