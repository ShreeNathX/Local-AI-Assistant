"""
slm.tokenizer.bpe — Byte-Level Byte-Pair Encoding (BBPE) Tokenizer built from scratch.

Features
--------
- 100% Out-Of-Vocabulary immunity (all 256 initial bytes are representable).
- Special token handling for assistant chat, reasoning tags, and tool calling.
- Full reversible roundtrip fidelity: decode(encode(text)) == text.
- Clean serialization and loading without external tokenizer library runtime dependencies.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import regex as re

from slm.core.exceptions import TokenizerError, TokenizerNotFoundError
from slm.core.logging import get_logger
from slm.tokenizer.constants import (
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    NUM_SPECIAL_TOKENS,
    PAD_TOKEN,
    PRE_TOKENIZE_REGEX,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    bytes_to_unicode,
    unicode_to_bytes,
)

logger = get_logger("tokenizer.bpe")


class ByteLevelBPETokenizer:
    """
    Byte-Level Byte-Pair Encoding Tokenizer.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = unicode_to_bytes()

        self.special_tokens_list = special_tokens or list(SPECIAL_TOKENS)
        self.special_token_to_id: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}

        for idx, token in enumerate(self.special_tokens_list):
            self.special_token_to_id[token] = idx
            self.id_to_special_token[idx] = token

        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self._cache: Dict[str, List[str]] = {}

        if vocab is not None and merges is not None:
            self.vocab = dict(vocab)
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            self.merges_list = list(merges)
            self.bpe_ranks = {pair: rank for rank, pair in enumerate(self.merges_list)}
        else:
            self._init_base_vocab()


    def _init_base_vocab(self) -> None:
        """Initialize vocabulary with special tokens followed by all 256 byte tokens."""
        self.vocab.clear()
        self.id_to_token.clear()

        # 1. Special tokens
        for token, idx in self.special_token_to_id.items():
            self.vocab[token] = idx
            self.id_to_token[idx] = token

        # 2. 256 initial byte tokens
        start_id = len(self.special_token_to_id)
        for b in range(256):
            char_token = self.byte_encoder[b]
            token_id = start_id + b
            self.vocab[char_token] = token_id
            self.id_to_token[token_id] = char_token

        self.merges_list = []
        self.bpe_ranks = {}
        self._cache.clear()

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.special_token_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.special_token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.special_token_to_id[EOS_TOKEN]

    def _tokenize_word_to_bytes(self, word: str) -> List[str]:
        """Convert a pre-tokenized word chunk to a list of byte-encoded characters."""
        return [self.byte_encoder[b] for b in word.encode("utf-8")]

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 8192,
        min_frequency: int = 2,
    ) -> None:
        """
        Train Byte-Pair Encoding merges from an iterator of texts.

        Parameters
        ----------
        iterator : Iterable[str]
            Corpus text chunks or documents.
        vocab_size : int
            Target total vocabulary size (must be >= 256 + num_special_tokens).
        min_frequency : int
            Minimum frequency for a pair to be considered for merging.
        """
        min_vocab = len(self.special_token_to_id) + 256
        if vocab_size < min_vocab:
            raise TokenizerError(
                f"Requested vocab_size ({vocab_size}) is smaller than minimum required ({min_vocab})"
            )

        logger.info(f"Starting BPE training: target vocab_size={vocab_size}, min_freq={min_frequency}")
        self._init_base_vocab()

        # Step 1: Pre-tokenize and count word frequencies
        word_counts: Counter[str] = Counter()
        for doc in iterator:
            for match in PRE_TOKENIZE_REGEX.finditer(doc):
                word = match.group(0)
                if word:
                    word_counts[word] += 1

        logger.info(f"Unique pre-tokenized words extracted: {len(word_counts)}")

        # Step 2: Split each word into sequence of base byte characters
        splits: Dict[str, List[str]] = {
            word: self._tokenize_word_to_bytes(word) for word in word_counts
        }

        # Step 3: Iteratively merge top pairs
        num_merges = vocab_size - len(self.vocab)
        merges: List[Tuple[str, str]] = []

        for merge_idx in range(num_merges):
            pair_counts: Counter[Tuple[str, str]] = Counter()

            for word, freq in word_counts.items():
                word_pieces = splits[word]
                for i in range(len(word_pieces) - 1):
                    pair = (word_pieces[i], word_pieces[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                logger.warning(f"No further pairs found to merge at merge {merge_idx}.")
                break

            best_pair, highest_freq = pair_counts.most_common(1)[0]
            if highest_freq < min_frequency:
                logger.info(
                    f"Highest pair frequency ({highest_freq}) is below threshold ({min_frequency}). "
                    f"Stopping training at vocab size {len(self.vocab)}."
                )
                break

            # Create new merged token
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token
            merges.append(best_pair)
            self.bpe_ranks[best_pair] = len(self.bpe_ranks)

            # Update splits with the new merge
            for word in word_counts:
                word_pieces = splits[word]
                if len(word_pieces) < 2:
                    continue

                new_pieces: List[str] = []
                i = 0
                while i < len(word_pieces):
                    if (
                        i < len(word_pieces) - 1
                        and word_pieces[i] == best_pair[0]
                        and word_pieces[i + 1] == best_pair[1]
                    ):
                        new_pieces.append(new_token)
                        i += 2
                    else:
                        new_pieces.append(word_pieces[i])
                        i += 1
                splits[word] = new_pieces

        self.merges_list = merges
        self._cache.clear()
        logger.info(f"BPE training complete. Final vocabulary size: {len(self.vocab)}")

    def _bpe(self, word_pieces: List[str]) -> List[str]:
        """Apply learned BPE merges greedily using merge ranks."""
        if len(word_pieces) <= 1:
            return word_pieces

        pieces = list(word_pieces)
        while True:
            # Find all adjacent pairs in current sequence
            pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
            # Find the pair with the lowest merge rank (highest priority)
            mergeable = [
                (self.bpe_ranks[pair], pair)
                for pair in pairs
                if pair in self.bpe_ranks
            ]
            if not mergeable:
                break

            _, best_pair = min(mergeable, key=lambda x: x[0])

            # Merge occurrences of best_pair in pieces
            new_pieces: List[str] = []
            i = 0
            while i < len(pieces):
                if (
                    i < len(pieces) - 1
                    and pieces[i] == best_pair[0]
                    and pieces[i + 1] == best_pair[1]
                ):
                    new_pieces.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = new_pieces
            if len(pieces) <= 1:
                break

        return pieces

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode string into token IDs.

        Preserves special tokens if they appear as exact substrings in text.
        """
        token_ids: List[int] = []

        if add_bos:
            token_ids.append(self.bos_id)

        # Regex to segment text into special tokens vs regular text
        special_pattern = "|".join(re.escape(tok) for tok in self.special_tokens_list)
        split_pattern = re.compile(f"({special_pattern})")
        parts = split_pattern.split(text)

        for part in parts:
            if not part:
                continue
            if part in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[part])
            else:
                for match in PRE_TOKENIZE_REGEX.finditer(part):
                    word = match.group(0)
                    if not word:
                        continue
                    if word not in self._cache:
                        raw_byte_chars = self._tokenize_word_to_bytes(word)
                        self._cache[word] = self._bpe(raw_byte_chars)

                    for piece in self._cache[word]:
                        token_ids.append(self.vocab.get(piece, self.unk_id))

        if add_eos:
            token_ids.append(self.eos_id)

        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """
        Decode a sequence of token IDs back into a UTF-8 string.

        Guaranteed invertible byte reconstruction.
        """
        byte_chunks: List[int] = []
        special_tokens_set = set(self.special_token_to_id.values())

        for token_id in token_ids:
            if token_id in special_tokens_set:
                if not skip_special_tokens:
                    token_str = self.id_to_special_token[token_id]
                    byte_chunks.extend(token_str.encode("utf-8"))
            else:
                token_str = self.id_to_token.get(token_id, "")
                for char in token_str:
                    if char in self.byte_decoder:
                        byte_chunks.append(self.byte_decoder[char])

        return bytes(byte_chunks).decode("utf-8", errors="replace")

    def save(self, directory: Path | str) -> None:
        """Save vocabulary, merges, and configuration to disk."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = save_dir / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        merges_path = save_dir / "merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.1\n")
            for pair in self.merges_list:
                f.write(f"{pair[0]} {pair[1]}\n")

        config_path = save_dir / "tokenizer_config.json"
        config = {
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens_list,
            "special_token_ids": self.special_token_to_id,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved tokenizer ({len(self.vocab)} tokens) to {save_dir}")

    @classmethod
    def load(cls, directory: Path | str) -> ByteLevelBPETokenizer:
        """Load tokenizer from saved artifacts directory."""
        load_dir = Path(directory)
        vocab_path = load_dir / "vocab.json"
        merges_path = load_dir / "merges.txt"
        config_path = load_dir / "tokenizer_config.json"

        if not vocab_path.is_file() or not merges_path.is_file():
            raise TokenizerNotFoundError(str(load_dir))

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        merges: List[Tuple[str, str]] = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        special_tokens = None
        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                special_tokens = cfg.get("special_tokens")

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
