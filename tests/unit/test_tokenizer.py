"""
tests/unit/test_tokenizer.py
============================
Unit tests for the Byte-Level BPE Tokenizer from scratch.
"""

from pathlib import Path
import pytest

from slm.core.exceptions import TokenizerError, TokenizerNotFoundError
from slm.tokenizer.bpe import ByteLevelBPETokenizer
from slm.tokenizer.constants import (
    BOS_TOKEN,
    EOS_TOKEN,
    IM_END_TOKEN,
    IM_START_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    THOUGHT_TOKEN,
    TOOL_CALL_TOKEN,
)


@pytest.mark.unit
class TestTokenizerBaseVocab:
    """Test base byte vocabulary and special tokens setup."""

    def test_base_vocab_size(self):
        tok = ByteLevelBPETokenizer()
        # 10 special tokens + 256 byte tokens = 266
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 256
        assert tok.pad_id == 0
        assert tok.bos_id == 2
        assert tok.eos_id == 3

    def test_special_token_ids(self):
        tok = ByteLevelBPETokenizer()
        for i, token_name in enumerate(SPECIAL_TOKENS):
            assert tok.vocab[token_name] == i
            assert tok.id_to_token[i] == token_name


@pytest.mark.unit
class TestTokenizerRoundtrip:
    """Verify 100% reversible roundtrip fidelity across diverse inputs."""

    def setup_method(self):
        self.tok = ByteLevelBPETokenizer()

    @pytest.mark.parametrize(
        "sample",
        [
            "Hello, world! This is a test of the SLM tokenizer.",
            "def add(a: int, b: int) -> int:\n    return a + b",
            "Numbers and symbols: 1234567890 + - * / = % @ # $ ^ & * ( ) _ +",
            "Special characters: Café, naïve, résumé, Zürich.",
            "Multi-language: こんにちは世界, Привет мир, مرحبا بالعالم.",
            "Emojis: 🚀 🤖 🧠 ⚡ 💻 🔥",
            "Complex whitespace:\n\t\tIndented line\n\n\nTriple newline",
        ],
    )
    def test_exact_reconstruction(self, sample: str):
        token_ids = self.tok.encode(sample)
        reconstructed = self.tok.decode(token_ids)
        assert reconstructed == sample

    def test_bos_and_eos_flags(self):
        sample = "Test sentence."
        tokens = self.tok.encode(sample, add_bos=True, add_eos=True)
        assert tokens[0] == self.tok.bos_id
        assert tokens[-1] == self.tok.eos_id

        # Decode skipping special tokens gives original text
        assert self.tok.decode(tokens, skip_special_tokens=True) == sample

        # Decode including special tokens gives tokens enclosed
        assert self.tok.decode(tokens, skip_special_tokens=False) == f"<s>{sample}</s>"


@pytest.mark.unit
class TestSpecialTokensHandling:
    """Verify special delimiter tokens are treated atomically."""

    def test_chat_and_tool_tokens_preserved(self):
        tok = ByteLevelBPETokenizer()
        text = f"{IM_START_TOKEN}user\nWhat is 2+2?{IM_END_TOKEN}{IM_START_TOKEN}assistant\n{THOUGHT_TOKEN}Calculating...{IM_END_TOKEN}"
        token_ids = tok.encode(text)

        # Check special token IDs appear directly in the sequence
        assert tok.special_token_to_id[IM_START_TOKEN] in token_ids
        assert tok.special_token_to_id[IM_END_TOKEN] in token_ids
        assert tok.special_token_to_id[THOUGHT_TOKEN] in token_ids

        # Reconstructed matches exactly
        decoded = tok.decode(token_ids)
        assert decoded == text


@pytest.mark.unit
class TestBPETraining:
    """Test iterative BPE merge training algorithm."""

    def test_training_learns_merges(self):
        corpus = [
            "the cat sat on the mat",
            "the rat sat on the mat",
            "the bat sat on the mat",
            "the cat and the dog",
        ] * 10

        tok = ByteLevelBPETokenizer()
        initial_vocab_size = tok.vocab_size
        target_vocab = initial_vocab_size + 20

        tok.train_from_iterator(corpus, vocab_size=target_vocab, min_frequency=2)

        assert tok.vocab_size > initial_vocab_size
        assert len(tok.merges_list) > 0

        # "the" is the most frequent word, should be compressed into fewer tokens
        uncompressed_tok = ByteLevelBPETokenizer()
        uncompressed_len = len(uncompressed_tok.encode(" the"))
        trained_len = len(tok.encode(" the"))
        assert trained_len <= uncompressed_len

        # Reversible roundtrip must still hold
        sample = "the cat sat on the mat"
        assert tok.decode(tok.encode(sample)) == sample

    def test_training_vocab_size_too_small_raises(self):
        tok = ByteLevelBPETokenizer()
        with pytest.raises(TokenizerError):
            tok.train_from_iterator(["hello world"], vocab_size=100)


@pytest.mark.unit
class TestSaveAndLoad:
    """Test tokenizer persistence and reloading."""

    def test_save_load_fidelity(self, tmp_path: Path):
        corpus = ["machine learning algorithms require substantial data"] * 5
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(corpus, vocab_size=tok.vocab_size + 10, min_frequency=2)

        save_dir = tmp_path / "tokenizer_artifacts"
        tok.save(save_dir)

        assert (save_dir / "vocab.json").exists()
        assert (save_dir / "merges.txt").exists()
        assert (save_dir / "tokenizer_config.json").exists()

        # Load back
        loaded_tok = ByteLevelBPETokenizer.load(save_dir)
        assert loaded_tok.vocab_size == tok.vocab_size
        assert len(loaded_tok.merges_list) == len(tok.merges_list)

        test_phrase = "machine learning data"
        assert loaded_tok.encode(test_phrase) == tok.encode(test_phrase)
        assert loaded_tok.decode(loaded_tok.encode(test_phrase)) == test_phrase

    def test_load_nonexistent_dir_raises(self, tmp_path: Path):
        with pytest.raises(TokenizerNotFoundError):
            ByteLevelBPETokenizer.load(tmp_path / "non_existent")
