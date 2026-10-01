"""
slm.data.filter — Quality filtering, heuristic cleaners, and deduplication for SLM.

Provides domain-aware text sanitization, heuristic quality checks, line/n-gram
repetition detection, exact SHA-256 deduplication, and MinHash LSH near-deduplication.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Generator, Iterable, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from slm.core.logging import get_logger
from slm.data.schema import Document, Domain

logger = get_logger("data.filter")


class FilterConfig(BaseModel):
    """Configuration thresholds for document quality filtering."""

    model_config = ConfigDict(frozen=True)

    min_chars: int = Field(default=50, ge=1, description="Minimum character length")
    max_chars: int = Field(default=100_000, ge=100, description="Maximum character length")
    min_words: int = Field(default=10, ge=1, description="Minimum word count")
    max_symbol_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Max fraction of non-alphanumeric characters (for prose)"
    )
    max_repetition_ratio: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Max fraction of repeated 3-grams"
    )
    require_terminal_punct: bool = Field(
        default=True,
        description="Whether prose documents require at least one terminal punctuation mark (. ! ?)"
    )
    minhash_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Jaccard similarity threshold for near-duplicate rejection"
    )
    minhash_num_perm: int = Field(
        default=128,
        ge=16,
        le=512,
        description="Number of permutations in MinHash signatures"
    )


# Regex for control characters (retaining standard whitespace: \t, \n, \r)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MULTI_WHITESPACE_RE = re.compile(r"[ ]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")



def clean_text(text: str) -> str:
    """
    Sanitize text through Unicode NFC normalization and control character stripping.
    Preserves line-level indentation (tabs/spaces) while collapsing inline whitespace.

    Parameters
    ----------
    text : str
        Input string.

    Returns
    -------
    str
        Sanitized, normalized string.
    """
    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)
    # 2. Strip non-printable control characters
    sanitized = _CONTROL_CHAR_RE.sub("", normalized)

    # 3. Clean line by line: preserve leading indentation, collapse inline whitespace
    lines = []
    for line in sanitized.splitlines():
        stripped_left = line.lstrip(" \t")
        indent = line[: len(line) - len(stripped_left)]
        body = re.sub(r"[ \t]+", " ", stripped_left)
        lines.append(indent + body if body else "")
    sanitized = "\n".join(lines)

    # 4. Collapse excessive vertical newlines (>2 -> 2)
    sanitized = _MULTI_NEWLINE_RE.sub("\n\n", sanitized)
    return sanitized.strip()



def compute_repetition_ratio(text: str, n: int = 3) -> float:
    """
    Calculate the ratio of repeated n-grams over total n-grams in the text.

    High repetition indicates machine-generated spam or infinite loops.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


class QualityFilter:
    """
    Applies domain-aware heuristic checks to determine if a document is high quality.
    """

    def __init__(self, config: Optional[FilterConfig] = None) -> None:
        self.config = config or FilterConfig()

    def evaluate(self, doc: Document) -> Tuple[bool, str]:
        """
        Evaluate document quality.

        Returns
        -------
        Tuple[bool, str]
            (True, "pass") if document meets quality criteria, else (False, reason).
        """
        text = doc.text
        char_len = len(text)

        # 1. Length checks
        if char_len < self.config.min_chars:
            return False, f"too_short ({char_len} < {self.config.min_chars} chars)"
        if char_len > self.config.max_chars:
            return False, f"too_long ({char_len} > {self.config.max_chars} chars)"

        # 2. Word count
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        if word_count < self.config.min_words:
            return False, f"too_few_words ({word_count} < {self.config.min_words})"

        is_code = doc.domain in (Domain.CODE_PYTHON, Domain.CODE_GENERAL)

        # 3. Repetition check
        rep_ratio = compute_repetition_ratio(text)
        if rep_ratio > self.config.max_repetition_ratio:
            return False, f"high_repetition ({rep_ratio:.2f} > {self.config.max_repetition_ratio})"

        # 4. Prose-specific checks (bypassed for code)
        if not is_code:
            # Non-alphanumeric symbol ratio
            non_alphanumeric = sum(1 for c in text if not c.isalnum() and not c.isspace())
            symbol_ratio = non_alphanumeric / char_len if char_len > 0 else 1.0
            if symbol_ratio > self.config.max_symbol_ratio:
                return False, f"high_symbol_ratio ({symbol_ratio:.2f} > {self.config.max_symbol_ratio})"

            # Terminal punctuation check
            if self.config.require_terminal_punct:
                if not any(mark in text for mark in (".", "!", "?", "。")):
                    return False, "missing_terminal_punctuation"

        return True, "pass"


class Deduplicator:
    """
    High-performance two-stage deduplication engine:
    1. Stage 1: Exact duplicate detection via SHA-256 content hashes.
    2. Stage 2: Near-duplicate detection via MinHash LSH (datasketch).
    """

    def __init__(
        self,
        exact_dedup: bool = True,
        near_dedup: bool = True,
        minhash_threshold: float = 0.85,
        num_perm: int = 128,
    ) -> None:
        self.exact_dedup = exact_dedup
        self.near_dedup = near_dedup
        self.minhash_threshold = minhash_threshold
        self.num_perm = num_perm

        self._seen_hashes: Set[str] = set()
        self._lsh = None

        if self.near_dedup:
            try:
                from datasketch import MinHashLSH  # type: ignore
                self._lsh = MinHashLSH(threshold=self.minhash_threshold, num_perm=self.num_perm)
            except ImportError:
                logger.warning("datasketch not installed. Near-deduplication disabled.")
                self.near_dedup = False

    def is_duplicate(self, doc: Document) -> bool:
        """
        Check if document is an exact or near duplicate of any previously seen document.

        If not duplicate, registers the document in the seen index.
        """
        # 1. Exact match check
        content_hash = doc.content_hash()
        if self.exact_dedup:
            if content_hash in self._seen_hashes:
                return True

        # 2. Near duplicate check via MinHash LSH
        if self.near_dedup and self._lsh is not None:
            from datasketch import MinHash  # type: ignore

            minhash = MinHash(num_perm=self.num_perm)
            # Tokenize into word 3-grams for MinHash signature
            words = re.findall(r"\b\w+\b", doc.text.lower())
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    shingle = f"{words[i]} {words[i+1]} {words[i+2]}"
                    minhash.update(shingle.encode("utf-8"))

                matches = self._lsh.query(minhash)
                if matches:
                    return True

                # Not a duplicate -> store in LSH
                try:
                    self._lsh.insert(doc.id, minhash)
                except ValueError:
                    pass  # ID already exists

        # Register in exact seen set
        if self.exact_dedup:
            self._seen_hashes.add(content_hash)

        return False

    def reset(self) -> None:
        """Clear all stored hashes and LSH index."""
        self._seen_hashes.clear()
        if self.near_dedup and self._lsh is not None:
            from datasketch import MinHashLSH  # type: ignore
            self._lsh = MinHashLSH(threshold=self.minhash_threshold, num_perm=self.num_perm)


def filter_and_clean_stream(
    documents: Iterable[Document],
    filter_config: Optional[FilterConfig] = None,
    dedup: bool = True,
) -> Generator[Document, None, None]:
    """
    Generator pipeline that sanitizes text, filters out low-quality records,
    and removes exact/near duplicates.
    """
    cfg = filter_config or FilterConfig()
    q_filter = QualityFilter(cfg)
    deduplicator = Deduplicator(
        exact_dedup=dedup,
        near_dedup=dedup,
        minhash_threshold=cfg.minhash_threshold,
        num_perm=cfg.minhash_num_perm,
    ) if dedup else None

    for doc in documents:
        # Step 1: Clean and normalize text
        cleaned_text = clean_text(doc.text)
        if not cleaned_text:
            continue

        # Create cleaned document copy
        cleaned_doc = Document(
            id=doc.id,
            text=cleaned_text,
            source=doc.source,
            domain=doc.domain,
            license=doc.license,
            language=doc.language,
            metadata=doc.metadata,
            token_count=doc.token_count,
        )

        # Step 2: Quality heuristic checks
        passed, reason = q_filter.evaluate(cleaned_doc)
        if not passed:
            logger.debug(f"Document {doc.id} dropped: {reason}")
            continue

        # Step 3: Deduplication check
        if deduplicator and deduplicator.is_duplicate(cleaned_doc):
            logger.debug(f"Document {doc.id} dropped: duplicate")
            continue

        yield cleaned_doc
