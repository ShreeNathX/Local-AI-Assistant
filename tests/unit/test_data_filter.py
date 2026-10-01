"""
tests/unit/test_data_filter.py
==============================
Unit tests for slm.data.filter (cleaning, heuristics, and deduplication).
"""

import pytest

from slm.data.schema import Document, Domain, LicenseType
from slm.data.filter import (
    Deduplicator,
    FilterConfig,
    QualityFilter,
    clean_text,
    compute_repetition_ratio,
    filter_and_clean_stream,
)


@pytest.mark.unit
class TestCleanText:
    """Test text sanitization and normalization."""

    def test_strip_control_characters(self):
        raw = "Hello\x00\x07World\x1f! Normal text."
        cleaned = clean_text(raw)
        assert cleaned == "HelloWorld! Normal text."

    def test_preserves_tabs_and_newlines(self):
        raw = "Line 1\n\tIndented Line 2\nLine 3"
        cleaned = clean_text(raw)
        assert "Line 1\n\tIndented Line 2\nLine 3" in cleaned

    def test_collapses_excessive_whitespace(self):
        raw = "Word1     Word2   \t   Word3"
        cleaned = clean_text(raw)
        assert cleaned == "Word1 Word2 Word3"

    def test_collapses_excessive_newlines(self):
        raw = "Paragraph 1\n\n\n\n\n\nParagraph 2"
        cleaned = clean_text(raw)
        assert cleaned == "Paragraph 1\n\nParagraph 2"


@pytest.mark.unit
class TestRepetitionDetection:
    """Test n-gram repetition detection."""

    def test_natural_text_low_repetition(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Every good boy does fine in music school. "
            "Artificial intelligence is transforming how we write software."
        )
        ratio = compute_repetition_ratio(text)
        assert ratio < 0.15

    def test_repetitive_text_high_repetition(self):
        text = "buy now buy now buy now buy now buy now buy now buy now."
        ratio = compute_repetition_ratio(text)
        assert ratio > 0.40


@pytest.mark.unit
class TestQualityFilter:
    """Test heuristic quality filter."""

    def setup_method(self):
        self.q_filter = QualityFilter(
            FilterConfig(
                min_chars=30,
                max_chars=500,
                min_words=5,
                max_symbol_ratio=0.25,
                max_repetition_ratio=0.30,
                require_terminal_punct=True,
            )
        )

    def test_valid_document_passes(self):
        doc = Document(
            text="The little squirrel stored twenty acorns inside the hollow tree for the long winter.",
            source="test",
            domain=Domain.STORIES_LITERATURE,
        )
        passed, reason = self.q_filter.evaluate(doc)
        assert passed is True
        assert reason == "pass"

    def test_too_short_fails(self):
        doc = Document(
            text="Hi there.",
            source="test",
            domain=Domain.GENERAL_KNOWLEDGE,
        )
        passed, reason = self.q_filter.evaluate(doc)
        assert passed is False
        assert "too_short" in reason

    def test_missing_terminal_punctuation_fails(self):
        doc = Document(
            text="This is a sentence without any period or mark at the end of it all",
            source="test",
            domain=Domain.GENERAL_KNOWLEDGE,
        )
        passed, reason = self.q_filter.evaluate(doc)
        assert passed is False
        assert reason == "missing_terminal_punctuation"

    def test_code_domain_allows_no_terminal_punct_and_symbols(self):
        code_text = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "x = add(10, 20)\n"
            "print(x)"
        )
        doc = Document(
            text=code_text,
            source="test",
            domain=Domain.CODE_PYTHON,
        )
        passed, reason = self.q_filter.evaluate(doc)
        assert passed is True


@pytest.mark.unit
class TestDeduplication:
    """Test exact and near deduplication."""

    def test_exact_duplicate_detected(self):
        dedup = Deduplicator(exact_dedup=True, near_dedup=False)
        doc1 = Document(
            text="The golden sun set behind the quiet mountains in the far distance.",
            source="s1",
            domain=Domain.STORIES_LITERATURE,
        )
        doc2 = Document(
            text="The golden sun set behind the quiet mountains in the far distance.",
            source="s2",
            domain=Domain.STORIES_LITERATURE,
        )
        assert dedup.is_duplicate(doc1) is False
        assert dedup.is_duplicate(doc2) is True

    def test_near_duplicate_detected_by_minhash(self):
        dedup = Deduplicator(
            exact_dedup=False,
            near_dedup=True,
            minhash_threshold=0.75,
            num_perm=128,
        )
        doc1 = Document(
            text="A quick brown fox jumps over the lazy dog in the sunny meadow on Monday afternoon.",
            source="s1",
            domain=Domain.STORIES_LITERATURE,
        )
        # 1 word changed: "Monday" -> "Tuesday"
        doc2 = Document(
            text="A quick brown fox jumps over the lazy dog in the sunny meadow on Tuesday afternoon.",
            source="s2",
            domain=Domain.STORIES_LITERATURE,
        )
        assert dedup.is_duplicate(doc1) is False
        assert dedup.is_duplicate(doc2) is True

    def test_distinct_documents_not_flagged(self):
        dedup = Deduplicator(exact_dedup=True, near_dedup=True)
        doc1 = Document(
            text="Machine learning models are trained by optimizing empirical risk over large datasets.",
            source="s1",
            domain=Domain.GENERAL_KNOWLEDGE,
        )
        doc2 = Document(
            text="Baking sourdough bread requires flour, water, salt, wild yeast, and patience.",
            source="s2",
            domain=Domain.GENERAL_KNOWLEDGE,
        )
        assert dedup.is_duplicate(doc1) is False
        assert dedup.is_duplicate(doc2) is False


@pytest.mark.unit
class TestPipelineStream:
    """Test full stream filtering and cleaning pipeline."""

    def test_filter_and_clean_stream(self):
        docs = [
            # Valid doc
            Document(
                text="The solar system consists of the Sun and eight planets orbiting around it.",
                source="test",
                domain=Domain.GENERAL_KNOWLEDGE,
            ),
            # Invalid doc (too short)
            Document(
                text="Too short.",
                source="test",
                domain=Domain.GENERAL_KNOWLEDGE,
            ),
            # Exact duplicate of doc 1
            Document(
                text="The solar system consists of the Sun and eight planets orbiting around it.",
                source="test",
                domain=Domain.GENERAL_KNOWLEDGE,
            ),
            # Valid doc 2
            Document(
                text="Python is an interpreted, high-level and general-purpose programming language.",
                source="test",
                domain=Domain.GENERAL_KNOWLEDGE,
            ),
        ]

        cfg = FilterConfig(min_chars=25, min_words=5)
        filtered = list(filter_and_clean_stream(docs, filter_config=cfg, dedup=True))

        assert len(filtered) == 2
        texts = [d.text for d in filtered]
        assert "The solar system consists of the Sun and eight planets orbiting around it." in texts
        assert "Python is an interpreted, high-level and general-purpose programming language." in texts
