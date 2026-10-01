"""
tests/unit/test_data_schema.py
==============================
Unit tests for slm.data schemas, sources, and collectors.
"""

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from slm.data.schema import (
    DatasetManifest,
    DatasetSplit,
    Document,
    Domain,
    LicenseType,
)
from slm.data.sources import (
    PRETRAINING_SOURCES,
    get_source_config,
    list_available_sources,
    validate_mixture_weights,
)
from slm.data.collector import (
    LocalFileCollector,
    SyntheticOverfitCollector,
)


@pytest.mark.unit
class TestDocumentSchema:
    """Test suite for Document Pydantic model."""

    def test_valid_document_creation(self):
        doc = Document(
            text="This is a test document.",
            source="test_source",
            domain=Domain.GENERAL_KNOWLEDGE,
            license=LicenseType.MIT,
        )
        assert doc.text == "This is a test document."
        assert doc.source == "test_source"
        assert doc.domain == Domain.GENERAL_KNOWLEDGE
        assert doc.license == LicenseType.MIT
        assert doc.language == "en"
        assert len(doc.id) > 0

    def test_empty_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Document(
                text="",
                source="test",
                domain=Domain.GENERAL_KNOWLEDGE,
            )

    def test_whitespace_only_text_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Document(
                text="   \n\t  ",
                source="test",
                domain=Domain.GENERAL_KNOWLEDGE,
            )

    def test_json_line_roundtrip(self):
        doc = Document(
            text="Roundtrip test string with unicode: Café, naïve, 123.",
            source="roundtrip_source",
            domain=Domain.CODE_PYTHON,
            license=LicenseType.APACHE_2_0,
            metadata={"lines": 1, "test": True},
        )
        line = doc.to_json_line()
        reconstructed = Document.from_json_line(line)
        assert reconstructed.id == doc.id
        assert reconstructed.text == doc.text
        assert reconstructed.source == doc.source
        assert reconstructed.domain == doc.domain
        assert reconstructed.metadata == doc.metadata

    def test_content_hash(self):
        doc1 = Document(
            text="Identical content.",
            source="s1",
            domain=Domain.GENERAL_KNOWLEDGE,
        )
        doc2 = Document(
            text="Identical content.",
            source="s2",
            domain=Domain.STORIES_LITERATURE,
        )
        assert doc1.content_hash() == doc2.content_hash()

    def test_byte_size(self):
        doc = Document(
            text="Hello World",
            source="test",
            domain=Domain.GENERAL_KNOWLEDGE,
        )
        assert doc.byte_size() == 11


@pytest.mark.unit
class TestDatasetManifest:
    """Test suite for DatasetManifest."""

    def test_manifest_roundtrip_save_load(self, tmp_path: Path):
        manifest = DatasetManifest(
            name="slm_pretrain_v1",
            version="1.0.0",
            split=DatasetSplit.TRAIN,
            document_count=5000,
            total_bytes=1024000,
            domain_distribution={"general_knowledge": 0.5, "code_python": 0.5},
            source_licenses=[LicenseType.MIT, LicenseType.APACHE_2_0],
        )
        manifest_file = tmp_path / "manifest.json"
        manifest.save_json(manifest_file)
        assert manifest_file.exists()

        loaded = DatasetManifest.load_json(manifest_file)
        assert loaded.name == manifest.name
        assert loaded.document_count == 5000
        assert loaded.split == DatasetSplit.TRAIN


@pytest.mark.unit
class TestDataSources:
    """Test suite for pre-training data sources."""

    def test_sources_registered(self):
        sources = list_available_sources()
        assert "tinystories" in sources
        assert "code_python" in sources
        assert "math_reasoning" in sources

    def test_get_valid_source(self):
        cfg = get_source_config("tinystories")
        assert cfg.name == "tinystories"
        assert cfg.domain == Domain.STORIES_LITERATURE

    def test_get_invalid_source_raises(self):
        with pytest.raises(KeyError):
            get_source_config("non_existent_source_xyz")

    def test_mixture_weights_sum_to_one(self):
        all_sources = list(PRETRAINING_SOURCES.values())
        assert validate_mixture_weights(all_sources)


@pytest.mark.unit
class TestCollectors:
    """Test suite for collectors."""

    def test_synthetic_overfit_collector(self, tmp_path: Path):
        collector = SyntheticOverfitCollector(repeats=2)
        docs = list(collector.stream_documents())
        # 5 stories + 5 code + 5 math + 5 qa = 20 per repeat * 2 repeats = 40 docs
        assert len(docs) == 40
        assert all(isinstance(d, Document) for d in docs)

        domains = {d.domain for d in docs}
        assert Domain.STORIES_LITERATURE in domains
        assert Domain.CODE_PYTHON in domains
        assert Domain.MATH_REASONING in domains
        assert Domain.CONVERSATION_QA in domains

        # Test save_raw
        output_file = tmp_path / "synthetic_raw.jsonl"
        written = collector.save_raw(output_file, max_docs=10)
        assert written == 10
        assert output_file.exists()

    def test_local_file_collector(self, tmp_path: Path):
        doc1_file = tmp_path / "doc1.txt"
        doc1_file.write_text("This is local file 1 text.", encoding="utf-8")
        doc2_file = tmp_path / "doc2.txt"
        doc2_file.write_text("This is local file 2 text.", encoding="utf-8")

        collector = LocalFileCollector(
            path=tmp_path,
            source_name="test_local",
            domain=Domain.GENERAL_KNOWLEDGE,
            glob_pattern="*.txt",
        )
        docs = list(collector.stream_documents())
        assert len(docs) == 2
        texts = {d.text for d in docs}
        assert "This is local file 1 text." in texts
        assert "This is local file 2 text." in texts
