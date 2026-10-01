"""
slm.data.schema — Data models and schemas for pre-training and fine-tuning datasets.

This module provides strict Pydantic schemas for all text documents, dataset
manifests, and data splits used throughout the SLM data engineering pipeline.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Domain(str, Enum):
    """Domains of pre-training data."""
    GENERAL_KNOWLEDGE = "general_knowledge"
    STORIES_LITERATURE = "stories_literature"
    CODE_PYTHON = "code_python"
    CODE_GENERAL = "code_general"
    MATH_REASONING = "math_reasoning"
    CONVERSATION_QA = "conversation_qa"
    INSTRUCTION = "instruction"
    CUSTOM = "custom"


class LicenseType(str, Enum):
    """Allowed open-source licenses for data ingestion."""
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    BSD_3_CLAUSE = "BSD-3-Clause"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC0 = "CC0-1.0"
    PUBLIC_DOMAIN = "Public-Domain"
    OPEN_RAIL = "OpenRAIL"
    CUSTOM_PERMISSIVE = "Custom-Permissive"


class Document(BaseModel):
    """
    Standardized schema for a single text document in the SLM corpus.

    Attributes
    ----------
    id : str
        Unique document identifier (UUID or content hash).
    text : str
        Raw or cleaned text content.
    source : str
        Name of the data source (e.g. 'fineweb_edu', 'tinystories', 'starcoder').
    domain : Domain
        Categorical domain for mixture weighting.
    license : LicenseType
        License governing the text.
    language : str
        ISO 639-1 language code (default 'en').
    metadata : Dict[str, Any]
        Source-specific metadata (e.g. url, author, quality_score, score).
    token_count : Optional[int]
        Estimated or measured token count.
    created_at : str
        ISO timestamp when document record was created.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1, description="Document body text")
    source: str = Field(..., min_length=1, description="Source name identifier")
    domain: Domain = Field(..., description="Corpus domain category")
    license: LicenseType = Field(default=LicenseType.CC0, description="Data license")
    language: str = Field(default="en", max_length=10, description="Language code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    token_count: Optional[int] = Field(default=None, ge=0, description="Token count if known")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp"
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Document text cannot be empty or pure whitespace")
        return v

    def content_hash(self) -> str:
        """Calculate SHA-256 hash of the normalized document text."""
        return hashlib.sha256(self.text.strip().encode("utf-8")).hexdigest()

    def byte_size(self) -> int:
        """Return size in bytes of the UTF-8 text."""
        return len(self.text.encode("utf-8"))

    def to_json_line(self) -> str:
        """Serialize document as a single-line JSON string (JSONL format)."""
        return self.model_dump_json()

    @classmethod
    def from_json_line(cls, line: str) -> Document:
        """Parse a document from a JSON string."""
        return cls.model_validate_json(line.strip())


class DatasetSplit(str, Enum):
    """Dataset partition types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DatasetManifest(BaseModel):
    """
    Manifest describing a processed dataset artifact or collection.

    Used for provenance, reproducibility, and tracking data volumes across splits.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Dataset or mixture name")
    version: str = Field(default="1.0.0", description="Semantic version")
    split: DatasetSplit = Field(default=DatasetSplit.TRAIN, description="Target split")
    domain_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of total documents by domain"
    )
    document_count: int = Field(default=0, ge=0, description="Total number of documents")
    total_bytes: int = Field(default=0, ge=0, description="Total UTF-8 bytes")
    total_tokens: Optional[int] = Field(default=None, ge=0, description="Total token count")
    content_sha256: str = Field(default="", description="Digest of dataset contents")
    source_licenses: List[LicenseType] = Field(default_factory=list, description="All licenses present")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation timestamp"
    )

    def save_json(self, path: Path | str) -> None:
        """Save manifest to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, path: Path | str) -> DatasetManifest:
        """Load manifest from a JSON file."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"Manifest file not found: {target}")
        with open(target, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
