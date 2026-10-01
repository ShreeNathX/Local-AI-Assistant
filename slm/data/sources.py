"""
slm.data.sources — Pre-training corpus source registry and configurations.

Defines vetted, high-quality, openly licensed datasets and their mixture
weights for training the SLM model.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from slm.data.schema import Domain, LicenseType


class SourceConfig(BaseModel):
    """
    Configuration for a single pre-training data source.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Unique source identifier")
    hf_dataset_id: Optional[str] = Field(None, description="Hugging Face hub dataset ID")
    hf_subset: Optional[str] = Field(None, description="Subset / configuration name")
    hf_split: str = Field(default="train", description="Dataset split to consume")
    text_column: str = Field(default="text", description="Name of column containing body text")
    domain: Domain = Field(..., description="Corpus domain category")
    license: LicenseType = Field(..., description="Governing license")
    mixture_weight: float = Field(..., ge=0.0, le=1.0, description="Sampling weight in the corpus mixture")
    target_docs: Optional[int] = Field(None, ge=1, description="Max documents to ingest from source")
    description: str = Field(..., description="Human-readable description of source")


# ---------------------------------------------------------------------------
# Standard Pre-training Mixture Registry (Balanced for ~50M parameter SLM)
# ---------------------------------------------------------------------------

PRETRAINING_SOURCES: Dict[str, SourceConfig] = {
    "tinystories": SourceConfig(
        name="tinystories",
        hf_dataset_id="roneneldan/TinyStories",
        hf_subset=None,
        hf_split="train",
        text_column="text",
        domain=Domain.STORIES_LITERATURE,
        license=LicenseType.CC_BY_4_0,
        mixture_weight=0.25,
        target_docs=100_000,
        description="Synthetic stories with clean grammar, reasoning, and consistent vocabulary. Ideal for SLMs."
    ),
    "fineweb_edu": SourceConfig(
        name="fineweb_edu",
        hf_dataset_id="HuggingFaceFW/fineweb-edu",
        hf_subset="sample-10BT",
        hf_split="train",
        text_column="text",
        domain=Domain.GENERAL_KNOWLEDGE,
        license=LicenseType.OPEN_RAIL,
        mixture_weight=0.30,
        target_docs=150_000,
        description="High-quality educational web content filtered by educational quality classifier."
    ),
    "wikipedia_en": SourceConfig(
        name="wikipedia_en",
        hf_dataset_id="wikimedia/wikipedia",
        hf_subset="20231101.en",
        hf_split="train",
        text_column="text",
        domain=Domain.GENERAL_KNOWLEDGE,
        license=LicenseType.CC_BY_SA_4_0,
        mixture_weight=0.15,
        target_docs=50_000,
        description="English Wikipedia articles covering general encyclopedic knowledge."
    ),
    "code_python": SourceConfig(
        name="code_python",
        hf_dataset_id="bigcode/the-stack-smol",
        hf_subset="data/python",
        hf_split="train",
        text_column="content",
        domain=Domain.CODE_PYTHON,
        license=LicenseType.APACHE_2_0,
        mixture_weight=0.15,
        target_docs=50_000,
        description="Permissively licensed Python source code with docstrings and clean algorithms."
    ),
    "math_reasoning": SourceConfig(
        name="math_reasoning",
        hf_dataset_id="openai/gsm8k",
        hf_subset="main",
        hf_split="train",
        text_column="question",
        domain=Domain.MATH_REASONING,
        license=LicenseType.MIT,
        mixture_weight=0.10,
        target_docs=25_000,
        description="Grade-school math problems with chain-of-thought solutions for basic arithmetic and reasoning."
    ),
    "dialogue_qa": SourceConfig(
        name="dialogue_qa",
        hf_dataset_id="Open-Orca/OpenOrca",
        hf_subset=None,
        hf_split="train",
        text_column="response",
        domain=Domain.CONVERSATION_QA,
        license=LicenseType.MIT,
        mixture_weight=0.05,
        target_docs=20_000,
        description="High-quality instruction and explanation pairs for natural conversation."
    ),
}


def get_source_config(name: str) -> SourceConfig:
    """Retrieve a vetted source configuration by name."""
    if name not in PRETRAINING_SOURCES:
        valid = ", ".join(PRETRAINING_SOURCES.keys())
        raise KeyError(f"Unknown data source '{name}'. Available sources: {valid}")
    return PRETRAINING_SOURCES[name]


def list_available_sources() -> List[str]:
    """Return all registered source names."""
    return list(PRETRAINING_SOURCES.keys())


def validate_mixture_weights(sources: List[SourceConfig], tolerance: float = 1e-4) -> bool:
    """Ensure mixture weights across active sources sum to approximately 1.0."""
    total = sum(s.mixture_weight for s in sources)
    return abs(total - 1.0) <= tolerance
