"""
slm.data — Data engineering, collection, cleaning, and preparation pipeline.

Modules
-------
schema      Standard document and dataset manifest schemas.
sources     Corpus source configurations and licensing registry.
collector   Streaming data ingestion and sampling.
filter      Quality filters, heuristics, and toxicity/dedup filters.
"""

from slm.data.schema import (
    DatasetManifest,
    DatasetSplit,
    Document,
    Domain,
    LicenseType,
)
from slm.data.sources import (
    PRETRAINING_SOURCES,
    SourceConfig,
    get_source_config,
    list_available_sources,
    validate_mixture_weights,
)
from slm.data.collector import (
    BaseCollector,
    LocalFileCollector,
    StreamingHFCollector,
    SyntheticOverfitCollector,
)
from slm.data.filter import (
    Deduplicator,
    FilterConfig,
    QualityFilter,
    clean_text,
    compute_repetition_ratio,
    filter_and_clean_stream,
)

__all__ = [
    "Document",
    "Domain",
    "LicenseType",
    "DatasetSplit",
    "DatasetManifest",
    "SourceConfig",
    "PRETRAINING_SOURCES",
    "get_source_config",
    "list_available_sources",
    "validate_mixture_weights",
    "BaseCollector",
    "LocalFileCollector",
    "StreamingHFCollector",
    "SyntheticOverfitCollector",
    "FilterConfig",
    "QualityFilter",
    "Deduplicator",
    "clean_text",
    "compute_repetition_ratio",
    "filter_and_clean_stream",
]
