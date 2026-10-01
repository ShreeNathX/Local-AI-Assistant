"""
slm.core — Core infrastructure for the SLM project.

Modules
-------
config      Config loading, validation, and dataclasses.
logging     Structured, per-component logging (loguru-backed).
exceptions  All project-specific exception classes.
"""

from slm.core.config import (
    load_config,
    load_model_config,
    load_training_config,
    load_tokenizer_config,
    ProjectConfig,
    ModelConfig,
    TrainingConfig,
    TokenizerConfig,
)
from slm.core.logging import get_logger, configure_logging
from slm.core.exceptions import (
    SLMError,
    ConfigError,
    ConfigValueError,
    ConfigFileNotFoundError,
    DataError,
    DataFileNotFoundError,
    DataCorruptedError,
    DataValidationError,
    DataCollectionError,
    TokenizerError,

    TokenizerNotFoundError,
    ModelError,
    CheckpointNotFoundError,
    TrainingError,
    NaNLossError,
    GradientExplosionError,
    OutOfMemoryError,
    EnvironmentError,
    CUDANotAvailableError,
    InsufficientVRAMError,
    InsufficientDiskError,
    ExitCode,
)

__all__ = [
    # Config
    "load_config",
    "load_model_config",
    "load_training_config",
    "load_tokenizer_config",
    "ProjectConfig",
    "ModelConfig",
    "TrainingConfig",
    "TokenizerConfig",
    # Logging
    "get_logger",
    "configure_logging",
    # Exceptions
    "SLMError",
    "ConfigError",
    "ConfigValueError",
    "ConfigFileNotFoundError",
    "DataError",
    "DataFileNotFoundError",
    "DataCorruptedError",
    "DataValidationError",
    "DataCollectionError",
    "TokenizerError",

    "TokenizerNotFoundError",
    "ModelError",
    "CheckpointNotFoundError",
    "TrainingError",
    "NaNLossError",
    "GradientExplosionError",
    "OutOfMemoryError",
    "EnvironmentError",
    "CUDANotAvailableError",
    "InsufficientVRAMError",
    "InsufficientDiskError",
    "ExitCode",
]
