"""
slm/core/exceptions.py
======================
All project-specific exceptions.

Design principles:
  - Every exception carries: what, where, why, how-to-fix.
  - No bare ``except Exception`` in the codebase.
  - Exceptions are hierarchical so callers can catch at the
    right level of specificity.
  - Exit codes are defined for CLI scripts.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class SLMError(Exception):
    """
    Root exception for the SLM project.

    All project exceptions inherit from this class so callers
    can catch ``SLMError`` to handle any project error.

    Attributes
    ----------
    message : str
        Human-readable description of what went wrong.
    component : str
        Which component raised the error (e.g. "DataLoader", "Tokenizer").
    suggestion : str
        How the user can fix the issue.
    """

    def __init__(
        self,
        message: str,
        component: str = "SLM",
        suggestion: str = "",
    ) -> None:
        self.message   = message
        self.component = component
        self.suggestion = suggestion
        full = f"[{component}] {message}"
        if suggestion:
            full += f"\n  → Fix: {suggestion}"
        super().__init__(full)


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigError(SLMError):
    """Raised when a configuration file is missing, invalid, or contains
    an incompatible value."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message, component="Config", suggestion=suggestion)


class ConfigValueError(ConfigError):
    """Raised when a configuration value is out of range or the wrong type."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when a required configuration file does not exist."""


# ---------------------------------------------------------------------------
# Environment / hardware errors
# ---------------------------------------------------------------------------

class EnvironmentError(SLMError):  # noqa: A001 (shadows builtin intentionally)
    """Raised when the runtime environment does not meet requirements
    (e.g., wrong Python version, missing CUDA, insufficient VRAM)."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message, component="Environment", suggestion=suggestion)


class CUDANotAvailableError(EnvironmentError):
    """Raised when CUDA is required but not available."""

    def __init__(self, suggestion: str = "") -> None:
        super().__init__(
            "CUDA is not available.",
            suggestion=suggestion or (
                "Install PyTorch with CUDA support: "
                "pip install torch==2.5.1+cu121 "
                "--index-url https://download.pytorch.org/whl/cu121"
            ),
        )


class InsufficientVRAMError(EnvironmentError):
    """Raised when GPU VRAM is insufficient for the requested model/batch."""

    def __init__(self, required_gb: float, available_gb: float) -> None:
        super().__init__(
            f"Insufficient GPU VRAM: need {required_gb:.1f} GB, "
            f"found {available_gb:.1f} GB.",
            suggestion=(
                "Reduce batch_size, use gradient_accumulation_steps, "
                "enable mixed_precision=fp16, or use a smaller model config."
            ),
        )


class InsufficientDiskError(EnvironmentError):
    """Raised when free disk space is below the required minimum."""

    def __init__(self, required_gb: float, available_gb: float, path: str = "") -> None:
        loc = f" on {path}" if path else ""
        super().__init__(
            f"Insufficient disk space{loc}: "
            f"need {required_gb:.1f} GB, found {available_gb:.1f} GB.",
            suggestion="Free up disk space or move the project to a larger drive.",
        )


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------

class DataError(SLMError):
    """Base class for data-related errors."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message, component="Data", suggestion=suggestion)


class DataFileNotFoundError(DataError):
    """Raised when a required data file does not exist."""


class DataCorruptedError(DataError):
    """Raised when a data file exists but its content is corrupted or
    does not match its expected checksum / schema."""


class DataValidationError(DataError):
    """Raised when a dataset fails validation thresholds (e.g., too many
    invalid records, wrong encoding, empty documents)."""


class DataVersionError(DataError):
    """Raised when dataset version tags are missing or incompatible."""


class DataCollectionError(DataError):
    """Raised when data collection or streaming ingestion fails."""



# ---------------------------------------------------------------------------
# Tokenizer errors
# ---------------------------------------------------------------------------

class TokenizerError(SLMError):
    """Base class for tokenizer-related errors."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message, component="Tokenizer", suggestion=suggestion)


class TokenizerNotFoundError(TokenizerError):
    """Raised when the tokenizer directory or files do not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Tokenizer not found at: {path}",
            suggestion="Train the tokenizer first: python scripts/train_tokenizer.py",
        )


class TokenizerVersionMismatchError(TokenizerError):
    """Raised when the loaded tokenizer version does not match the model's
    expected tokenizer version (would silently corrupt inference)."""


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------

class ModelError(SLMError):
    """Base class for model-related errors."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message, component="Model", suggestion=suggestion)


class ModelConfigError(ModelError):
    """Raised when model configuration values are invalid or inconsistent."""


class CheckpointNotFoundError(ModelError):
    """Raised when a checkpoint path is specified but does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Checkpoint not found: {path}",
            suggestion=(
                "Verify the checkpoint path in configs/training.yaml "
                "or remove resume_from to start fresh."
            ),
        )


class CheckpointCorruptedError(ModelError):
    """Raised when a checkpoint file exists but cannot be loaded."""


class ParameterCountMismatchError(ModelError):
    """Raised when the actual model parameter count differs significantly
    from the expected count (indicates architecture bug)."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"Parameter count mismatch: expected ~{expected:,}, got {actual:,}.",
            suggestion=(
                "Check the model configuration (n_layers, d_model, n_heads). "
                "Verify no layers were accidentally duplicated or omitted."
            ),
        )


# ---------------------------------------------------------------------------
# Training errors
# ---------------------------------------------------------------------------

class TrainingError(SLMError):
    """Base class for training-related errors."""

    def __init__(self, message: str, suggestion: str = "") -> None:
        super().__init__(message, component="Training", suggestion=suggestion)


class NaNLossError(TrainingError):
    """
    Raised immediately when NaN or Inf loss is detected.
    Training MUST stop — do not continue from a corrupted state.
    """

    def __init__(self, step: int, loss_value: float) -> None:
        super().__init__(
            f"NaN/Inf loss detected at step {step}: loss={loss_value}",
            suggestion=(
                "1. Reduce learning_rate. "
                "2. Increase warmup_steps. "
                "3. Lower grad_clip threshold. "
                "4. Check dataset for corrupted batches (all-zero, all-pad). "
                "5. Resume from the last valid checkpoint."
            ),
        )


class GradientExplosionError(TrainingError):
    """Raised when gradient norm exceeds a critical threshold after clipping."""

    def __init__(self, step: int, grad_norm: float, clip_value: float) -> None:
        super().__init__(
            f"Gradient explosion at step {step}: "
            f"grad_norm={grad_norm:.2f}, clip={clip_value}",
            suggestion=(
                "Reduce learning_rate or increase warmup_steps. "
                "Consider reducing the model size or sequence length."
            ),
        )


class OutOfMemoryError(TrainingError):
    """Raised when a CUDA out-of-memory error is caught during training."""

    def __init__(self, step: int, suggestion: str = "") -> None:
        super().__init__(
            f"CUDA out-of-memory at step {step}.",
            suggestion=suggestion or (
                "Reduce batch_size. Increase gradient_accumulation_steps. "
                "Enable mixed_precision=fp16. Reduce context_length."
            ),
        )


# ---------------------------------------------------------------------------
# Tool / API errors
# ---------------------------------------------------------------------------

class ToolError(SLMError):
    """Base class for tool-calling errors."""

    def __init__(self, tool_name: str, message: str, suggestion: str = "") -> None:
        super().__init__(
            f"Tool '{tool_name}': {message}",
            component="Tools",
            suggestion=suggestion,
        )


class ToolTimeoutError(ToolError):
    """Raised when a tool call exceeds its allowed time budget."""


class ToolPermissionError(ToolError):
    """Raised when a tool call is attempted without proper authorization."""


class WebSearchError(ToolError):
    """Raised when a web search fails (network, rate limit, no results)."""

    def __init__(self, message: str) -> None:
        super().__init__(
            tool_name="WebSearch",
            message=message,
            suggestion=(
                "Check internet connection, API key, and rate limits. "
                "The system will fall back to in-context SLM knowledge."
            ),
        )


# ---------------------------------------------------------------------------
# Exit codes (for CLI scripts)
# ---------------------------------------------------------------------------

class ExitCode:
    """Standard exit codes for CLI scripts."""
    OK                  = 0
    GENERAL_ERROR       = 1
    CONFIG_ERROR        = 2
    ENVIRONMENT_ERROR   = 3
    DATA_ERROR          = 4
    MODEL_ERROR         = 5
    TRAINING_ERROR      = 6
    CHECKPOINT_ERROR    = 7
