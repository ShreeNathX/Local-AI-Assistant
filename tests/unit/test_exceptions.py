"""
tests/unit/test_exceptions.py
==============================
Unit tests for slm.core.exceptions.

Tests:
  - All exception classes can be instantiated
  - Exception messages contain component name
  - Exception messages contain suggestion when provided
  - Hierarchy is correct (isinstance checks)
  - ExitCode constants are unique integers
"""

import pytest
from slm.core.exceptions import (
    SLMError,
    ConfigError,
    ConfigValueError,
    ConfigFileNotFoundError,
    DataError,
    DataFileNotFoundError,
    DataCorruptedError,
    DataValidationError,
    TokenizerError,
    TokenizerNotFoundError,
    ModelError,
    CheckpointNotFoundError,
    ParameterCountMismatchError,
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


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class TestSLMError:
    def test_basic_instantiation(self):
        err = SLMError("something failed", component="Test")
        assert "Test" in str(err)
        assert "something failed" in str(err)

    def test_suggestion_in_message(self):
        err = SLMError("failed", component="Test", suggestion="do X instead")
        assert "do X instead" in str(err)

    def test_no_suggestion(self):
        err = SLMError("failed", component="Test")
        assert "→ Fix:" not in str(err)

    def test_attributes(self):
        err = SLMError("msg", component="Comp", suggestion="fix it")
        assert err.message == "msg"
        assert err.component == "Comp"
        assert err.suggestion == "fix it"


# ---------------------------------------------------------------------------
# Config exceptions
# ---------------------------------------------------------------------------

class TestConfigExceptions:
    def test_config_error_is_slm_error(self):
        err = ConfigError("bad config")
        assert isinstance(err, SLMError)

    def test_config_value_error_is_config_error(self):
        err = ConfigValueError("bad value")
        assert isinstance(err, ConfigError)

    def test_config_file_not_found_is_config_error(self):
        err = ConfigFileNotFoundError("no file")
        assert isinstance(err, ConfigError)

    def test_component_label(self):
        err = ConfigError("x")
        assert "Config" in str(err)


# ---------------------------------------------------------------------------
# Data exceptions
# ---------------------------------------------------------------------------

class TestDataExceptions:
    def test_data_error_hierarchy(self):
        assert issubclass(DataFileNotFoundError, DataError)
        assert issubclass(DataCorruptedError, DataError)
        assert issubclass(DataValidationError, DataError)
        assert issubclass(DataError, SLMError)

    def test_component_label(self):
        err = DataError("oops")
        assert "Data" in str(err)


# ---------------------------------------------------------------------------
# Tokenizer exceptions
# ---------------------------------------------------------------------------

class TestTokenizerExceptions:
    def test_tokenizer_not_found(self):
        err = TokenizerNotFoundError("/some/path")
        assert "/some/path" in str(err)
        assert isinstance(err, TokenizerError)
        assert isinstance(err, SLMError)

    def test_suggestion_present(self):
        err = TokenizerNotFoundError("/p")
        assert "train_tokenizer" in err.suggestion


# ---------------------------------------------------------------------------
# Model / checkpoint exceptions
# ---------------------------------------------------------------------------

class TestModelExceptions:
    def test_checkpoint_not_found(self):
        err = CheckpointNotFoundError("/path/to/ckpt")
        assert "/path/to/ckpt" in str(err)
        assert isinstance(err, ModelError)

    def test_parameter_count_mismatch(self):
        err = ParameterCountMismatchError(expected=50_000_000, actual=48_000_000)
        assert "50,000,000" in str(err)
        assert "48,000,000" in str(err)
        assert isinstance(err, ModelError)


# ---------------------------------------------------------------------------
# Training exceptions
# ---------------------------------------------------------------------------

class TestTrainingExceptions:
    def test_nan_loss_error(self):
        err = NaNLossError(step=1500, loss_value=float("nan"))
        assert "1500" in str(err)
        assert isinstance(err, TrainingError)

    def test_gradient_explosion_error(self):
        err = GradientExplosionError(step=100, grad_norm=150.3, clip_value=1.0)
        assert "100" in str(err)
        assert isinstance(err, TrainingError)

    def test_oom_error(self):
        err = OutOfMemoryError(step=200)
        assert "200" in str(err)
        assert isinstance(err, TrainingError)


# ---------------------------------------------------------------------------
# Environment exceptions
# ---------------------------------------------------------------------------

class TestEnvironmentExceptions:
    def test_cuda_not_available(self):
        err = CUDANotAvailableError()
        assert isinstance(err, EnvironmentError)
        assert "CUDA" in str(err)
        assert len(err.suggestion) > 0

    def test_insufficient_vram(self):
        err = InsufficientVRAMError(required_gb=8.0, available_gb=4.0)
        assert "8.0" in str(err)
        assert "4.0" in str(err)
        assert isinstance(err, EnvironmentError)

    def test_insufficient_disk(self):
        err = InsufficientDiskError(required_gb=100.0, available_gb=30.0, path="C:\\")
        assert "100.0" in str(err)
        assert "30.0" in str(err)
        assert "C:\\" in str(err)


# ---------------------------------------------------------------------------
# ExitCode
# ---------------------------------------------------------------------------

class TestExitCode:
    def test_exit_codes_are_integers(self):
        codes = [
            ExitCode.OK,
            ExitCode.GENERAL_ERROR,
            ExitCode.CONFIG_ERROR,
            ExitCode.ENVIRONMENT_ERROR,
            ExitCode.DATA_ERROR,
            ExitCode.MODEL_ERROR,
            ExitCode.TRAINING_ERROR,
            ExitCode.CHECKPOINT_ERROR,
        ]
        assert all(isinstance(c, int) for c in codes)

    def test_exit_codes_are_unique(self):
        codes = [
            ExitCode.OK,
            ExitCode.GENERAL_ERROR,
            ExitCode.CONFIG_ERROR,
            ExitCode.ENVIRONMENT_ERROR,
            ExitCode.DATA_ERROR,
            ExitCode.MODEL_ERROR,
            ExitCode.TRAINING_ERROR,
            ExitCode.CHECKPOINT_ERROR,
        ]
        assert len(codes) == len(set(codes))

    def test_ok_is_zero(self):
        assert ExitCode.OK == 0
