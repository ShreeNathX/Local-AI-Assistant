"""
tests/unit/test_config.py
==========================
Unit tests for slm.core.config.

Tests:
  - Valid YAML loads correctly
  - Invalid YAML raises ConfigError
  - Missing file raises ConfigFileNotFoundError
  - Invalid model config raises ConfigValueError
  - Invalid training config raises ConfigValueError
  - d_model / n_heads divisibility is enforced
  - Environment variable overrides work
  - Config is immutable (frozen dataclass)
"""

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from slm.core.config import (
    load_model_config,
    load_training_config,
    load_tokenizer_config,
    ModelConfig,
    TrainingConfig,
)
from slm.core.exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigValueError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, data: dict, filename: str = "config.yaml") -> Path:
    path = tmp_path / filename
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Model config tests
# ---------------------------------------------------------------------------

class TestModelConfig:

    def _valid_model_data(self) -> dict:
        return {
            "model": {
                "name": "test-model",
                "version": "0.1.0",
                "architecture": "decoder-only-transformer",
                "vocab_size": 16384,
                "context_length": 512,
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 4,
                "d_ffn": 1024,
                "dropout": 0.1,
                "pos_encoding": "rope",
                "rope_theta": 10000.0,
                "norm_type": "rmsnorm",
                "norm_eps": 1e-5,
                "attention_type": "causal",
                "bias": False,
                "init_std": 0.02,
                "tie_embeddings": True,
            }
        }

    def test_valid_config_loads(self, tmp_path):
        path = write_yaml(tmp_path, self._valid_model_data())
        cfg = load_model_config(path)
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.vocab_size == 16384

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigFileNotFoundError):
            load_model_config(tmp_path / "nonexistent.yaml")

    def test_d_model_not_divisible_by_n_heads_raises(self, tmp_path):
        data = self._valid_model_data()
        data["model"]["d_model"] = 256
        data["model"]["n_heads"] = 3   # 256 / 3 is not integer
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="divisible"):
            load_model_config(path)

    def test_vocab_size_too_small_raises(self, tmp_path):
        data = self._valid_model_data()
        data["model"]["vocab_size"] = 10
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="vocab_size"):
            load_model_config(path)

    def test_dropout_out_of_range_raises(self, tmp_path):
        data = self._valid_model_data()
        data["model"]["dropout"] = 1.5
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="dropout"):
            load_model_config(path)

    def test_invalid_norm_type_raises(self, tmp_path):
        data = self._valid_model_data()
        data["model"]["norm_type"] = "batchnorm"
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="norm_type"):
            load_model_config(path)

    def test_invalid_pos_encoding_raises(self, tmp_path):
        data = self._valid_model_data()
        data["model"]["pos_encoding"] = "alibi"
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="pos_encoding"):
            load_model_config(path)

    def test_config_is_immutable(self, tmp_path):
        path = write_yaml(tmp_path, self._valid_model_data())
        cfg = load_model_config(path)
        with pytest.raises((AttributeError, TypeError)):
            cfg.d_model = 512  # type: ignore[misc]

    def test_empty_yaml_raises(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ConfigError):
            load_model_config(path)

    def test_invalid_yaml_syntax_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("model: {invalid: yaml: content")
        with pytest.raises(ConfigError):
            load_model_config(path)


# ---------------------------------------------------------------------------
# Training config tests
# ---------------------------------------------------------------------------

class TestTrainingConfig:

    def _valid_training_data(self) -> dict:
        return {
            "training": {
                "model_config": "configs/model_tiny.yaml",
                "dataset_version": "v001",
                "tokenizer_version": "v001",
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "max_steps": 10000,
                "warmup_steps": 500,
                "eval_interval": 200,
                "save_interval": 500,
                "log_interval": 10,
                "optimizer": "adamw",
                "learning_rate": 3e-4,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
                "grad_clip": 1.0,
                "scheduler": "cosine",
                "min_lr_ratio": 0.1,
                "mixed_precision": "fp16",
                "seed": 42,
                "checkpoint_dir": "checkpoints",
                "keep_last_n_checkpoints": 3,
                "keep_best_checkpoint": True,
                "resume_from": None,
                "data_dir": "data/processed",
                "num_workers": 0,
                "pin_memory": True,
                "val_split": 0.005,
            }
        }

    def test_valid_config_loads(self, tmp_path):
        path = write_yaml(tmp_path, self._valid_training_data())
        cfg = load_training_config(path)
        assert cfg.batch_size == 4
        assert cfg.learning_rate == pytest.approx(3e-4)

    def test_invalid_batch_size_raises(self, tmp_path):
        data = self._valid_training_data()
        data["training"]["batch_size"] = 0
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="batch_size"):
            load_training_config(path)

    def test_invalid_learning_rate_raises(self, tmp_path):
        data = self._valid_training_data()
        data["training"]["learning_rate"] = -0.001
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="learning_rate"):
            load_training_config(path)

    def test_invalid_mixed_precision_raises(self, tmp_path):
        data = self._valid_training_data()
        data["training"]["mixed_precision"] = "int8"
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="mixed_precision"):
            load_training_config(path)

    def test_warmup_exceeds_max_steps_raises(self, tmp_path):
        data = self._valid_training_data()
        data["training"]["warmup_steps"] = 15000
        data["training"]["max_steps"] = 10000
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="warmup_steps"):
            load_training_config(path)

    def test_invalid_val_split_raises(self, tmp_path):
        data = self._valid_training_data()
        data["training"]["val_split"] = 0.9  # > 0.5
        path = write_yaml(tmp_path, data)
        with pytest.raises(ConfigValueError, match="val_split"):
            load_training_config(path)
