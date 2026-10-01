"""
slm/core/config.py
==================
Configuration loading, validation, and management.

Rules:
  - All configuration is loaded from YAML files.
  - Environment variables prefixed with SLM_ override YAML values.
  - Invalid or missing required values raise ConfigError immediately.
  - No silent defaults for critical values (vocab_size, context_length, etc.).
  - Config is immutable after load (frozen dataclass).

Usage:
    from slm.core.config import load_config, ProjectConfig

    cfg = load_config("configs/project.yaml")
    print(cfg.paths.checkpoint_dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from slm.core.exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigValueError,
)
from slm.core.logging import get_logger

logger = get_logger("Config")


# ---------------------------------------------------------------------------
# Dataclasses — typed, validated configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathsConfig:
    data_root:       str = "data"
    raw_data:        str = "data/raw"
    interim_data:    str = "data/interim"
    processed_data:  str = "data/processed"
    validation_dir:  str = "data/validation"
    licenses_dir:    str = "data/licenses"
    tokenizer_dir:   str = "tokenizer/artifacts"
    checkpoint_dir:  str = "checkpoints"
    logs_dir:        str = "logs"
    eval_dir:        str = "evaluation/results"


@dataclass(frozen=True)
class LoggingConfig:
    level:     str = "INFO"
    format:    str = "text"
    rotation:  str = "10 MB"
    retention: str = "30 days"


@dataclass(frozen=True)
class HardwareConfig:
    device:      str  = "auto"
    num_workers: int  = 0
    pin_memory:  bool = True


@dataclass(frozen=True)
class ProjectConfig:
    name:        str = "slm-project"
    version:     str = "0.1.0"
    description: str = ""
    author:      str = ""
    seed:        int = 42

    paths:    PathsConfig    = field(default_factory=PathsConfig)
    logging:  LoggingConfig  = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    # Project root — resolved at load time
    root: str = ""


@dataclass(frozen=True)
class ModelConfig:
    name:             str   = "slm-tiny"
    version:          str   = "0.1.0"
    architecture:     str   = "decoder-only-transformer"
    vocab_size:       int   = 32000
    context_length:   int   = 512
    d_model:          int   = 256
    n_layers:         int   = 4
    n_heads:          int   = 4
    d_ffn:            int   = 1024
    dropout:          float = 0.1
    pos_encoding:     str   = "rope"
    rope_theta:       float = 10000.0
    norm_type:        str   = "rmsnorm"
    norm_eps:         float = 1e-5
    attention_type:   str   = "causal"
    bias:             bool  = False
    init_std:         float = 0.02
    tie_embeddings:   bool  = True


@dataclass(frozen=True)
class TrainingConfig:
    model_config:               str   = "configs/model_50m.yaml"
    dataset_version:            str   = "v001"
    tokenizer_version:          str   = "v001"
    batch_size:                 int   = 4
    gradient_accumulation_steps: int  = 8
    max_steps:                  int   = 100000
    warmup_steps:               int   = 1000
    eval_interval:              int   = 500
    save_interval:              int   = 1000
    log_interval:               int   = 10
    optimizer:                  str   = "adamw"
    learning_rate:              float = 3e-4
    weight_decay:               float = 0.1
    beta1:                      float = 0.9
    beta2:                      float = 0.95
    eps:                        float = 1e-8
    grad_clip:                  float = 1.0
    scheduler:                  str   = "cosine"
    min_lr_ratio:               float = 0.1
    mixed_precision:            str   = "fp16"
    seed:                       int   = 42
    checkpoint_dir:             str   = "checkpoints"
    keep_last_n_checkpoints:    int   = 3
    keep_best_checkpoint:       bool  = True
    resume_from:                str | None = None
    data_dir:                   str   = "data/processed"
    num_workers:                int   = 0
    pin_memory:                 bool  = True
    val_split:                  float = 0.005


@dataclass(frozen=True)
class TokenizerConfig:
    version:        str       = "v001"
    algorithm:      str       = "bpe"
    vocab_size:     int       = 32000
    unk_token:      str       = "[UNK]"
    bos_token:      str       = "[BOS]"
    eos_token:      str       = "[EOS]"
    pad_token:      str       = "[PAD]"
    sep_token:      str       = "[SEP]"
    mask_token:     str       = "[MASK]"
    training_files: list[str] = field(default_factory=list)
    min_frequency:  int       = 2
    max_token_length: int     = 15
    output_dir:     str       = "tokenizer/artifacts/v001"


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its content as a dict.

    Raises
    ------
    ConfigFileNotFoundError
        If the file does not exist.
    ConfigError
        If the file cannot be parsed as valid YAML.
    """
    if not path.exists():
        raise ConfigFileNotFoundError(
            f"Configuration file not found: {path}",
            suggestion=f"Create the file or copy from the provided template.",
        )
    if not path.is_file():
        raise ConfigError(
            f"Configuration path is not a file: {path}",
            suggestion="Provide a path to a .yaml file, not a directory.",
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse YAML file: {path}\n  {exc}",
            suggestion="Validate the YAML syntax at https://yaml-online-parser.appspot.com/",
        ) from exc

    if data is None:
        raise ConfigError(
            f"Configuration file is empty: {path}",
            suggestion="Populate the file with required configuration keys.",
        )
    if not isinstance(data, dict):
        raise ConfigError(
            f"Configuration file must be a YAML mapping (dict): {path}",
            suggestion="Ensure the top-level structure is key: value pairs.",
        )
    return data


def _apply_env_overrides(data: dict[str, Any], prefix: str = "SLM_") -> dict[str, Any]:
    """
    Override config values from environment variables.

    Convention: SLM_SEED=123 overrides config["seed"] = 123.
    Nested keys are not supported via env — only top-level scalars.
    """
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            # Attempt type coercion based on existing value type
            existing = data.get(config_key)
            try:
                if isinstance(existing, bool):
                    data[config_key] = value.lower() in ("1", "true", "yes")
                elif isinstance(existing, int):
                    data[config_key] = int(value)
                elif isinstance(existing, float):
                    data[config_key] = float(value)
                else:
                    data[config_key] = value
                logger.debug(
                    f"Env override: {key} → config[{config_key}] = {data[config_key]}"
                )
            except (ValueError, TypeError):
                raise ConfigValueError(
                    f"Environment variable {key}={value!r} cannot be coerced "
                    f"to type {type(existing).__name__}.",
                    suggestion=f"Ensure {key} is set to a valid {type(existing).__name__} value.",
                )
    return data


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_model_config(cfg: ModelConfig) -> None:
    """Raise ConfigValueError if the model config contains invalid values."""
    errors: list[str] = []

    if cfg.d_model % cfg.n_heads != 0:
        errors.append(
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads}). "
            f"head_dim = {cfg.d_model}/{cfg.n_heads} = {cfg.d_model / cfg.n_heads:.1f} (not integer)."
        )
    if cfg.vocab_size < 256:
        errors.append(f"vocab_size ({cfg.vocab_size}) is unreasonably small. Minimum: 256.")
    if cfg.vocab_size > 200000:
        errors.append(f"vocab_size ({cfg.vocab_size}) is very large. Typical range: 16K–100K.")
    if cfg.context_length < 64:
        errors.append(f"context_length ({cfg.context_length}) is too small. Minimum: 64.")
    if cfg.n_layers < 1:
        errors.append(f"n_layers ({cfg.n_layers}) must be >= 1.")
    if cfg.dropout < 0.0 or cfg.dropout >= 1.0:
        errors.append(f"dropout ({cfg.dropout}) must be in [0.0, 1.0).")
    if cfg.norm_type not in ("rmsnorm", "layernorm"):
        errors.append(f"norm_type ({cfg.norm_type!r}) must be 'rmsnorm' or 'layernorm'.")
    if cfg.pos_encoding not in ("rope", "learned", "sinusoidal"):
        errors.append(
            f"pos_encoding ({cfg.pos_encoding!r}) must be 'rope', 'learned', or 'sinusoidal'."
        )

    if errors:
        raise ConfigValueError(
            "Model configuration validation failed:\n  " + "\n  ".join(errors),
            suggestion="Review configs/model_*.yaml and correct the listed fields.",
        )


def _validate_training_config(cfg: TrainingConfig) -> None:
    """Raise ConfigValueError if the training config contains invalid values."""
    errors: list[str] = []

    if cfg.batch_size < 1:
        errors.append(f"batch_size ({cfg.batch_size}) must be >= 1.")
    if cfg.learning_rate <= 0 or cfg.learning_rate > 0.1:
        errors.append(
            f"learning_rate ({cfg.learning_rate}) is out of reasonable range (0, 0.1]."
        )
    if cfg.grad_clip <= 0:
        errors.append(f"grad_clip ({cfg.grad_clip}) must be > 0.")
    if cfg.mixed_precision not in ("fp32", "fp16", "bf16"):
        errors.append(
            f"mixed_precision ({cfg.mixed_precision!r}) must be 'fp32', 'fp16', or 'bf16'."
        )
    if cfg.val_split <= 0 or cfg.val_split >= 0.5:
        errors.append(f"val_split ({cfg.val_split}) must be in (0.0, 0.5).")
    if cfg.warmup_steps >= cfg.max_steps:
        errors.append(
            f"warmup_steps ({cfg.warmup_steps}) must be < max_steps ({cfg.max_steps})."
        )

    if errors:
        raise ConfigValueError(
            "Training configuration validation failed:\n  " + "\n  ".join(errors),
            suggestion="Review configs/training.yaml and correct the listed fields.",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, project_root: str | Path | None = None) -> ProjectConfig:
    """
    Load and validate the project configuration.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the project YAML config (configs/project.yaml).
    project_root : str or Path, optional
        Project root directory. Defaults to the parent of yaml_path's parent.

    Returns
    -------
    ProjectConfig
        Validated, immutable project configuration.
    """
    yaml_path = Path(yaml_path).resolve()
    root      = Path(project_root).resolve() if project_root else yaml_path.parent.parent

    raw = _load_yaml(yaml_path)
    raw = _apply_env_overrides(raw)

    project_data  = raw.get("project", {})
    paths_data    = raw.get("paths", {})
    logging_data  = raw.get("logging", {})
    hardware_data = raw.get("hardware", {})

    try:
        cfg = ProjectConfig(
            name        = project_data.get("name", "slm-project"),
            version     = project_data.get("version", "0.1.0"),
            description = project_data.get("description", ""),
            author      = project_data.get("author", ""),
            seed        = int(raw.get("seed", 42)),
            root        = str(root),
            paths       = PathsConfig(**{
                k: v for k, v in paths_data.items()
                if k in PathsConfig.__dataclass_fields__
            }),
            logging     = LoggingConfig(**{
                k: v for k, v in logging_data.items()
                if k in LoggingConfig.__dataclass_fields__
            }),
            hardware    = HardwareConfig(**{
                k: v for k, v in hardware_data.items()
                if k in HardwareConfig.__dataclass_fields__
            }),
        )
    except TypeError as exc:
        raise ConfigError(
            f"Configuration construction failed: {exc}",
            suggestion="Check for unexpected keys in configs/project.yaml.",
        ) from exc

    logger.info(
        "Project config loaded",
        name=cfg.name,
        version=cfg.version,
        seed=cfg.seed,
    )
    return cfg


def load_model_config(yaml_path: str | Path) -> ModelConfig:
    """Load and validate a model configuration file."""
    yaml_path = Path(yaml_path).resolve()
    raw = _load_yaml(yaml_path)
    model_data = raw.get("model", raw)  # support both nested and flat

    try:
        cfg = ModelConfig(**{
            k: v for k, v in model_data.items()
            if k in ModelConfig.__dataclass_fields__
        })
    except TypeError as exc:
        raise ConfigError(
            f"Model config construction failed: {exc}",
            suggestion=f"Check for unexpected keys in {yaml_path}.",
        ) from exc

    _validate_model_config(cfg)
    logger.info(
        "Model config loaded",
        name=cfg.name,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
    )
    return cfg


def load_training_config(yaml_path: str | Path) -> TrainingConfig:
    """Load and validate a training configuration file."""
    yaml_path = Path(yaml_path).resolve()
    raw = _load_yaml(yaml_path)
    train_data = raw.get("training", raw)

    try:
        cfg = TrainingConfig(**{
            k: v for k, v in train_data.items()
            if k in TrainingConfig.__dataclass_fields__
        })
    except TypeError as exc:
        raise ConfigError(
            f"Training config construction failed: {exc}",
            suggestion=f"Check for unexpected keys in {yaml_path}.",
        ) from exc

    _validate_training_config(cfg)
    logger.info(
        "Training config loaded",
        lr=cfg.learning_rate,
        batch_size=cfg.batch_size,
        grad_accum=cfg.gradient_accumulation_steps,
        max_steps=cfg.max_steps,
        mixed_precision=cfg.mixed_precision,
    )
    return cfg


def load_tokenizer_config(yaml_path: str | Path) -> TokenizerConfig:
    """Load and validate a tokenizer configuration file."""
    yaml_path = Path(yaml_path).resolve()
    raw = _load_yaml(yaml_path)
    tok_data = raw.get("tokenizer", raw)

    special_tokens = tok_data.pop("special_tokens", {})
    tok_data.update(special_tokens)  # flatten into tokenizer config

    try:
        cfg = TokenizerConfig(**{
            k: v for k, v in tok_data.items()
            if k in TokenizerConfig.__dataclass_fields__
        })
    except TypeError as exc:
        raise ConfigError(
            f"Tokenizer config construction failed: {exc}",
            suggestion=f"Check for unexpected keys in {yaml_path}.",
        ) from exc

    if cfg.vocab_size < 1000:
        raise ConfigValueError(
            f"vocab_size ({cfg.vocab_size}) is unreasonably small for a BPE tokenizer.",
            suggestion="Set vocab_size >= 8000. Recommended: 16000–32000.",
        )

    logger.info(
        "Tokenizer config loaded",
        version=cfg.version,
        algorithm=cfg.algorithm,
        vocab_size=cfg.vocab_size,
    )
    return cfg
