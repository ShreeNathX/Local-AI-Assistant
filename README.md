# SLM Project

> A Small Language Model (SLM) built from scratch — for educational and engineering purposes.

## Project Status

| Phase | Name | Status |
|---|---|---|
| 0 | Requirements & Planning | PASSED |
| 1 | Hardware & Environment Validation | IN PROGRESS |
| 2 | Repository & Configuration | IN PROGRESS |
| 3–28 | Subsequent phases | NOT STARTED |

## Quick Start

### 1. Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- Git

### 2. Clone and Set Up

```powershell
git clone <repo-url>
cd slm-project

# Create virtual environment and install all dependencies
powershell -ExecutionPolicy Bypass -File scripts\install_env.ps1

# Activate the environment
.venv\Scripts\Activate.ps1
```

### 3. Copy environment configuration

```powershell
copy .env.example .env
# Edit .env to add your API keys (optional for early phases)
```

### 4. Run Hardware Check

```powershell
python scripts/hardware_check.py --save-report
```

Expected output (PASS on a CUDA-capable system):
```
[PASS]  Python           3.11.9
[PASS]  GPU              NVIDIA GeForce GTX 1650 | 4.0 GB VRAM
[PASS]  CUDA (PyTorch)   CUDA 12.1
...
Overall: PASS — Safe to proceed.
```

### 5. Run Tests

```powershell
# Install development dependencies first
pip install -r requirements-dev.txt

# Run all unit tests
pytest tests/unit/

# Run with coverage
pytest tests/unit/ --cov=slm --cov-report=term-missing
```

## Repository Structure

```
slm-project/
├── configs/         # YAML configuration files (model, training, tokenizer)
├── data/            # Data pipeline (raw → interim → processed)
├── slm/             # Importable Python package
│   ├── core/        # Config, logging, exceptions
│   ├── model/       # Transformer architecture (Phase 10+)
│   ├── tokenizer/   # BPE tokenizer (Phase 9+)
│   ├── training/    # Training loop (Phase 11+)
│   ├── evaluation/  # Metrics and evaluation (Phase 15+)
│   ├── inference/   # Text generation (Phase 13+)
│   ├── tools/       # External tools: search, calculator (Phase 21+)
│   ├── rag/         # Retrieval-augmented generation (Phase 23+)
│   ├── router/      # Intent routing (Phase 24+)
│   └── api/         # REST API (Phase 25+)
├── scripts/         # Runnable scripts (hardware_check, install_env)
├── tests/           # Unit, integration, regression tests
├── docs/            # Documentation
├── checkpoints/     # Model checkpoints (git-ignored)
├── logs/            # Structured logs (git-ignored)
└── tokenizer/       # Tokenizer training and artifacts
```

## Development Rules

1. **Never skip a phase** — every phase must PASS before the next begins.
2. **Never ignore errors** — stop, diagnose, fix, re-test.
3. **Never invent results** — all metrics must be measured.
4. **Fail fast** — missing files, bad configs, and wrong types raise exceptions immediately.
5. **Raw data is immutable** — never modify files in `data/raw/`.

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB | 8+ GB |
| Disk | 50 GB free | 200+ GB free |
| CUDA | 11.8 | 12.1+ |

## Phase Gate Definition of Done

A phase is only PASSED when:
- [ ] Implementation committed to git
- [ ] Unit tests pass
- [ ] Integration tests pass (where applicable)
- [ ] Smoke test passes
- [ ] Validation report generated
- [ ] Errors handled (no bare `except`)
- [ ] Structured logs confirmed
- [ ] Output artifact versioned
- [ ] Reproducibility confirmed (fixed seed)
- [ ] Documentation updated

## License

This project is for educational purposes. All training data must be used in compliance with their respective licenses. See `data/licenses/` for details.
