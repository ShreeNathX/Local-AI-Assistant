# Setup Guide

## Prerequisites

| Requirement | Minimum | How to Check |
|---|---|---|
| Python | 3.10 | `python --version` |
| NVIDIA Driver | 520+ | `nvidia-smi` |
| CUDA Toolkit | 11.8+ | `nvcc --version` |
| Git | Any | `git --version` |
| Free Disk | 50 GB | `Get-PSDrive C` |

## Installation

### 1. Create virtual environment

```powershell
cd slm-project
powershell -ExecutionPolicy Bypass -File scripts\install_env.ps1
```

This script:
- Creates `.venv/` using Python 3.11.9
- Installs PyTorch 2.5.1 with CUDA 12.1 support
- Installs all `requirements.txt` dependencies

### 2. Activate environment

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt should show `(.venv)`.

### 3. Install as editable package

```powershell
pip install -e .
```

This allows `from slm.core import ...` imports anywhere.

### 4. Install dev dependencies

```powershell
pip install -r requirements-dev.txt
```

### 5. Copy environment config

```powershell
copy .env.example .env
```

Edit `.env` to add API keys when needed (Phase 22+).

## Hardware Check

```powershell
python scripts/hardware_check.py --save-report
```

A report will be saved to `logs/hardware_check_<timestamp>.json`.

## Running Tests

```powershell
# Unit tests only (fast, no GPU)
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=slm --cov-report=term-missing

# All tests
pytest -v
```

## Troubleshooting

### PyTorch: CUDA not available after install

```
python -c "import torch; print(torch.cuda.is_available())"
# Returns: False
```

**Causes and fixes:**

1. **CPU-only PyTorch installed:** Reinstall with CUDA index:
   ```powershell
   pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

2. **NVIDIA driver too old:** Check `nvidia-smi`. Update driver from NVIDIA website.

3. **CUDA toolkit mismatch:** PyTorch 2.5.1+cu121 requires CUDA 12.1+.
   Check: `nvidia-smi | findstr "CUDA Version"`

4. **Wrong venv active:** Confirm `.venv` is active before running.

### Import errors (`ModuleNotFoundError`)

```
python -c "from slm.core import get_logger"
# ModuleNotFoundError: No module named 'slm'
```

**Fix:** Install in editable mode:
```powershell
pip install -e .
```

### OneDrive sync causing install failures

OneDrive syncing the project directory during pip install can cause timeouts.

**Fix:** Pause OneDrive sync before running `install_env.ps1`:
- System tray → OneDrive icon → Pause syncing for 2 hours
