# =============================================================
# SLM Project — scripts/install_env.ps1
#
# Creates a Python virtual environment and installs all
# dependencies including CUDA-enabled PyTorch.
#
# Usage:
#   cd <project-root>
#   powershell -ExecutionPolicy Bypass -File scripts\install_env.ps1
#
# Requirements:
#   - Python 3.10+ installed and on PATH
#   - NVIDIA GPU with driver >= 520
#   - Internet connection
# =============================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$VENV_DIR = Join-Path $PROJECT_ROOT ".venv"
$PYTHON = "python"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SLM Project — Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Verify Python version ---
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
try {
    $py_ver = & $PYTHON --version 2>&1
    Write-Host "      Found: $py_ver" -ForegroundColor Green
    if ($py_ver -notmatch "Python 3\.(1[0-9]|[2-9][0-9])") {
        throw "Python 3.10+ is required. Found: $py_ver"
    }
} catch {
    Write-Host "FAIL: $_" -ForegroundColor Red
    exit 1
}

# --- Step 2: Create virtual environment ---
Write-Host "[2/6] Creating virtual environment at .venv ..." -ForegroundColor Yellow
if (Test-Path $VENV_DIR) {
    Write-Host "      .venv already exists — skipping creation." -ForegroundColor DarkYellow
} else {
    & $PYTHON -m venv $VENV_DIR
    Write-Host "      Created: .venv" -ForegroundColor Green
}

$PIP = Join-Path $VENV_DIR "Scripts\pip.exe"
$PYTHON_VENV = Join-Path $VENV_DIR "Scripts\python.exe"

# --- Step 3: Upgrade pip inside venv ---
Write-Host "[3/6] Upgrading pip inside venv..." -ForegroundColor Yellow
& $PIP install --upgrade pip --quiet
Write-Host "      pip upgraded." -ForegroundColor Green

# --- Step 4: Install PyTorch with CUDA 12.1 ---
Write-Host "[4/6] Installing PyTorch 2.5.1 + CUDA 12.1..." -ForegroundColor Yellow
Write-Host "      (This may take 5-10 minutes on first run)" -ForegroundColor DarkGray
& $PIP install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 `
    --index-url https://download.pytorch.org/whl/cu121 `
    --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: PyTorch CUDA installation failed." -ForegroundColor Red
    Write-Host "      Check internet connection and PyTorch CUDA compatibility." -ForegroundColor Red
    exit 1
}
Write-Host "      PyTorch installed." -ForegroundColor Green

# --- Step 5: Install project requirements ---
Write-Host "[5/6] Installing requirements.txt..." -ForegroundColor Yellow
& $PIP install -r (Join-Path $PROJECT_ROOT "requirements.txt") --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: requirements.txt installation failed." -ForegroundColor Red
    exit 1
}
Write-Host "      requirements.txt installed." -ForegroundColor Green

# --- Step 6: Verify PyTorch CUDA ---
Write-Host "[6/6] Verifying PyTorch CUDA availability..." -ForegroundColor Yellow
$cuda_check = & $PYTHON_VENV -c "import torch; print(torch.cuda.is_available())" 2>&1
if ($cuda_check -eq "True") {
    Write-Host "      CUDA: AVAILABLE" -ForegroundColor Green
} else {
    Write-Host "WARNING: CUDA not available after installation." -ForegroundColor DarkYellow
    Write-Host "         Run scripts\hardware_check.py for full diagnostics." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Environment setup complete." -ForegroundColor Green
Write-Host ""
Write-Host "  Activate the environment:" -ForegroundColor White
Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  Next step:" -ForegroundColor White
Write-Host "    python scripts\hardware_check.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
