#!/usr/bin/env python3
"""
SLM Project — Hardware Diagnostic Script
=========================================
Phase 1, Step 1 — MUST PASS before any training begins.

Checks:
  - Python version
  - Operating system
  - CPU (model, cores, threads)
  - RAM (total, available)
  - GPU (model, VRAM, CUDA compute capability)
  - CUDA (driver version, PyTorch CUDA build)
  - PyTorch (version, CUDA availability)
  - Disk space (free on project drive)
  - Key package availability

Reports each item as:
  [PASS]    — meets minimum requirement
  [WARN]    — works but with caveats
  [FAIL]    — does not meet minimum requirement
  [INFO]    — informational only

Exits with code 0 on PASS/WARN, code 1 on any FAIL.

Usage:
  python scripts/hardware_check.py
  python scripts/hardware_check.py --json       # machine-readable JSON output
  python scripts/hardware_check.py --save-report # save report to logs/
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Constants — minimum requirements
# ---------------------------------------------------------------------------

MIN_PYTHON = (3, 10)
MIN_RAM_GB = 8.0
WARN_RAM_GB = 12.0
MIN_VRAM_GB = 4.0
WARN_VRAM_GB = 6.0
MIN_DISK_GB = 50.0
WARN_DISK_GB = 100.0
MIN_PYTORCH_MAJOR = 2
MIN_PYTORCH_MINOR = 3

# VRAM → max practical model size (fp16) — informational only
VRAM_MODEL_TABLE = [
    (4.0, "~20M parameters — tiny model only"),
    (6.0, "~50M parameters — small SLM"),
    (8.0, "~100M parameters — target SLM"),
    (12.0, "~200M parameters"),
    (16.0, "~300M–500M parameters"),
    (24.0, "~1B+ parameters with LoRA"),
]

# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

PASS  = "PASS"
WARN  = "WARN"
FAIL  = "FAIL"
INFO  = "INFO"
SKIP  = "SKIP"

STATUS_COLORS = {
    PASS: "\033[92m",   # green
    WARN: "\033[93m",   # yellow
    FAIL: "\033[91m",   # red
    INFO: "\033[94m",   # blue
    SKIP: "\033[90m",   # grey
    "RESET": "\033[0m",
    "BOLD":  "\033[1m",
}

USE_COLOR = sys.stdout.isatty() and platform.system() != "Windows" or (
    platform.system() == "Windows" and os.environ.get("TERM") is not None
)


def _color(text: str, color_key: str) -> str:
    if not USE_COLOR:
        return text
    return f"{STATUS_COLORS.get(color_key, '')}{text}{STATUS_COLORS['RESET']}"


def _bold(text: str) -> str:
    if not USE_COLOR:
        return text
    return f"{STATUS_COLORS['BOLD']}{text}{STATUS_COLORS['RESET']}"


def _format_row(status: str, label: str, value: str, note: str = "") -> str:
    tag = f"[{status:4s}]"
    colored_tag = _color(tag, status)
    label_col = f"{label:<28}"
    note_str = f"  # {note}" if note else ""
    return f"  {colored_tag}  {label_col}  {value}{note_str}"


# ---------------------------------------------------------------------------
# Individual checks — each returns (status, value_str, note_str, raw_data)
# ---------------------------------------------------------------------------

def check_python() -> tuple[str, str, str, dict]:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    data = {"major": v.major, "minor": v.minor, "micro": v.micro, "executable": sys.executable}
    if (v.major, v.minor) >= MIN_PYTHON:
        return PASS, version_str, "", data
    return FAIL, version_str, f"Requires Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+", data


def check_os() -> tuple[str, str, str, dict]:
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    full = f"{system} {release} ({machine})"
    data = {"system": system, "release": release, "version": version, "machine": machine}
    return INFO, full, "", data


def check_cpu() -> tuple[str, str, str, dict]:
    import psutil
    cores_physical = psutil.cpu_count(logical=False) or 0
    cores_logical  = psutil.cpu_count(logical=True)  or 0
    freq = psutil.cpu_freq()
    freq_str = f"{freq.max:.0f} MHz max" if freq else "freq unknown"

    try:
        # Try to get CPU brand
        if platform.system() == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and l.strip() != "Name"]
            cpu_name = lines[0] if lines else "Unknown"
        else:
            result = subprocess.run(
                ["cat", "/proc/cpuinfo"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "model name" in line.lower():
                    cpu_name = line.split(":", 1)[1].strip()
                    break
            else:
                cpu_name = platform.processor() or "Unknown"
    except Exception:
        cpu_name = platform.processor() or "Unknown"

    value = f"{cpu_name} | {cores_physical}C/{cores_logical}T | {freq_str}"
    data = {
        "name": cpu_name,
        "physical_cores": cores_physical,
        "logical_cores": cores_logical,
        "freq_max_mhz": freq.max if freq else None,
    }
    status = PASS if cores_logical >= 4 else WARN
    note = "" if cores_logical >= 4 else "Recommend 4+ logical cores for DataLoader workers"
    return status, value, note, data


def check_ram() -> tuple[str, str, str, dict]:
    import psutil
    vm = psutil.virtual_memory()
    total_gb    = vm.total   / (1024**3)
    available_gb = vm.available / (1024**3)
    used_pct    = vm.percent

    value = f"{total_gb:.1f} GB total | {available_gb:.1f} GB available | {used_pct:.0f}% used"
    data  = {"total_gb": round(total_gb, 2), "available_gb": round(available_gb, 2), "used_pct": used_pct}

    if total_gb < MIN_RAM_GB:
        return FAIL, value, f"Minimum {MIN_RAM_GB:.0f} GB required", data
    if total_gb < WARN_RAM_GB:
        return WARN, value, f"Recommend {WARN_RAM_GB:.0f}+ GB for comfortable training", data
    return PASS, value, "", data


def check_pytorch() -> tuple[str, str, str, dict]:
    try:
        import torch
        version = torch.__version__
        cuda_build = torch.version.cuda
        parts = version.split("+")[0].split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0

        value = f"{version}"
        data = {"version": version, "cuda_build": cuda_build, "major": major, "minor": minor}

        if (major, minor) < (MIN_PYTORCH_MAJOR, MIN_PYTORCH_MINOR):
            return WARN, value, f"Recommend PyTorch {MIN_PYTORCH_MAJOR}.{MIN_PYTORCH_MINOR}+", data
        if cuda_build is None or "+cpu" in version:
            return WARN, value, "CPU-only build — CUDA training not available", data
        return PASS, value, f"CUDA build: {cuda_build}", data

    except ImportError:
        return FAIL, "NOT INSTALLED", "Run: pip install torch", {}


def check_cuda() -> tuple[str, str, str, dict]:
    try:
        import torch

        if not torch.cuda.is_available():
            # Try to explain why
            cuda_build = torch.version.cuda
            if cuda_build is None:
                note = "PyTorch is CPU-only build. Reinstall with CUDA index."
            else:
                note = "CUDA not available — check NVIDIA driver and CUDA toolkit installation."
            return FAIL, "UNAVAILABLE", note, {"available": False}

        device_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda or "unknown"
        value = f"CUDA {cuda_version} | {device_count} device(s)"
        data  = {"available": True, "version": cuda_version, "device_count": device_count}
        return PASS, value, "", data

    except ImportError:
        return SKIP, "N/A", "PyTorch not installed", {}


def check_gpu() -> tuple[str, str, str, dict]:
    """Check GPU via PyTorch CUDA + fallback to nvidia-smi."""
    try:
        import torch

        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                cap = f"{props.major}.{props.minor}"
                devices.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(vram_gb, 2),
                    "compute_capability": cap,
                    "multiprocessors": props.multi_processor_count,
                })

            if not devices:
                return WARN, "CUDA available but no devices found", "", {"devices": []}

            # Report primary GPU
            gpu = devices[0]
            vram_gb = gpu["vram_gb"]
            value = (
                f"{gpu['name']} | {vram_gb:.1f} GB VRAM | "
                f"Compute {gpu['compute_capability']} | "
                f"{gpu['multiprocessors']} SMs"
            )

            # Determine max model recommendation
            model_rec = "Unknown"
            for threshold, rec in VRAM_MODEL_TABLE:
                if vram_gb >= threshold:
                    model_rec = rec

            if vram_gb < MIN_VRAM_GB:
                return FAIL, value, f"Minimum {MIN_VRAM_GB:.0f} GB VRAM required for training", {"devices": devices}
            if vram_gb < WARN_VRAM_GB:
                return WARN, value, f"Limited VRAM — max model size: {model_rec}", {"devices": devices}
            return PASS, value, f"Recommended model: {model_rec}", {"devices": devices}

    except ImportError:
        pass

    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0] if len(parts) > 0 else "Unknown"
            mem  = parts[1] if len(parts) > 1 else "?"
            drv  = parts[2] if len(parts) > 2 else "?"
            cap  = parts[3] if len(parts) > 3 else "?"
            # Parse VRAM
            vram_gb = 0.0
            try:
                vram_gb = float(mem.split()[0]) / 1024.0  # MiB → GiB
            except (ValueError, IndexError):
                pass

            value = f"{name} | {vram_gb:.1f} GB VRAM | Driver {drv} | Compute {cap}"
            note  = "Detected via nvidia-smi (PyTorch CUDA not available)"
            data  = {"name": name, "vram_gb": round(vram_gb, 2), "driver": drv, "compute_cap": cap,
                     "source": "nvidia-smi"}
            return WARN, value, note, data

        return WARN, "nvidia-smi returned no output", "GPU may not be present", {}

    except FileNotFoundError:
        return WARN, "NOT FOUND", "nvidia-smi not on PATH — NVIDIA GPU may not be installed", {}
    except subprocess.TimeoutExpired:
        return WARN, "TIMEOUT", "nvidia-smi timed out", {}


def check_disk() -> tuple[str, str, str, dict]:
    import psutil
    # Use project root drive
    project_root = Path(__file__).resolve().parent.parent
    drive = Path(project_root.anchor)
    try:
        usage = psutil.disk_usage(str(drive))
        free_gb  = usage.free  / (1024**3)
        total_gb = usage.total / (1024**3)
        used_pct = usage.percent
        value = f"{free_gb:.1f} GB free / {total_gb:.1f} GB total ({used_pct:.0f}% used) on {drive}"
        data  = {"drive": str(drive), "free_gb": round(free_gb, 2),
                 "total_gb": round(total_gb, 2), "used_pct": used_pct}
        if free_gb < MIN_DISK_GB:
            return FAIL, value, f"Minimum {MIN_DISK_GB:.0f} GB free required (datasets + checkpoints)", data
        if free_gb < WARN_DISK_GB:
            return WARN, value, f"Recommend {WARN_DISK_GB:.0f}+ GB free — datasets can be large", data
        return PASS, value, "", data
    except Exception as exc:
        return WARN, str(exc), "Could not read disk usage", {}


def check_package(name: str, import_name: str | None = None) -> tuple[str, str, str, dict]:
    mod_name = import_name or name
    try:
        mod = __import__(mod_name)
        version = getattr(mod, "__version__", "unknown")
        return PASS, version, "", {"package": name, "version": version}
    except ImportError:
        return FAIL, "NOT INSTALLED", f"pip install {name}", {"package": name, "version": None}


def check_nvidia_driver() -> tuple[str, str, str, dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            driver = result.stdout.strip().splitlines()[0].strip()
            data = {"driver_version": driver}
            try:
                major = int(driver.split(".")[0])
                if major >= 520:
                    return PASS, f"Driver {driver}", "", data
                return WARN, f"Driver {driver}", "Recommend NVIDIA Driver >= 520 for CUDA 12.x", data
            except ValueError:
                return INFO, f"Driver {driver}", "", data
        return WARN, "nvidia-smi call failed", "", {}
    except FileNotFoundError:
        return WARN, "nvidia-smi not found", "NVIDIA driver may not be installed", {}
    except subprocess.TimeoutExpired:
        return WARN, "nvidia-smi timeout", "", {}


# ---------------------------------------------------------------------------
# Main diagnostic runner
# ---------------------------------------------------------------------------

def run_diagnostics() -> dict[str, Any]:
    results: dict[str, Any] = {}
    rows: list[tuple[str, str, str, str]] = []  # (status, label, value, note)

    checks = [
        ("python",          "Python",           check_python),
        ("os",              "OS",               check_os),
        ("cpu",             "CPU",              check_cpu),
        ("ram",             "RAM",              check_ram),
        ("nvidia_driver",   "NVIDIA Driver",    check_nvidia_driver),
        ("pytorch",         "PyTorch",          check_pytorch),
        ("cuda",            "CUDA (PyTorch)",   check_cuda),
        ("gpu",             "GPU",              check_gpu),
        ("disk",            "Disk (free)",      check_disk),
    ]

    package_checks = [
        ("numpy",       None),
        ("tokenizers",  None),
        ("transformers",None),
        ("datasets",    None),
        ("psutil",      None),
        ("rich",        None),
        ("loguru",      None),
        ("tqdm",        None),
        ("yaml",        "yaml"),
        ("dotenv",      "dotenv"),
    ]

    print()
    print(_bold("=" * 64))
    print(_bold("  SLM Project — Hardware & Environment Diagnostic"))
    print(_bold(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print(_bold("=" * 64))
    print()

    # --- System checks ---
    print(_bold("  SYSTEM"))
    print(_bold("  " + "-" * 60))
    for key, label, fn in checks:
        try:
            status, value, note, data = fn()
        except Exception as exc:
            status, value, note, data = FAIL, "ERROR", str(exc), {}
        results[key] = {"status": status, "value": value, "note": note, "data": data}
        rows.append((status, label, value, note))
        print(_format_row(status, label, value, note))
    print()

    # --- Package checks ---
    print(_bold("  PACKAGES"))
    print(_bold("  " + "-" * 60))
    pkg_results: dict[str, Any] = {}
    for pkg_name, import_name in package_checks:
        try:
            status, value, note, data = check_package(pkg_name, import_name)
        except Exception as exc:
            status, value, note, data = FAIL, "ERROR", str(exc), {}
        pkg_results[pkg_name] = {"status": status, "version": value}
        rows.append((status, f"pkg:{pkg_name}", value, note))
        print(_format_row(status, f"pkg:{pkg_name}", value, note))
    results["packages"] = pkg_results
    print()

    # --- Overall verdict ---
    fail_count = sum(1 for (s, *_) in rows if s == FAIL)
    warn_count = sum(1 for (s, *_) in rows if s == WARN)

    print(_bold("=" * 64))
    if fail_count > 0:
        verdict = FAIL
        verdict_msg = (
            f"FAIL — {fail_count} critical issue(s) detected.\n"
            f"  Resolve all [FAIL] items before proceeding."
        )
    elif warn_count > 0:
        verdict = WARN
        verdict_msg = (
            f"WARN — {warn_count} warning(s) detected.\n"
            f"  Review [WARN] items. Training may be limited."
        )
    else:
        verdict = PASS
        verdict_msg = "PASS — All checks passed. Safe to proceed."

    print(f"\n  Overall: {_color(verdict_msg, verdict)}\n")

    # --- GPU-based model recommendation ---
    gpu_data = results.get("gpu", {}).get("data", {})
    devices  = gpu_data.get("devices", [])
    if devices:
        vram = devices[0].get("vram_gb", 0.0)
        rec = "Unknown"
        for threshold, label in VRAM_MODEL_TABLE:
            if vram >= threshold:
                rec = label
        print(f"  {_bold('Model Recommendation based on VRAM:')}")
        print(f"    VRAM: {vram:.1f} GB -> Max practical model: {rec}")
        print()
    elif results.get("gpu", {}).get("data", {}).get("vram_gb"):
        vram = results["gpu"]["data"]["vram_gb"]
        rec = "Unknown"
        for threshold, label in VRAM_MODEL_TABLE:
            if vram >= threshold:
                rec = label
        print(f"  {_bold('Model Recommendation based on VRAM (nvidia-smi):')}")
        print(f"    VRAM: {vram:.1f} GB -> Max practical model: {rec}")
        print()


    print(_bold("=" * 64))
    print()

    results["summary"] = {
        "verdict":    verdict,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "timestamp":  datetime.datetime.now().isoformat(),
    }
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SLM Project — Hardware & Environment Diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Exit codes:
          0 — PASS or WARN (safe to proceed with caution)
          1 — FAIL (must resolve issues before continuing)
        """)
    )
    parser.add_argument("--json",        action="store_true", help="Output results as JSON")
    parser.add_argument("--save-report", action="store_true", help="Save report to logs/hardware_check_<timestamp>.json")
    args = parser.parse_args()

    results = run_diagnostics()

    if args.json:
        print(json.dumps(results, indent=2, default=str))

    if args.save_report:
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = logs_dir / f"hardware_check_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Report saved to: {report_path}")
        print()

    verdict = results.get("summary", {}).get("verdict", FAIL)
    sys.exit(0 if verdict in (PASS, WARN) else 1)


if __name__ == "__main__":
    main()
