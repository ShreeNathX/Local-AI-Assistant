"""
setup.py — makes the slm package pip-installable in development mode.

Install with:
    pip install -e .

This allows imports like:
    from slm.core import get_logger

without needing to add the project root to PYTHONPATH manually.
"""

from setuptools import setup, find_packages

setup(
    name="slm",
    version="0.1.0",
    description="Small Language Model built from scratch",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*", "docs*"]),
    install_requires=[],  # managed via requirements.txt
    extras_require={
        "dev": [],        # managed via requirements-dev.txt
    },
)
