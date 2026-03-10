# tests/conftest.py
"""
Shared pytest configuration for ASR test suite.

Sets up:
- Path resolution so tests can import from the ASR directory
- Common fixtures reused across test files
- Skip markers for tests requiring optional packages
"""
import os
import sys
import pytest

# Add parent directory (speechbrain_recipe/ASR) to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests that require a CUDA-capable GPU"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests that take > 10 seconds (backbone download etc.)"
    )
    config.addinivalue_line(
        "markers", "onnx: marks tests that require onnx and onnxruntime"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU and slow tests unless --gpu / --runslow flags are passed."""
    skip_gpu = pytest.mark.skip(reason="requires --gpu flag to run")
    skip_slow = pytest.mark.skip(reason="requires --runslow flag for long tests")

    for item in items:
        if "gpu" in item.keywords and not config.getoption("--gpu", default=False):
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not config.getoption("--runslow", default=False):
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption("--gpu", action="store_true", default=False,
                     help="Run GPU-requiring tests")
    parser.addoption("--runslow", action="store_true", default=False,
                     help="Run slow tests (downloads backbone model)")
