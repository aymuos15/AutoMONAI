"""
Pytest configuration and shared fixtures for tests.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def torch_device():
    """Provide a torch device for the entire test session."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def use_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA (deselect with '-m \"not cuda\"')"
    )
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")
