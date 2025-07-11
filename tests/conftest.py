"""
Pytest configuration file for Pinocchio tests.
"""
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add the project root to the Python path to ensure correct imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import fixtures here if needed


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_sessions_dir(temp_dir):
    """Create a temporary directory for sessions."""
    sessions_dir = temp_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for tests."""
    temp_file_path = temp_dir / "test_file.txt"
    with open(temp_file_path, "w") as f:
        f.write("Test content")
    return temp_file_path


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Return a sample configuration for testing."""
    return {
        "app": {
            "name": "pinocchio-test",
            "version": "0.1.0",
            "debug": True,
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "test-key",
        },
        "memory": {
            "storage_path": "./data",
            "max_items": 1000,
        },
        "session": {
            "timeout": 3600,
            "auto_save": True,
        },
    }


@pytest.fixture
def test_data_dir():
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def ensure_test_data_dir(test_data_dir):
    """Ensure the test data directory exists."""
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir
