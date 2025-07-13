"""
Testing utilities for Pinocchio modules.

WARNING: This module is ONLY for test code. Do NOT import or use in production modules.

This module provides common testing utilities that can be used across
different test modules to reduce code duplication and improve test consistency.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock

from pinocchio.utils import (
    cleanup_temp_files,
    create_temp_directory,
    create_temp_file,
    ensure_directory,
    safe_read_json,
    safe_write_json,
)


def create_test_temp_dir() -> Path:
    """
    Create a temporary directory for testing.

    Returns:
        Path to temporary directory
    """
    return create_temp_directory(prefix="pinocchio_test_")


def create_test_temp_file(content: str = "test content", suffix: str = ".txt") -> Path:
    """
    Create a temporary file for testing.

    Args:
        content: File content
        suffix: File suffix

    Returns:
        Path to temporary file
    """
    temp_file = create_temp_file(suffix=suffix)
    with open(temp_file, "w") as f:
        f.write(content)
    return temp_file


def create_test_json_file(data: Dict[str, Any], filename: str = "test.json") -> Path:
    """
    Create a temporary JSON file for testing.

    Args:
        data: Data to write to JSON file
        filename: Name of the file

    Returns:
        Path to JSON file
    """
    temp_dir = create_test_temp_dir()
    json_file = temp_dir / filename
    success = safe_write_json(data, json_file)
    if not success:
        raise RuntimeError(f"Failed to create test JSON file: {json_file}")
    return json_file


def load_test_json_file(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load JSON data from a test file.

    Args:
        file_path: Path to JSON file

    Returns:
        JSON data or None if failed
    """
    return safe_read_json(file_path)


def assert_dict_structure(
    data: Dict[str, Any],
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None,
) -> None:
    """
    Assert that a dictionary has the required structure.

    Args:
        data: Dictionary to check
        required_keys: List of required keys
        optional_keys: List of optional keys
    """
    if optional_keys is None:
        optional_keys = []

    # Check required keys
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"

    # Check that all keys are either required or optional
    all_valid_keys = set(required_keys) | set(optional_keys)
    for key in data.keys():
        assert key in all_valid_keys, f"Unexpected key: {key}"


def assert_json_file_structure(
    file_path: Union[str, Path], required_keys: List[str]
) -> None:
    """
    Assert that a JSON file has the required structure.

    Args:
        file_path: Path to JSON file
        required_keys: List of required keys
    """
    data = safe_read_json(file_path)
    assert data is not None, f"Failed to read JSON file: {file_path}"
    assert_dict_structure(data, required_keys)


def create_test_logger(
    name: str = "test_logger", level: int = logging.DEBUG
) -> logging.Logger:
    """
    Create a test logger.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Add a simple handler for testing
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def cleanup_test_files(*file_paths: Union[str, Path]) -> None:
    """
    Clean up test files.

    Args:
        *file_paths: File paths to clean up
    """
    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logging.warning(f"Failed to clean up test file {path}: {e}")


def create_test_directory_structure(base_path: Path) -> Dict[str, Path]:
    """
    Create a test directory structure.

    Args:
        base_path: Base directory path

    Returns:
        Dictionary mapping directory names to paths
    """
    ensure_directory(base_path)

    directories = {
        "sessions": base_path / "sessions",
        "memory": base_path / "memory",
        "prompts": base_path / "prompts",
        "knowledge": base_path / "knowledge",
        "logs": base_path / "logs",
    }

    for name, dir_path in directories.items():
        ensure_directory(dir_path)

    return directories
