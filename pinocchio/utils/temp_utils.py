"""Temporary file utilities for Pinocchio modules."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union


def create_temp_file(
    content: str = "",
    suffix: str = ".tmp",
    prefix: str = "pinocchio_",
    directory: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a temporary file with optional content.

    Args:
        content: Content to write to the file
        suffix: File suffix
        prefix: File prefix
        directory: Directory to create file in (optional)

    Returns:
        Path to the created temporary file
    """
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        prefix=prefix,
        dir=str(directory) if directory else None,
        delete=False,
    ) as f:
        if content:
            f.write(content)
        return Path(f.name)


def create_temp_directory(
    prefix: str = "pinocchio_",
    directory: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a temporary directory.

    Args:
        prefix: Directory prefix
        directory: Parent directory to create temp dir in (optional)

    Returns:
        Path to the created temporary directory
    """
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

    temp_dir = tempfile.mkdtemp(
        prefix=prefix,
        dir=str(directory) if directory else None,
    )
    return Path(temp_dir)


def cleanup_temp_files(*file_paths: Union[str, Path]) -> None:
    """
    Clean up temporary files.

    Args:
        *file_paths: Paths to files to delete
    """
    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass  # Ignore errors if file is already deleted


def cleanup_temp_directories(*dir_paths: Union[str, Path]) -> None:
    """
    Clean up temporary directories.

    Args:
        *dir_paths: Paths to directories to delete
    """
    for dir_path in dir_paths:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            try:
                import shutil

                shutil.rmtree(path)
            except OSError:
                pass  # Ignore errors if directory is already deleted


def get_temp_file_path(
    suffix: str = ".tmp",
    prefix: str = "pinocchio_",
    directory: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Get a temporary file path without creating the file.

    Args:
        suffix: File suffix
        prefix: File prefix
        directory: Directory for the file (optional)

    Returns:
        Path to the temporary file location
    """
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

    fd, path = tempfile.mkstemp(
        suffix=suffix,
        prefix=prefix,
        dir=str(directory) if directory else None,
    )
    os.close(fd)  # Close the file descriptor
    return Path(path)
