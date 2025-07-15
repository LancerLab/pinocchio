"""File operation utilities for Pinocchio."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .string_utils import sanitize_filename

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        Path object of the directory
    """
    path_obj = Path(path)
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path_obj}")
        return path_obj
    except Exception as e:
        logger.error(f"Failed to create directory {path_obj}: {e}")
        raise


def safe_write_json(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    indent: int = 2,
    backup: bool = True,
) -> bool:
    """
    Safely write data to JSON file with backup.

    Args:
        data: Data to write
        file_path: Target file path
        indent: JSON indentation
        backup: Whether to create backup if file exists

    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)

    def default(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle TaskResult objects
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    try:
        # Create directory if needed
        ensure_directory(file_path.parent)

        # Create backup if file exists and backup is requested
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(
                f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            file_path.rename(backup_path)
            logger.debug(f"Created backup: {backup_path}")

        # Write new file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=default)

        logger.debug(f"Successfully wrote JSON to: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}: {e}")
        return False


def safe_read_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Safely read JSON file.

    Args:
        file_path: File path to read

    Returns:
        Parsed JSON data or None if failed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Successfully read JSON from: {file_path}")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to read JSON from {file_path}: {e}")
        return None


def safe_write_text(
    content: str,
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    backup: bool = True,
) -> bool:
    """
    Safely write text to file with backup.

    Args:
        content: Text content to write
        file_path: Target file path
        encoding: File encoding
        backup: Whether to create backup if file exists

    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)

    try:
        # Create directory if needed
        ensure_directory(file_path.parent)

        # Create backup if file exists and backup is requested
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(
                f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}{file_path.suffix}'
            )
            file_path.rename(backup_path)
            logger.debug(f"Created backup: {backup_path}")

        # Write new file
        with open(file_path, "w", encoding=encoding) as f:
            f.write(content)

        logger.debug(f"Successfully wrote text to: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to write text to {file_path}: {e}")
        return False


def safe_read_text(
    file_path: Union[str, Path], encoding: str = "utf-8"
) -> Optional[str]:
    """
    Safely read text file.

    Args:
        file_path: File path to read
        encoding: File encoding

    Returns:
        File content or None if failed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return None

    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        logger.debug(f"Successfully read text from: {file_path}")
        return content

    except Exception as e:
        logger.error(f"Failed to read text from {file_path}: {e}")
        return None


def get_unique_filename(base_path: Union[str, Path], extension: str = "") -> Path:
    """
    Get unique filename by adding counter if file exists.

    Args:
        base_path: Base file path
        extension: File extension (optional)

    Returns:
        Unique file path
    """
    base_path = Path(base_path)
    if extension and not extension.startswith("."):
        extension = f".{extension}"

    # If no extension provided, use the original extension
    if not extension and base_path.suffix:
        extension = base_path.suffix
        base_path = base_path.with_suffix("")

    counter = 0
    while True:
        if counter == 0:
            candidate = base_path.with_suffix(extension)
        else:
            candidate = base_path.with_suffix(f"_{counter}{extension}")

        if not candidate.exists():
            return candidate

        counter += 1
        if counter > 10000:  # Prevent infinite loop
            raise ValueError("Too many files with similar names")


def cleanup_old_files(
    directory: Union[str, Path], pattern: str = "*.backup.*", keep_count: int = 5
) -> int:
    """
    Clean up old backup files, keeping only the most recent ones.

    Args:
        directory: Directory to clean up
        pattern: File pattern to match
        keep_count: Number of files to keep

    Returns:
        Number of files removed
    """
    directory = Path(directory)

    if not directory.exists():
        return 0

    try:
        # Find matching files
        files = list(directory.glob(pattern))

        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Remove old files
        removed_count = 0
        for file in files[keep_count:]:
            try:
                file.unlink()
                removed_count += 1
                logger.debug(f"Removed old file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file}: {e}")

        return removed_count

    except Exception as e:
        logger.error(f"Failed to cleanup old files in {directory}: {e}")
        return 0


def get_file_info(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Get file information including size, modification time, etc.

    Args:
        file_path: File path to inspect

    Returns:
        File information dictionary or None if failed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return None

    try:
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
        }
    except Exception as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        return None


def get_output_path(
    logs_root: str, output_type: str, session_id: str = None, filename: str = None
) -> str:
    """Get standardized output path for code/bin/analysis/verbose/record, optionally by session_id."""
    from pathlib import Path

    base = Path(logs_root) / output_type
    if session_id:
        base = base / session_id
    if filename:
        base = base / filename
    base.parent.mkdir(parents=True, exist_ok=True)
    return str(base)


def get_operator_name_from_task(task_description: str) -> str:
    """Extract operator name from task description (e.g., 'matrix multiplication', 'conv2d')."""
    # Simple rule: take the first English word or short phrase
    match = re.search(r"([a-zA-Z0-9_\-]+)", task_description)
    if match:
        return sanitize_filename(match.group(1).lower())
    return "unknown_op"
