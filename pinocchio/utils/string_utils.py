"""String utilities for Pinocchio modules."""

import logging
import re
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(" .")
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed_file"
    return sanitized


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extract code blocks from markdown text.

    Args:
        text: Text containing code blocks
        language: Specific language to extract (optional)

    Returns:
        List of code block contents
    """
    pattern = r"```(\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    code_blocks = []
    for lang, code in matches:
        if language is None or lang == language:
            code_blocks.append(code.strip())

    return code_blocks


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Text to normalize

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    return text.strip()


def split_safe(
    text: str, delimiter: str = "\n", max_parts: Optional[int] = None
) -> List[str]:
    """
    Safely split text by delimiter with optional max parts.

    Args:
        text: Text to split
        delimiter: Delimiter to split by
        max_parts: Maximum number of parts to split into

    Returns:
        List of text parts
    """
    if not text:
        return []

    parts = text.split(delimiter)

    if max_parts is not None:
        # Join remaining parts if we exceed max_parts
        if len(parts) > max_parts:
            parts = parts[: max_parts - 1] + [delimiter.join(parts[max_parts - 1 :])]

    return [part.strip() for part in parts if part.strip()]


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human readable string.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted file size string
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"


def is_valid_json(text: str) -> bool:
    """
    Check if text is valid JSON.

    Args:
        text: Text to check

    Returns:
        True if valid JSON, False otherwise
    """
    import json

    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.

    Args:
        text: Text to extract URLs from

    Returns:
        List of URLs found
    """
    url_pattern = r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?"
    return re.findall(url_pattern, text)


def remove_ansi_escape_codes(text: str) -> str:
    """
    Remove ANSI escape codes from text.

    Args:
        text: Text with ANSI codes

    Returns:
        Text without ANSI codes
    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)
