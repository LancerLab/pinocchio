"""Validation utilities for Pinocchio modules."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def validate_file_path(path: Union[str, Path]) -> bool:
    """
    Validate if file path is safe and accessible.

    Args:
        path: File path to validate

    Returns:
        True if path is valid, False otherwise
    """
    try:
        path_obj = Path(path)
        # Check if path is absolute and within reasonable bounds
        if path_obj.is_absolute():
            # Prevent access to system directories
            forbidden_prefixes = [
                "/etc",
                "/var",
                "/usr",
                "/bin",
                "/sbin",
                "/sys",
                "/proc",
            ]
            for prefix in forbidden_prefixes:
                if str(path_obj).startswith(prefix):
                    logger.warning(f"Access to system directory blocked: {path}")
                    return False

        # Check for path traversal attempts
        if ".." in str(path_obj):
            logger.warning(f"Path traversal attempt detected: {path}")
            return False

        return True
    except Exception as e:
        logger.error(f"Path validation failed for {path}: {e}")
        return False


def validate_json_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate JSON structure has required keys.

    Args:
        data: JSON data to validate
        required_keys: List of required keys

    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(data, dict):
        logger.error("Data is not a dictionary")
        return False

    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        logger.error(f"Missing required keys: {missing_keys}")
        return False

    return True


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        True if email is valid, False otherwise
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid, False otherwise
    """
    pattern = r"^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$"
    return bool(re.match(pattern, url))


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema.

    Args:
        config: Configuration to validate
        schema: Schema definition

    Returns:
        True if config matches schema, False otherwise
    """

    def validate_section(data: Any, schema_def: Any) -> bool:
        if isinstance(schema_def, dict):
            if not isinstance(data, dict):
                return False
            for key, value in schema_def.items():
                if key not in data:
                    if schema_def.get("required", True):
                        return False
                elif not validate_section(data[key], value):
                    return False
        elif isinstance(schema_def, list):
            if not isinstance(data, list):
                return False
            for item in data:
                if not validate_section(item, schema_def[0]):
                    return False
        elif isinstance(schema_def, type):
            if not isinstance(data, schema_def):
                return False
        return True

    return validate_section(config, schema)


def validate_agent_response_format(response: Dict[str, Any]) -> bool:
    """
    Validate agent response format.

    Args:
        response: Agent response to validate

    Returns:
        True if response format is valid, False otherwise
    """
    required_fields = ["success", "agent_type"]
    optional_fields = ["output", "error_message", "request_id"]

    # Check required fields
    for field in required_fields:
        if field not in response:
            logger.error(f"Missing required field in agent response: {field}")
            return False

    # Check field types
    if not isinstance(response["success"], bool):
        logger.error("Field 'success' must be boolean")
        return False

    if not isinstance(response["agent_type"], str):
        logger.error("Field 'agent_type' must be string")
        return False

    return True


def validate_session_data(session_data: Dict[str, Any]) -> bool:
    """
    Validate session data structure.

    Args:
        session_data: Session data to validate

    Returns:
        True if session data is valid, False otherwise
    """
    required_fields = ["session_id", "created_at", "status"]

    for field in required_fields:
        if field not in session_data:
            logger.error(f"Missing required field in session data: {field}")
            return False

    # Validate session_id format
    if not re.match(r"^[a-f0-9]{8,}$", session_data["session_id"]):
        logger.error("Invalid session_id format")
        return False

    # Validate status
    valid_statuses = ["active", "completed", "failed", "cancelled"]
    if session_data["status"] not in valid_statuses:
        logger.error(f"Invalid session status: {session_data['status']}")
        return False

    return True


def validate_memory_data(memory_data: Dict[str, Any]) -> bool:
    """
    Validate memory data structure.

    Args:
        memory_data: Memory data to validate

    Returns:
        True if memory data is valid, False otherwise
    """
    required_fields = ["memory_id", "agent_type", "content", "created_at"]

    for field in required_fields:
        if field not in memory_data:
            logger.error(f"Missing required field in memory data: {field}")
            return False

    # Validate content is not empty
    if not memory_data["content"]:
        logger.error("Memory content cannot be empty")
        return False

    return True


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input text.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""

    # Remove null bytes and control characters
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\r\t")

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate

    Returns:
        True if API key is valid, False otherwise
    """
    if not isinstance(api_key, str):
        return False

    # Check minimum length
    if len(api_key) < 10:
        return False

    # Check for common patterns
    if api_key.startswith("sk-") or api_key.startswith("pk_"):
        return True

    # Allow other formats but ensure they're not empty or too short
    return len(api_key.strip()) >= 10


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

    Raises:
        AssertionError: If structure validation fails
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
