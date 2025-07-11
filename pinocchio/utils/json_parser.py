"""JSON parsing utilities for Pinocchio."""

import json
import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def safe_json_parse(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string with error handling.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not json_str or not isinstance(json_str, str):
        logger.warning("Invalid JSON input: empty or non-string")
        return None

    try:
        # Remove common formatting issues
        cleaned_str = json_str.strip()
        if cleaned_str.startswith("```json"):
            # Remove markdown code block formatting
            cleaned_str = cleaned_str.replace("```json", "").replace("```", "").strip()

        result = json.loads(cleaned_str)

        # Ensure result is a dictionary
        if isinstance(result, dict):
            return result
        else:
            logger.warning(f"JSON parsed but result is not a dict: {type(result)}")
            return {"content": result}

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e}")
        return None


def parse_structured_output(response: str) -> Dict[str, Any]:
    """
    Parse LLM response and extract structured output.

    Args:
        response: Raw LLM response string

    Returns:
        Structured dictionary with parsed content
    """
    if not response:
        return {"error": "Empty response"}

    # Try to parse as JSON first
    parsed = safe_json_parse(response)
    if parsed:
        return parsed

    # If JSON parsing fails, return as plain text
    return {"content": response.strip(), "format": "plain_text", "parsed": False}


def format_json_response(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format dictionary as pretty JSON string.

    Args:
        data: Dictionary to format
        indent: JSON indentation level

    Returns:
        Formatted JSON string
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"JSON formatting failed: {e}")
        return str(data)


def validate_agent_response(response: Dict[str, Any]) -> bool:
    """
    Validate agent response structure.

    Args:
        response: Agent response dictionary

    Returns:
        True if response is valid, False otherwise
    """
    required_fields = ["agent_type", "success"]

    if not isinstance(response, dict):
        return False

    for field in required_fields:
        if field not in response:
            logger.warning(f"Missing required field in agent response: {field}")
            return False

    return True


def extract_code_from_response(response: Union[str, Dict[str, Any]]) -> Optional[str]:
    """
    Extract code content from agent response.

    Args:
        response: Agent response (string or dict)

    Returns:
        Extracted code string or None
    """
    if isinstance(response, str):
        # Try to extract code from markdown blocks
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Code blocks are at odd indices
                    # Remove language identifier if present
                    lines = part.strip().split("\n")
                    if lines and not lines[0].strip().startswith(("/", "#", "*")):
                        lines = lines[1:]  # Remove language line
                    return "\n".join(lines)
        return response.strip()

    elif isinstance(response, dict):
        # Look for common code fields
        code_fields = ["code", "content", "result", "output"]
        for field in code_fields:
            if field in response and isinstance(response[field], str):
                return response[field].strip()

    return None
