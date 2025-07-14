"""JSON parsing utilities for Pinocchio."""

import json
import logging
import re
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def _extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from text using bracket counting.
    Returns the substring or None if not found.
    """
    start = text.find("{")
    if start == -1:
        return None
    count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            count += 1
        elif text[i] == "}":
            count -= 1
            if count == 0:
                return text[start : i + 1]
    return None


def _try_recover_truncated_json(text: str) -> Optional[dict]:
    """
    Attempt to recover a truncated JSON object by completing missing right braces.
    """
    first = text.find("{")
    if first == -1:
        return None
    # Count the number of right braces needed to complete
    count = 0
    for i in range(first, len(text)):
        if text[i] == "{":
            count += 1
        elif text[i] == "}":
            count -= 1
    # Need count right braces
    candidate = text[first:] + ("}" * count if count > 0 else "")
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            logger.info("Recovered JSON object by auto-completing brackets.")
            return result
    except Exception as e:
        logger.warning(f"Failed to recover truncated JSON by bracket completion: {e}")
    return None


def safe_json_parse(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON string with error handling. Now supports extracting JSON from extra text and partial JSON recovery.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not json_str or not isinstance(json_str, str):
        logger.warning("Invalid JSON input: empty or non-string")
        return None

    # Remove markdown code block formatting
    cleaned_str = json_str.strip()
    if cleaned_str.startswith("```json"):
        cleaned_str = cleaned_str.replace("```json", "").replace("```", "").strip()

    # Try direct parse first
    try:
        result = json.loads(cleaned_str)
        if isinstance(result, dict):
            return result
        else:
            logger.warning(f"JSON parsed but result is not a dict: {type(result)}")
            return {"content": result}
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parsing failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during JSON parsing: {e}")
        return None

    # Try to extract first JSON object from text using bracket counting
    json_candidate = _extract_first_json_object(cleaned_str)
    if json_candidate:
        try:
            result = json.loads(json_candidate)
            if isinstance(result, dict):
                logger.info("Recovered JSON object from extra text.")
                return result
        except Exception as e:
            logger.warning(f"Failed to parse extracted JSON object: {e}")

    # Try to recover from partial JSON (truncated at end, auto-complete brackets)
    recovered = _try_recover_truncated_json(cleaned_str)
    if recovered:
        return recovered

    logger.error("JSON parsing failed after all recovery attempts.")
    return None


def parse_structured_output(response: str) -> Dict[str, Any]:
    """
    Parse LLM response and extract structured output. Now tries to extract JSON from extra text before falling back to plain text.

    Args:
        response: Raw LLM response string

    Returns:
        Structured dictionary with parsed content
    """
    if not response:
        return {"error": "Empty response"}

    # Try to parse as JSON first (with recovery)
    parsed = safe_json_parse(response)
    if parsed:
        return parsed

    # Try to extract JSON from text using bracket counting (if not already tried)
    json_candidate = _extract_first_json_object(response)
    if json_candidate:
        try:
            result = json.loads(json_candidate)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    # If all fails, return as plain text
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
