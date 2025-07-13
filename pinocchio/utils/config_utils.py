"""Configuration utilities for Pinocchio modules."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .file_utils import safe_read_json, safe_write_json


def create_test_config(
    config_data: Dict[str, Any],
    config_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Create a test configuration file.

    Args:
        config_data: Configuration data to write
        config_path: Path for the config file (optional)

    Returns:
        Path to the created configuration file
    """
    if config_path is None:
        from .temp_utils import create_temp_file

        config_path = create_temp_file(suffix=".json", prefix="test_config_")

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    success = safe_write_json(config_data, config_path)
    if not success:
        raise RuntimeError(f"Failed to write config to {config_path}")

    return config_path


def load_test_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a test configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration data
    """
    config_path = Path(config_path)
    data = safe_read_json(config_path)
    if data is None:
        raise RuntimeError(f"Failed to read config from {config_path}")
    return data


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


def create_default_test_config() -> Dict[str, Any]:
    """
    Create a default test configuration.

    Returns:
        Default test configuration
    """
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "session": {
            "store_dir": "./test_sessions",
            "max_sessions": 100,
        },
        "agents": {
            "generator": {
                "enabled": True,
                "timeout": 300,
            },
            "optimizer": {
                "enabled": True,
                "timeout": 300,
            },
            "debugger": {
                "enabled": True,
                "timeout": 300,
            },
            "evaluator": {
                "enabled": True,
                "timeout": 300,
            },
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        Validated configuration

    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ["llm", "session", "agents"]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate LLM configuration
    llm_config = config["llm"]
    required_llm_keys = ["provider", "model"]
    for key in required_llm_keys:
        if key not in llm_config:
            raise ValueError(f"Missing required LLM configuration key: {key}")

    # Validate session configuration
    session_config = config["session"]
    if "store_dir" not in session_config:
        raise ValueError("Missing required session configuration key: store_dir")

    # Validate agents configuration
    agents_config = config["agents"]
    required_agents = ["generator", "optimizer", "debugger", "evaluator"]
    for agent in required_agents:
        if agent not in agents_config:
            raise ValueError(f"Missing required agent configuration: {agent}")

    return config


def create_minimal_test_config() -> Dict[str, Any]:
    """
    Create a minimal test configuration for basic testing.

    Returns:
        Minimal test configuration
    """
    return {
        "llm": {
            "provider": "mock",
            "model": "test-model",
        },
        "session": {
            "store_dir": "./test_sessions",
        },
        "agents": {
            "generator": {"enabled": True},
            "optimizer": {"enabled": True},
            "debugger": {"enabled": True},
            "evaluator": {"enabled": True},
        },
    }
