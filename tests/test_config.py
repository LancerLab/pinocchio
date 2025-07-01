"""Tests for the configuration management module."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from pinocchio.config import (
    ConfigError,
    ConfigFileError,
    CredentialsError,
    credentials,
    settings,
)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration dictionary."""
    return {
        "app": {
            "name": "test-app",
            "debug": True,
        },
        "database": {
            "host": "localhost",
            "port": 5432,
        },
    }


@pytest.fixture
def config_file(tmp_path: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary config file."""
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(sample_config, f)
    return config_file


def test_settings_load_from_dict(sample_config: Dict[str, Any]):
    """Test loading configuration from dictionary."""
    settings.load_from_dict(sample_config)
    assert settings.get("app.name") == "test-app"
    assert settings.get("database.port") == 5432


def test_settings_load_from_file(config_file: Path):
    """Test loading configuration from file."""
    settings.load_from_file(config_file)
    assert settings.get("app.name") == "test-app"
    assert settings.get("database.port") == 5432


def test_settings_load_from_env():
    """Test loading configuration from environment variables."""
    os.environ["PINOCCHIO_TEST_KEY"] = "test-value"
    os.environ["PINOCCHIO_TEST_JSON"] = '{"key": "value"}'

    settings.load_from_env()
    assert settings.get("test_key") == "test-value"
    assert settings.get("test_json") == {"key": "value"}

    del os.environ["PINOCCHIO_TEST_KEY"]
    del os.environ["PINOCCHIO_TEST_JSON"]


def test_settings_default_values():
    """Test default value mechanism."""
    defaults = {"key1": "default1", "key2": "default2"}
    settings.load_defaults(defaults)

    assert settings.get("key1") == "default1"
    settings.set("key1", "value1")
    assert settings.get("key1") == "value1"


def test_settings_nonexistent_key():
    """Test getting nonexistent configuration key."""
    assert settings.get("nonexistent") is None
    assert settings.get("nonexistent", "default") == "default"


def test_settings_nested_dict():
    """Test nested dictionary support."""
    nested = {"level1": {"level2": {"key": "value"}}}
    settings.load_from_dict(nested)
    assert settings.get("level1.level2.key") == "value"


def test_settings_invalid_file():
    """Test loading from invalid file."""
    with pytest.raises(ConfigFileError):
        settings.load_from_file("nonexistent.json")


def test_credentials_load_from_env():
    """Test loading credentials from environment variables."""
    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    credentials.load_from_env()
    assert credentials.get("openai_api_key") == "sk-test-key"
    del os.environ["OPENAI_API_KEY"]


def test_credentials_mask_value():
    """Test credential value masking."""
    assert credentials.mask_value("sk-test-key") == "sk-********"
    assert credentials.mask_value("") == ""


def test_credentials_unknown_key():
    """Test accessing unknown sensitive key."""
    with pytest.raises(CredentialsError):
        credentials.get("unknown_key")


def test_credentials_require():
    """Test requiring credential value."""
    # Clear any existing credentials
    credentials._credentials.clear()

    with pytest.raises(CredentialsError):
        credentials.require("openai_api_key")

    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    credentials.load_from_env()
    assert credentials.require("openai_api_key") == "sk-test-key"
    del os.environ["OPENAI_API_KEY"]
