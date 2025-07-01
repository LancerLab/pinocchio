"""Tests for the configuration management module."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from pinocchio.config import (
    ConfigFileError,
    ConfigLoader,
    ConfigSchema,
    ConfigValidationError,
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


@pytest.fixture
def yaml_config_file(tmp_path: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary YAML config file."""
    if not YAML_AVAILABLE:
        pytest.skip("PyYAML not installed")
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def credentials_file(tmp_path: Path) -> Path:
    """Create a temporary credentials file."""
    creds_file = tmp_path / "credentials.json"
    creds_data = {
        "openai_api_key": {"api_key": "sk-test-openai"},
        "anthropic_api_key": {"api_key": "sk-test-anthropic"},
    }
    with open(creds_file, "w") as f:
        json.dump(creds_data, f)
    return creds_file


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
    # Clear existing config
    settings._config.clear()

    # Test simple key
    os.environ["PINOCCHIO_TEST_KEY"] = "test-value"
    # Test JSON value
    os.environ["PINOCCHIO_TEST_JSON"] = '{"key": "value"}'
    # Test nested key
    os.environ["PINOCCHIO_APP_NAME"] = "env-app"

    settings.load_from_env()
    assert settings.get("test.key") == "test-value"
    assert settings.get("test.json") == {"key": "value"}
    assert settings.get("app.name") == "env-app"

    # Clean up
    del os.environ["PINOCCHIO_TEST_KEY"]
    del os.environ["PINOCCHIO_TEST_JSON"]
    del os.environ["PINOCCHIO_APP_NAME"]


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


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
def test_settings_load_from_yaml(yaml_config_file: Path):
    """Test loading configuration from YAML file."""
    settings.load_from_file(yaml_config_file)
    assert settings.get("app.name") == "test-app"
    assert settings.get("database.port") == 5432


def test_settings_export_config(tmp_path: Path, sample_config: Dict[str, Any]):
    """Test exporting configuration to file."""
    settings.load_from_dict(sample_config)

    # Export to JSON
    json_file = tmp_path / "export.json"
    settings.export_config(str(json_file), "json")
    assert json_file.exists()

    # Verify exported content
    with open(json_file, "r") as f:
        exported = json.load(f)
    assert "app" in exported
    assert exported["app"]["name"] == "test-app"

    # Export to YAML if available
    if YAML_AVAILABLE:
        yaml_file = tmp_path / "export.yaml"
        settings.export_config(str(yaml_file), "yaml")
        assert yaml_file.exists()

        # Verify exported content
        with open(yaml_file, "r") as f:
            exported = yaml.safe_load(f)
        assert "app" in exported
        assert exported["app"]["name"] == "test-app"


def test_credentials_load_from_file(credentials_file: Path):
    """Test loading credentials from file."""
    credentials._credentials.clear()
    credentials.load_from_file(str(credentials_file))
    assert credentials.get("openai_api_key") == "sk-test-openai"
    assert credentials.get("anthropic_api_key") == "sk-test-anthropic"


def test_credentials_save_to_file(tmp_path: Path):
    """Test saving credentials to file."""
    credentials._credentials.clear()
    credentials.set("openai_api_key", "sk-test-save")

    save_path = tmp_path / "saved_creds.json"
    credentials.save_to_file(str(save_path))
    assert save_path.exists()

    # Verify saved content
    with open(save_path, "r") as f:
        saved = json.load(f)
    assert "openai_api_key" in saved
    assert saved["openai_api_key"]["api_key"] == "sk-test-save"


class TestConfigSchema(ConfigSchema):
    """Test configuration schema."""

    name: str
    value: int
    optional: str = "default"


def test_config_schema_validation():
    """Test configuration schema validation."""
    # Valid config
    valid_config = {"name": "test", "value": 42}
    validated = TestConfigSchema.validate_config(valid_config)
    assert validated["name"] == "test"
    assert validated["value"] == 42
    assert validated["optional"] == "default"

    # Invalid config (wrong type)
    invalid_config = {"name": "test", "value": "not-an-int"}
    with pytest.raises(ConfigValidationError):
        TestConfigSchema.validate_config(invalid_config)

    # Invalid config (missing required field)
    invalid_config = {"name": "test"}
    with pytest.raises(ConfigValidationError):
        TestConfigSchema.validate_config(invalid_config)


def test_config_schema_get_schema():
    """Test getting JSON schema from config schema."""
    schema = TestConfigSchema.get_schema()
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["value"]["type"] == "integer"
    assert schema["properties"]["optional"]["default"] == "default"


def test_config_loader(sample_config: Dict[str, Any], config_file: Path):
    """Test configuration loader."""
    # Create a loader
    loader = ConfigLoader(settings)

    # Load with defaults
    config = loader.load_config(default_config=sample_config)
    assert config["app"]["name"] == "test-app"

    # Load from file
    config = loader.load_config(config_files=[str(config_file)])
    assert config["app"]["name"] == "test-app"

    # Test environment variable precedence
    try:
        os.environ["PINOCCHIO_APP_NAME"] = "env-app"
        # Create a new loader to ensure clean state
        new_loader = ConfigLoader(settings)
        config = new_loader.load_config(
            default_config=sample_config, config_files=[str(config_file)]
        )
        assert config["app"]["name"] == "env-app"
    finally:
        # Clean up environment variable
        if "PINOCCHIO_APP_NAME" in os.environ:
            del os.environ["PINOCCHIO_APP_NAME"]


def test_config_loader_with_schema():
    """Test configuration loader with schema validation."""
    # Create a loader with schema
    loader = ConfigLoader(settings, TestConfigSchema)

    # Valid config
    valid_config = {"name": "test", "value": 42}
    config = loader.load_config(default_config=valid_config)
    assert config["name"] == "test"
    assert config["optional"] == "default"

    # Invalid config
    settings._config.clear()
    invalid_config = {"name": "test", "value": "not-an-int"}
    with pytest.raises(ConfigValidationError):
        loader.load_config(default_config=invalid_config)


def test_config_diff():
    """Test configuration diff calculation."""
    # Create a loader
    loader = ConfigLoader(settings)

    # Test simple diff
    config1 = {"a": 1, "b": 2, "c": 3}
    config2 = {"a": 1, "b": 4, "d": 5}

    diff = loader.config_diff(config1, config2)
    assert "a" not in diff  # unchanged
    assert diff["b"] == {"from": 2, "to": 4}  # changed
    assert diff["c"] == {"removed": 3}  # removed
    assert diff["d"] == {"added": 5}  # added

    # Test nested diff
    config1 = {"nested": {"a": 1, "b": 2}}
    config2 = {"nested": {"a": 1, "c": 3}}

    diff = loader.config_diff(config1, config2)
    assert "nested" in diff
    assert "a" not in diff["nested"]  # unchanged
    assert diff["nested"]["b"] == {"removed": 2}  # removed
    assert diff["nested"]["c"] == {"added": 3}  # added
