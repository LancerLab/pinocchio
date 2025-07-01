"""
Configuration management module for Pinocchio.

This module provides a centralized configuration management system that supports:
1. Loading configuration from JSON and YAML files
2. Environment variable overrides
3. Simple get/set interface
4. Default value mechanism
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigError(Exception):
    """Base class for configuration related errors."""

    pass


class ConfigFileError(ConfigError):
    """Raised when there are issues with configuration files."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    pass


class Settings:
    """Central configuration manager for Pinocchio."""

    def __init__(self) -> None:
        """Initialize the settings manager with empty configuration."""
        self._config: Dict[str, Any] = {}
        self._config_sources: Dict[str, str] = {}  # Tracks where each config came from
        self._defaults: Dict[str, Any] = {}

    def load_defaults(self, defaults: Dict[str, Any]) -> None:
        """
        Load default configuration values.

        Args:
            defaults: Dictionary containing default configuration values
        """
        self._defaults = defaults.copy()
        # Apply defaults that don't have values set
        for key, value in self._defaults.items():
            if key not in self._config:
                self._set_value(key, value, "default")

    def load_from_dict(self, config_dict: Dict[str, Any], source: str = "dict") -> None:
        """
        Load configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration values
            source: Source identifier for tracking
        """
        for key, value in config_dict.items():
            self._set_value(key, value, source)

    def load_from_env(self, prefix: str = "PINOCCHIO_") -> None:
        """
        Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variables to consider
        """
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable name to config key
                # Example: PINOCCHIO_APP_NAME -> app.name
                config_key = key[len(prefix) :].lower()
                config_key = config_key.replace("_", ".")

                # Try to parse JSON values from environment
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value

                self._set_value(config_key, parsed_value, "environment")

    def _load_json_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.

        Args:
            filepath: Path to the JSON file

        Returns:
            Dictionary containing configuration values

        Raises:
            ConfigFileError: If file cannot be parsed as JSON
        """
        try:
            with open(filepath, "r") as f:
                result = json.load(f)
                if not isinstance(result, dict):
                    raise ConfigFileError("JSON file must contain a dictionary")
                return dict(result)
        except json.JSONDecodeError as e:
            raise ConfigFileError(f"Invalid JSON in configuration file: {e}")

    def _load_yaml_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            filepath: Path to the YAML file

        Returns:
            Dictionary containing configuration values

        Raises:
            ConfigFileError: If PyYAML is not installed or file cannot be parsed
        """
        try:
            import yaml
        except ImportError:
            raise ConfigFileError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml"
            )

        try:
            with open(filepath, "r") as f:
                result = yaml.safe_load(f)
                if not isinstance(result, dict):
                    raise ConfigFileError("YAML file must contain a dictionary")
                return dict(result)
        except Exception as e:
            raise ConfigFileError(f"Error reading YAML configuration file: {e}")

    def load_from_file(
        self, filepath: Union[str, Path], format: Optional[str] = None
    ) -> None:
        """
        Load configuration from a file (JSON or YAML).

        Args:
            filepath: Path to the configuration file
            format: File format ('json', 'yaml', or None to infer from extension)

        Raises:
            ConfigFileError: If file cannot be read or parsed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise ConfigFileError(f"Configuration file not found: {filepath}")

        # Determine format from extension if not specified
        if format is None:
            format = filepath.suffix.lstrip(".").lower()
            if format not in ["json", "yaml", "yml"]:
                format = "json"  # Default to JSON if extension is not recognized

        try:
            if format == "json":
                config_dict = self._load_json_file(filepath)
            elif format in ["yaml", "yml"]:
                config_dict = self._load_yaml_file(filepath)
            else:
                raise ConfigFileError(f"Unsupported configuration format: {format}")
        except ConfigFileError:
            raise
        except Exception as e:
            raise ConfigFileError(f"Error reading configuration file: {e}")

        self.load_from_dict(config_dict, f"file:{filepath}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested configs)
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        # First try to get from config
        value = self._get_nested(self._config, key)
        if value is not None:
            return value

        # Then try defaults
        value = self._get_nested(self._defaults, key)
        if value is not None:
            return value

        return default

    def set(self, key: str, value: Any, source: str = "runtime") -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key (supports dot notation for nested configs)
            value: Configuration value
            source: Source identifier for tracking
        """
        self._set_value(key, value, source)

    def _set_value(self, key: str, value: Any, source: str) -> None:
        """
        Internal method to set a configuration value with source tracking.

        Args:
            key: Configuration key (supports dot notation for nested configs)
            value: Configuration value
            source: Source identifier for tracking
        """
        keys = key.split(".")
        config = self._config

        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value
        self._config_sources[key] = source

    def _get_nested(self, config: Dict[str, Any], key: str) -> Optional[Any]:
        """
        Get a value from nested dictionary using dot notation.

        Args:
            config: Configuration dictionary
            key: Key in dot notation (e.g., 'database.host')

        Returns:
            Value if found, None otherwise
        """
        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()

    def get_source(self, key: str) -> Optional[str]:
        """
        Get the source of a configuration value.

        Args:
            key: Configuration key

        Returns:
            Source identifier or None if key not found
        """
        return self._config_sources.get(key)

    def _export_json(self, filepath: str, config: Dict[str, Any]) -> None:
        """
        Write the configuration to a JSON file.

        Args:
            filepath: Path to export the configuration to
            config: Configuration dictionary to export
        """
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

    def _export_yaml(self, filepath: str, config: Dict[str, Any]) -> None:
        """
        Write the configuration to a YAML file.

        Args:
            filepath: Path to export configuration to
            config: Configuration dictionary to export

        Raises:
            ConfigFileError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ConfigFileError(
                "PyYAML is required for YAML support. Install with: pip install pyyaml"
            )

        with open(filepath, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def export_config(
        self, filepath: str, format: str = "json", include_defaults: bool = True
    ) -> None:
        """
        Write current configuration to a file.

        Args:
            filepath: Path to export configuration to
            format: File format ('json' or 'yaml')
            include_defaults: Whether to include default values

        Raises:
            ConfigFileError: If export fails
        """
        config = (
            self.get_all()
            if include_defaults
            else {
                k: v
                for k, v in self.get_all().items()
                if self.get_source(k) != "default"
            }
        )

        try:
            if format.lower() == "json":
                self._export_json(filepath, config)
            elif format.lower() in ["yaml", "yml"]:
                self._export_yaml(filepath, config)
            else:
                raise ConfigFileError(f"Unsupported format: {format}")
        except ConfigFileError:
            raise
        except Exception as e:
            raise ConfigFileError(f"Error exporting configuration: {e}")


# Global settings instance
settings = Settings()
