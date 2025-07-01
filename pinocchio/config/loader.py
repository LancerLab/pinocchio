"""
Configuration loader module for Pinocchio.

This module provides utilities for loading configuration from multiple sources with precedence.
"""

from typing import Any, Dict, List, Optional, Type

from .schema import ConfigSchema
from .settings import Settings


class ConfigLoader:
    """Utility for loading configuration from multiple sources with precedence."""

    def __init__(
        self, settings: Settings, config_schema: Optional[Type[ConfigSchema]] = None
    ):
        """
        Initialize the configuration loader.

        Args:
            settings: Settings instance to use
            config_schema: Optional schema for validation
        """
        self.settings = settings
        self.config_schema = config_schema

    def load_config(
        self,
        default_config: Optional[Dict[str, Any]] = None,
        config_files: Optional[List[str]] = None,
        env_prefix: str = "PINOCCHIO_",
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Load configuration from multiple sources with precedence.

        Precedence order (highest to lowest):
        1. Environment variables
        2. User config file
        3. Default config file
        4. Default values

        Args:
            default_config: Default configuration values
            config_files: List of configuration files to try (in order of precedence)
            env_prefix: Prefix for environment variables
            validate: Whether to validate the configuration

        Returns:
            Complete configuration dictionary
        """
        # Clear existing configuration to avoid interference
        self.settings._config.clear()
        self.settings._config_sources.clear()

        # Start with default config
        if default_config:
            self.settings.load_from_dict(default_config, "defaults")

        # Load from config files
        if config_files:
            for config_file in config_files:
                try:
                    self.settings.load_from_file(config_file)
                except Exception:
                    pass  # Skip missing or invalid files

        # Load from environment variables (highest precedence)
        self.settings.load_from_env(prefix=env_prefix)

        # Get the complete config
        config = self.settings.get_all()

        # Validate if requested and schema is available
        if validate and self.config_schema:
            config = self.config_schema.validate_config(config)

        return config

    def config_diff(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the difference between two configurations.

        Args:
            config1: First configuration
            config2: Second configuration

        Returns:
            Dictionary of differences
        """
        diff = {}

        # Find keys in config2 that differ from config1
        for key, value in config2.items():
            if key not in config1:
                diff[key] = {"added": value}
            elif isinstance(value, dict) and isinstance(config1[key], dict):
                nested_diff = self._nested_diff(config1[key], value)
                if nested_diff:
                    diff[key] = nested_diff
            elif config1[key] != value:
                diff[key] = {"from": config1[key], "to": value}

        # Find keys in config1 that are not in config2
        for key in config1:
            if key not in config2:
                diff[key] = {"removed": config1[key]}

        return diff

    def _nested_diff(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the difference between two nested dictionaries.

        Args:
            dict1: First dictionary
            dict2: Second dictionary

        Returns:
            Dictionary of differences
        """
        diff = {}

        # Find keys in dict2 that differ from dict1
        for key, value in dict2.items():
            if key not in dict1:
                diff[key] = {"added": value}
            elif isinstance(value, dict) and isinstance(dict1[key], dict):
                nested_diff = self._nested_diff(dict1[key], value)
                if nested_diff:
                    diff[key] = nested_diff
            elif dict1[key] != value:
                diff[key] = {"from": dict1[key], "to": value}

        # Find keys in dict1 that are not in dict2
        for key in dict1:
            if key not in dict2:
                diff[key] = {"removed": dict1[key]}

        return diff
