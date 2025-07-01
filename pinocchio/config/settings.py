"""
Configuration management module for Pinocchio.

This module provides a centralized configuration management system that supports:
1. Loading configuration from JSON files
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
    
    def __init__(self):
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
                config_key = key[len(prefix):].lower()
                # Try to parse JSON values from environment
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                self._set_value(config_key, parsed_value, "environment")
                
    def load_from_file(self, filepath: Union[str, Path]) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the JSON configuration file
            
        Raises:
            ConfigFileError: If file cannot be read or parsed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise ConfigFileError(f"Configuration file not found: {filepath}")
            
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigFileError(f"Invalid JSON in configuration file: {e}")
        except IOError as e:
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
        keys = key.split('.')
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
        keys = key.split('.')
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

# Global settings instance
settings = Settings()
