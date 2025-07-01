"""
Configuration management package for Pinocchio.

This package provides centralized configuration management with support for:
1. JSON and YAML file based configuration
2. Environment variable overrides
3. Secure credentials handling
4. Configuration validation
5. Default values
"""

from .credentials import CredentialsError, credentials
from .loader import ConfigLoader
from .schema import AgentsConfig, AppConfig, ConfigSchema, LLMConfig
from .settings import ConfigError, ConfigFileError, ConfigValidationError, settings

__all__ = [
    "settings",
    "credentials",
    "ConfigError",
    "ConfigFileError",
    "ConfigValidationError",
    "CredentialsError",
    "ConfigSchema",
    "LLMConfig",
    "AgentsConfig",
    "AppConfig",
    "ConfigLoader",
]
