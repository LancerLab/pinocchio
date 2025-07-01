"""
Configuration management package for Pinocchio.

This package provides centralized configuration management with support for:
1. JSON file based configuration
2. Environment variable overrides
3. Secure credentials handling
4. Default values
"""

from .settings import (
    settings,
    ConfigError,
    ConfigFileError,
    ConfigValidationError,
)
from .credentials import credentials, CredentialsError

__all__ = [
    'settings',
    'credentials',
    'ConfigError',
    'ConfigFileError',
    'ConfigValidationError',
    'CredentialsError',
]
