"""
Credentials management module for Pinocchio.

This module provides secure handling of sensitive configuration values like API keys.
It ensures that sensitive values are:
1. Only loaded from secure sources (environment variables or secure files)
2. Never logged or exposed in error messages
3. Properly masked when displayed
"""

import os
from pathlib import Path
from typing import Dict, Optional

from .settings import ConfigError, settings


class CredentialsError(ConfigError):
    """Raised when there are issues with credentials."""

    pass


class Credentials:
    """Manager for sensitive configuration values."""

    # List of known sensitive configuration keys
    SENSITIVE_KEYS = {
        "openai_api_key",
        "anthropic_api_key",
        "github_token",
    }

    # Environment variable mapping
    ENV_MAPPING = {
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "github_token": "GITHUB_TOKEN",
    }

    def __init__(self):
        """Initialize the credentials manager."""
        self._credentials: Dict[str, str] = {}

    def load_from_env(self) -> None:
        """
        Load credentials from environment variables.

        This is the preferred way to load sensitive values.
        """
        self._credentials.clear()  # Clear existing credentials
        for key, env_var in self.ENV_MAPPING.items():
            if env_var in os.environ:
                self._credentials[key] = os.environ[env_var]
                # Store in settings but mark as sensitive
                settings.set(key, os.environ[env_var], "environment:sensitive")

    def get(self, key: str) -> Optional[str]:
        """
        Get a credential value.

        Args:
            key: Credential key

        Returns:
            Credential value or None if not found

        Raises:
            CredentialsError: If trying to access an unknown sensitive key
        """
        if key not in self.SENSITIVE_KEYS:
            raise CredentialsError(f"Unknown sensitive key: {key}")

        return self._credentials.get(key)

    def require(self, key: str) -> str:
        """
        Get a required credential value.

        Args:
            key: Credential key

        Returns:
            Credential value

        Raises:
            CredentialsError: If credential is not found
        """
        value = self.get(key)
        if value is None:
            env_var = self.ENV_MAPPING.get(key, key.upper())
            raise CredentialsError(
                f"Required credential '{key}' not found. "
                f"Please set the {env_var} environment variable."
            )
        return value

    def is_sensitive(self, key: str) -> bool:
        """
        Check if a key is considered sensitive.

        Args:
            key: Configuration key

        Returns:
            True if key is sensitive, False otherwise
        """
        return key in self.SENSITIVE_KEYS

    def mask_value(self, value: str) -> str:
        """
        Mask a sensitive value for display.

        Args:
            value: Value to mask

        Returns:
            Masked value (e.g., "sk-***")
        """
        if not value:
            return ""
        prefix = value[:3]
        masked_length = len(value) - len(prefix)
        return f"{prefix}{'*' * masked_length}"


# Global credentials instance
credentials = Credentials()
