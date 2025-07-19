"""Configuration manager for Pinocchio."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from ..utils.file_utils import ensure_directory, safe_read_json, safe_write_json
from .models import (
    LLMConfigEntry,
    LLMConfigList,
    LLMProvider,
    PinocchioConfig,
    SessionConfig,
    StorageConfig,
)

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""

    pass


class ConfigManager:
    """Configuration manager for Pinocchio."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = self._load_config()

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        # Check environment variable first
        env_config = os.getenv("PINOCCHIO_CONFIG_FILE")
        if env_config and os.path.exists(env_config):
            return env_config

        # Try to find config in current directory or project root
        possible_paths = [
            "pinocchio.json",
            "config/pinocchio.json",
            "pinocchio/config/pinocchio.json",
            str(Path.home() / ".pinocchio" / "config.json"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # Return default path for creation
        return "pinocchio.json"

    def _load_config(self) -> PinocchioConfig:
        """Load configuration from file using Pydantic models."""
        default_config = PinocchioConfig()

        if os.path.exists(self.config_file):
            try:
                file_data = safe_read_json(self.config_file)
                if file_data is not None:
                    # Create config from file data, merging with defaults
                    config = PinocchioConfig(**file_data)
                    logger.info(f"Configuration loaded from {self.config_file}")
                    return config
                else:
                    raise ConfigError(
                        f"Failed to load config from {self.config_file}: Invalid JSON format"
                    )
            except Exception as e:
                # Don't fallback to default config, instead raise the error
                if isinstance(e, ConfigError):
                    raise
                else:
                    raise ConfigError(
                        f"Configuration validation failed for {self.config_file}: {e}"
                    )
        else:
            # Create default config file
            try:
                # Use utils function to ensure directory exists
                ensure_directory(Path(self.config_file).parent)
                # Use utils function for safe JSON writing
                success = safe_write_json(default_config.model_dump(), self.config_file)
                if success:
                    logger.info(
                        f"Created default configuration file: {self.config_file}"
                    )
                    return default_config
                else:
                    raise ConfigError(
                        f"Failed to create config file: {self.config_file}"
                    )
            except Exception as e:
                raise ConfigError(f"Failed to create config file: {e}")

    def get_llm_config(self) -> LLMConfigEntry:
        """Get the best LLM configuration (auto select by priority)."""
        # First try to use the new llms array
        if hasattr(self.config, "llms") and self.config.llms:
            # Use the first LLM in the array as default
            return self.config.llms[0]

        # Fallback to legacy llm field
        llm = self.config.llm
        if isinstance(llm, list):
            # Convert legacy list to LLMConfigEntry objects with generated IDs
            llm_entries = []
            for i, item in enumerate(llm):
                if isinstance(item, dict):
                    # Add id if not present
                    if "id" not in item:
                        item["id"] = f"legacy_{i}"
                    llm_entries.append(LLMConfigEntry(**item))
                elif isinstance(item, LLMConfigEntry):
                    llm_entries.append(item)
            if llm_entries:
                return llm_entries[0]  # Return first one as default
        elif isinstance(llm, dict):
            # Single dict, add id if not present
            if "id" not in llm:
                llm["id"] = "legacy_main"
            return LLMConfigEntry(**llm)

        # If no valid config found, create a default
        return LLMConfigEntry(
            id="default",
            provider=LLMProvider.CUSTOM,
            base_url="http://localhost:8001",
            model_name="default",
            priority=10,
        )

    def get_all_llm_configs(self):
        """Return all LLM configs as a list of LLMConfigEntry."""
        # First try to use the new llms array
        if hasattr(self.config, "llms") and self.config.llms:
            return self.config.llms

        # Fallback to legacy llm field
        llm = self.config.llm
        if isinstance(llm, list):
            return [
                LLMConfigEntry(**item) if not isinstance(item, LLMConfigEntry) else item
                for item in llm
            ]
        elif isinstance(llm, LLMConfigList):
            return llm.llms
        elif isinstance(llm, LLMConfigEntry):
            return [llm]
        elif isinstance(llm, dict):
            return [LLMConfigEntry(**llm)]
        else:
            return []

    def get_agent_config(self, agent_type: str) -> Optional[Any]:
        """Get agent configuration."""
        agents = self.config.agents
        return getattr(agents, agent_type, None)

    def get_agent_llm_config(self, agent_type: str) -> LLMConfigEntry:
        """Get agent-specific LLM configuration with fallback to global config."""
        # Get agent config
        agent_config = self.get_agent_config(agent_type)
        if agent_config and hasattr(agent_config, "llm") and agent_config.llm:
            # Agent has specified an LLM ID, find it in the llms array
            llm_id = agent_config.llm
            if hasattr(self.config, "llms") and self.config.llms:
                for llm_config in self.config.llms:
                    if llm_config.id == llm_id:
                        return llm_config
                logger.warning(
                    f"LLM config with id '{llm_id}' not found for agent '{agent_type}', using default"
                )

        # Fallback to global config
        return self.get_llm_config()

    def get_llm_config_by_id(self, llm_id: str) -> Optional[LLMConfigEntry]:
        """Get LLM configuration by ID."""
        if hasattr(self.config, "llms") and self.config.llms:
            for llm_config in self.config.llms:
                if llm_config.id == llm_id:
                    return llm_config
        return None

    def get_session_config(self) -> SessionConfig:
        """Get session configuration."""
        return self.config.session

    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        return self.config.storage

    def get_logs_path(self) -> str:
        """Get logs root path from storage config."""
        return self.config.storage.logs_path

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_verbose_config(self) -> dict:
        """Get verbose logging configuration."""
        verbose_config = self.get("verbose", {})
        if isinstance(verbose_config, dict):
            return verbose_config
        else:
            # If verbose config is a Pydantic model, convert to dict
            return (
                verbose_config.model_dump()
                if hasattr(verbose_config, "model_dump")
                else {}
            )

    def is_verbose_enabled(self) -> bool:
        """Check if verbose logging is enabled."""
        return self.get("verbose.enabled", False)

    def get_verbose_mode(self) -> str:
        """Get verbose mode (development/production/debug)."""
        return self.get("verbose.mode", "production")

    def get_verbose_level(self) -> str:
        """Get verbose level (minimal/detailed/maximum)."""
        return self.get("verbose.level", "minimal")

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split(".")
        config_dict = self.config.model_dump()

        # Navigate to the target location
        current = config_dict
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

        # Recreate the config object
        self.config = PinocchioConfig(**config_dict)

    def save(self) -> None:
        """Save configuration to file."""
        try:
            # Use utils function to ensure directory exists
            ensure_directory(Path(self.config_file).parent)
            # Use utils function for safe JSON writing
            success = safe_write_json(self.config.model_dump(), self.config_file)
            if success:
                logger.info(f"Configuration saved to {self.config_file}")
            else:
                logger.error(f"Failed to save configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def reload(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()

    def validate_config(self) -> bool:
        """Validate current configuration."""
        try:
            # Pydantic will validate on creation
            PinocchioConfig(**self.config.model_dump())
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def get_config_value(config, key, default=None):
    """
    Safely get a nested config value from a Pydantic model or dict using dot notation.
    """
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k, default)
        elif hasattr(value, k):
            value = getattr(value, k, default)
        else:
            return default
        if value is None:
            return default
    return value
