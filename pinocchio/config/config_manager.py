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
    PinocchioConfig,
    SessionConfig,
    StorageConfig,
)

logger = logging.getLogger(__name__)


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
                    logger.warning(f"Failed to load config from {self.config_file}")
                    return default_config
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                return default_config
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
                else:
                    logger.warning(f"Failed to create config file: {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to create config file: {e}")

            return default_config

    def get_llm_config(self) -> LLMConfigEntry:
        """Get the best LLM configuration (auto select by priority)."""
        llm = self.config.llm
        if isinstance(llm, list):
            # Already a list of dicts (from JSON), parse to LLMConfigList
            llm_list = LLMConfigList(
                llms=[
                    LLMConfigEntry(**item)
                    if not isinstance(item, LLMConfigEntry)
                    else item
                    for item in llm
                ]
            )
            return llm_list.get_best_llm()
        elif isinstance(llm, LLMConfigList):
            return llm.get_best_llm()
        elif isinstance(llm, LLMConfigEntry):
            return llm
        elif isinstance(llm, dict):
            # Single dict, treat as one LLMConfigEntry
            return LLMConfigEntry(**llm)
        else:
            raise ValueError("Invalid LLM config format")

    def get_all_llm_configs(self):
        """Return all LLM configs as a list of LLMConfigEntry."""
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
        # Try to get agent-specific config
        agent_llm_key = f"llm_{agent_type}"
        agent_config = getattr(self.config, agent_llm_key, None)

        if agent_config is not None:
            return agent_config

        # Fallback to global config
        return self.get_llm_config()

    def get_session_config(self) -> SessionConfig:
        """Get session configuration."""
        return self.config.session

    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration."""
        return self.config.storage

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
