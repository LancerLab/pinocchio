"""
Configuration schema validation module for Pinocchio.

This module provides schema validation for configuration values using Pydantic.
It ensures that configuration values meet expected formats and constraints.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError

from .settings import ConfigValidationError


class ConfigSchema(BaseModel):
    """Base class for configuration schemas."""

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against the schema.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Validated configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            validated = cls(**config)
            return validated.model_dump()
        except ValidationError as e:
            raise ConfigValidationError(str(e))

    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """Get the JSON schema for this configuration."""
        return cls.model_json_schema()


# Example schema classes
class LLMConfig(ConfigSchema):
    """Configuration schema for LLM settings."""

    provider: str
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000


class AgentsConfig(ConfigSchema):
    """Configuration schema for agents settings."""

    generator: Dict[str, Any] = {}
    debugger: Dict[str, Any] = {}
    optimizer: Dict[str, Any] = {}
    evaluator: Dict[str, Any] = {}


class AppConfig(ConfigSchema):
    """Main application configuration schema."""

    llm: Optional[LLMConfig] = None
    agents: Optional[AgentsConfig] = None
    log_level: str = "INFO"
    memory_path: str = "./memory"
    knowledge_path: str = "./knowledge"
