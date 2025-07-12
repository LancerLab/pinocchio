"""Pydantic data models for Pinocchio configuration."""

from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class LLMProvider(str, Enum):
    """LLM provider types."""

    CUSTOM = "custom"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class LLMConfig(BaseModel):
    """LLM configuration data model."""

    provider: LLMProvider = Field(
        default=LLMProvider.CUSTOM, description="LLM provider type"
    )

    base_url: str = Field(
        default="http://localhost:8001", description="Base URL for the LLM service"
    )

    model_name: str = Field(default="default", description="Model name or identifier")

    timeout: int = Field(
        default=120, ge=1, le=600, description="Request timeout in seconds"
    )

    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retry attempts"
    )

    api_key: Optional[str] = Field(
        default=None, description="API key for the LLM service"
    )

    headers: Optional[Dict[str, str]] = Field(
        default=None, description="Additional headers for requests"
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v):
        """Validate base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout value."""
        if v < 1:
            raise ValueError("timeout must be at least 1 second")
        if v > 600:
            raise ValueError("timeout must not exceed 600 seconds")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v):
        """Validate max retries value."""
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        if v > 10:
            raise ValueError("max_retries must not exceed 10")
        return v

    model_config = {
        "extra": "forbid",  # Prevent additional fields
        "validate_assignment": True,  # Validate on assignment
    }


class AgentConfig(BaseModel):
    """Agent configuration data model."""

    enabled: bool = Field(default=True, description="Whether the agent is enabled")

    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retry attempts"
    )

    timeout: Optional[int] = Field(
        default=None, ge=1, le=600, description="Agent-specific timeout in seconds"
    )

    model_config = {"extra": "forbid", "validate_assignment": True}


class AgentsConfig(BaseModel):
    """Agents configuration data model."""

    generator: AgentConfig = Field(
        default_factory=AgentConfig, description="Generator agent configuration"
    )

    debugger: AgentConfig = Field(
        default_factory=AgentConfig, description="Debugger agent configuration"
    )

    optimizer: AgentConfig = Field(
        default_factory=AgentConfig, description="Optimizer agent configuration"
    )

    evaluator: AgentConfig = Field(
        default_factory=AgentConfig, description="Evaluator agent configuration"
    )

    model_config = {"extra": "forbid", "validate_assignment": True}


class SessionConfig(BaseModel):
    """Session configuration data model."""

    auto_save: bool = Field(default=True, description="Whether to auto-save sessions")

    save_interval_seconds: int = Field(
        default=300, ge=1, le=3600, description="Auto-save interval in seconds"
    )

    max_session_size_mb: int = Field(
        default=100, ge=1, le=1000, description="Maximum session size in MB"
    )

    model_config = {"extra": "forbid", "validate_assignment": True}


class StorageConfig(BaseModel):
    """Storage configuration data model."""

    sessions_path: str = Field(
        default="./sessions", description="Path to sessions directory"
    )

    memories_path: str = Field(
        default="./memories", description="Path to memories directory"
    )

    knowledge_path: str = Field(
        default="./knowledge", description="Path to knowledge directory"
    )

    model_config = {"extra": "forbid", "validate_assignment": True}


class PinocchioConfig(BaseModel):
    """Main Pinocchio configuration data model."""

    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")

    agents: AgentsConfig = Field(
        default_factory=AgentsConfig, description="Agents configuration"
    )

    session: SessionConfig = Field(
        default_factory=SessionConfig, description="Session configuration"
    )

    storage: StorageConfig = Field(
        default_factory=StorageConfig, description="Storage configuration"
    )

    model_config = {"extra": "forbid", "validate_assignment": True}
