"""Pydantic data models for Pinocchio configuration."""

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""

    CUSTOM = "custom"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class LLMConfigEntry(BaseModel):
    """Configuration entry for a single LLM provider."""

    provider: LLMProvider = Field(..., description="LLM provider type")
    base_url: Optional[str] = Field(
        None, description="Base URL for the LLM service (for custom)"
    )
    model_name: str = Field(..., description="Model name or identifier")
    timeout: int = Field(
        default=120, ge=1, le=600, description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retry attempts"
    )
    api_key: Optional[str] = Field(
        None, description="API key for the LLM service (for openai/anthropic)"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None, description="Additional headers for requests"
    )
    priority: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Priority for auto selection, lower is higher priority",
    )
    label: Optional[str] = Field(None, description="Optional label for this LLM config")

    def __str__(self):
        """Return string representation of the LLM config entry."""
        return f"LLMConfigEntry(provider={self.provider}, model_name={self.model_name}, base_url={self.base_url}, priority={self.priority})"

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v, values):
        """Validate base_url for custom providers."""
        provider = values.data.get("provider")
        if provider == LLMProvider.CUSTOM and (
            not v or not v.startswith(("http://", "https://"))
        ):
            raise ValueError(
                "base_url must be set and start with http:// or https:// for custom provider"
            )
        return v

    model_config = {"extra": "forbid", "validate_assignment": True}


class LLMConfigList(BaseModel):
    """List of LLM configurations with priority-based selection."""

    llms: List[LLMConfigEntry] = Field(..., description="List of LLM configurations")

    def get_best_llm(self) -> LLMConfigEntry:
        """Get the best LLM configuration based on priority and provider ranking."""

        # Sort by priority (lower is better), custom > openai > anthropic > mock
        def sort_key(entry: LLMConfigEntry):
            provider_rank = {
                LLMProvider.CUSTOM: 0,
                LLMProvider.OPENAI: 1,
                LLMProvider.ANTHROPIC: 2,
                LLMProvider.MOCK: 3,
            }.get(entry.provider, 99)
            return (entry.priority, provider_rank)

        return sorted(self.llms, key=sort_key)[0]

    model_config = {"extra": "forbid", "validate_assignment": True}


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


class DebugRepairConfig(BaseModel):
    """Debug repair configuration data model."""

    max_repair_attempts: int = Field(
        default=3, ge=1, le=10, description="Maximum number of repair attempts"
    )

    model_config = {"extra": "forbid", "validate_assignment": True}


class OptimizationConfig(BaseModel):
    """Optimization configuration data model."""

    max_optimisation_rounds: int = Field(
        default=3, ge=1, le=10, description="Maximum number of optimization rounds"
    )

    optimizer_enabled: bool = Field(
        default=True, description="Whether optimizer is enabled"
    )

    model_config = {"extra": "forbid", "validate_assignment": True}


class PinocchioConfig(BaseModel):
    """Main Pinocchio configuration data model."""

    llm: Union[LLMConfigEntry, List[LLMConfigEntry]] = Field(
        default_factory=lambda: [
            LLMConfigEntry(
                provider=LLMProvider.CUSTOM,
                base_url="http://localhost:8001",
                model_name="default",
                priority=1,
            )
        ],
        description="LLM configuration(s)",
    )
    agents: AgentsConfig = Field(
        default_factory=AgentsConfig, description="Agents configuration"
    )
    session: SessionConfig = Field(
        default_factory=SessionConfig, description="Session configuration"
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig, description="Storage configuration"
    )
    debug_repair: DebugRepairConfig = Field(
        default_factory=DebugRepairConfig, description="Debug repair configuration"
    )
    optimization: OptimizationConfig = Field(
        default_factory=OptimizationConfig, description="Optimization configuration"
    )
    model_config = {"extra": "forbid", "validate_assignment": True}
