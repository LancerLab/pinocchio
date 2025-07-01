"""
Exceptions module for the Pinocchio multi-agent system.

This module defines a hierarchy of custom exception classes used throughout
the Pinocchio system, providing standardized error reporting and handling.
"""
from typing import Dict, Any, Optional


class PinocchioError(Exception):
    """Base exception class for all Pinocchio-specific errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a new PinocchioError.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional contextual information about the error
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to a dictionary representation."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


# Module-specific exceptions

class ConfigError(PinocchioError):
    """Errors related to configuration loading and validation."""

    def __init__(self, message: str, error_code: str = "CONFIG_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class LLMError(PinocchioError):
    """Errors related to LLM API calls and processing."""

    def __init__(self, message: str, error_code: str = "LLM_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class WorkflowError(PinocchioError):
    """Errors related to workflow execution and orchestration."""

    def __init__(self, message: str, error_code: str = "WORKFLOW_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class MemoryError(PinocchioError):
    """Errors related to memory operations."""

    def __init__(self, message: str, error_code: str = "MEMORY_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class SessionError(PinocchioError):
    """Errors related to session management."""

    def __init__(self, message: str, error_code: str = "SESSION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class AgentError(PinocchioError):
    """Errors related to agent operations."""

    def __init__(self, message: str, error_code: str = "AGENT_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class KnowledgeError(PinocchioError):
    """Errors related to knowledge base operations."""

    def __init__(self, message: str, error_code: str = "KNOWLEDGE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class PromptError(PinocchioError):
    """Errors related to prompt template processing."""

    def __init__(self, message: str, error_code: str = "PROMPT_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


# LLM-specific exceptions

class LLMAPIError(LLMError):
    """Errors related to LLM API communication."""

    def __init__(self, message: str, error_code: str = "LLM_API_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class LLMRateLimitError(LLMError):
    """Errors related to LLM API rate limiting."""

    def __init__(self, message: str, error_code: str = "LLM_RATE_LIMIT_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class LLMAuthenticationError(LLMError):
    """Errors related to LLM API authentication."""

    def __init__(self, message: str, error_code: str = "LLM_AUTH_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class LLMContentFilterError(LLMError):
    """Errors related to LLM content filtering."""

    def __init__(self, message: str, error_code: str = "LLM_CONTENT_FILTER_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class LLMTimeoutError(LLMError):
    """Errors related to LLM API request timeouts."""

    def __init__(self, message: str, error_code: str = "LLM_TIMEOUT_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class LLMQuotaExceededError(LLMError):
    """Errors related to LLM API quota or budget exceeded."""

    def __init__(self, message: str, error_code: str = "LLM_QUOTA_EXCEEDED", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class LLMInvalidRequestError(LLMError):
    """Errors related to invalid requests to LLM API."""

    def __init__(self, message: str, error_code: str = "LLM_INVALID_REQUEST", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


# Config-specific exceptions

class ConfigFileNotFoundError(ConfigError):
    """Error raised when a configuration file is not found."""

    def __init__(self, message: str, error_code: str = "CONFIG_FILE_NOT_FOUND", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class ConfigValidationError(ConfigError):
    """Error raised when configuration validation fails."""

    def __init__(self, message: str, error_code: str = "CONFIG_VALIDATION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class ConfigKeyError(ConfigError):
    """Error raised when a required configuration key is missing."""

    def __init__(self, message: str, error_code: str = "CONFIG_KEY_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class ConfigTypeError(ConfigError):
    """Error raised when a configuration value has an incorrect type."""

    def __init__(self, message: str, error_code: str = "CONFIG_TYPE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


# Workflow-specific exceptions

class WorkflowTaskError(WorkflowError):
    """Error raised when a workflow task fails."""

    def __init__(self, message: str, error_code: str = "WORKFLOW_TASK_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class WorkflowDefinitionError(WorkflowError):
    """Error raised when a workflow definition is invalid."""

    def __init__(self, message: str, error_code: str = "WORKFLOW_DEFINITION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class WorkflowTimeoutError(WorkflowError):
    """Error raised when a workflow execution times out."""

    def __init__(self, message: str, error_code: str = "WORKFLOW_TIMEOUT", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class WorkflowCancelledError(WorkflowError):
    """Error raised when a workflow is cancelled."""

    def __init__(self, message: str, error_code: str = "WORKFLOW_CANCELLED", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


# Knowledge-specific exceptions

class KnowledgeResourceNotFoundError(KnowledgeError):
    """Error raised when a knowledge resource is not found."""

    def __init__(self, message: str, error_code: str = "KNOWLEDGE_RESOURCE_NOT_FOUND", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class KnowledgeResourceInvalidError(KnowledgeError):
    """Error raised when a knowledge resource is invalid."""

    def __init__(self, message: str, error_code: str = "KNOWLEDGE_RESOURCE_INVALID", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class KnowledgeVersionError(KnowledgeError):
    """Error raised when there's an issue with knowledge resource versioning."""

    def __init__(self, message: str, error_code: str = "KNOWLEDGE_VERSION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


# Agent-specific exceptions

class AgentInitializationError(AgentError):
    """Error raised when an agent fails to initialize."""

    def __init__(self, message: str, error_code: str = "AGENT_INITIALIZATION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class AgentExecutionError(AgentError):
    """Error raised when an agent fails during execution."""

    def __init__(self, message: str, error_code: str = "AGENT_EXECUTION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class AgentTimeoutError(AgentError):
    """Error raised when an agent execution times out."""

    def __init__(self, message: str, error_code: str = "AGENT_TIMEOUT", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


# Memory-specific exceptions

class MemoryStorageError(MemoryError):
    """Error raised when memory storage operations fail."""

    def __init__(self, message: str, error_code: str = "MEMORY_STORAGE_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class MemoryRetrievalError(MemoryError):
    """Error raised when memory retrieval operations fail."""

    def __init__(self, message: str, error_code: str = "MEMORY_RETRIEVAL_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class MemoryCorruptionError(MemoryError):
    """Error raised when memory data is corrupted."""

    def __init__(self, message: str, error_code: str = "MEMORY_CORRUPTION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details) 