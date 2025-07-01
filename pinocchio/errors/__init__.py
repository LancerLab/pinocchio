"""
Errors module for the Pinocchio multi-agent system.

This module provides standardized error handling and exception management
across the Pinocchio system.
"""

# Import and re-export common exceptions for convenience
from .exceptions import (
    # Base exception
    PinocchioError,
    
    # Module-specific exceptions
    ConfigError,
    LLMError,
    WorkflowError,
    MemoryError,
    SessionError,
    AgentError,
    KnowledgeError,
    PromptError,
    
    # LLM-specific exceptions
    LLMAPIError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMTimeoutError,
    LLMQuotaExceededError,
    LLMInvalidRequestError,
    
    # Config-specific exceptions
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigKeyError,
    ConfigTypeError,
    
    # Workflow-specific exceptions
    WorkflowTaskError,
    WorkflowDefinitionError,
    WorkflowTimeoutError,
    WorkflowCancelledError,
    
    # Knowledge-specific exceptions
    KnowledgeResourceNotFoundError,
    KnowledgeResourceInvalidError,
    KnowledgeVersionError,
    
    # Agent-specific exceptions
    AgentInitializationError,
    AgentExecutionError,
    AgentTimeoutError,
    
    # Memory-specific exceptions
    MemoryStorageError,
    MemoryRetrievalError,
    MemoryCorruptionError,
)

# Import and re-export error handling utilities
from .handlers import (
    handle_errors,
    error_context,
    retry,
    global_error_handler,
    CircuitBreaker,
    CircuitBreakerOpenError,
)

# Import and re-export error reporting tools
from .reporting import (
    ErrorReporter,
    ErrorMetricsCollector,
)

# Import and re-export logging utilities
from .logging import (
    JsonFormatter,
    PinocchioLogFormatter,
    setup_logging,
    get_logger,
)

__all__ = [
    # Base exception
    'PinocchioError',
    
    # Module-specific exceptions
    'ConfigError',
    'LLMError',
    'WorkflowError',
    'MemoryError',
    'SessionError',
    'AgentError',
    'KnowledgeError',
    'PromptError',
    
    # LLM-specific exceptions
    'LLMAPIError',
    'LLMRateLimitError',
    'LLMAuthenticationError',
    'LLMContentFilterError',
    'LLMTimeoutError',
    'LLMQuotaExceededError',
    'LLMInvalidRequestError',
    
    # Config-specific exceptions
    'ConfigFileNotFoundError',
    'ConfigValidationError',
    'ConfigKeyError',
    'ConfigTypeError',
    
    # Workflow-specific exceptions
    'WorkflowTaskError',
    'WorkflowDefinitionError',
    'WorkflowTimeoutError',
    'WorkflowCancelledError',
    
    # Knowledge-specific exceptions
    'KnowledgeResourceNotFoundError',
    'KnowledgeResourceInvalidError',
    'KnowledgeVersionError',
    
    # Agent-specific exceptions
    'AgentInitializationError',
    'AgentExecutionError',
    'AgentTimeoutError',
    
    # Memory-specific exceptions
    'MemoryStorageError',
    'MemoryRetrievalError',
    'MemoryCorruptionError',
    
    # Error handling utilities
    'handle_errors',
    'error_context',
    'retry',
    'global_error_handler',
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    
    # Error reporting tools
    'ErrorReporter',
    'ErrorMetricsCollector',
    
    # Logging utilities
    'JsonFormatter',
    'PinocchioLogFormatter',
    'setup_logging',
    'get_logger',
] 