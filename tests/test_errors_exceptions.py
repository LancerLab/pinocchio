"""
Unit tests for the errors exceptions module.
"""
import pytest

from pinocchio.errors import (  # Base exception; Module-specific exceptions; LLM-specific exceptions; Config-specific exceptions; Workflow-specific exceptions; Knowledge-specific exceptions; Agent-specific exceptions; Memory-specific exceptions; Circuit breaker exception
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentTimeoutError,
    CircuitBreakerOpenError,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigKeyError,
    ConfigTypeError,
    ConfigValidationError,
    KnowledgeError,
    KnowledgeResourceInvalidError,
    KnowledgeResourceNotFoundError,
    KnowledgeVersionError,
    LLMAPIError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMError,
    LLMInvalidRequestError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTimeoutError,
    MemoryCorruptionError,
    MemoryError,
    MemoryRetrievalError,
    MemoryStorageError,
    PinocchioError,
    PromptError,
    SessionError,
    WorkflowCancelledError,
    WorkflowDefinitionError,
    WorkflowError,
    WorkflowTaskError,
    WorkflowTimeoutError,
)


class TestBaseException:
    """Tests for the base PinocchioError class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        error = PinocchioError("Test message")
        assert error.message == "Test message"
        assert error.error_code == "UNKNOWN_ERROR"
        assert error.details == {}
        assert str(error) == "Test message"

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        error = PinocchioError(
            message="Custom message",
            error_code="CUSTOM_ERROR",
            details={"key": "value"},
        )
        assert error.message == "Custom message"
        assert error.error_code == "CUSTOM_ERROR"
        assert error.details == {"key": "value"}

    def test_to_dict(self):
        """Test to_dict method."""
        error = PinocchioError(
            message="Test message", error_code="TEST_ERROR", details={"key": "value"}
        )
        error_dict = error.to_dict()
        assert error_dict == {
            "message": "Test message",
            "error_code": "TEST_ERROR",
            "details": {"key": "value"},
        }


class TestModuleSpecificExceptions:
    """Tests for module-specific exception classes."""

    def test_config_error(self):
        """Test ConfigError."""
        error = ConfigError("Config error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "CONFIG_ERROR"
        assert error.message == "Config error"

        # Test with custom error code and details
        error = ConfigError(
            "Custom config error",
            error_code="CUSTOM_CONFIG_ERROR",
            details={"config_file": "config.json"},
        )
        assert error.error_code == "CUSTOM_CONFIG_ERROR"
        assert error.details == {"config_file": "config.json"}

    def test_llm_error(self):
        """Test LLMError."""
        error = LLMError("LLM error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "LLM_ERROR"
        assert error.message == "LLM error"

    def test_workflow_error(self):
        """Test WorkflowError."""
        error = WorkflowError("Workflow error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "WORKFLOW_ERROR"
        assert error.message == "Workflow error"

    def test_memory_error(self):
        """Test MemoryError."""
        error = MemoryError("Memory error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "MEMORY_ERROR"
        assert error.message == "Memory error"

    def test_session_error(self):
        """Test SessionError."""
        error = SessionError("Session error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "SESSION_ERROR"
        assert error.message == "Session error"

    def test_agent_error(self):
        """Test AgentError."""
        error = AgentError("Agent error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "AGENT_ERROR"
        assert error.message == "Agent error"

    def test_knowledge_error(self):
        """Test KnowledgeError."""
        error = KnowledgeError("Knowledge error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "KNOWLEDGE_ERROR"
        assert error.message == "Knowledge error"

    def test_prompt_error(self):
        """Test PromptError."""
        error = PromptError("Prompt error")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "PROMPT_ERROR"
        assert error.message == "Prompt error"


class TestLLMSpecificExceptions:
    """Tests for LLM-specific exception classes."""

    def test_llm_api_error(self):
        """Test LLMAPIError."""
        error = LLMAPIError("API error")
        assert isinstance(error, LLMError)
        assert isinstance(error, PinocchioError)
        assert error.error_code == "LLM_API_ERROR"
        assert error.message == "API error"

    def test_llm_rate_limit_error(self):
        """Test LLMRateLimitError."""
        error = LLMRateLimitError("Rate limit exceeded")
        assert isinstance(error, LLMError)
        assert error.error_code == "LLM_RATE_LIMIT_ERROR"
        assert error.message == "Rate limit exceeded"

    def test_llm_authentication_error(self):
        """Test LLMAuthenticationError."""
        error = LLMAuthenticationError("Authentication failed")
        assert isinstance(error, LLMError)
        assert error.error_code == "LLM_AUTH_ERROR"
        assert error.message == "Authentication failed"

    def test_llm_content_filter_error(self):
        """Test LLMContentFilterError."""
        error = LLMContentFilterError("Content filtered")
        assert isinstance(error, LLMError)
        assert error.error_code == "LLM_CONTENT_FILTER_ERROR"
        assert error.message == "Content filtered"

    def test_llm_timeout_error(self):
        """Test LLMTimeoutError."""
        error = LLMTimeoutError("Request timed out")
        assert isinstance(error, LLMError)
        assert error.error_code == "LLM_TIMEOUT_ERROR"
        assert error.message == "Request timed out"

    def test_llm_quota_exceeded_error(self):
        """Test LLMQuotaExceededError."""
        error = LLMQuotaExceededError("Quota exceeded")
        assert isinstance(error, LLMError)
        assert error.error_code == "LLM_QUOTA_EXCEEDED"
        assert error.message == "Quota exceeded"

    def test_llm_invalid_request_error(self):
        """Test LLMInvalidRequestError."""
        error = LLMInvalidRequestError("Invalid request")
        assert isinstance(error, LLMError)
        assert error.error_code == "LLM_INVALID_REQUEST"
        assert error.message == "Invalid request"


class TestConfigSpecificExceptions:
    """Tests for config-specific exception classes."""

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError."""
        error = ConfigFileNotFoundError("Config file not found")
        assert isinstance(error, ConfigError)
        assert error.error_code == "CONFIG_FILE_NOT_FOUND"
        assert error.message == "Config file not found"

    def test_config_validation_error(self):
        """Test ConfigValidationError."""
        error = ConfigValidationError("Validation failed")
        assert isinstance(error, ConfigError)
        assert error.error_code == "CONFIG_VALIDATION_ERROR"
        assert error.message == "Validation failed"

    def test_config_key_error(self):
        """Test ConfigKeyError."""
        error = ConfigKeyError("Missing key")
        assert isinstance(error, ConfigError)
        assert error.error_code == "CONFIG_KEY_ERROR"
        assert error.message == "Missing key"

    def test_config_type_error(self):
        """Test ConfigTypeError."""
        error = ConfigTypeError("Invalid type")
        assert isinstance(error, ConfigError)
        assert error.error_code == "CONFIG_TYPE_ERROR"
        assert error.message == "Invalid type"


class TestWorkflowSpecificExceptions:
    """Tests for workflow-specific exception classes."""

    def test_workflow_task_error(self):
        """Test WorkflowTaskError."""
        error = WorkflowTaskError("Task failed")
        assert isinstance(error, WorkflowError)
        assert error.error_code == "WORKFLOW_TASK_ERROR"
        assert error.message == "Task failed"

    def test_workflow_definition_error(self):
        """Test WorkflowDefinitionError."""
        error = WorkflowDefinitionError("Invalid definition")
        assert isinstance(error, WorkflowError)
        assert error.error_code == "WORKFLOW_DEFINITION_ERROR"
        assert error.message == "Invalid definition"

    def test_workflow_timeout_error(self):
        """Test WorkflowTimeoutError."""
        error = WorkflowTimeoutError("Workflow timed out")
        assert isinstance(error, WorkflowError)
        assert error.error_code == "WORKFLOW_TIMEOUT"
        assert error.message == "Workflow timed out"

    def test_workflow_cancelled_error(self):
        """Test WorkflowCancelledError."""
        error = WorkflowCancelledError("Workflow cancelled")
        assert isinstance(error, WorkflowError)
        assert error.error_code == "WORKFLOW_CANCELLED"
        assert error.message == "Workflow cancelled"


class TestKnowledgeSpecificExceptions:
    """Tests for knowledge-specific exception classes."""

    def test_knowledge_resource_not_found_error(self):
        """Test KnowledgeResourceNotFoundError."""
        error = KnowledgeResourceNotFoundError("Resource not found")
        assert isinstance(error, KnowledgeError)
        assert error.error_code == "KNOWLEDGE_RESOURCE_NOT_FOUND"
        assert error.message == "Resource not found"

    def test_knowledge_resource_invalid_error(self):
        """Test KnowledgeResourceInvalidError."""
        error = KnowledgeResourceInvalidError("Invalid resource")
        assert isinstance(error, KnowledgeError)
        assert error.error_code == "KNOWLEDGE_RESOURCE_INVALID"
        assert error.message == "Invalid resource"

    def test_knowledge_version_error(self):
        """Test KnowledgeVersionError."""
        error = KnowledgeVersionError("Version error")
        assert isinstance(error, KnowledgeError)
        assert error.error_code == "KNOWLEDGE_VERSION_ERROR"
        assert error.message == "Version error"


class TestAgentSpecificExceptions:
    """Tests for agent-specific exception classes."""

    def test_agent_initialization_error(self):
        """Test AgentInitializationError."""
        error = AgentInitializationError("Initialization failed")
        assert isinstance(error, AgentError)
        assert error.error_code == "AGENT_INITIALIZATION_ERROR"
        assert error.message == "Initialization failed"

    def test_agent_execution_error(self):
        """Test AgentExecutionError."""
        error = AgentExecutionError("Execution failed")
        assert isinstance(error, AgentError)
        assert error.error_code == "AGENT_EXECUTION_ERROR"
        assert error.message == "Execution failed"

    def test_agent_timeout_error(self):
        """Test AgentTimeoutError."""
        error = AgentTimeoutError("Agent timed out")
        assert isinstance(error, AgentError)
        assert error.error_code == "AGENT_TIMEOUT"
        assert error.message == "Agent timed out"


class TestMemorySpecificExceptions:
    """Tests for memory-specific exception classes."""

    def test_memory_storage_error(self):
        """Test MemoryStorageError."""
        error = MemoryStorageError("Storage error")
        assert isinstance(error, MemoryError)
        assert error.error_code == "MEMORY_STORAGE_ERROR"
        assert error.message == "Storage error"

    def test_memory_retrieval_error(self):
        """Test MemoryRetrievalError."""
        error = MemoryRetrievalError("Retrieval error")
        assert isinstance(error, MemoryError)
        assert error.error_code == "MEMORY_RETRIEVAL_ERROR"
        assert error.message == "Retrieval error"

    def test_memory_corruption_error(self):
        """Test MemoryCorruptionError."""
        error = MemoryCorruptionError("Corruption detected")
        assert isinstance(error, MemoryError)
        assert error.error_code == "MEMORY_CORRUPTION_ERROR"
        assert error.message == "Corruption detected"


class TestCircuitBreakerException:
    """Tests for circuit breaker exception."""

    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError."""
        error = CircuitBreakerOpenError("Circuit breaker is open")
        assert isinstance(error, PinocchioError)
        assert error.error_code == "CIRCUIT_BREAKER_OPEN"
        assert error.message == "Circuit breaker is open"

        # Test with details
        error = CircuitBreakerOpenError(
            "Circuit breaker is open",
            details={"failure_count": 5, "recovery_timeout": 60},
        )
        assert error.details == {"failure_count": 5, "recovery_timeout": 60}
