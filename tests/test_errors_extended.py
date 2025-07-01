"""
Unit tests for the extended error handling functionality.
"""
import io
import json
import logging
import sys
import time
from unittest.mock import patch

import pytest

from pinocchio.errors import (
    # Circuit breaker
    CircuitBreaker,
    CircuitBreakerOpenError,
    
    # Specific exceptions
    LLMTimeoutError,
    ConfigValidationError,
    WorkflowTaskError,
    
    # Logging
    JsonFormatter,
    PinocchioLogFormatter,
)


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""
    
    def test_circuit_breaker_normal_operation(self):
        """Test that circuit breaker allows calls when closed."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        # Should execute normally
        result = cb.execute(lambda: "success")
        assert result == "success"
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        cb = CircuitBreaker("test", failure_threshold=3)
        
        # Cause failures
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.execute(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        # Circuit should be open now
        assert cb.is_open
        assert not cb.is_closed
        
        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            cb.execute(lambda: "success")
    
    def test_circuit_breaker_half_open(self):
        """Test that circuit breaker transitions to half-open after timeout."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        # Cause failures to open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.execute(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        assert cb.is_open
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Execute successful call - should transition to half-open and then closed
        result = cb.execute(lambda: "recovered")
        assert result == "recovered"
        assert cb.is_closed
    
    def test_circuit_breaker_remains_open_after_half_open_failure(self):
        """Test that circuit breaker goes back to open after failure in half-open state."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0.1)
        
        # Cause failures to open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.execute(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        assert cb.is_open
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Execute failing call during half-open state
        with pytest.raises(ValueError):
            cb.execute(lambda: (_ for _ in ()).throw(ValueError("still failing")))
        
        # Circuit should be open again
        assert cb.is_open
        
        # Should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            cb.execute(lambda: "success")
    
    def test_circuit_breaker_manual_reset(self):
        """Test manually resetting the circuit breaker."""
        cb = CircuitBreaker("test", failure_threshold=2)
        
        # Cause failures to open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.execute(lambda: (_ for _ in ()).throw(ValueError("test error")))
        
        assert cb.is_open
        
        # Manually reset
        cb.reset()
        
        # Should be closed and allow calls
        assert cb.is_closed
        result = cb.execute(lambda: "reset success")
        assert result == "reset success"


class TestSpecificExceptions:
    """Tests for the specific exception classes."""
    
    def test_llm_timeout_error(self):
        """Test LLMTimeoutError."""
        error = LLMTimeoutError("Request timed out", details={"timeout": 30})
        assert error.error_code == "LLM_TIMEOUT_ERROR"
        assert error.message == "Request timed out"
        assert error.details == {"timeout": 30}
        
        # Test to_dict
        error_dict = error.to_dict()
        assert error_dict["error_code"] == "LLM_TIMEOUT_ERROR"
        assert error_dict["message"] == "Request timed out"
        assert error_dict["details"] == {"timeout": 30}
    
    def test_config_validation_error(self):
        """Test ConfigValidationError."""
        error = ConfigValidationError(
            "Invalid configuration", 
            details={"field": "api_key", "reason": "missing"}
        )
        assert error.error_code == "CONFIG_VALIDATION_ERROR"
        assert error.message == "Invalid configuration"
        assert error.details == {"field": "api_key", "reason": "missing"}
    
    def test_workflow_task_error(self):
        """Test WorkflowTaskError."""
        error = WorkflowTaskError(
            "Task execution failed", 
            details={"task_id": "123", "step": "preprocessing"}
        )
        assert error.error_code == "WORKFLOW_TASK_ERROR"
        assert error.message == "Task execution failed"
        assert error.details == {"task_id": "123", "step": "preprocessing"}


class TestLogFormatters:
    """Tests for the log formatters."""
    
    def test_json_formatter(self):
        """Test JsonFormatter produces valid JSON."""
        formatter = JsonFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Should be valid JSON
        log_data = json.loads(formatted)
        
        # Check fields
        assert log_data["level"] == "ERROR"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert "path" in log_data
    
    def test_json_formatter_with_error_context(self):
        """Test JsonFormatter with error context."""
        formatter = JsonFormatter()
        
        # Create a log record with error context
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_file.py",
            lineno=42,
            msg="Test error",
            args=(),
            exc_info=None
        )
        record.error_context = {
            "function": "test_function",
            "error_code": "TEST_ERROR"
        }
        
        # Format the record
        formatted = formatter.format(record)
        
        # Should be valid JSON with error context
        log_data = json.loads(formatted)
        assert "error_context" in log_data
        assert log_data["error_context"]["function"] == "test_function"
        assert log_data["error_context"]["error_code"] == "TEST_ERROR"
    
    def test_pinocchio_log_formatter(self):
        """Test PinocchioLogFormatter."""
        formatter = PinocchioLogFormatter(use_colors=False)
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test_file.py",
            lineno=42,
            msg="Test warning",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check basic format
        assert "[WARNING]" in formatted
        assert "test_logger" in formatted
        assert "Test warning" in formatted
    
    def test_pinocchio_log_formatter_with_exception(self):
        """Test PinocchioLogFormatter with exception information."""
        formatter = PinocchioLogFormatter(use_colors=False)
        
        # Create exception info
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()
        
        # Create a log record with exception info
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_file.py",
            lineno=42,
            msg="Exception occurred",
            args=(),
            exc_info=exc_info
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check exception format
        assert "Exception occurred" in formatted
        assert "Exception:" in formatted
        assert "ValueError: Test exception" in formatted 