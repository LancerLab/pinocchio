"""
Unit tests for the errors handlers module.
"""
import time
import logging
from unittest.mock import patch, MagicMock, call

import pytest

from pinocchio.errors import (
    PinocchioError,
    LLMError,
    LLMRateLimitError,
    CircuitBreakerOpenError,
    handle_errors,
    error_context,
    retry,
    global_error_handler,
    CircuitBreaker
)


class TestHandleErrors:
    """Tests for the handle_errors decorator."""

    def test_handle_errors_no_exception(self):
        """Test handle_errors when no exception is raised."""
        @handle_errors()
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"

    def test_handle_errors_with_standard_exception(self):
        """Test handle_errors with a standard exception."""
        @handle_errors(reraise=True)
        def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError) as excinfo:
            test_function()
        
        assert str(excinfo.value) == "Test error"

    def test_handle_errors_with_pinocchio_error(self):
        """Test handle_errors with a PinocchioError."""
        @handle_errors(reraise=True)
        def test_function():
            raise PinocchioError(
                "Test error",
                error_code="TEST_ERROR",
                details={"key": "value"}
            )
        
        with pytest.raises(PinocchioError) as excinfo:
            test_function()
        
        assert excinfo.value.error_code == "TEST_ERROR"
        assert excinfo.value.details == {"key": "value"}

    def test_handle_errors_with_fallback_value(self):
        """Test handle_errors with a fallback value."""
        @handle_errors(fallback_value="default", reraise=False)
        def test_function():
            raise ValueError("Test error")
        
        result = test_function()
        assert result == "default"

    def test_handle_errors_with_log_level(self):
        """Test handle_errors with custom log level."""
        with patch("logging.Logger.log") as mock_log:
            @handle_errors(log_level=logging.WARNING, reraise=True)
            def test_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                test_function()
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[0] == logging.WARNING
            assert "Test error" in args[1]


class TestErrorContext:
    """Tests for the error_context context manager."""

    def test_error_context_no_exception(self):
        """Test error_context when no exception is raised."""
        with error_context("test_operation"):
            result = "success"
        
        assert result == "success"

    def test_error_context_with_standard_exception(self):
        """Test error_context with a standard exception."""
        with pytest.raises(ValueError) as excinfo:
            with error_context("test_operation"):
                raise ValueError("Test error")
        
        assert str(excinfo.value) == "Test error"

    def test_error_context_with_pinocchio_error(self):
        """Test error_context with a PinocchioError."""
        with pytest.raises(PinocchioError) as excinfo:
            with error_context("test_operation"):
                raise PinocchioError(
                    "Test error",
                    error_code="TEST_ERROR",
                    details={"key": "value"}
                )
        
        assert excinfo.value.error_code == "TEST_ERROR"
        assert excinfo.value.details == {"key": "value"}

    def test_error_context_with_no_reraise(self):
        """Test error_context with reraise=False."""
        with patch("logging.Logger.log") as mock_log:
            with error_context("test_operation", reraise=False):
                raise ValueError("Test error")
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert "Test error" in args[1]

    def test_error_context_with_custom_log_level(self):
        """Test error_context with custom log level."""
        with patch("logging.Logger.log") as mock_log:
            with pytest.raises(ValueError):
                with error_context("test_operation", log_level=logging.WARNING):
                    raise ValueError("Test error")
            
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[0] == logging.WARNING


class TestRetry:
    """Tests for the retry decorator."""

    def test_retry_no_exception(self):
        """Test retry when no exception is raised."""
        mock_function = MagicMock(return_value="success")
        
        @retry()
        def test_function():
            return mock_function()
        
        result = test_function()
        
        assert result == "success"
        mock_function.assert_called_once()

    def test_retry_with_temporary_exception(self):
        """Test retry with a temporary exception that eventually succeeds."""
        mock_function = MagicMock(side_effect=[ValueError("Attempt 1"), "success"])
        mock_sleep = MagicMock()
        
        with patch("time.sleep", mock_sleep):
            @retry(max_retries=3)
            def test_function():
                return mock_function()
            
            result = test_function()
            
            assert result == "success"
            assert mock_function.call_count == 2
            mock_sleep.assert_called_once()

    def test_retry_with_persistent_exception(self):
        """Test retry with a persistent exception that always fails."""
        mock_function = MagicMock(side_effect=ValueError("Persistent error"))
        mock_sleep = MagicMock()
        
        with patch("time.sleep", mock_sleep):
            @retry(max_retries=3)
            def test_function():
                return mock_function()
            
            with pytest.raises(ValueError) as excinfo:
                test_function()
            
            assert str(excinfo.value) == "Persistent error"
            assert mock_function.call_count == 4  # Initial + 3 retries
            assert mock_sleep.call_count == 3

    def test_retry_with_specific_exceptions(self):
        """Test retry with specific exception types."""
        mock_function = MagicMock(side_effect=[ValueError("Retry this"), TypeError("Don't retry this")])
        mock_sleep = MagicMock()
        
        with patch("time.sleep", mock_sleep):
            @retry(exceptions_to_retry=[ValueError])
            def test_function():
                return mock_function()
            
            with pytest.raises(TypeError) as excinfo:
                test_function()
            
            assert str(excinfo.value) == "Don't retry this"
            assert mock_function.call_count == 2
            mock_sleep.assert_called_once()

    def test_retry_with_backoff(self):
        """Test retry with exponential backoff."""
        mock_function = MagicMock(side_effect=[ValueError("Attempt 1"), ValueError("Attempt 2"), "success"])
        mock_sleep = MagicMock()
        
        with patch("time.sleep", mock_sleep):
            @retry(max_retries=4, backoff_factor=2.0)
            def test_function():
                return mock_function()
            
            result = test_function()
            
            assert result == "success"
            assert mock_function.call_count == 3
            
            # Check that sleep was called with increasing delays
            assert mock_sleep.call_count == 2
            assert mock_sleep.call_args_list[0] == call(1.0)
            assert mock_sleep.call_args_list[1] == call(2.0)  # 1.0 * 2.0


class TestGlobalErrorHandler:
    """Tests for the global_error_handler function."""

    def test_global_error_handler_with_standard_exception(self):
        """Test global_error_handler with a standard exception."""
        with patch("logging.Logger.critical") as mock_critical:
            error = ValueError("Test error")
            exc_type = type(error)
            exc_value = error
            exc_traceback = None
            
            global_error_handler(exc_type, exc_value, exc_traceback)
            
            mock_critical.assert_called_once()
            args, kwargs = mock_critical.call_args
            assert "Unhandled exception" in args[0]
            assert "ValueError: Test error" in args[0]

    def test_global_error_handler_with_keyboard_interrupt(self):
        """Test global_error_handler with KeyboardInterrupt."""
        with patch("sys.__excepthook__") as mock_excepthook:
            error = KeyboardInterrupt()
            exc_type = type(error)
            exc_value = error
            exc_traceback = None
            
            global_error_handler(exc_type, exc_value, exc_traceback)
            
            mock_excepthook.assert_called_once_with(exc_type, exc_value, exc_traceback)


class TestCircuitBreaker:
    """Tests for the CircuitBreaker class."""

    def test_circuit_breaker_closed_state(self):
        """Test CircuitBreaker in closed state."""
        breaker = CircuitBreaker(name="test-breaker")
        
        # Circuit is initially closed
        assert breaker.is_closed
        
        # Function should execute normally
        result = breaker.execute(lambda: "success")
        assert result == "success"

    def test_circuit_breaker_open_state(self):
        """Test CircuitBreaker transitions to open state after failures."""
        breaker = CircuitBreaker(
            name="test-breaker", 
            failure_threshold=2
        )
        
        # Cause failures to open the circuit
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 1")))
        
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 2")))
        
        # Circuit should be open now
        assert breaker.is_open
        
        # Further calls should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            breaker.execute(lambda: "success")

    def test_circuit_breaker_half_open_state(self):
        """Test CircuitBreaker transitions to half-open state after timeout."""
        breaker = CircuitBreaker(
            name="test-breaker", 
            failure_threshold=2, 
            recovery_timeout=0.1
        )
        
        # Cause failures to open the circuit
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 1")))
        
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 2")))
        
        # Circuit should be open now
        assert breaker.is_open
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should be allowed (half-open state)
        result = breaker.execute(lambda: "success")
        assert result == "success"
        
        # Circuit should be closed again after successful call
        assert breaker.is_closed

    def test_circuit_breaker_remains_open_after_half_open_failure(self):
        """Test CircuitBreaker remains open if call fails in half-open state."""
        breaker = CircuitBreaker(
            name="test-breaker", 
            failure_threshold=2, 
            recovery_timeout=0.1
        )
        
        # Cause failures to open the circuit
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 1")))
        
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 2")))
        
        # Circuit should be open now
        assert breaker.is_open
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Circuit should be half-open now, but call fails
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure in half-open")))
        
        # Circuit should be open again
        assert breaker.is_open

    def test_circuit_breaker_reset(self):
        """Test CircuitBreaker reset method."""
        breaker = CircuitBreaker(
            name="test-breaker", 
            failure_threshold=2
        )
        
        # Cause failures to open the circuit
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 1")))
        
        with pytest.raises(ValueError):
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("Failure 2")))
        
        # Circuit should be open now
        assert breaker.is_open
        
        # Reset the circuit breaker
        breaker.reset()
        
        # Circuit should be closed again
        assert breaker.is_closed
        assert breaker.failure_count == 0
        
        # Function should execute normally
        result = breaker.execute(lambda: "success")
        assert result == "success" 