"""
Unit tests for the errors module.
"""
import logging
import pytest
from contextlib import contextmanager
from typing import Dict, Any, Optional, List

from pinocchio.errors import (
    PinocchioError,
    ConfigError,
    LLMError,
    WorkflowError,
    handle_errors,
    error_context,
    retry,
    ErrorReporter,
    ErrorMetricsCollector,
)


class TestExceptions:
    """Tests for exception classes."""

    def test_base_exception(self):
        """Test the base PinocchioError class."""
        error = PinocchioError("Test error")
        assert str(error) == "Test error"
        assert error.error_code == "UNKNOWN_ERROR"
        assert error.details == {}

        # Test with custom error code and details
        details = {"key": "value"}
        error = PinocchioError("Test error", error_code="CUSTOM_ERROR", details=details)
        assert error.error_code == "CUSTOM_ERROR"
        assert error.details == details

    def test_to_dict(self):
        """Test the to_dict method."""
        details = {"key": "value"}
        error = PinocchioError("Test error", error_code="CUSTOM_ERROR", details=details)
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "CUSTOM_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == details

    def test_module_specific_exceptions(self):
        """Test module-specific exception classes."""
        # Test ConfigError
        config_error = ConfigError("Config error")
        assert isinstance(config_error, PinocchioError)
        assert config_error.error_code == "CONFIG_ERROR"

        # Test LLMError
        llm_error = LLMError("LLM error")
        assert isinstance(llm_error, PinocchioError)
        assert llm_error.error_code == "LLM_ERROR"

        # Test WorkflowError
        workflow_error = WorkflowError("Workflow error")
        assert isinstance(workflow_error, PinocchioError)
        assert workflow_error.error_code == "WORKFLOW_ERROR"


class TestErrorHandlers:
    """Tests for error handling utilities."""

    def test_handle_errors_decorator(self):
        """Test the handle_errors decorator."""
        @handle_errors(fallback_value="fallback")
        def function_that_raises():
            raise ValueError("Test error")

        @handle_errors(reraise=True)
        def function_that_reraises():
            raise ValueError("Test error")

        # Test fallback
        result = function_that_raises()
        assert result == "fallback"

        # Test reraise
        with pytest.raises(ValueError):
            function_that_reraises()

    def test_error_context(self):
        """Test the error_context context manager."""
        # Test no error case
        with error_context("test_context") as ctx:
            pass
        assert not ctx.error_occurred

        # Test error case with reraise=False
        with error_context("test_context", reraise=False) as ctx:
            raise ValueError("Test error")
        assert ctx.error_occurred
        assert isinstance(ctx.exception, ValueError)
        assert "Test error" in str(ctx.exception)

        # Test error case with reraise=True
        with pytest.raises(ValueError):
            with error_context("test_context", reraise=True) as ctx:
                raise ValueError("Test error")

    def test_retry_decorator(self):
        """Test the retry decorator."""
        call_count = 0

        @retry(max_retries=3, backoff_factor=0.1)
        def function_that_fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        # Function should succeed after two retries
        result = function_that_fails_twice()
        assert result == "success"
        assert call_count == 3

        # Test max retries exceeded
        call_count = 0

        @retry(max_retries=2, backoff_factor=0.1)
        def function_that_always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            function_that_always_fails()
        assert call_count == 3  # Initial call + 2 retries


class TestErrorReporting:
    """Tests for error reporting tools."""

    def test_error_reporter(self):
        """Test the ErrorReporter class."""
        reporter = ErrorReporter()
        
        # Test recording errors
        error = ValueError("Test error")
        reporter.record_error(error)
        assert len(reporter.errors) == 1
        
        # Test error summary
        summary = reporter.get_error_summary()
        assert summary["total_errors"] == 1
        assert summary["error_counts"]["ValueError"] == 1
        assert summary["latest_error"]["error_type"] == "ValueError"
        
        # Test recording PinocchioError
        pinocchio_error = PinocchioError("Pinocchio error", "TEST_ERROR", {"detail": "value"})
        reporter.record_error(pinocchio_error)
        assert len(reporter.errors) == 2
        
        # Test get_errors_by_type
        value_errors = reporter.get_errors_by_type("ValueError")
        assert len(value_errors) == 1
        assert value_errors[0]["message"] == "Test error"
        
        # Test clear_errors
        reporter.clear_errors()
        assert len(reporter.errors) == 0
        assert reporter.get_error_summary()["total_errors"] == 0

    def test_error_metrics_collector(self):
        """Test the ErrorMetricsCollector class."""
        collector = ErrorMetricsCollector()
        
        # Test recording error metrics
        collector.record_error("ValueError")
        collector.record_error("ValueError")
        collector.record_error("TypeError")
        
        # Test calculate_error_rates
        rates = collector.calculate_error_rates()
        assert len(rates["ValueError"]) == 1
        assert len(rates["TypeError"]) == 1
        
        # Test clear_metrics
        collector.clear_metrics()
        rates = collector.calculate_error_rates()
        assert not rates
        
        # Test recording after clearing
        collector.record_error("ValueError")
        rates = collector.calculate_error_rates()
        assert len(rates["ValueError"]) == 1 