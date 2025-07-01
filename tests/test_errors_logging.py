"""
Unit tests for the errors logging module.
"""
import io
import json
import logging
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from pinocchio.errors.logging import (
    JsonFormatter,
    PinocchioLogFormatter,
    setup_logging,
    get_logger
)


class TestJsonFormatter:
    """Tests for the JsonFormatter class."""

    def test_json_formatter_all_options(self):
        """Test JsonFormatter with all configuration options."""
        formatter = JsonFormatter(
            include_timestamp=True,
            include_level=True,
            include_name=True,
            include_path=True,
            include_process=True,
            include_thread=True,
            custom_field="custom_value"
        )
        
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
        assert "timestamp" in log_data
        assert log_data["level"] == "ERROR"
        assert log_data["logger"] == "test_logger"
        assert log_data["path"] == "test_file.py:42"
        assert "process" in log_data
        assert "thread" in log_data
        assert log_data["custom_field"] == "custom_value"

    def test_json_formatter_minimal_options(self):
        """Test JsonFormatter with minimal configuration options."""
        formatter = JsonFormatter(
            include_timestamp=False,
            include_level=False,
            include_name=False,
            include_path=False,
            include_process=False,
            include_thread=False
        )
        
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
        assert "timestamp" not in log_data
        assert "level" not in log_data
        assert "logger" not in log_data
        assert "path" not in log_data
        assert "process" not in log_data
        assert "thread" not in log_data
        assert log_data["message"] == "Test message"

    def test_json_formatter_with_exception_info(self):
        """Test JsonFormatter with exception information."""
        formatter = JsonFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
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
            
            # Should be valid JSON with exception info
            log_data = json.loads(formatted)
            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert log_data["exception"]["message"] == "Test exception"
            assert isinstance(log_data["exception"]["traceback"], list)

    def test_json_formatter_with_additional_fields(self):
        """Test JsonFormatter with additional fields from record.__dict__."""
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
        
        # Add custom fields
        record.custom_field1 = "value1"
        record.custom_field2 = {"nested": "value2"}
        
        # Format the record
        formatted = formatter.format(record)
        
        # Should be valid JSON with custom fields
        log_data = json.loads(formatted)
        assert log_data["custom_field1"] == "value1"
        assert log_data["custom_field2"] == {"nested": "value2"}


class TestPinocchioLogFormatter:
    """Tests for the PinocchioLogFormatter class."""

    def test_formatter_with_colors(self):
        """Test PinocchioLogFormatter with colors enabled."""
        formatter = PinocchioLogFormatter(use_colors=True)
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_file.py",
            lineno=42,
            msg="Test error",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check that color codes are included
        assert "\033[91mERROR\033[0m" in formatted
        
    def test_formatter_without_colors(self):
        """Test PinocchioLogFormatter with colors disabled."""
        formatter = PinocchioLogFormatter(use_colors=False)
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_file.py",
            lineno=42,
            msg="Test error",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check that color codes are not included
        assert "\033[" not in formatted
        
    def test_formatter_with_error_context(self):
        """Test PinocchioLogFormatter with error context."""
        formatter = PinocchioLogFormatter(use_colors=False)
        
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
            "error_code": "TEST_ERROR",
            "traceback": "Long traceback text..."
        }
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check that error context is included
        assert "Error Context:" in formatted
        assert "function: test_function" in formatted
        assert "error_code: TEST_ERROR" in formatted
        assert "traceback: <traceback omitted>" in formatted

    def test_formatter_custom_format(self):
        """Test PinocchioLogFormatter with custom format string."""
        formatter = PinocchioLogFormatter(
            fmt="%(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            use_colors=False
        )
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check format
        assert formatted == "INFO - Test message"


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging_basic(self):
        """Test basic setup_logging functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            json_log_file = os.path.join(temp_dir, "test.json")
            
            # Setup logging
            with patch("logging.StreamHandler") as mock_stream_handler, \
                 patch("logging.FileHandler") as mock_file_handler, \
                 patch("logging.getLogger") as mock_get_logger:
                
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(
                    log_level=logging.DEBUG,
                    log_file=log_file,
                    json_log_file=json_log_file
                )
                
                # Check that root logger was configured correctly
                mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)
                mock_root_logger.removeHandler.assert_not_called()  # No handlers to remove
                
                # Check that handlers were added
                assert mock_stream_handler.call_count == 1
                assert mock_file_handler.call_count == 2  # Regular log file and JSON log file
                
                # Check that formatters were set
                mock_stream_handler.return_value.setFormatter.assert_called_once()
                mock_file_handler.return_value.setFormatter.assert_called()
                
                # Check that handlers were added to root logger
                assert mock_root_logger.addHandler.call_count == 3

    def test_setup_logging_no_console(self):
        """Test setup_logging with console output disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            json_log_file = os.path.join(temp_dir, "test.json")
            
            # Setup logging
            with patch("logging.StreamHandler") as mock_stream_handler, \
                 patch("logging.FileHandler") as mock_file_handler, \
                 patch("logging.getLogger") as mock_get_logger:
                
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(
                    console_output=False,
                    log_file=log_file,
                    json_log_file=json_log_file
                )
                
                # Check that StreamHandler was not called
                mock_stream_handler.assert_not_called()
                
                # Check that FileHandler was still called
                assert mock_file_handler.call_count == 2

    def test_setup_logging_no_file(self):
        """Test setup_logging with file output disabled."""
        # Setup logging
        with patch("logging.StreamHandler") as mock_stream_handler, \
             patch("logging.FileHandler") as mock_file_handler, \
             patch("logging.getLogger") as mock_get_logger:
            
            mock_root_logger = MagicMock()
            mock_get_logger.return_value = mock_root_logger
            
            setup_logging(
                file_output=False,
                json_output=False
            )
            
            # Check that StreamHandler was called
            mock_stream_handler.assert_called_once()
            
            # Check that FileHandler was not called
            mock_file_handler.assert_not_called()


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_get_logger(self):
        """Test get_logger function."""
        with patch("logging.getLogger") as mock_get_logger:
            logger = get_logger("test_module")
            mock_get_logger.assert_called_once_with("test_module") 