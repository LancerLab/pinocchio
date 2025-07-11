"""
Logging utilities for the Pinocchio multi-agent system.

This module provides formatters and handlers for structured logging
of errors and other events.
"""
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Type


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    This formatter outputs log records as JSON objects, which can be
    easily parsed by log analysis tools.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_name: bool = True,
        include_path: bool = True,
        include_process: bool = False,
        include_thread: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the JSON formatter.

        Args:
            include_timestamp: Whether to include timestamp in the output
            include_level: Whether to include log level in the output
            include_name: Whether to include logger name in the output
            include_path: Whether to include file path and line number in the output
            include_process: Whether to include process ID in the output
            include_thread: Whether to include thread ID in the output
            **kwargs: Additional fields to include in every log record
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_name = include_name
        self.include_path = include_path
        self.include_process = include_process
        self.include_thread = include_thread
        self.additional_fields = kwargs

    def _add_basic_fields(
        self, record: logging.LogRecord, log_data: Dict[str, Any]
    ) -> None:
        """Add basic fields to log data."""
        # Always include the message
        log_data["message"] = record.getMessage()

        # Add optional fields based on configuration
        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(record.created).isoformat()

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_name:
            log_data["logger"] = record.name

        if self.include_path:
            log_data["path"] = f"{record.pathname}:{record.lineno}"

        if self.include_process:
            log_data["process"] = record.process

        if self.include_thread:
            log_data["thread"] = record.thread

    def _add_exception_info(
        self, record: logging.LogRecord, log_data: Dict[str, Any]
    ) -> None:
        """Add exception information to log data."""
        if record.exc_info:
            exc_type: Optional[Type[BaseException]] = record.exc_info[0]
            exc_value: Optional[BaseException] = record.exc_info[1]
            exc_tb = record.exc_info[2]

            if exc_type and exc_value:
                log_data["exception"] = {
                    "type": exc_type.__name__,
                    "message": str(exc_value),
                    "traceback": traceback.format_exception_only(exc_type, exc_value)
                    + (traceback.format_tb(exc_tb) if exc_tb else []),
                }

    def _add_additional_fields(
        self, record: logging.LogRecord, log_data: Dict[str, Any]
    ) -> None:
        """Add additional fields to log data."""
        # Add error context if present
        if hasattr(record, "error_context") and record.error_context:
            log_data["error_context"] = record.error_context

        # Add any additional fields from constructor
        log_data.update(self.additional_fields)

        # Add any additional fields from the record
        excluded_fields = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "id",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "error_context",
        }

        for key, value in record.__dict__.items():
            if key not in excluded_fields:
                log_data[key] = value

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.

        Args:
            record: The log record to format

        Returns:
            JSON string representation of the log record
        """
        log_data: Dict[str, Any] = {}

        self._add_basic_fields(record, log_data)
        self._add_exception_info(record, log_data)
        self._add_additional_fields(record, log_data)

        return json.dumps(log_data)


class PinocchioLogFormatter(logging.Formatter):
    """
    Custom formatter for Pinocchio logs.

    This formatter provides readable, colorized console output with
    contextual information for errors.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[91m\033[1m",  # Bold Red
        "RESET": "\033[0m",  # Reset
    }

    def __init__(
        self,
        fmt: str = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        use_colors: bool = True,
    ) -> None:
        """
        Initialize the formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            use_colors: Whether to use ANSI colors in output
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.

        Args:
            record: The log record to format

        Returns:
            Formatted log message
        """
        # Make a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Apply color to level name if enabled
        levelname = record_copy.levelname
        if self.use_colors and levelname in self.COLORS:
            # Apply color directly to the levelname
            record_copy.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        # Format the basic message
        result = super().format(record_copy)

        # Add exception info if present
        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            if exc_type and exc_value:
                if not result.endswith("\n"):
                    result += "\n"
                result += "Exception: " + str(exc_value)

                # Add traceback
                tb_lines = traceback.format_exception(*record.exc_info)
                result += "\n" + "".join(tb_lines)

        # Add error context if present
        if hasattr(record, "error_context") and record.error_context:
            result += "\nError Context:"
            for key, value in record.error_context.items():
                # Truncate large values like tracebacks
                if key == "traceback" and isinstance(value, str) and len(value) > 100:
                    result += f"\n  {key}: <traceback omitted>"
                else:
                    result += f"\n  {key}: {value}"

        return result


def setup_logging(
    log_level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    json_output: bool = True,
    log_file: str = "pinocchio.log",
    json_log_file: str = "pinocchio.json.log",
) -> None:
    """
    Set up logging for the Pinocchio system.

    Args:
        log_level: Minimum log level to record
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        json_output: Whether to output structured JSON logs
        log_file: Path to the log file
        json_log_file: Path to the JSON log file
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(PinocchioLogFormatter(use_colors=True))
        root_logger.addHandler(console_handler)

    # File handler
    if file_output:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(PinocchioLogFormatter(use_colors=False))
        root_logger.addHandler(file_handler)

    # JSON file handler
    if json_output:
        # Create directory if it doesn't exist
        json_log_dir = os.path.dirname(json_log_file)
        if json_log_dir and not os.path.exists(json_log_dir):
            os.makedirs(json_log_dir)

        json_handler = logging.FileHandler(json_log_file)
        json_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(json_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger (typically the module name)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
