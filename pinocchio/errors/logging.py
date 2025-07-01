"""
Logging utilities for the Pinocchio multi-agent system.

This module provides logging formatters and handlers for structured logging,
particularly focused on error reporting and monitoring.
"""
import json
import logging
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """
    Format log records as JSON objects.
    
    This formatter converts log records to JSON format, making them easier
    to parse and analyze with log management tools.
    """

    def __init__(
        self, 
        include_timestamp: bool = True,
        include_level: bool = True,
        include_name: bool = True,
        include_path: bool = True,
        include_process: bool = False,
        include_thread: bool = False,
        **kwargs: Any
    ):
        """
        Initialize a new JsonFormatter.
        
        Args:
            include_timestamp: Whether to include timestamp in the log
            include_level: Whether to include log level in the log
            include_name: Whether to include logger name in the log
            include_path: Whether to include file path and line number in the log
            include_process: Whether to include process ID in the log
            include_thread: Whether to include thread ID in the log
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

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON string representation of the log record
        """
        log_data: Dict[str, Any] = {}
        
        # Include standard fields based on configuration
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
        
        # Add the log message
        log_data["message"] = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add any extra attributes from the record
        if hasattr(record, "error_context") and record.error_context:
            log_data["error_context"] = record.error_context
        
        # Add any additional fields from record.__dict__
        for key, value in record.__dict__.items():
            if key not in [
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "id", "levelname", "levelno", "lineno", "module",
                "msecs", "message", "msg", "name", "pathname", "process",
                "processName", "relativeCreated", "stack_info", "thread", "threadName",
                "error_context"
            ]:
                log_data[key] = value
        
        # Add any additional fields specified in the constructor
        log_data.update(self.additional_fields)
        
        return json.dumps(log_data)


class PinocchioLogFormatter(logging.Formatter):
    """
    A custom formatter for Pinocchio logs with color support.
    
    This formatter provides a human-readable format for console output,
    with optional color coding for different log levels.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'RESET': '\033[0m'  # Reset
    }
    
    def __init__(
        self, 
        fmt: Optional[str] = None, 
        datefmt: Optional[str] = None, 
        use_colors: bool = True
    ):
        """
        Initialize a new PinocchioLogFormatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
            use_colors: Whether to use colors in the output
        """
        if fmt is None:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"
            
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with optional color coding.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted log message
        """
        # Save original levelname
        original_levelname = record.levelname
        
        # Apply color if enabled
        if self.use_colors and record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted_message = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        
        # Add exception info in a structured way if present
        if record.exc_info:
            formatted_exception = self._format_exception(record)
            formatted_message = f"{formatted_message}\n{formatted_exception}"
        
        # Add error context if present
        if hasattr(record, "error_context") and record.error_context:
            formatted_context = self._format_error_context(record.error_context)
            formatted_message = f"{formatted_message}\n{formatted_context}"
            
        return formatted_message
    
    def _format_exception(self, record: logging.LogRecord) -> str:
        """Format exception information with indentation and color."""
        if not record.exc_info:
            return ""
            
        exc_type, exc_value, exc_tb = record.exc_info
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        
        if self.use_colors:
            return f"{self.COLORS['ERROR']}Exception:\n{tb_text}{self.COLORS['RESET']}"
        else:
            return f"Exception:\n{tb_text}"
    
    def _format_error_context(self, error_context: Dict[str, Any]) -> str:
        """Format error context with indentation."""
        context_lines = ["Error Context:"]
        for key, value in error_context.items():
            if key == "traceback" and isinstance(value, str):
                context_lines.append(f"  {key}: <traceback omitted>")
            else:
                context_lines.append(f"  {key}: {value}")
                
        context_text = "\n".join(context_lines)
        
        if self.use_colors:
            return f"{self.COLORS['WARNING']}{context_text}{self.COLORS['RESET']}"
        else:
            return context_text


def setup_logging(
    log_level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    json_output: bool = True,
    log_file: str = "pinocchio.log",
    json_log_file: str = "pinocchio_errors.json",
    use_colors: bool = True
) -> None:
    """
    Configure logging for the Pinocchio system.
    
    This function sets up logging with appropriate formatters and handlers
    based on the specified configuration.
    
    Args:
        log_level: The minimum logging level
        console_output: Whether to output logs to the console
        file_output: Whether to output logs to a file
        json_output: Whether to output structured JSON logs
        log_file: Path to the log file
        json_log_file: Path to the JSON log file
        use_colors: Whether to use colors in console output
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if enabled
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(PinocchioLogFormatter(use_colors=use_colors))
        root_logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if file_output:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(PinocchioLogFormatter(use_colors=False))
        root_logger.addHandler(file_handler)
    
    # Add JSON handler if enabled
    if json_output:
        json_handler = logging.FileHandler(json_log_file)
        json_handler.setFormatter(JsonFormatter())
        json_handler.setLevel(logging.ERROR)  # Only log errors and above in JSON format
        root_logger.addHandler(json_handler)
    
    # Log that logging has been configured
    logging.info("Logging configured")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    This is a convenience function to get a logger with the specified name,
    ensuring that it's properly configured.
    
    Args:
        name: The name of the logger
        
    Returns:
        A logger instance
    """
    return logging.getLogger(name) 