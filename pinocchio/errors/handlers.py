"""
Error handling utilities for the Pinocchio multi-agent system.

This module provides decorators and context managers for standardized
error handling across the application.
"""
import functools
import logging
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Type, TypeVar, cast

from .exceptions import PinocchioError

# Type variable for generic function return type
T = TypeVar("T")

# Configure logger
logger = logging.getLogger(__name__)


def handle_errors(
    fallback_value: Any = None, reraise: bool = False, log_level: int = logging.ERROR
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorate a function with error handling.

    Args:
        fallback_value: Value to return if an error occurs and reraise is False
        reraise: Whether to reraise the exception after handling
        log_level: Logging level for error messages

    Returns:
        Decorated function with error handling
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract function context for better error reporting
                func_name = func.__name__
                module_name = func.__module__

                # Get traceback information
                tb = traceback.format_exc()

                # Log the error with context
                error_context = {
                    "function": func_name,
                    "module": module_name,
                    "args": repr(args),
                    "kwargs": repr(kwargs),
                    "traceback": tb,
                }

                if isinstance(e, PinocchioError):
                    error_context.update(e.to_dict())

                logger.log(
                    log_level,
                    f"Error in {module_name}.{func_name}: {str(e)}",
                    extra={"error_context": error_context},
                )

                # Reraise or return fallback
                if reraise:
                    raise
                return cast(T, fallback_value)

        return wrapper

    return decorator


@contextmanager
def error_context(
    context_name: str, reraise: bool = True, log_level: int = logging.ERROR
) -> Generator[Any, None, None]:
    """
    Provide a context manager for handling errors in a block of code.

    Args:
        context_name: Name of the context for logging
        reraise: Whether to reraise the exception after handling
        log_level: Logging level for error messages

    Yields:
        Context object with error information
    """

    class Context:
        def __init__(self) -> None:
            self.error_occurred = False
            self.exception: Optional[Exception] = None
            self.traceback: Optional[str] = None

    context = Context()

    try:
        yield context
    except Exception as e:
        context.error_occurred = True
        context.exception = e
        context.traceback = traceback.format_exc()

        # Log the error with context
        error_context = {"context_name": context_name, "traceback": context.traceback}

        if isinstance(e, PinocchioError):
            error_context.update(e.to_dict())

        logger.log(
            log_level,
            f"Error in context '{context_name}': {str(e)}",
            extra={"error_context": error_context},
        )

        if reraise:
            raise


def retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions_to_retry: Optional[List[Type[Exception]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorate a function to retry on exception with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor to multiply delay by on each retry
        exceptions_to_retry: List of exception types to retry on (defaults to all)

    Returns:
        Decorated function with retry logic
    """
    if exceptions_to_retry is None:
        exceptions_to_retry = [Exception]

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            retries = 0
            delay = 1.0

            while True:
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions_to_retry) as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Maximum retries ({max_retries}) exceeded for {func.__name__}",
                            extra={"last_exception": str(e)},
                        )
                        raise

                    # Calculate delay with exponential backoff
                    wait_time = delay * (backoff_factor ** (retries - 1))

                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.2f}s due to: {str(e)}"
                    )

                    time.sleep(wait_time)

        return wrapper

    return decorator


def global_error_handler(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: Optional[Any],
) -> None:
    """
    Handle unhandled exceptions globally.

    Args:
        exc_type: Exception type
        exc_value: Exception instance
        exc_traceback: Exception traceback
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't handle keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Get traceback as string
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    # Log the unhandled exception
    logger.critical(
        f"Unhandled exception: {exc_type.__name__}: {str(exc_value)}",
        extra={"traceback": tb_str, "timestamp": datetime.now().isoformat()},
    )

    # You can add additional handling here, such as:
    # - Sending error notifications
    # - Graceful shutdown
    # - Writing to a crash log


class CircuitBreakerOpenError(PinocchioError):
    """Error raised when a circuit breaker is open."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a CircuitBreakerOpenError.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", details=details)


class CircuitBreaker:
    """
    Implements the circuit breaker pattern to prevent repeated calls to failing services.

    The circuit breaker has three states:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, calls are blocked
    - HALF-OPEN: Testing if service has recovered, limited calls pass through

    This pattern helps prevent cascading failures and allows failing services time to recover.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 1,
    ) -> None:
        """
        Initialize a new CircuitBreaker.

        Args:
            name: Name of this circuit breaker (for logging)
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Seconds to wait before trying to recover (half-open)
            half_open_max_calls: Maximum number of calls to allow in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker.{name}")

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function

        Raises:
            CircuitBreakerOpenError: If the circuit is open
            Any exception raised by the function
        """
        self._check_state()

        try:
            result = func(*args, **kwargs)

            # Success - reset on success if in HALF-OPEN state
            if self.state == "HALF-OPEN":
                self.reset()
                self.logger.info(
                    f"Circuit breaker '{self.name}' reset to CLOSED after successful call"
                )

            return result

        except Exception as e:
            self._record_failure()
            self.logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {str(e)}",
                extra={"exception": str(e), "state": self.state},
            )
            raise

    def _check_state(self) -> None:
        """
        Check the current state of the circuit breaker and determine if calls should proceed.

        Raises:
            CircuitBreakerOpenError: If the circuit is open
        """
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = "HALF-OPEN"
                self.half_open_calls = 0
                self.logger.info(
                    f"Circuit breaker '{self.name}' switched from OPEN to HALF-OPEN"
                )
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN",
                    details={
                        "failure_count": self.failure_count,
                        "last_failure_time": self.last_failure_time,
                        "recovery_timeout": self.recovery_timeout,
                    },
                )

        if self.state == "HALF-OPEN":
            # Only allow a limited number of calls in half-open state
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is HALF-OPEN and maximum calls exceeded",
                    details={
                        "half_open_calls": self.half_open_calls,
                        "half_open_max_calls": self.half_open_max_calls,
                    },
                )
            self.half_open_calls += 1

    def _record_failure(self) -> None:
        """Record a failure and update circuit breaker state."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker '{self.name}' switched from CLOSED to OPEN "
                f"after {self.failure_count} failures"
            )
        elif self.state == "HALF-OPEN":
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker '{self.name}' switched from HALF-OPEN back to OPEN after failure"
            )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self.state = "CLOSED"
        self.logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    @property
    def is_open(self) -> bool:
        """Check if the circuit breaker is open."""
        return self.state == "OPEN"

    @property
    def is_closed(self) -> bool:
        """Check if the circuit breaker is closed."""
        return self.state == "CLOSED"

    @property
    def is_half_open(self) -> bool:
        """Check if the circuit breaker is half-open."""
        return self.state == "HALF-OPEN"
