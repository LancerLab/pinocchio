"""
Error reporting and analysis tools for the Pinocchio multi-agent system.

This module provides utilities for collecting, analyzing, and reporting errors
that occur during system operation.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .exceptions import PinocchioError


class ErrorReporter:
    """Collects and reports errors for analysis."""

    def __init__(self) -> None:
        """Initialize a new ErrorReporter."""
        self.errors: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def record_error(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an error with its context.

        Args:
            error: The exception that occurred
            context: Additional context information
        """
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error.__class__.__name__,
            "message": str(error),
            "context": context or {},
        }

        if isinstance(error, PinocchioError):
            error_data.update(error.to_dict())

        self.errors.append(error_data)
        self.logger.debug(f"Recorded error: {error_data['error_type']}")

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of recorded errors.

        Returns:
            Dictionary containing error statistics
        """
        if not self.errors:
            return {"total_errors": 0, "error_counts": {}, "latest_error": None}

        error_counts: Dict[str, int] = defaultdict(int)
        for error in self.errors:
            error_type = error["error_type"]
            error_counts[error_type] += 1

        return {
            "total_errors": len(self.errors),
            "error_counts": dict(error_counts),
            "latest_error": self.errors[-1] if self.errors else None,
        }

    def get_errors_by_type(self, error_type: str) -> List[Dict[str, Any]]:
        """
        Get all errors of a specific type.

        Args:
            error_type: The error type to filter by

        Returns:
            List of errors matching the specified type
        """
        return [error for error in self.errors if error["error_type"] == error_type]

    def get_errors_in_timeframe(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get errors that occurred within a specific timeframe.

        Args:
            start_time: Start of the timeframe
            end_time: End of the timeframe

        Returns:
            List of errors within the specified timeframe
        """
        return [
            error
            for error in self.errors
            if start_time <= datetime.fromisoformat(error["timestamp"]) <= end_time
        ]

    def export_to_json(self, filepath: str) -> None:
        """
        Export recorded errors to a JSON file.

        Args:
            filepath: Path to the output file
        """
        with open(filepath, "w") as f:
            json.dump(self.errors, f, indent=2)
        self.logger.info(f"Exported {len(self.errors)} error records to {filepath}")

    def clear_errors(self) -> None:
        """Clear all recorded errors."""
        self.errors = []
        self.logger.debug("Cleared error records")


class ErrorMetricsCollector:
    """Collects error metrics for monitoring."""

    def __init__(self) -> None:
        """Initialize a new ErrorMetricsCollector."""
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_rates: Dict[str, List[tuple[float, int]]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def record_error(self, error_type: str) -> None:
        """
        Record an error occurrence.

        Args:
            error_type: Type of the error that occurred
        """
        self.error_counts[error_type] += 1
        self.logger.debug(f"Recorded error metric for {error_type}")

    def calculate_error_rates(
        self, window_size: int = 60
    ) -> Dict[str, List[tuple[float, int]]]:
        """
        Calculate error rates over the specified window size.

        Args:
            window_size: Window size in seconds for rate calculation

        Returns:
            Dictionary mapping error types to lists of (timestamp, count) tuples
        """
        current_time = datetime.now().timestamp()

        # Record error rates
        for error_type, count in self.error_counts.items():
            self.error_rates[error_type].append((current_time, count))

        # Prune old data points
        for error_type in list(self.error_rates.keys()):
            self.error_rates[error_type] = [
                (t, c)
                for t, c in self.error_rates[error_type]
                if current_time - t <= window_size
            ]

        # Reset counts
        self.error_counts = defaultdict(int)

        return dict(self.error_rates)

    def get_error_rate(self, error_type: str, window_size: int = 60) -> float:
        """
        Get the error rate for a specific error type.

        Args:
            error_type: Type of error to calculate rate for
            window_size: Window size in seconds for rate calculation

        Returns:
            Error rate (errors per second) within the specified window
        """
        current_time = datetime.now().timestamp()

        # Filter data points within the window
        if error_type not in self.error_rates:
            return 0.0

        data_points = [
            (t, c)
            for t, c in self.error_rates.get(error_type, [])
            if current_time - t <= window_size
        ]

        if not data_points:
            return 0.0

        # Calculate total errors in the window
        total_errors = sum(count for _, count in data_points)

        # If we just called calculate_error_rates, the counts were reset
        if total_errors == 0:
            return 0.0

        # Calculate error rate (errors per second)
        return total_errors / window_size if window_size > 0 else 0.0

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.error_counts = defaultdict(int)
        self.error_rates = defaultdict(list)
        self.logger.debug("Cleared error metrics")
