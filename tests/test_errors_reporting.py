"""
Unit tests for the errors reporting module.
"""
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.errors import ErrorMetricsCollector, ErrorReporter, PinocchioError


class TestErrorReporter:
    """Tests for the ErrorReporter class."""

    def test_record_error_standard_exception(self):
        """Test recording a standard exception."""
        reporter = ErrorReporter()
        error = ValueError("Test error")

        reporter.record_error(error)

        assert len(reporter.errors) == 1
        assert reporter.errors[0]["error_type"] == "ValueError"
        assert reporter.errors[0]["message"] == "Test error"
        assert "timestamp" in reporter.errors[0]
        assert reporter.errors[0]["context"] == {}

    def test_record_error_with_context(self):
        """Test recording an error with context."""
        reporter = ErrorReporter()
        error = ValueError("Test error")
        context = {"function": "test_function", "input": "test_input"}

        reporter.record_error(error, context)

        assert len(reporter.errors) == 1
        assert reporter.errors[0]["context"] == context

    def test_record_error_pinocchio_error(self):
        """Test recording a PinocchioError."""
        reporter = ErrorReporter()
        error = PinocchioError(
            "Test error", error_code="TEST_ERROR", details={"key": "value"}
        )

        reporter.record_error(error)

        assert len(reporter.errors) == 1
        assert reporter.errors[0]["error_type"] == "PinocchioError"
        assert reporter.errors[0]["message"] == "Test error"
        assert reporter.errors[0]["error_code"] == "TEST_ERROR"
        assert reporter.errors[0]["details"] == {"key": "value"}

    def test_get_error_summary_empty(self):
        """Test getting error summary when no errors are recorded."""
        reporter = ErrorReporter()

        summary = reporter.get_error_summary()

        assert summary["total_errors"] == 0
        assert summary["error_counts"] == {}
        assert summary["latest_error"] is None

    def test_get_error_summary_with_errors(self):
        """Test getting error summary with multiple errors."""
        reporter = ErrorReporter()

        # Record multiple errors
        reporter.record_error(ValueError("Error 1"))
        reporter.record_error(TypeError("Error 2"))
        reporter.record_error(ValueError("Error 3"))

        summary = reporter.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["error_counts"] == {"ValueError": 2, "TypeError": 1}
        assert summary["latest_error"]["message"] == "Error 3"

    def test_get_errors_by_type(self):
        """Test filtering errors by type."""
        reporter = ErrorReporter()

        # Record multiple errors
        reporter.record_error(ValueError("Error 1"))
        reporter.record_error(TypeError("Error 2"))
        reporter.record_error(ValueError("Error 3"))

        value_errors = reporter.get_errors_by_type("ValueError")
        type_errors = reporter.get_errors_by_type("TypeError")
        other_errors = reporter.get_errors_by_type("KeyError")

        assert len(value_errors) == 2
        assert value_errors[0]["message"] == "Error 1"
        assert value_errors[1]["message"] == "Error 3"

        assert len(type_errors) == 1
        assert type_errors[0]["message"] == "Error 2"

        assert len(other_errors) == 0

    def test_get_errors_in_timeframe(self):
        """Test filtering errors by timeframe."""
        reporter = ErrorReporter()

        # Create timestamps
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        two_hours_ago = now - timedelta(hours=2)

        # Mock errors with different timestamps
        error1 = {
            "timestamp": two_hours_ago.isoformat(),
            "error_type": "ValueError",
            "message": "Error 1",
            "context": {},
        }

        error2 = {
            "timestamp": one_hour_ago.isoformat(),
            "error_type": "TypeError",
            "message": "Error 2",
            "context": {},
        }

        error3 = {
            "timestamp": now.isoformat(),
            "error_type": "ValueError",
            "message": "Error 3",
            "context": {},
        }

        # Add errors directly to the list
        reporter.errors = [error1, error2, error3]

        # Test filtering by timeframe
        errors_last_hour = reporter.get_errors_in_timeframe(
            one_hour_ago - timedelta(minutes=1), now + timedelta(minutes=1)
        )

        errors_last_two_hours = reporter.get_errors_in_timeframe(
            two_hours_ago - timedelta(minutes=1), now + timedelta(minutes=1)
        )

        errors_future = reporter.get_errors_in_timeframe(
            now + timedelta(hours=1), now + timedelta(hours=2)
        )

        assert len(errors_last_hour) == 2
        assert errors_last_hour[0]["message"] == "Error 2"
        assert errors_last_hour[1]["message"] == "Error 3"

        assert len(errors_last_two_hours) == 3

        assert len(errors_future) == 0

    def test_export_to_json(self):
        """Test exporting errors to JSON file."""
        reporter = ErrorReporter()

        # Record errors
        reporter.record_error(ValueError("Error 1"))
        reporter.record_error(TypeError("Error 2"))

        # Export to temporary file
        with tempfile.NamedTemporaryFile(suffix=".json") as temp_file:
            reporter.export_to_json(temp_file.name)

            # Read the file back
            with open(temp_file.name, "r") as f:
                exported_data = json.load(f)

            assert len(exported_data) == 2
            assert exported_data[0]["message"] == "Error 1"
            assert exported_data[1]["message"] == "Error 2"

    def test_clear_errors(self):
        """Test clearing errors."""
        reporter = ErrorReporter()

        # Record errors
        reporter.record_error(ValueError("Error 1"))
        reporter.record_error(TypeError("Error 2"))

        assert len(reporter.errors) == 2

        # Clear errors
        reporter.clear_errors()

        assert len(reporter.errors) == 0
        assert reporter.get_error_summary()["total_errors"] == 0


class TestErrorMetricsCollector:
    """Tests for the ErrorMetricsCollector class."""

    def test_record_error(self):
        """Test recording error metrics."""
        collector = ErrorMetricsCollector()

        # Record errors
        collector.record_error("ValueError")
        collector.record_error("ValueError")
        collector.record_error("TypeError")

        # Check internal state
        assert collector.error_counts["ValueError"] == 2
        assert collector.error_counts["TypeError"] == 1

    def test_calculate_error_rates(self):
        """Test calculating error rates."""
        collector = ErrorMetricsCollector()

        # Record errors
        collector.record_error("ValueError")
        collector.record_error("ValueError")
        collector.record_error("TypeError")

        # Calculate rates
        with patch("datetime.datetime") as mock_datetime:
            # Mock the current time
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.timestamp.return_value = mock_now.timestamp()

            rates = collector.calculate_error_rates(window_size=60)

            # Check rates
            assert "ValueError" in rates
            assert "TypeError" in rates
            assert len(rates["ValueError"]) == 1
            assert len(rates["TypeError"]) == 1
            assert rates["ValueError"][0][1] == 2  # Count
            assert rates["TypeError"][0][1] == 1  # Count

            # Check that error counts were reset
            assert collector.error_counts["ValueError"] == 0
            assert collector.error_counts["TypeError"] == 0

    def test_calculate_error_rates_prunes_old_data(self):
        """Test that calculate_error_rates prunes old data points."""
        collector = ErrorMetricsCollector()

        # Mock the datetime.now() to return a fixed timestamp
        with patch("datetime.datetime") as mock_datetime:
            # Set up a fixed current time
            current_time = 1000000.0
            mock_now = MagicMock()
            mock_now.timestamp.return_value = current_time
            mock_datetime.now.return_value = mock_now

            # Set up error rates with old and new data points
            old_time = current_time - 120  # 2 minutes ago (outside window)
            recent_time = current_time - 30  # 30 seconds ago (inside window)

            # Add some initial data to error_rates
            collector.error_rates = {
                "ValueError": [
                    (old_time, 5),  # Old data point (should be pruned)
                    (recent_time, 3),  # Recent data point (should be kept)
                ],
                "TypeError": [(old_time, 2)],  # Old data point (should be pruned)
            }

            # Record new errors
            collector.record_error("ValueError")
            collector.record_error("TypeError")

            # Calculate rates with 60-second window
            rates = collector.calculate_error_rates(window_size=60)

            # Check that old data points were pruned
            # Only the recent point and the new point should remain for ValueError
            assert len(rates["ValueError"]) == 1  # Only the recent point remains

            # For TypeError, the old point should be pruned and only the new point remains
            assert len(rates["TypeError"]) == 1  # Only the new point remains

    def test_get_error_rate(self):
        """Test getting error rate for a specific error type."""
        collector = ErrorMetricsCollector()

        # Set up error rates
        now = datetime.now().timestamp()
        collector.error_rates = {
            "ValueError": [
                (now - 30, 3),  # 30 seconds ago, 3 errors
                (now - 15, 2),  # 15 seconds ago, 2 errors
            ],
            "TypeError": [(now - 45, 1)],  # 45 seconds ago, 1 error
        }

        # Test with window_size=60 (should include all data points)
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value.timestamp.return_value = now

            # Total errors: 5 (3 + 2) in 60 seconds = 0.083 errors/second
            value_error_rate = collector.get_error_rate("ValueError", window_size=60)
            # Total errors: 1 in 60 seconds = 0.017 errors/second
            type_error_rate = collector.get_error_rate("TypeError", window_size=60)
            # No errors recorded
            key_error_rate = collector.get_error_rate("KeyError", window_size=60)

            assert value_error_rate > 0
            assert type_error_rate > 0
            assert key_error_rate == 0.0

    def test_get_error_rate_with_small_window(self):
        """Test getting error rate with a small window size."""
        collector = ErrorMetricsCollector()

        # Set up error rates
        now = datetime.now().timestamp()
        collector.error_rates = {
            "ValueError": [
                (now - 30, 3),  # 30 seconds ago, 3 errors
                (now - 15, 2),  # 15 seconds ago, 2 errors
            ],
            "TypeError": [(now - 45, 1)],  # 45 seconds ago, 1 error
        }

        # Test with window_size=20 (should only include the most recent data point)
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value.timestamp.return_value = now

            # Only includes the data point from 15 seconds ago: 2 errors in 20 seconds = 0.1 errors/second
            value_error_rate = collector.get_error_rate("ValueError", window_size=20)
            # No data points within the window
            type_error_rate = collector.get_error_rate("TypeError", window_size=20)

            assert value_error_rate > 0
            assert type_error_rate == 0.0

    def test_clear_metrics(self):
        """Test clearing metrics."""
        collector = ErrorMetricsCollector()

        # Record errors and calculate rates
        collector.record_error("ValueError")
        collector.record_error("TypeError")
        collector.calculate_error_rates()

        # Verify that data exists
        assert len(collector.error_rates["ValueError"]) > 0
        assert len(collector.error_rates["TypeError"]) > 0

        # Clear metrics
        collector.clear_metrics()

        # Verify that data was cleared
        assert len(collector.error_counts) == 0
        assert len(collector.error_rates) == 0
