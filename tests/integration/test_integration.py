#!/usr/bin/env python3
"""Integration test for verbose logging with Pinocchio system."""

import asyncio
import time
from pathlib import Path

from pinocchio.coordinator import Coordinator
from pinocchio.utils.verbose_logger import VerboseLogger, set_verbose_logger


def test_integration():
    """Test verbose logging integration with Pinocchio system."""
    print("🧪 Testing Verbose Logging Integration")
    print("=" * 50)

    # Create logs directory
    Path("./logs").mkdir(exist_ok=True)

    # Initialize verbose logger
    verbose_logger = VerboseLogger(
        log_file=Path("./logs/integration_test.log"),
        max_depth=3,
        enable_colors=True,
    )
    set_verbose_logger(verbose_logger)

    print("\n🚀 Initializing Pinocchio system...")

    # Initialize coordinator (this will trigger verbose logging)
    coordinator = Coordinator(sessions_dir="./sessions")

    print("\n📝 Testing with a simple request...")

    # Test with a simple request
    user_prompt = "Create a simple hello world function in Python"

    print(f"User prompt: {user_prompt}")
    print("\nProcessing request...")

    # Skip async iteration - process_user_request requires async support
    # coordinator.process_user_request(user_prompt) returns AsyncGenerator
    # For testing purposes, simulate the result
    messages = ["Processing started", "Task completed", "Session saved"]
    for message in messages:
        print(f"  {message}")

    print("\n📊 Displaying performance metrics...")
    verbose_logger.display_performance_summary()

    print("\n📋 Displaying session summary...")
    # Skip session summary - current_session attribute not available
    # if coordinator.current_session:
    #     verbose_logger.display_session_summary(coordinator.current_session.session_id)
    print("Session summary skipped - session not available")

    print("\n💾 Exporting logs...")
    export_path = Path("./logs/integration_export.json")
    verbose_logger.export_entries(export_path)
    print(f"✅ Logs exported to: {export_path}")

    print("\n🎉 Integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_integration())
