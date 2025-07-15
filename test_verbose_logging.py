#!/usr/bin/env python3
"""Test script for verbose logging functionality."""

import asyncio
import time
from pathlib import Path

from pinocchio.utils.verbose_logger import (
    LogLevel,
    VerboseLogger,
    get_verbose_logger,
    set_verbose_logger,
)


async def test_verbose_logging():
    """Test verbose logging functionality."""
    print("🧪 Testing Verbose Logging System")
    print("=" * 50)

    # Create logs directory
    Path("./logs").mkdir(exist_ok=True)

    # Initialize verbose logger
    verbose_logger = VerboseLogger(
        log_file=Path("./logs/test_verbose.log"),
        max_depth=3,
        enable_colors=True,
    )
    set_verbose_logger(verbose_logger)

    print("\n📝 Testing basic logging...")

    # Test different log levels
    verbose_logger.log(
        LogLevel.INFO,
        "test",
        "This is an info message",
        data={"test_key": "test_value", "number": 42},
    )

    verbose_logger.log(
        LogLevel.WARNING,
        "test",
        "This is a warning message",
        data={"warning_type": "test_warning"},
    )

    verbose_logger.log(
        LogLevel.ERROR,
        "test",
        "This is an error message",
        data={"error_code": 500, "error_type": "test_error"},
    )

    print("\n🤖 Testing agent activity logging...")

    # Test agent activity
    verbose_logger.log_agent_activity(
        "generator",
        "Code generation started",
        data={"task_id": "task_001", "language": "python", "complexity": "high"},
        session_id="session_123",
        step_id="step_001",
        duration_ms=150.5,
    )

    verbose_logger.log_agent_activity(
        "optimizer",
        "Code optimization completed",
        data={
            "task_id": "task_002",
            "optimizations_applied": ["loop_unrolling", "memory_access"],
            "performance_improvement": "15%",
        },
        session_id="session_123",
        step_id="step_002",
        duration_ms=320.7,
    )

    print("\n🎯 Testing coordinator activity logging...")

    # Test coordinator activity
    verbose_logger.log_coordinator_activity(
        "Session started",
        data={
            "user_prompt": "Create a high-performance sorting algorithm",
            "session_id": "session_123",
            "agent_count": 4,
        },
        session_id="session_123",
        duration_ms=45.2,
    )

    verbose_logger.log_coordinator_activity(
        "Task plan created",
        data={
            "task_count": 3,
            "plan_id": "plan_456",
            "validation": {"valid": True, "warnings": []},
        },
        session_id="session_123",
        duration_ms=120.8,
    )

    print("\n🤖 Testing LLM activity logging...")

    # Test LLM activity
    verbose_logger.log_llm_activity(
        "Code generation request",
        request_data={"prompt_length": 1500, "model": "gpt-4", "temperature": 0.7},
        response_data={
            "response_length": 2500,
            "tokens_used": 1800,
            "completion_reason": "stop",
        },
        session_id="session_123",
        duration_ms=2500.3,
    )

    print("\n📊 Testing performance logging...")

    # Test performance metrics
    verbose_logger.log_performance("llm_call", 1500.5)
    verbose_logger.log_performance("agent_execution", 320.7)
    verbose_logger.log_performance("task_planning", 120.8)
    verbose_logger.log_performance("llm_call", 1800.2)
    verbose_logger.log_performance("agent_execution", 450.1)

    print("\n📈 Displaying performance summary...")
    verbose_logger.display_performance_summary()

    print("\n📋 Displaying session summary...")
    verbose_logger.display_session_summary("session_123")

    print("\n💾 Testing log export...")
    export_path = Path("./logs/verbose_export_test.json")
    verbose_logger.export_entries(export_path)
    print(f"✅ Logs exported to: {export_path}")

    print("\n🎉 Verbose logging test completed!")
    print("Check the logs directory for generated files.")


if __name__ == "__main__":
    asyncio.run(test_verbose_logging())
