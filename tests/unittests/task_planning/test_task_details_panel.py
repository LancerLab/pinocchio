#!/usr/bin/env python3
"""
Core tests for task details panel functionality
"""

import pytest

from pinocchio.cli.main import PinocchioCLI
from pinocchio.data_models.task_planning import AgentType, Task, TaskPriority
from pinocchio.task_planning.task_executor import TaskExecutor


def test_task_details_panel_buffering():
    """Test task details panel buffering mechanism."""
    cli = PinocchioCLI()

    # Test start marker
    cli.add_message("📋 Task Details:")
    assert cli.is_collecting_task_details
    assert len(cli.task_details_buffer) == 1

    # Test content buffering
    cli.add_message("   📋 Description: Test task")
    assert cli.is_collecting_task_details
    assert len(cli.task_details_buffer) == 2

    # Test end marker
    cli.add_message("<<END_TASK_DETAILS>>")
    assert not cli.is_collecting_task_details
    assert len(cli.task_details_buffer) == 0


def test_task_details_content_generation():
    """Test task details content generation."""
    executor = TaskExecutor()

    task = Task(
        task_id="test_task",
        agent_type=AgentType.GENERATOR,
        task_description="Generate code",
        priority=TaskPriority.CRITICAL,
        input_data={"instruction": "Generate high-performance code"},
    )

    details = executor._generate_task_details_panel(task)

    # Verify essential content
    content_text = "\n".join(details)
    assert "📋 Description:" in content_text
    assert "🎯 Priority:" in content_text
    assert "💡 Detailed Instruction:" in content_text


def test_task_details_panel_with_session_prefix():
    """Test task details panel with session prefix."""
    cli = PinocchioCLI()

    messages = [
        "[session_123] 📋 Task Details:",
        "[session_123]    📋 Description: Test",
        "[session_123] <<END_TASK_DETAILS>>",
    ]

    for message in messages:
        cli.add_message(message)

    assert not cli.is_collecting_task_details
    assert len(cli.task_details_buffer) == 0


if __name__ == "__main__":
    # Run tests
    test_task_details_panel_buffering()
    test_task_details_content_generation()
    test_task_details_panel_with_session_prefix()
    print("✅ All core task details panel tests passed!")
