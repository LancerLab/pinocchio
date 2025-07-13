#!/usr/bin/env python3
"""
Test task details panel functionality
"""

import asyncio

from pinocchio.cli.main import PinocchioCLI
from pinocchio.data_models.task_planning import (
    AgentType,
    Task,
    TaskDependency,
    TaskPlan,
    TaskPriority,
)
from pinocchio.task_planning.task_executor import TaskExecutor


class TestTaskDetailsPanel:
    """Test task details panel functionality."""

    def test_task_details_panel_buffering(self):
        """Test that task details are properly buffered and displayed as a panel."""
        cli = PinocchioCLI()

        # Simulate task details messages
        messages = [
            "📋 Task Details:",
            "   📋 Description: [Round 1] Compile and debug generated code",
            "   🎯 Priority: 🔴 critical",
            "   🔗 Dependencies: task_1",
            "   📋 Requirements: error_handling=True",
            "   💡 Detailed Instruction:",
            "      Analyze the generated code for potential issues, errors, and improvements.",
            "      Debugging Focus:",
            "      - Syntax errors and compatibility issues",
            "      - Logic errors and edge cases",
            "      - Performance bottlenecks",
            "      - Memory access patterns",
            "      - Error handling and validation",
            "      Provide:",
            "      - Detailed analysis of issues found",
            "      - Specific fixes with explanations",
            "      - Improved code version",
            "      - Recommendations for robustness",
            "",
            "<<END_TASK_DETAILS>>",
        ]

        # Process messages through CLI
        for message in messages:
            cli.add_message(message)

        # Verify that buffering state is reset
        assert not cli.is_collecting_task_details
        assert len(cli.task_details_buffer) == 0

    def test_task_details_panel_content_generation(self):
        """Test that TaskExecutor generates proper task details panel content."""
        executor = TaskExecutor()

        # Create a test task
        task = Task(
            task_id="task_1",
            agent_type=AgentType.DEBUGGER,
            task_description="Debug the generated code",
            priority=TaskPriority.CRITICAL,
            input_data={
                "instruction": "Analyze the generated code for potential issues, errors, and improvements."
            },
            dependencies=[TaskDependency(task_id="task_0", dependency_type="required")],
            requirements={"error_handling": True},
        )

        # Generate task details panel content
        details = executor._generate_task_details_panel(task)

        # Verify content structure
        assert len(details) > 0
        assert any("📋 Description:" in line for line in details)
        assert any("🎯 Priority:" in line for line in details)
        assert any("🔗 Dependencies:" in line for line in details)
        assert any("📋 Requirements:" in line for line in details)
        assert any("💡 Detailed Instruction:" in line for line in details)

    def test_task_details_panel_marker_detection(self):
        """Test that CLI properly detects task details start and end markers."""
        cli = PinocchioCLI()

        # Test start marker detection
        start_message = "📋 Task Details:"
        cli.add_message(start_message)
        assert cli.is_collecting_task_details
        assert len(cli.task_details_buffer) == 1

        # Test end marker detection
        end_message = "<<END_TASK_DETAILS>>"
        cli.add_message(end_message)
        assert not cli.is_collecting_task_details
        assert len(cli.task_details_buffer) == 0

    def test_task_details_panel_buffering_with_prefix(self):
        """Test that task details buffering works with session prefixes."""
        cli = PinocchioCLI()

        # Simulate messages with session prefix
        messages = [
            "[session_abc123] 📋 Task Details:",
            "[session_abc123]    📋 Description: Test task",
            "[session_abc123]    🎯 Priority: 🔴 critical",
            "[session_abc123] <<END_TASK_DETAILS>>",
        ]

        # Process messages
        for message in messages:
            cli.add_message(message)

        # Verify buffering completed
        assert not cli.is_collecting_task_details
        assert len(cli.task_details_buffer) == 0

    def test_task_details_panel_empty_content(self):
        """Test that empty task details are handled properly."""
        executor = TaskExecutor()

        # Create a minimal task
        task = Task(
            task_id="task_1",
            agent_type=AgentType.GENERATOR,
            task_description="Generate code",
            priority=TaskPriority.MEDIUM,
        )

        # Generate task details panel content
        details = executor._generate_task_details_panel(task)

        # Should at least contain description
        assert len(details) > 0
        assert any("📋 Description:" in line for line in details)


async def test_task_details_panel_integration():
    """Integration test for task details panel functionality."""
    cli = PinocchioCLI()

    print("=== Task Details Panel Integration Test ===")

    # Simulate complete task details flow
    messages = [
        "📋 Task Details:",
        "   📋 Description: [Round 1] Generate high-performance code",
        "   🎯 Priority: 🔴 critical",
        "   💡 Detailed Instruction:",
        "      Generate high-performance Choreo DSL operator code based on the user request.",
        "      Focus on:",
        "      - Performance optimization",
        "      - Memory efficiency",
        "      - Code correctness",
        "",
        "<<END_TASK_DETAILS>>",
    ]

    # Process messages
    for message in messages:
        cli.add_message(message)

    print("✅ Task details panel integration test completed")


if __name__ == "__main__":
    # Run unit tests
    test_instance = TestTaskDetailsPanel()
    test_instance.test_task_details_panel_buffering()
    test_instance.test_task_details_panel_content_generation()
    test_instance.test_task_details_panel_marker_detection()
    test_instance.test_task_details_panel_buffering_with_prefix()
    test_instance.test_task_details_panel_empty_content()

    # Run integration test
    asyncio.run(test_task_details_panel_integration())

    print("✅ All task details panel tests passed!")
