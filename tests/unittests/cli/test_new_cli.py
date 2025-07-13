#!/usr/bin/env python3
"""Tests for the new Pinocchio CLI functionality."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from pinocchio.cli.main import PinocchioCLI
from pinocchio.data_models.task_planning import AgentType, TaskPriority
from tests.utils import create_test_task


class TestNewCLI:
    """Test the new PinocchioCLI functionality."""

    @pytest.fixture
    def cli(self):
        """Create a new CLI instance for testing."""
        return PinocchioCLI()

    def test_cli_initialization(self, cli):
        """Test CLI initialization."""
        assert cli.console is not None
        assert cli.coordinator is None
        assert cli.messages == []
        assert cli.tasks == []
        assert cli.last_task_plan is None
        assert cli.task_plan_overview_buffer == []
        assert cli.is_collecting_task_plan_overview is False

    def test_task_plan_overview_buffering_with_session_prefix(self, cli):
        """Test task plan overview buffering with session ID prefix."""
        # Simulate messages with session ID prefix (like from Coordinator)
        messages = [
            "[session_abc123] 📋 Task Plan Overview:",
            "  1. ⚡ GENERATOR (🔴 critical)",
            "     📝 [Round 1] hello",
            "     💡 Instruction: Generate high-performance Choreo DSL operator code based on the user request.",
            "     🔗 Dependencies: ",
            "",
            "  2. 🔧 DEBUGGER (🔴 critical)",
            "     📝 [Round 1] Compile and debug generated code",
            "     💡 Instruction: Analyze the generated code for potential issues, errors, and improvements.",
            "     🔗 Dependencies: task_1",
            "",
            "  3. 🚀 OPTIMIZER (🟡 high)",
            "     📝 [Round 1] Optimise code for: performance and efficiency",
            "     💡 Instruction: Analyze and optimize the generated Choreo DSL code for better performance.",
            "     🔗 Dependencies: task_2",
            "",
            "<<END_TASK_PLAN>>",
        ]

        # Process messages
        for message in messages:
            cli.add_message(message)

        # Verify buffering worked correctly
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []

    def test_task_plan_overview_buffering_without_session_prefix(self, cli):
        """Test task plan overview buffering without session ID prefix."""
        # Simulate messages without session ID prefix
        messages = [
            "📋 Task Plan Overview:",
            "  1. ⚡ GENERATOR (🔴 critical)",
            "     📝 [Round 1] hello",
            "     💡 Instruction: Generate high-performance Choreo DSL operator code based on the user request.",
            "     🔗 Dependencies: ",
            "",
            "  2. 🔧 DEBUGGER (🔴 critical)",
            "     📝 [Round 1] Compile and debug generated code",
            "     💡 Instruction: Analyze the generated code for potential issues, errors, and improvements.",
            "     🔗 Dependencies: task_1",
            "",
            "  3. 🚀 OPTIMIZER (🟡 high)",
            "     📝 [Round 1] Optimise code for: performance and efficiency",
            "     💡 Instruction: Analyze and optimize the generated Choreo DSL code for better performance.",
            "     🔗 Dependencies: task_2",
            "",
            "<<END_TASK_PLAN>>",
        ]

        # Process messages
        for message in messages:
            cli.add_message(message)

        # Verify buffering worked correctly
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []

    def test_regular_message_processing(self, cli):
        """Test regular message processing (not task plan overview)."""
        # Test regular messages
        regular_messages = [
            "Hello, this is a regular message",
            "Another regular message",
            "Coordinator: Starting task execution...",
            "Generator: Processing request...",
        ]

        for message in regular_messages:
            cli.add_message(message)

        # Verify regular processing
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []
        assert len(cli.messages) > 0

    def test_json_message_detection(self, cli):
        """Test JSON message detection and processing."""
        json_messages = [
            '{"test": "json", "data": "value"}',
            '{"agent": "generator", "result": {"code": "test"}}',
            '{"status": "success", "output": {"message": "test"}}',
        ]

        for message in json_messages:
            cli.add_message(message)

        # Verify JSON messages are processed
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []

    def test_task_update_processing(self, cli):
        """Test task update processing."""
        task = create_test_task(
            task_id="test-1",
            description="Test task",
            agent_type=AgentType.GENERATOR,
            priority=TaskPriority.CRITICAL,
        )

        cli.print_task_update(task)

        # Verify task update was processed
        assert not cli.is_collecting_task_plan_overview

    def test_task_plan_processing(self, cli):
        """Test task plan processing."""
        tasks = [
            create_test_task(
                task_id="task-1",
                description="Generate code",
                agent_type=AgentType.GENERATOR,
                priority=TaskPriority.CRITICAL,
            ),
            create_test_task(
                task_id="task-2",
                description="Debug code",
                agent_type=AgentType.DEBUGGER,
                priority=TaskPriority.HIGH,
            ),
        ]

        cli.print_task_plan(tasks, "test-plan-123")

        # Verify task plan was processed
        assert cli.last_task_plan == (tasks, "test-plan-123")

    def test_coordinator_message_processing(self, cli):
        """Test coordinator message processing."""
        coordinator_messages = [
            "Coordinator: Starting session...",
            "Coordinator: Task plan created successfully",
            "Coordinator: Execution completed",
        ]

        for message in coordinator_messages:
            cli.add_coordinator_message(message)

        # Verify coordinator messages are processed
        assert not cli.is_collecting_task_plan_overview

    def test_agent_message_processing(self, cli):
        """Test agent message processing."""
        agent_messages = [
            "Generator: Analyzing request...",
            "Debugger: Checking code for issues...",
            "Optimizer: Applying optimizations...",
        ]

        for message in agent_messages:
            cli.add_agent_message(message, "Generator")

        # Verify agent messages are processed
        assert not cli.is_collecting_task_plan_overview

    def test_user_prompt_processing(self, cli):
        """Test user prompt processing."""
        user_prompts = [
            "Write a matrix multiplication operator",
            "Generate a convolution operator",
            "Create a pooling operator",
        ]

        for prompt in user_prompts:
            cli.add_user_prompt(prompt)

        # Verify user prompts are processed
        assert not cli.is_collecting_task_plan_overview

    def test_buffering_state_management(self, cli):
        """Test buffering state management."""
        # Start buffering
        cli.add_message("📋 Task Plan Overview:")
        assert cli.is_collecting_task_plan_overview
        assert len(cli.task_plan_overview_buffer) == 1

        # Add more messages
        cli.add_message("  Task 1: Generate code")
        cli.add_message("  Task 2: Debug code")
        assert cli.is_collecting_task_plan_overview
        assert len(cli.task_plan_overview_buffer) == 3

        # End buffering
        cli.add_message("<<END_TASK_PLAN>>")
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []

    def test_buffering_with_session_prefix(self, cli):
        """Test buffering with session ID prefix."""
        # Start buffering with session prefix
        cli.add_message("[session_abc123] 📋 Task Plan Overview:")
        assert cli.is_collecting_task_plan_overview
        assert len(cli.task_plan_overview_buffer) == 1

        # Add more messages
        cli.add_message("  Task 1: Generate code")
        cli.add_message("  Task 2: Debug code")
        assert cli.is_collecting_task_plan_overview
        assert len(cli.task_plan_overview_buffer) == 3

        # End buffering
        cli.add_message("<<END_TASK_PLAN>>")
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    @pytest.fixture
    def cli(self):
        """Create a new CLI instance for testing."""
        return PinocchioCLI()

    def test_complete_workflow_simulation(self, cli):
        """Test a complete workflow simulation."""
        # Simulate a complete workflow with mixed message types
        workflow_messages = [
            # User prompt
            "User: Write a matrix multiplication operator",
            # Coordinator messages
            "[session_abc123] Session started",
            "[session_abc123] 🤖 Creating intelligent task plan...",
            "[session_abc123] ✅ Task plan created: 3 tasks",
            # Task plan overview
            "[session_abc123] 📋 Task Plan Overview:",
            "  1. ⚡ GENERATOR (🔴 critical)",
            "     📝 [Round 1] Write a matrix multiplication operator",
            "     💡 Instruction: Generate high-performance Choreo DSL operator code.",
            "     🔗 Dependencies: ",
            "",
            "  2. 🔧 DEBUGGER (🔴 critical)",
            "     📝 [Round 1] Compile and debug generated code",
            "     💡 Instruction: Analyze the generated code for potential issues.",
            "     🔗 Dependencies: task_1",
            "",
            "  3. 🚀 OPTIMIZER (🟡 high)",
            "     📝 [Round 1] Optimise code for: performance and efficiency",
            "     💡 Instruction: Analyze and optimize the generated Choreo DSL code.",
            "     🔗 Dependencies: task_2",
            "",
            "<<END_TASK_PLAN>>",
            # Agent messages
            "Generator: Analyzing request...",
            "Generator: Generating code...",
            "Debugger: Checking code for issues...",
            "Optimizer: Applying optimizations...",
            # Task updates
            "Task update: Generator completed",
            "Task update: Debugger completed",
            "Task update: Optimizer completed",
            # Final result
            "[session_abc123] Session completed successfully",
        ]

        # Process all messages
        for message in workflow_messages:
            cli.add_message(message)

        # Verify final state
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []
        assert len(cli.messages) > 0

    def test_error_handling_in_buffering(self, cli):
        """Test error handling during buffering."""
        # Start buffering
        cli.add_message("📋 Task Plan Overview:")
        assert cli.is_collecting_task_plan_overview

        # Add some messages
        cli.add_message("  Task 1: Generate code")
        cli.add_message("  Task 2: Debug code")

        # Simulate an error (no END marker)
        # The buffering should continue until END marker is received
        assert cli.is_collecting_task_plan_overview
        assert len(cli.task_plan_overview_buffer) == 3

        # Force flush (simulate end of processing)
        cli._flush_task_plan_overview()
        assert not cli.is_collecting_task_plan_overview
        assert cli.task_plan_overview_buffer == []


if __name__ == "__main__":
    pytest.main([__file__])
