"""End-to-end tests for MVP workflow."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from pinocchio.coordinator import Coordinator, process_simple_request
from pinocchio.llm.mock_client import MockLLMClient
from pinocchio.session_logger import SessionLogger


class TestMVPWorkflow:
    """Test MVP end-to-end workflow."""

    @pytest.fixture
    def temp_sessions_dir(self):
        """Create temporary sessions directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def coordinator(self, temp_sessions_dir):
        """Create coordinator with temporary directory."""
        mock_client = MockLLMClient(response_delay_ms=10, failure_rate=0.0)
        return Coordinator(llm_client=mock_client, sessions_dir=temp_sessions_dir)

    @pytest.mark.asyncio
    async def test_simple_code_generation(self, coordinator):
        """Test simple code generation workflow."""
        user_prompt = "编写一个conv2d算子"

        messages = []
        async for message in coordinator.process_user_request(user_prompt):
            messages.append(message)

        # Verify workflow completed
        assert len(messages) > 0

        # Check for expected messages
        message_text = "\n".join(messages)
        assert "Session started" in message_text
        assert "Task plan created" in message_text
        assert "Executing task task_1: generator" in message_text
        assert "Task task_1 completed successfully" in message_text
        assert "Code generation completed" in message_text
        assert "Session completed successfully" in message_text

        # Verify code was generated
        assert "```choreo" in message_text
        assert "func" in message_text

        # Check session was saved
        sessions = coordinator.get_session_history()
        assert len(sessions) == 1
        assert sessions[0]["user_prompt"] == user_prompt
        assert sessions[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_matrix_multiplication_generation(self, coordinator):
        """Test matrix multiplication code generation."""
        user_prompt = "生成一个高性能的矩阵乘法算子"

        messages = []
        async for message in coordinator.process_user_request(user_prompt):
            messages.append(message)

        message_text = "\n".join(messages)
        print(f"Generated messages: {message_text}")

        # Verify successful completion
        assert (
            "Session completed successfully" in message_text
            or "Code generation completed" in message_text
        )

        # Verify code content
        assert "```choreo" in message_text
        # Check for matrix-related content in the generated code
        # The generated code should contain matrix-related keywords
        assert any(
            keyword in message_text.lower()
            for keyword in ["matrix", "matmul", "matmul_kernel", "conv_kernel"]
        )

    @pytest.mark.asyncio
    async def test_workflow_with_performance_requirements(self, coordinator):
        """Test workflow with specific performance requirements."""
        user_prompt = "创建一个快速且内存高效的元素级加法算子"

        messages = []
        async for message in coordinator.process_user_request(user_prompt):
            messages.append(message)

        message_text = "\n".join(messages)

        # Verify successful completion
        assert (
            "Session completed successfully" in message_text
            or "Code generation completed" in message_text
        )

        # Check that optimization information is displayed
        assert (
            "Optimizations applied" in message_text
            or "optimization" in message_text.lower()
        )

    @pytest.mark.asyncio
    async def test_session_persistence(self, coordinator, temp_sessions_dir):
        """Test that sessions are properly saved and can be loaded."""
        user_prompt = "编写一个简单的算子"

        # Process request
        messages = []
        async for message in coordinator.process_user_request(user_prompt):
            messages.append(message)

        # Check session file was created
        sessions_dir = Path(temp_sessions_dir)
        session_files = list(sessions_dir.glob("session_*.json"))
        assert len(session_files) == 1

        # Load session and verify data
        session_file = session_files[0]
        loaded_session = SessionLogger.load_from_file(str(session_file))

        assert loaded_session.user_prompt == user_prompt
        assert loaded_session.completed_at is not None
        assert len(loaded_session.summary_logs) > 0
        assert len(loaded_session.communication_logs) > 0
        assert loaded_session.metadata.get("status") == "completed"

    @pytest.mark.asyncio
    async def test_coordinator_statistics(self, coordinator):
        """Test coordinator statistics tracking."""
        initial_stats = coordinator.get_stats()
        assert initial_stats["total_sessions"] == 0
        assert initial_stats["successful_sessions"] == 0
        assert initial_stats["success_rate"] == 0.0

        # Process successful request
        messages = []
        async for message in coordinator.process_user_request("编写一个算子"):
            messages.append(message)

        # Check updated stats
        stats = coordinator.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["successful_sessions"] == 1
        assert stats["success_rate"] == 1.0
        # Note: current_session_id and agent_stats are no longer available in the new implementation
        pass

    @pytest.mark.asyncio
    async def test_error_handling_with_llm_failure(self, temp_sessions_dir):
        """Test error handling when LLM fails."""
        # Create coordinator with high failure rate
        mock_client = MockLLMClient(response_delay_ms=10, failure_rate=1.0)
        coordinator = Coordinator(
            llm_client=mock_client, sessions_dir=temp_sessions_dir
        )

        messages = []
        async for message in coordinator.process_user_request("编写一个算子"):
            messages.append(message)

        message_text = "\n".join(messages)

        # Should handle failure gracefully
        assert "Session started" in message_text
        assert "failed" in message_text.lower() or "error" in message_text.lower()

        # Session should be marked as failed
        stats = coordinator.get_stats()
        assert stats["total_sessions"] == 1
        assert stats["successful_sessions"] == 0

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, coordinator):
        """Test multiple sessions handling."""
        prompts = ["编写一个conv2d算子", "生成矩阵乘法算子", "创建加法算子"]

        for prompt in prompts:
            messages = []
            async for message in coordinator.process_user_request(prompt):
                messages.append(message)

        # Check all sessions were processed
        stats = coordinator.get_stats()
        assert stats["total_sessions"] == len(prompts)
        assert stats["successful_sessions"] == len(prompts)

        # Check session history
        sessions = coordinator.get_session_history()
        assert len(sessions) == len(prompts)

        # Verify each session has unique ID
        session_ids = [s["session_id"] for s in sessions]
        assert len(set(session_ids)) == len(prompts)

    def test_requirement_extraction(self, coordinator):
        """Test requirement extraction from user prompts."""
        # This test is no longer applicable as requirements are now extracted
        # by the task planner instead of the coordinator
        pass

    def test_optimization_goal_extraction(self, coordinator):
        """Test optimization goal extraction from user prompts."""
        # This test is no longer applicable as optimization goals are now extracted
        # by the task planner instead of the coordinator
        pass


class TestConvenienceFunction:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_process_simple_request(self):
        """Test simple request processing function."""
        result = await process_simple_request("编写一个简单的算子")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Session started" in result
        assert "Session completed" in result
        assert "Code generation completed" in result


class TestSessionLogger:
    """Test SessionLogger functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_session_creation(self, temp_dir):
        """Test session creation and basic functionality."""
        user_prompt = "测试会话"
        session = SessionLogger(user_prompt, temp_dir)

        assert session.session_id.startswith("session_")
        assert session.user_prompt == user_prompt
        assert session.created_at is not None
        assert session.completed_at is None
        assert len(session.summary_logs) == 0
        assert len(session.communication_logs) == 0

    def test_session_logging(self, temp_dir):
        """Test session logging functionality."""
        session = SessionLogger("测试", temp_dir)

        # Test summary logging
        message = session.log_summary("测试消息")
        assert session.session_id in message
        assert "测试消息" in message
        assert len(session.summary_logs) == 1

        # Test communication logging
        session.log_communication(
            step_id="step_1",
            agent_type="generator",
            request={"test": "request"},
            response={"test": "response"},
        )
        assert len(session.communication_logs) == 1

        comm_log = session.communication_logs[0]
        assert comm_log["step_id"] == "step_1"
        assert comm_log["agent_type"] == "generator"
        assert comm_log["request"]["test"] == "request"
        assert comm_log["response"]["test"] == "response"

    def test_session_persistence(self, temp_dir):
        """Test session save and load functionality."""
        user_prompt = "持久化测试"
        session = SessionLogger(user_prompt, temp_dir)

        # Add some data
        session.log_summary("测试消息1")
        session.log_summary("测试消息2")
        session.log_communication("step_1", "generator", {"req": 1}, {"resp": 1})
        session.update_context("key1", "value1")
        session.add_metadata("meta1", "meta_value1")
        session.complete_session("completed")

        # Save session
        file_path = session.save_to_file()
        assert file_path.exists()

        # Load session
        loaded_session = SessionLogger.load_from_file(str(file_path))

        # Verify data
        assert loaded_session.session_id == session.session_id
        assert loaded_session.user_prompt == user_prompt
        assert len(loaded_session.summary_logs) == 2
        assert len(loaded_session.communication_logs) == 1
        assert loaded_session.context["key1"] == "value1"
        assert loaded_session.metadata["meta1"] == "meta_value1"
        assert loaded_session.metadata["status"] == "completed"
        assert loaded_session.completed_at is not None

    def test_session_listing(self, temp_dir):
        """Test session listing functionality."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = SessionLogger(f"测试{i}", temp_dir)
            session.complete_session("completed")
            session.save_to_file()
            sessions.append(session)

        # List sessions
        session_list = SessionLogger.list_sessions(temp_dir)

        assert len(session_list) == 3

        # Verify session data
        for i, session_info in enumerate(session_list):
            assert "session_id" in session_info
            assert "user_prompt" in session_info
            assert "created_at" in session_info
            assert "status" in session_info
            assert session_info["status"] == "completed"
