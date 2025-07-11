"""Unit tests for Coordinator."""

import shutil
import tempfile
from unittest.mock import AsyncMock, Mock

import pytest

from pinocchio.coordinator import Coordinator
from pinocchio.llm.mock_client import MockLLMClient


class TestCoordinator:
    """Test Coordinator functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return MockLLMClient(response_delay_ms=1, failure_rate=0.0)

    @pytest.fixture
    def coordinator(self, mock_llm_client, temp_dir):
        """Create coordinator with mocks."""
        return Coordinator(llm_client=mock_llm_client, sessions_dir=temp_dir)

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.llm_client is not None
        assert coordinator.generator_agent is not None
        assert coordinator.total_sessions == 0
        assert coordinator.successful_sessions == 0
        assert coordinator.current_session is None

    def test_requirement_extraction(self, coordinator):
        """Test requirement extraction from prompts."""
        test_cases = [
            ("conv2d算子", {"operation_type": "convolution"}),
            ("矩阵乘法", {"operation_type": "matrix_multiplication"}),
            ("快速加法", {"operation_type": "element_wise_addition"}),
            ("高性能conv", {"operation_type": "convolution", "performance": "high"}),
            ("内存高效tensor", {"memory_efficient": True, "data_type": "tensor"}),
        ]

        for prompt, expected in test_cases:
            requirements = coordinator._extract_requirements(prompt)

            for key, value in expected.items():
                assert requirements[key] == value

    def test_optimization_goal_extraction(self, coordinator):
        """Test optimization goal extraction."""
        test_cases = [
            ("快速算子", ["maximize_throughput"]),
            ("内存优化", ["minimize_memory_usage"]),
            ("并行处理", ["enable_parallelization"]),
            ("cache优化", ["optimize_cache_locality"]),
            ("普通算子", ["maximize_throughput", "optimize_cache_locality"]),  # default
        ]

        for prompt, expected_goals in test_cases:
            goals = coordinator._extract_optimization_goals(prompt)

            for goal in expected_goals:
                assert goal in goals

    def test_simple_plan_generation(self, coordinator):
        """Test simple plan generation."""
        user_prompt = "编写一个conv2d算子"
        plan = coordinator._generate_simple_plan(user_prompt)

        assert len(plan) == 1
        assert plan[0]["agent_type"] == "generator"
        assert plan[0]["task_description"] == user_prompt
        assert "requirements" in plan[0]
        assert "optimization_goals" in plan[0]

    @pytest.mark.asyncio
    async def test_agent_step_execution(self, coordinator):
        """Test agent step execution."""
        step_id = "test_step"
        agent_type = "generator"
        task_description = "测试任务"

        # Create a session first
        coordinator.current_session = Mock()
        coordinator.current_session.session_id = "test_session"
        coordinator.current_session.get_context.return_value = {"test": "context"}
        coordinator.current_session.log_communication = Mock()

        # Execute step
        result = await coordinator._execute_agent_step(
            step_id, agent_type, task_description
        )

        # Verify result
        assert result is not None
        assert hasattr(result, "success")
        assert hasattr(result, "output")

        # Verify communication was logged
        coordinator.current_session.log_communication.assert_called_once()

    def test_stats_tracking(self, coordinator):
        """Test statistics tracking."""
        initial_stats = coordinator.get_stats()

        assert initial_stats["total_sessions"] == 0
        assert initial_stats["successful_sessions"] == 0
        assert initial_stats["success_rate"] == 0.0
        assert initial_stats["current_session_id"] is None
        assert "generator" in initial_stats["agent_stats"]

    def test_stats_reset(self, coordinator):
        """Test statistics reset."""
        # Manually set some stats
        coordinator.total_sessions = 5
        coordinator.successful_sessions = 3

        coordinator.reset_stats()

        assert coordinator.total_sessions == 0
        assert coordinator.successful_sessions == 0

    def test_session_history_empty(self, coordinator):
        """Test session history when no sessions exist."""
        history = coordinator.get_session_history()
        assert isinstance(history, list)
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_load_nonexistent_session(self, coordinator):
        """Test loading a session that doesn't exist."""
        result = await coordinator.load_session("nonexistent_session")
        assert result is None
