"""Unit tests for agents."""

from unittest.mock import AsyncMock, Mock

import pytest

from pinocchio.agents.base import Agent, AgentWithRetry
from pinocchio.agents.debugger import DebuggerAgent
from pinocchio.agents.evaluator import EvaluatorAgent
from pinocchio.agents.generator import GeneratorAgent
from pinocchio.agents.optimizer import OptimizerAgent
from pinocchio.llm.custom_llm_client import CustomLLMClient
from pinocchio.llm.mock_client import MockLLMClient
from tests.utils import (
    assert_task_plan_valid,
    assert_task_valid,
    create_mock_llm_client,
    create_test_task,
    create_test_task_plan,
)


class TestGeneratorAgent:
    """Test GeneratorAgent functionality."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return MockLLMClient(response_delay_ms=1, failure_rate=0.0)

    @pytest.fixture
    def generator_agent(self, mock_llm_client):
        """Create generator agent."""
        return GeneratorAgent(mock_llm_client)

    def test_generator_initialization(self, generator_agent):
        """Test generator agent initialization."""
        assert generator_agent.agent_type == "generator"
        assert generator_agent.llm_client is not None
        assert generator_agent.call_count == 0
        assert generator_agent.max_retries == 3

    @pytest.mark.asyncio
    async def test_generator_execute_success(self, generator_agent):
        """Test successful code generation."""
        request = {
            "request_id": "test_request",
            "task_description": "编写一个conv2d算子",
            "requirements": {"performance": "high"},
            "optimization_goals": ["maximize_throughput"],
        }

        result = await generator_agent.execute(request)

        assert result.success == True
        assert result.agent_type == "generator"
        assert result.request_id == "test_request"
        assert "code" in result.output
        assert "explanation" in result.output
        assert "optimization_techniques" in result.output
        assert result.processing_time_ms is not None

    @pytest.mark.asyncio
    async def test_generator_execute_with_llm_failure(self):
        """Test generator with LLM failure."""
        # Create generator with failing LLM client
        failing_client = MockLLMClient(response_delay_ms=1, failure_rate=1.0)
        generator = GeneratorAgent(failing_client, max_retries=2)

        request = {"request_id": "test_request", "task_description": "编写一个算子"}

        result = await generator.execute(request)

        assert result.success == False
        assert result.error_message is not None
        assert "LLM call failed" in result.error_message

    def test_build_generation_prompt(self, generator_agent):
        """Test prompt building for generation."""
        request = {
            "task_description": "编写conv2d算子",
            "requirements": {"performance": "high", "memory_efficient": True},
            "optimization_goals": ["maximize_throughput", "minimize_memory_usage"],
            "context": {"previous_step": "planning"},
        }

        prompt = generator_agent._build_generation_prompt(request)

        assert "编写conv2d算子" in prompt
        assert "performance: high" in prompt
        assert "memory_efficient: True" in prompt
        assert "maximize_throughput" in prompt
        assert "minimize_memory_usage" in prompt
        assert "Context:" in prompt
        assert "Instructions for Code Generation" in prompt

    def test_process_generation_response(self, generator_agent):
        """Test response processing."""
        llm_response = {
            "success": True,
            "output": {
                "code": "func test() { }",
                "language": "choreo_dsl",
                "explanation": "测试代码",
                "optimization_techniques": ["basic_loops"],
            },
        }

        request = {
            "request_id": "test",
            "task_description": "测试",
            "timestamp": "2024-01-01T00:00:00",
        }

        output = generator_agent._process_generation_response(llm_response, request)

        assert output["code"] == "func test() { }"
        assert output["language"] == "choreo_dsl"
        assert output["explanation"] == "测试代码"
        assert output["optimization_techniques"] == ["basic_loops"]
        assert "generation_metadata" in output
        assert output["generation_metadata"]["agent_type"] == "generator"

    def test_generate_simple_code(self, generator_agent):
        """Test simple code generation."""
        test_cases = [
            ("conv2d算子", "convolution"),
            ("矩阵乘法", "matrix_multiplication"),
            ("加法算子", "addition"),
        ]

        for description, expected_op in test_cases:
            result = generator_agent.generate_simple_code(description)

            assert "code" in result
            assert "language" in result
            assert result["language"] == "choreo_dsl"
            assert expected_op in result["code"]
            assert "func" in result["code"]


class TestBaseAgent:
    """Test base Agent functionality."""

    class TestAgent(Agent):
        """Test implementation of Agent."""

        def __init__(self, agent_type: str, llm_client):
            super().__init__(agent_type, llm_client)

        async def execute(self, request):
            return self._create_response(
                request_id=request.get("request_id", "test"),
                success=True,
                output={"test": "output"},
            )

        def _get_agent_instructions(self):
            return "Test agent instructions"

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.complete = AsyncMock(return_value='{"test": "response"}')
        return client

    @pytest.fixture
    def test_agent(self, mock_llm_client):
        """Create test agent."""
        return self.TestAgent("test", mock_llm_client)

    def test_agent_initialization(self, test_agent):
        """Test agent initialization."""
        assert test_agent.agent_type == "test"
        assert test_agent.call_count == 0
        assert test_agent.total_processing_time == 0.0

    @pytest.mark.asyncio
    async def test_llm_call(self, test_agent):
        """Test LLM call functionality."""
        result = await test_agent._call_llm("test prompt")

        assert isinstance(result, dict)
        assert test_agent.call_count == 1
        assert test_agent.total_processing_time > 0

    def test_build_prompt(self, test_agent):
        """Test prompt building."""
        request = {"task_description": "test task", "context": {"key": "value"}}

        prompt = test_agent._build_prompt(request)

        assert "test task" in prompt
        assert "test agent" in prompt
        assert "key" in prompt
        assert "value" in prompt
        assert "Test agent instructions" in prompt

    def test_create_response(self, test_agent):
        """Test response creation."""
        response = test_agent._create_response(
            request_id="test_id",
            success=True,
            output={"result": "success"},
            processing_time_ms=100,
        )

        assert response.agent_type == "test"
        assert response.success == True
        assert response.request_id == "test_id"
        assert response.output["result"] == "success"
        assert response.processing_time_ms == 100

    def test_handle_error(self, test_agent):
        """Test error handling."""
        error = Exception("Test error")
        response = test_agent._handle_error("test_id", error)

        assert response.success == False
        assert response.error_message == "Test error"
        assert response.request_id == "test_id"

    def test_stats_functionality(self, test_agent):
        """Test statistics functionality."""
        # Initial stats
        stats = test_agent.get_stats()
        assert stats["call_count"] == 0
        assert stats["average_processing_time_ms"] == 0.0

        # Simulate some processing
        test_agent.call_count = 3
        test_agent.total_processing_time = 300.0

        stats = test_agent.get_stats()
        assert stats["call_count"] == 3
        assert stats["average_processing_time_ms"] == 100.0

        # Reset stats
        test_agent.reset_stats()
        assert test_agent.call_count == 0
        assert test_agent.total_processing_time == 0.0


class TestAgentWithRetry:
    """Test AgentWithRetry functionality."""

    class TestRetryAgent(AgentWithRetry):
        """Test implementation of AgentWithRetry."""

        def __init__(
            self,
            agent_type: str,
            llm_client,
            max_retries: int = 3,
            retry_delay: float = 1.0,
        ):
            super().__init__(agent_type, llm_client, max_retries, retry_delay)

        async def execute(self, request):
            return self._create_response(
                request_id=request.get("request_id", "test"),
                success=True,
                output={"test": "output"},
            )

        def _get_agent_instructions(self):
            return "Test retry agent instructions"

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client that fails sometimes."""
        client = Mock()
        client.complete = AsyncMock(
            side_effect=[
                Exception("First failure"),
                Exception("Second failure"),
                '{"success": true, "output": "final success"}',
            ]
        )
        return client

    @pytest.fixture
    def retry_agent(self, mock_llm_client):
        """Create retry agent."""
        return self.TestRetryAgent(
            "retry_test", mock_llm_client, max_retries=3, retry_delay=0.001
        )

    @pytest.mark.asyncio
    async def test_retry_mechanism_success(self, retry_agent):
        """Test retry mechanism with eventual success."""
        result = await retry_agent._call_llm_with_retry("test prompt")

        # Should succeed after retries
        assert result is not None
        assert retry_agent.llm_client.complete.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_mechanism_final_failure(self):
        """Test retry mechanism with final failure."""
        # Create client that always fails
        failing_client = Mock()
        failing_client.complete = AsyncMock(side_effect=Exception("Always fails"))

        retry_agent = self.TestRetryAgent(
            "retry_test", failing_client, max_retries=2, retry_delay=0.001
        )

        with pytest.raises(Exception) as exc_info:
            await retry_agent._call_llm_with_retry("test prompt")

        assert "LLM call failed after 3 attempts" in str(exc_info.value)
        assert failing_client.complete.call_count == 3


def test_generator_agent_auto_llm():
    agent = GeneratorAgent()
    assert isinstance(agent.llm_client, CustomLLMClient)
    # Check config fields (model_name, base_url, etc.)
    assert hasattr(agent.llm_client, "model_name")
    assert hasattr(agent.llm_client, "base_url")


def test_optimizer_agent_auto_llm():
    agent = OptimizerAgent()
    assert isinstance(agent.llm_client, CustomLLMClient)
    assert hasattr(agent.llm_client, "model_name")
    assert hasattr(agent.llm_client, "base_url")


def test_debugger_agent_auto_llm():
    agent = DebuggerAgent()
    assert isinstance(agent.llm_client, CustomLLMClient)
    assert hasattr(agent.llm_client, "model_name")
    assert hasattr(agent.llm_client, "base_url")


def test_evaluator_agent_auto_llm():
    agent = EvaluatorAgent()
    assert isinstance(agent.llm_client, CustomLLMClient)
    assert hasattr(agent.llm_client, "model_name")
    assert hasattr(agent.llm_client, "base_url")
