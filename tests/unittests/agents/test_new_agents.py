"""Tests for the new agent implementations (Optimizer, Debugger, Evaluator)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pinocchio.agents import (
    DebuggerAgent,
    EvaluatorAgent,
    GeneratorAgent,
    OptimizerAgent,
)
from tests.utils import (
    assert_task_plan_valid,
    assert_task_valid,
    create_mock_llm_client,
    create_test_task,
    create_test_task_plan,
)


class TestOptimizerAgent:
    """Test cases for OptimizerAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock(
            return_value='{"success": true, "output": {"optimization_suggestions": []}}'
        )
        return client

    @pytest.fixture
    def optimizer_agent(self, mock_llm_client):
        """Create an OptimizerAgent instance."""
        return OptimizerAgent(mock_llm_client)

    def test_optimizer_initialization(self, optimizer_agent):
        """Test OptimizerAgent initialization."""
        assert optimizer_agent.agent_type == "optimizer"
        assert optimizer_agent.call_count == 0

    @pytest.mark.asyncio
    async def test_optimizer_execute_success(self, optimizer_agent):
        """Test successful optimization execution."""
        request = {
            "code": "func test() { for i in range(10) { compute(i); } }",
            "optimization_goals": ["performance", "memory_efficiency"],
            "request_id": "test_123",
        }

        response = await optimizer_agent.execute(request)

        assert response.success is True
        assert response.agent_type == "optimizer"
        assert "optimization_suggestions" in response.output

    def test_optimizer_analyze_code_performance(self, optimizer_agent):
        """Test synchronous code performance analysis."""
        code = "func test() { for i in range(10) { compute(i); } }"
        goals = ["performance", "memory_efficiency"]

        result = optimizer_agent.analyze_code_performance(code, goals)

        assert isinstance(result, dict)
        assert "optimization_suggestions" in result

    def test_optimizer_get_optimization_suggestions(self, optimizer_agent):
        """Test getting optimization suggestions."""
        code = "func test() { for i in range(10) { compute(i); } }"

        suggestions = optimizer_agent.get_optimization_suggestions(code)

        assert isinstance(suggestions, list)

    def test_optimizer_get_optimized_code(self, optimizer_agent):
        """Test getting optimized code."""
        code = "func test() { for i in range(10) { compute(i); } }"

        optimized_code = optimizer_agent.get_optimized_code(code)

        assert isinstance(optimized_code, str)
        assert len(optimized_code) > 0


class TestDebuggerAgent:
    """Test cases for DebuggerAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock(
            return_value='{"success": true, "output": {"issues_found": []}}'
        )
        return client

    @pytest.fixture
    def debugger_agent(self, mock_llm_client):
        """Create a DebuggerAgent instance."""
        return DebuggerAgent(mock_llm_client)

    def test_debugger_initialization(self, debugger_agent):
        """Test DebuggerAgent initialization."""
        assert debugger_agent.agent_type == "debugger"
        assert debugger_agent.call_count == 0

    @pytest.mark.asyncio
    async def test_debugger_execute_success(self, debugger_agent):
        """Test successful debugging execution."""
        request = {
            "code": "func test() { for i in range(10) { compute(i); } }",
            "error_message": "Index out of bounds",
            "request_id": "test_123",
        }

        response = await debugger_agent.execute(request)

        assert response.success is True
        assert response.agent_type == "debugger"
        assert "issues_found" in response.output

    def test_debugger_analyze_code_issues(self, debugger_agent):
        """Test synchronous code issue analysis."""
        code = "func test() { for i in range(10) { compute(i); } }"
        error_message = "Index out of bounds"

        result = debugger_agent.analyze_code_issues(code, error_message)

        assert isinstance(result, dict)
        assert "issues_found" in result

    def test_debugger_get_code_issues(self, debugger_agent):
        """Test getting code issues."""
        code = "func test() { for i in range(10) { compute(i); } }"

        issues = debugger_agent.get_code_issues(code)

        assert isinstance(issues, list)

    def test_debugger_get_debugged_code(self, debugger_agent):
        """Test getting debugged code."""
        code = "func test() { for i in range(10) { compute(i); } }"
        error_message = "Index out of bounds"

        debugged_code = debugger_agent.get_debugged_code(code, error_message)

        assert isinstance(debugged_code, str)
        assert len(debugged_code) > 0

    def test_debugger_get_code_health_score(self, debugger_agent):
        """Test getting code health score."""
        code = "func test() { for i in range(10) { compute(i); } }"

        health_score = debugger_agent.get_code_health_score(code)

        assert isinstance(health_score, dict)
        assert "overall_health" in health_score


class TestEvaluatorAgent:
    """Test cases for EvaluatorAgent."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = AsyncMock()
        client.complete = AsyncMock(
            return_value='{"success": true, "output": {"performance_analysis": {}}}'
        )
        return client

    @pytest.fixture
    def evaluator_agent(self, mock_llm_client):
        """Create an EvaluatorAgent instance."""
        return EvaluatorAgent(mock_llm_client)

    def test_evaluator_initialization(self, evaluator_agent):
        """Test EvaluatorAgent initialization."""
        assert evaluator_agent.agent_type == "evaluator"
        assert evaluator_agent.call_count == 0

    @pytest.mark.asyncio
    async def test_evaluator_execute_success(self, evaluator_agent):
        """Test successful evaluation execution."""
        request = {
            "code": "func test() { for i in range(10) { compute(i); } }",
            "performance_metrics": {"execution_time": "1ms"},
            "request_id": "test_123",
        }

        response = await evaluator_agent.execute(request)

        assert response.success is True
        assert response.agent_type == "evaluator"
        assert "performance_analysis" in response.output

    def test_evaluator_evaluate_performance(self, evaluator_agent):
        """Test synchronous performance evaluation."""
        code = "func test() { for i in range(10) { compute(i); } }"
        metrics = {"execution_time": "1ms", "memory_usage": "1MB"}

        result = evaluator_agent.evaluate_performance(code, metrics)

        assert isinstance(result, dict)
        assert "performance_analysis" in result

    def test_evaluator_get_performance_analysis(self, evaluator_agent):
        """Test getting performance analysis."""
        code = "func test() { for i in range(10) { compute(i); } }"

        analysis = evaluator_agent.get_performance_analysis(code)

        assert isinstance(analysis, dict)

    def test_evaluator_get_optimization_recommendations(self, evaluator_agent):
        """Test getting optimization recommendations."""
        code = "func test() { for i in range(10) { compute(i); } }"

        recommendations = evaluator_agent.get_optimization_recommendations(code)

        assert isinstance(recommendations, list)

    def test_evaluator_get_performance_score(self, evaluator_agent):
        """Test getting performance score."""
        code = "func test() { for i in range(10) { compute(i); } }"

        score = evaluator_agent.get_performance_score(code)

        assert isinstance(score, dict)
        assert "performance_metrics" in score
        assert "bottlenecks" in score

    def test_evaluator_compare_with_baseline(self, evaluator_agent):
        """Test comparing with baseline metrics."""
        code = "func test() { for i in range(10) { compute(i); } }"
        baseline = {"execution_time": "2ms", "memory_usage": "2MB"}

        comparison = evaluator_agent.compare_with_baseline(code, baseline)

        assert isinstance(comparison, dict)


class TestAgentIntegration:
    """Integration tests for agent interactions."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client for integration tests."""
        client = AsyncMock()
        client.complete = AsyncMock(
            return_value='{"success": true, "output": {"test": "data"}}'
        )
        return client

    def test_agent_workflow_integration(self, mock_llm_client):
        """Test basic workflow integration between agents."""
        # Create all agents
        generator = GeneratorAgent(mock_llm_client)
        optimizer = OptimizerAgent(mock_llm_client)
        debugger = DebuggerAgent(mock_llm_client)
        evaluator = EvaluatorAgent(mock_llm_client)

        # Test that all agents can be created and have correct types
        assert generator.agent_type == "generator"
        assert optimizer.agent_type == "optimizer"
        assert debugger.agent_type == "debugger"
        assert evaluator.agent_type == "evaluator"

        # Test that all agents have the required methods
        assert hasattr(generator, "execute")
        assert hasattr(optimizer, "execute")
        assert hasattr(debugger, "execute")
        assert hasattr(evaluator, "execute")

    def test_agent_error_handling(self, mock_llm_client):
        """Test error handling in agents."""
        # Configure mock to raise an exception
        mock_llm_client.complete.side_effect = Exception("LLM call failed")

        optimizer = OptimizerAgent(mock_llm_client)
        debugger = DebuggerAgent(mock_llm_client)
        evaluator = EvaluatorAgent(mock_llm_client)

        # Test error handling for each agent
        for agent in [optimizer, debugger, evaluator]:
            result = (
                agent.analyze_code_performance("test code")
                if hasattr(agent, "analyze_code_performance")
                else agent.analyze_code_issues("test code")
                if hasattr(agent, "analyze_code_issues")
                else agent.evaluate_performance("test code")
            )

            # Should return empty dict or fallback structure on error
            assert isinstance(result, dict)
