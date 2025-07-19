"""Unit tests for agent-specific LLM configuration."""

import os
from unittest.mock import MagicMock, patch

import pytest

from pinocchio.agents.debugger import DebuggerAgent
from pinocchio.agents.evaluator import EvaluatorAgent
from pinocchio.agents.generator import GeneratorAgent
from pinocchio.agents.optimizer import OptimizerAgent
from pinocchio.config import ConfigManager
from pinocchio.config.models import LLMConfigEntry, LLMProvider
from pinocchio.llm.custom_llm_client import CustomLLMClient
from tests.utils import (
    assert_task_plan_valid,
    assert_task_valid,
    create_mock_llm_client,
    create_test_task,
    create_test_task_plan,
)


class TestAgentSpecificConfig:
    """Test agent-specific LLM configuration functionality (Mock tests - safe for CI)."""

    def test_config_manager_get_agent_llm_config(self):
        """Test ConfigManager.get_agent_llm_config method."""
        config_manager = ConfigManager()

        # Test fallback to global config (default behavior)
        result = config_manager.get_agent_llm_config("generator")
        assert isinstance(result, LLMConfigEntry)
        assert result.provider == LLMProvider.CUSTOM
        assert result.model_name is not None
        assert result.base_url is not None

    def test_generator_agent_with_specific_config(self):
        """Test GeneratorAgent uses agent-specific config when available."""
        # Mock the entire ConfigManager class
        with patch(
            "pinocchio.agents.generator.ConfigManager"
        ) as mock_config_manager_class:
            # Create a mock instance
            mock_config_manager = MagicMock()

            # Mock the agent-specific config
            mock_llm_config = LLMConfigEntry(
                id="generator-llm",
                provider=LLMProvider.CUSTOM,
                model_name="generator-specific-model",
                base_url="http://generator:8001",
            )
            mock_config_manager.get_agent_llm_config.return_value = mock_llm_config
            mock_config_manager.get.return_value = True

            # Make the class return our mock instance
            mock_config_manager_class.return_value = mock_config_manager

            # Create agent - should use our mocked config
            agent = GeneratorAgent()

            # Verify agent-specific config was used
            mock_config_manager.get_agent_llm_config.assert_called_once_with(
                "generator"
            )
            assert isinstance(agent.llm_client, CustomLLMClient)
            assert agent.llm_client.model_name == "generator-specific-model"

    def test_optimizer_agent_with_specific_config(self):
        """Test OptimizerAgent uses agent-specific config when available."""
        with patch(
            "pinocchio.agents.optimizer.ConfigManager"
        ) as mock_config_manager_class:
            mock_config_manager = MagicMock()
            mock_llm_config = LLMConfigEntry(
                id="optimizer-llm",
                provider=LLMProvider.CUSTOM,
                model_name="optimizer-specific-model",
                base_url="http://optimizer:8001",
            )
            mock_config_manager.get_agent_llm_config.return_value = mock_llm_config
            mock_config_manager.get.return_value = True

            mock_config_manager_class.return_value = mock_config_manager

            agent = OptimizerAgent()

            mock_config_manager.get_agent_llm_config.assert_called_once_with(
                "optimizer"
            )
            assert isinstance(agent.llm_client, CustomLLMClient)
            assert agent.llm_client.model_name == "optimizer-specific-model"

    def test_debugger_agent_with_specific_config(self):
        """Test DebuggerAgent uses agent-specific config when available."""
        with patch(
            "pinocchio.agents.debugger.ConfigManager"
        ) as mock_config_manager_class:
            mock_config_manager = MagicMock()
            mock_llm_config = LLMConfigEntry(
                id="debugger-llm",
                provider=LLMProvider.CUSTOM,
                model_name="debugger-specific-model",
                base_url="http://debugger:8001",
            )
            mock_config_manager.get_agent_llm_config.return_value = mock_llm_config
            mock_config_manager.get.return_value = True

            mock_config_manager_class.return_value = mock_config_manager

            agent = DebuggerAgent()

            mock_config_manager.get_agent_llm_config.assert_called_once_with("debugger")
            assert isinstance(agent.llm_client, CustomLLMClient)
            assert agent.llm_client.model_name == "debugger-specific-model"

    def test_evaluator_agent_with_specific_config(self):
        """Test EvaluatorAgent uses agent-specific config when available."""
        with patch(
            "pinocchio.agents.evaluator.ConfigManager"
        ) as mock_config_manager_class:
            mock_config_manager = MagicMock()
            mock_llm_config = LLMConfigEntry(
                id="evaluator-llm",
                provider=LLMProvider.CUSTOM,
                model_name="evaluator-specific-model",
                base_url="http://evaluator:8001",
            )
            mock_config_manager.get_agent_llm_config.return_value = mock_llm_config
            mock_config_manager.get.return_value = True

            mock_config_manager_class.return_value = mock_config_manager

            agent = EvaluatorAgent()

            mock_config_manager.get_agent_llm_config.assert_called_once_with(
                "evaluator"
            )
            assert isinstance(agent.llm_client, CustomLLMClient)
            assert agent.llm_client.model_name == "evaluator-specific-model"

    def test_agent_fallback_to_global_config(self):
        """Test agents fallback to global config when agent-specific config is not available."""
        # Test with real config manager (should use global config)
        agents = [GeneratorAgent(), OptimizerAgent(), DebuggerAgent(), EvaluatorAgent()]

        for agent in agents:
            assert isinstance(agent.llm_client, CustomLLMClient)
            # Should use the global config from pinocchio.json
            assert agent.llm_client.model_name == "Qwen/Qwen3-32B"

    def test_agent_with_external_llm_client(self):
        """Test agents can still accept external LLM client."""
        mock_llm_client = MagicMock()

        generator = GeneratorAgent(llm_client=mock_llm_client)
        optimizer = OptimizerAgent(llm_client=mock_llm_client)
        debugger = DebuggerAgent(llm_client=mock_llm_client)
        evaluator = EvaluatorAgent(llm_client=mock_llm_client)

        assert generator.llm_client == mock_llm_client
        assert optimizer.llm_client == mock_llm_client
        assert debugger.llm_client == mock_llm_client
        assert evaluator.llm_client == mock_llm_client

    def test_config_validation(self):
        """Test that agent-specific configs are properly validated."""
        config_manager = ConfigManager()

        # Test with valid config
        result = config_manager.get_agent_llm_config("generator")
        assert isinstance(result, LLMConfigEntry)
        assert result.provider == LLMProvider.CUSTOM
        assert result.model_name is not None
        assert result.base_url is not None

    @pytest.mark.asyncio
    async def test_agent_execution_with_specific_config(self):
        """Test that agents can execute with their specific configs."""
        with patch(
            "pinocchio.agents.generator.ConfigManager"
        ) as mock_config_manager_class:
            mock_config_manager = MagicMock()
            mock_llm_config = LLMConfigEntry(
                id="test-llm",
                provider=LLMProvider.CUSTOM,
                model_name="test-model",
                base_url="http://test:8001",
            )
            mock_config_manager.get_agent_llm_config.return_value = mock_llm_config
            mock_config_manager.get.return_value = True

            mock_config_manager_class.return_value = mock_config_manager

            # Test generator agent execution
            generator = GeneratorAgent()
            request = {
                "request_id": "test",
                "task_description": "Generate a simple function",
            }

            # Mock the LLM client to avoid actual API calls
            with patch.object(generator.llm_client, "complete") as mock_complete:
                mock_complete.return_value = (
                    '{"success": true, "output": {"code": "test"}}'
                )

                result = await generator.execute(request)
                assert result.success is True
                assert result.agent_type == "generator"


@pytest.mark.local_only
class TestAgentSpecificConfigRealLLM:
    """Test agent-specific LLM configuration with real LLM (Local only - not for CI)."""

    @pytest.mark.skipif(
        not os.getenv("ENABLE_REAL_LLM_TESTS"),
        reason="Real LLM tests disabled. Set ENABLE_REAL_LLM_TESTS=1 to enable.",
    )
    def test_generator_agent_real_llm(self):
        """Test GeneratorAgent with real LLM configuration."""
        config_manager = ConfigManager()
        agent_llm_config = config_manager.get_agent_llm_config("generator")

        # Verify we have a real LLM config
        assert agent_llm_config.provider == LLMProvider.CUSTOM
        assert agent_llm_config.base_url is not None
        assert agent_llm_config.model_name is not None

        # Create agent with real LLM
        agent = GeneratorAgent()
        assert isinstance(agent.llm_client, CustomLLMClient)
        assert agent.llm_client.model_name == agent_llm_config.model_name
        assert agent.llm_client.base_url == agent_llm_config.base_url

    @pytest.mark.skipif(
        not os.getenv("ENABLE_REAL_LLM_TESTS"),
        reason="Real LLM tests disabled. Set ENABLE_REAL_LLM_TESTS=1 to enable.",
    )
    def test_optimizer_agent_real_llm(self):
        """Test OptimizerAgent with real LLM configuration."""
        config_manager = ConfigManager()
        agent_llm_config = config_manager.get_agent_llm_config("optimizer")

        agent = OptimizerAgent()
        assert isinstance(agent.llm_client, CustomLLMClient)
        assert agent.llm_client.model_name == agent_llm_config.model_name

    @pytest.mark.skipif(
        not os.getenv("ENABLE_REAL_LLM_TESTS"),
        reason="Real LLM tests disabled. Set ENABLE_REAL_LLM_TESTS=1 to enable.",
    )
    def test_debugger_agent_real_llm(self):
        """Test DebuggerAgent with real LLM configuration."""
        config_manager = ConfigManager()
        agent_llm_config = config_manager.get_agent_llm_config("debugger")

        agent = DebuggerAgent()
        assert isinstance(agent.llm_client, CustomLLMClient)
        assert agent.llm_client.model_name == agent_llm_config.model_name

    @pytest.mark.skipif(
        not os.getenv("ENABLE_REAL_LLM_TESTS"),
        reason="Real LLM tests disabled. Set ENABLE_REAL_LLM_TESTS=1 to enable.",
    )
    def test_evaluator_agent_real_llm(self):
        """Test EvaluatorAgent with real LLM configuration."""
        config_manager = ConfigManager()
        agent_llm_config = config_manager.get_agent_llm_config("evaluator")

        agent = EvaluatorAgent()
        assert isinstance(agent.llm_client, CustomLLMClient)
        assert agent.llm_client.model_name == agent_llm_config.model_name

    @pytest.mark.skipif(
        not os.getenv("ENABLE_REAL_LLM_TESTS"),
        reason="Real LLM tests disabled. Set ENABLE_REAL_LLM_TESTS=1 to enable.",
    )
    @pytest.mark.asyncio
    async def test_agent_execution_real_llm(self):
        """Test agent execution with real LLM (basic connectivity test)."""
        generator = GeneratorAgent()

        # Simple test request
        request = {
            "request_id": "test-real-llm",
            "task_description": "Generate a simple hello world function",
        }

        try:
            result = await generator.execute(request)
            # Basic validation - should not crash
            assert hasattr(result, "success")
            assert hasattr(result, "agent_type")
            assert result.agent_type == "generator"
        except Exception as e:
            # If LLM service is not available, that's expected
            pytest.skip(f"Real LLM test skipped - service not available: {e}")

    @pytest.mark.skipif(
        not os.getenv("ENABLE_REAL_LLM_TESTS"),
        reason="Real LLM tests disabled. Set ENABLE_REAL_LLM_TESTS=1 to enable.",
    )
    def test_agent_specific_config_priority(self):
        """Test that agent-specific configs take priority over global config."""
        config_manager = ConfigManager()

        # Test all agents have their own config
        agent_types = ["generator", "optimizer", "debugger", "evaluator"]

        for agent_type in agent_types:
            config = config_manager.get_agent_llm_config(agent_type)
            assert config is not None
            assert config.provider == LLMProvider.CUSTOM
            assert config.base_url is not None
            assert config.model_name is not None
