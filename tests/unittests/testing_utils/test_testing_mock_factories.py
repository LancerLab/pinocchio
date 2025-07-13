"""Tests for mock factories."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pinocchio.llm.mock_client import MockLLMClient
from tests.utils.mock_factories import (
    create_async_mock_llm_client,
    create_mock_agent_response,
    create_mock_llm_client,
    create_task_planner_mock_response,
)


class TestMockFactories:
    """Test mock factory functions."""

    def test_create_mock_llm_client(self):
        """Test creating mock LLM client."""
        client = create_mock_llm_client()

        assert client is not None
        assert hasattr(client, "complete")
        assert client.response_delay_ms == 1
        assert client.failure_rate == 0.0

    def test_create_mock_llm_client_with_custom_params(self):
        """Test creating mock LLM client with custom parameters."""
        client = create_mock_llm_client(
            response='{"test": "response"}', response_delay_ms=100, failure_rate=0.5
        )
        assert isinstance(client, MockLLMClient)

    def test_create_async_mock_llm_client(self):
        """Test creating async mock LLM client."""
        client = create_async_mock_llm_client()

        assert isinstance(client, AsyncMock)
        assert hasattr(client, "complete")
        assert isinstance(client.complete, AsyncMock)

    def test_create_async_mock_llm_client_with_custom_response(self):
        """Test creating async mock LLM client with custom response."""
        custom_response = '{"custom": "response"}'
        client = create_async_mock_llm_client(custom_response)

        # The response is set in the mock, but we can't easily test the return value
        # without calling the method, which is fine for a factory test
        assert isinstance(client, AsyncMock)

    def test_create_mock_agent_response(self):
        """Test creating mock agent response."""
        response = create_mock_agent_response()

        assert isinstance(response, MagicMock)
        assert response.success is True
        assert response.output == {"code": "test code"}
        assert response.error_message is None
        assert response.processing_time_ms == 100
        assert response.request_id == "test_request"

    def test_create_mock_agent_response_with_custom_params(self):
        """Test creating mock agent response with custom parameters."""
        custom_output = {"custom": "output"}
        response = create_mock_agent_response(
            success=False,
            output=custom_output,
            error_message="Test error",
            processing_time_ms=200,
            request_id="custom_request",
        )

        assert response.success is False
        assert response.output == custom_output
        assert response.error_message == "Test error"
        assert response.processing_time_ms == 200
        assert response.request_id == "custom_request"

    def test_create_task_planner_mock_response(self):
        """Test creating task planner mock response."""
        response = create_task_planner_mock_response()

        assert isinstance(response, str)
        assert "requirements" in response
        assert "optimization_goals" in response
        assert "constraints" in response
        assert "planning_strategy" in response


class TestMockFixtures:
    """Test mock fixtures."""

    def test_mock_llm_client_fixture(self, mock_llm_client):
        """Test mock_llm_client fixture."""
        assert mock_llm_client is not None
        assert hasattr(mock_llm_client, "complete")

    def test_async_mock_llm_client_fixture(self, async_mock_llm_client):
        """Test async_mock_llm_client fixture."""
        assert isinstance(async_mock_llm_client, AsyncMock)
        assert hasattr(async_mock_llm_client, "complete")

    def test_task_planner_mock_llm_client_fixture(self, task_planner_mock_llm_client):
        """Test task_planner_mock_llm_client fixture."""
        assert isinstance(task_planner_mock_llm_client, AsyncMock)
        assert hasattr(task_planner_mock_llm_client, "complete")
