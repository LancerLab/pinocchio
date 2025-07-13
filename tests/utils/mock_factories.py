"""Mock factories for testing."""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from pinocchio.llm.mock_client import MockLLMClient


def create_mock_llm_client(
    response: str = '{"success": true, "output": {"code": "test code"}}',
    delay_ms: int = 1,
    failure_rate: float = 0.0,
) -> MockLLMClient:
    """
    Create a mock LLM client for testing.

    Args:
        response: Mock response string
        delay_ms: Response delay in milliseconds
        failure_rate: Failure rate (0.0 to 1.0)

    Returns:
        MockLLMClient instance
    """
    return MockLLMClient(response_delay_ms=delay_ms, failure_rate=failure_rate)


def create_async_mock_llm_client(
    response: str = '{"success": true, "output": {"code": "test code"}}',
) -> AsyncMock:
    """
    Create an async mock LLM client for testing.

    Args:
        response: Mock response string

    Returns:
        AsyncMock client instance
    """
    client = AsyncMock()
    client.complete = AsyncMock(return_value=response)
    return client


def create_mock_agent_response(
    success: bool = True,
    output: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    processing_time_ms: int = 100,
    request_id: str = "test_request",
) -> MagicMock:
    """
    Create a mock agent response.

    Args:
        success: Whether the response indicates success
        output: Response output data
        error_message: Error message if any
        processing_time_ms: Processing time in milliseconds
        request_id: Request identifier

    Returns:
        MagicMock response object
    """
    if output is None:
        output = {"code": "test code"}

    return MagicMock(
        success=success,
        output=output,
        error_message=error_message,
        processing_time_ms=processing_time_ms,
        request_id=request_id,
    )


def create_task_planner_mock_response() -> str:
    """
    Create a mock response for TaskPlanner analysis.

    Returns:
        JSON string representing task planner analysis
    """
    return """{
        "requirements": {
            "primary_goal": "Generate a matrix multiplication function",
            "secondary_goals": ["efficient implementation"],
            "code_requirements": ["efficient_data_structures", "performance_optimization"]
        },
        "optimization_goals": ["performance", "memory_efficiency"],
        "constraints": ["readability"],
        "user_preferences": {},
        "planning_strategy": "standard"
    }"""
