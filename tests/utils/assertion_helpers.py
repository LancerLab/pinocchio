"""Assertion helpers for Pinocchio tests."""

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from pinocchio.data_models.task_planning import (
    AgentType,
    Task,
    TaskPlan,
    TaskPriority,
    TaskStatus,
)
from pinocchio.session import Session, SessionStatus


def assert_task_valid(
    task: Task, expected_agent_type: Optional[AgentType] = None
) -> None:
    """
    Assert that a task is valid.

    Args:
        task: Task to validate
        expected_agent_type: Expected agent type (optional)
    """
    assert isinstance(task, Task)
    assert task.task_id is not None
    assert task.task_description is not None
    assert task.agent_type is not None
    assert task.priority is not None
    assert task.status is not None

    if expected_agent_type:
        assert task.agent_type == expected_agent_type


def assert_task_plan_valid(
    plan: TaskPlan, expected_task_count: Optional[int] = None
) -> None:
    """
    Assert that a task plan is valid.

    Args:
        plan: Task plan to validate
        expected_task_count: Expected number of tasks (optional)
    """
    assert isinstance(plan, TaskPlan)
    assert plan.plan_id is not None
    assert plan.user_request is not None
    assert isinstance(plan.tasks, list)

    if expected_task_count is not None:
        assert len(plan.tasks) == expected_task_count

    for task in plan.tasks:
        assert_task_valid(task)


def assert_session_valid(
    session: Session, expected_status: Optional[SessionStatus] = None
) -> None:
    """
    Assert that a session is valid.

    Args:
        session: Session to validate
        expected_status: Expected session status (optional)
    """
    assert isinstance(session, Session)
    assert session.session_id is not None
    assert session.task_description is not None
    assert session.status is not None
    assert session.creation_time is not None

    if expected_status:
        assert session.status == expected_status


def assert_mock_called_with_args(
    mock: MagicMock,
    expected_args: List[Any] = None,
    expected_kwargs: Dict[str, Any] = None,
) -> None:
    """
    Assert that a mock was called with specific arguments.

    Args:
        mock: Mock object to check
        expected_args: Expected positional arguments
        expected_kwargs: Expected keyword arguments
    """
    assert mock.called

    if expected_args is not None:
        assert mock.call_args[0] == tuple(expected_args)

    if expected_kwargs is not None:
        actual_kwargs = mock.call_args[1] if len(mock.call_args) > 1 else {}
        for key, value in expected_kwargs.items():
            assert key in actual_kwargs
            assert actual_kwargs[key] == value


def assert_dict_contains_keys(data: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Assert that a dictionary contains required keys.

    Args:
        data: Dictionary to check
        required_keys: List of required keys
    """
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"


def assert_dict_has_values(
    data: Dict[str, Any], expected_values: Dict[str, Any]
) -> None:
    """
    Assert that a dictionary has specific values.

    Args:
        data: Dictionary to check
        expected_values: Expected key-value pairs
    """
    for key, expected_value in expected_values.items():
        assert key in data, f"Missing key: {key}"
        assert (
            data[key] == expected_value
        ), f"Value mismatch for key {key}: expected {expected_value}, got {data[key]}"


def assert_list_contains_items(items: List[Any], expected_items: List[Any]) -> None:
    """
    Assert that a list contains expected items.

    Args:
        items: List to check
        expected_items: Expected items
    """
    for expected_item in expected_items:
        assert expected_item in items, f"Expected item not found: {expected_item}"


def assert_list_has_length(items: List[Any], expected_length: int) -> None:
    """
    Assert that a list has expected length.

    Args:
        items: List to check
        expected_length: Expected length
    """
    assert (
        len(items) == expected_length
    ), f"Expected length {expected_length}, got {len(items)}"


def assert_string_contains(text: str, expected_substring: str) -> None:
    """
    Assert that a string contains a substring.

    Args:
        text: Text to check
        expected_substring: Expected substring
    """
    assert (
        expected_substring in text
    ), f"Expected substring '{expected_substring}' not found in '{text}'"


def assert_string_starts_with(text: str, expected_prefix: str) -> None:
    """
    Assert that a string starts with a prefix.

    Args:
        text: Text to check
        expected_prefix: Expected prefix
    """
    assert text.startswith(
        expected_prefix
    ), f"Expected prefix '{expected_prefix}', got '{text}'"


def assert_string_ends_with(text: str, expected_suffix: str) -> None:
    """
    Assert that a string ends with a suffix.

    Args:
        text: Text to check
        expected_suffix: Expected suffix
    """
    assert text.endswith(
        expected_suffix
    ), f"Expected suffix '{expected_suffix}', got '{text}'"


def assert_task_status(task: Task, expected_status: TaskStatus) -> None:
    """
    Assert that a task has expected status.

    Args:
        task: Task to check
        expected_status: Expected status
    """
    assert (
        task.status == expected_status
    ), f"Expected status {expected_status}, got {task.status}"


def assert_session_status(session: Session, expected_status: SessionStatus) -> None:
    """
    Assert that a session has expected status.

    Args:
        session: Session to check
        expected_status: Expected status
    """
    assert (
        session.status == expected_status
    ), f"Expected status {expected_status}, got {session.status}"


def assert_task_priority(task: Task, expected_priority: TaskPriority) -> None:
    """
    Assert that a task has expected priority.

    Args:
        task: Task to check
        expected_priority: Expected priority
    """
    assert (
        task.priority == expected_priority
    ), f"Expected priority {expected_priority}, got {task.priority}"


def assert_agent_type(task: Task, expected_agent_type: AgentType) -> None:
    """
    Assert that a task has expected agent type.

    Args:
        task: Task to check
        expected_agent_type: Expected agent type
    """
    assert (
        task.agent_type == expected_agent_type
    ), f"Expected agent type {expected_agent_type}, got {task.agent_type}"


def assert_task_dependencies(task: Task, expected_dependency_count: int) -> None:
    """
    Assert that a task has expected number of dependencies.

    Args:
        task: Task to check
        expected_dependency_count: Expected number of dependencies
    """
    actual_count = len(task.dependencies) if task.dependencies else 0
    assert (
        actual_count == expected_dependency_count
    ), f"Expected {expected_dependency_count} dependencies, got {actual_count}"


def assert_plan_completion_status(plan: TaskPlan, expected_completed: bool) -> None:
    """
    Assert that a plan has expected completion status.

    Args:
        plan: Plan to check
        expected_completed: Expected completion status
    """
    actual_completed = plan.is_completed()
    assert (
        actual_completed == expected_completed
    ), f"Expected completion status {expected_completed}, got {actual_completed}"


def assert_session_runtime(
    session: Session,
    min_runtime: Optional[float] = None,
    max_runtime: Optional[float] = None,
) -> None:
    """
    Assert that a session has runtime within expected range.

    Args:
        session: Session to check
        min_runtime: Minimum expected runtime (optional)
        max_runtime: Maximum expected runtime (optional)
    """
    if session.runtime_seconds is not None:
        if min_runtime is not None:
            assert (
                session.runtime_seconds >= min_runtime
            ), f"Runtime {session.runtime_seconds} is less than minimum {min_runtime}"
        if max_runtime is not None:
            assert (
                session.runtime_seconds <= max_runtime
            ), f"Runtime {session.runtime_seconds} is greater than maximum {max_runtime}"
    else:
        # If no runtime is set, it should be None for active sessions
        if session.status == SessionStatus.ACTIVE:
            assert session.runtime_seconds is None
        else:
            # For completed/failed sessions, runtime should be set
            assert (
                session.runtime_seconds is not None
            ), "Runtime should be set for completed/failed sessions"
