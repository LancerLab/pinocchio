"""Tests for assertion helpers."""

from unittest.mock import MagicMock

import pytest

from pinocchio.data_models.task_planning import AgentType, TaskPriority, TaskStatus
from pinocchio.session import SessionStatus
from tests.utils.assertion_helpers import (
    assert_agent_type,
    assert_dict_contains_keys,
    assert_dict_has_values,
    assert_list_contains_items,
    assert_list_has_length,
    assert_mock_called_with_args,
    assert_plan_completion_status,
    assert_session_runtime,
    assert_session_status,
    assert_session_valid,
    assert_string_contains,
    assert_string_ends_with,
    assert_string_starts_with,
    assert_task_dependencies,
    assert_task_plan_valid,
    assert_task_priority,
    assert_task_status,
    assert_task_valid,
)
from tests.utils.testing_data_factories import (
    create_completed_test_session,
    create_simple_task_plan,
    create_test_session,
    create_test_task,
    create_test_task_plan,
)


class TestAssertionHelpers:
    """Test assertion helper functions."""

    def test_assert_task_valid(self):
        """Test assert_task_valid function."""
        task = create_test_task()
        assert_task_valid(task)
        assert_task_valid(task, expected_agent_type=AgentType.GENERATOR)

    def test_assert_task_valid_with_invalid_task(self):
        """Test assert_task_valid with invalid task."""
        with pytest.raises(AssertionError):
            assert_task_valid(None)

    def test_assert_task_plan_valid(self):
        """Test assert_task_plan_valid function."""
        plan = create_simple_task_plan()
        assert_task_plan_valid(plan)
        assert_task_plan_valid(plan, expected_task_count=1)

    def test_assert_task_plan_valid_with_invalid_plan(self):
        """Test assert_task_plan_valid with invalid plan."""
        with pytest.raises(AssertionError):
            assert_task_plan_valid(None)

    def test_assert_session_valid(self):
        """Test assert_session_valid function."""
        session = create_test_session()
        assert_session_valid(session)
        assert_session_valid(session, expected_status=SessionStatus.ACTIVE)

    def test_assert_session_valid_with_invalid_session(self):
        """Test assert_session_valid with invalid session."""
        with pytest.raises(AssertionError):
            assert_session_valid(None)

    def test_assert_mock_called_with_args(self):
        """Test assert_mock_called_with_args function."""
        mock = MagicMock()
        mock("arg1", "arg2", kwarg1="value1", kwarg2="value2")

        assert_mock_called_with_args(mock)
        assert_mock_called_with_args(mock, expected_args=["arg1", "arg2"])
        assert_mock_called_with_args(mock, expected_kwargs={"kwarg1": "value1"})

    def test_assert_mock_called_with_args_not_called(self):
        """Test assert_mock_called_with_args with uncalled mock."""
        mock = MagicMock()
        with pytest.raises(AssertionError):
            assert_mock_called_with_args(mock)

    def test_assert_dict_contains_keys(self):
        """Test assert_dict_contains_keys function."""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        assert_dict_contains_keys(data, ["key1", "key2"])
        assert_dict_contains_keys(data, ["key1", "key2", "key3"])

    def test_assert_dict_contains_keys_missing(self):
        """Test assert_dict_contains_keys with missing key."""
        data = {"key1": "value1"}
        with pytest.raises(AssertionError):
            assert_dict_contains_keys(data, ["key1", "missing_key"])

    def test_assert_dict_has_values(self):
        """Test assert_dict_has_values function."""
        data = {"key1": "value1", "key2": "value2"}
        assert_dict_has_values(data, {"key1": "value1"})
        assert_dict_has_values(data, {"key1": "value1", "key2": "value2"})

    def test_assert_dict_has_values_mismatch(self):
        """Test assert_dict_has_values with value mismatch."""
        data = {"key1": "value1"}
        with pytest.raises(AssertionError):
            assert_dict_has_values(data, {"key1": "wrong_value"})

    def test_assert_list_contains_items(self):
        """Test assert_list_contains_items function."""
        items = [1, 2, 3, 4, 5]
        assert_list_contains_items(items, [1, 3, 5])
        assert_list_contains_items(items, [2, 4])

    def test_assert_list_contains_items_missing(self):
        """Test assert_list_contains_items with missing item."""
        items = [1, 2, 3]
        with pytest.raises(AssertionError):
            assert_list_contains_items(items, [1, 4])

    def test_assert_list_has_length(self):
        """Test assert_list_has_length function."""
        items = [1, 2, 3]
        assert_list_has_length(items, 3)
        assert_list_has_length([], 0)

    def test_assert_list_has_length_mismatch(self):
        """Test assert_list_has_length with length mismatch."""
        items = [1, 2, 3]
        with pytest.raises(AssertionError):
            assert_list_has_length(items, 5)

    def test_assert_string_contains(self):
        """Test assert_string_contains function."""
        text = "Hello, world!"
        assert_string_contains(text, "Hello")
        assert_string_contains(text, "world")

    def test_assert_string_contains_missing(self):
        """Test assert_string_contains with missing substring."""
        text = "Hello, world!"
        with pytest.raises(AssertionError):
            assert_string_contains(text, "missing")

    def test_assert_string_starts_with(self):
        """Test assert_string_starts_with function."""
        text = "Hello, world!"
        assert_string_starts_with(text, "Hello")

    def test_assert_string_starts_with_wrong_prefix(self):
        """Test assert_string_starts_with with wrong prefix."""
        text = "Hello, world!"
        with pytest.raises(AssertionError):
            assert_string_starts_with(text, "World")

    def test_assert_string_ends_with(self):
        """Test assert_string_ends_with function."""
        text = "Hello, world!"
        assert_string_ends_with(text, "world!")

    def test_assert_string_ends_with_wrong_suffix(self):
        """Test assert_string_ends_with with wrong suffix."""
        text = "Hello, world!"
        with pytest.raises(AssertionError):
            assert_string_ends_with(text, "Hello")

    def test_assert_task_status(self):
        """Test assert_task_status function."""
        task = create_test_task()
        assert_task_status(task, TaskStatus.PENDING)

    def test_assert_task_status_mismatch(self):
        """Test assert_task_status with status mismatch."""
        task = create_test_task()
        with pytest.raises(AssertionError):
            assert_task_status(task, TaskStatus.COMPLETED)

    def test_assert_session_status(self):
        """Test assert_session_status function."""
        session = create_test_session()
        assert_session_status(session, SessionStatus.ACTIVE)

    def test_assert_session_status_mismatch(self):
        """Test assert_session_status with status mismatch."""
        session = create_test_session()
        with pytest.raises(AssertionError):
            assert_session_status(session, SessionStatus.COMPLETED)

    def test_assert_task_priority(self):
        """Test assert_task_priority function."""
        task = create_test_task(priority=TaskPriority.CRITICAL)
        assert_task_priority(task, TaskPriority.CRITICAL)

    def test_assert_task_priority_mismatch(self):
        """Test assert_task_priority with priority mismatch."""
        task = create_test_task()
        with pytest.raises(AssertionError):
            assert_task_priority(task, TaskPriority.HIGH)

    def test_assert_agent_type(self):
        """Test assert_agent_type function."""
        task = create_test_task()
        assert_agent_type(task, AgentType.GENERATOR)

    def test_assert_agent_type_mismatch(self):
        """Test assert_agent_type with agent type mismatch."""
        task = create_test_task()
        with pytest.raises(AssertionError):
            assert_agent_type(task, AgentType.OPTIMIZER)

    def test_assert_task_dependencies(self):
        """Test assert_task_dependencies function."""
        task = create_test_task()
        assert_task_dependencies(task, 0)

    def test_assert_task_dependencies_with_deps(self):
        """Test assert_task_dependencies with dependencies."""
        from tests.utils.testing_data_factories import create_test_task_dependency

        task = create_test_task(dependencies=[create_test_task_dependency("dep1")])
        assert_task_dependencies(task, 1)

    def test_assert_plan_completion_status(self):
        """Test assert_plan_completion_status function."""
        plan = create_simple_task_plan()
        assert_plan_completion_status(plan, False)  # Initially not completed

    def test_assert_session_runtime(self):
        """Test assert_session_runtime function."""
        session = create_test_session()
        # Active session should have no runtime
        assert_session_runtime(session)

    def test_assert_session_runtime_completed(self):
        """Test assert_session_runtime with completed session."""
        session = create_completed_test_session()
        # Completed session should have runtime
        assert session.runtime_seconds is not None
        assert_session_runtime(session, min_runtime=0.0)


class TestAssertionHelpersIntegration:
    """Integration tests for assertion helpers."""

    def test_multiple_assertions_on_task(self):
        """Test multiple assertions on a single task."""
        task = create_test_task(
            agent_type=AgentType.OPTIMIZER, priority=TaskPriority.HIGH
        )

        assert_task_valid(task)
        assert_agent_type(task, AgentType.OPTIMIZER)
        assert_task_priority(task, TaskPriority.HIGH)
        assert_task_status(task, TaskStatus.PENDING)
        assert_task_dependencies(task, 0)

    def test_multiple_assertions_on_session(self):
        """Test multiple assertions on a single session."""
        session = create_completed_test_session()

        assert_session_valid(session)
        assert_session_status(session, SessionStatus.COMPLETED)
        assert session.runtime_seconds is not None
        assert_session_runtime(session, min_runtime=0.0)

    def test_multiple_assertions_on_plan(self):
        """Test multiple assertions on a single plan."""
        plan = create_simple_task_plan()

        assert_task_plan_valid(plan)
        assert_task_plan_valid(plan, expected_task_count=1)
        assert_plan_completion_status(plan, False)

        # Check the task in the plan
        assert len(plan.tasks) == 1
        assert_task_valid(plan.tasks[0])
        assert_agent_type(plan.tasks[0], AgentType.GENERATOR)
