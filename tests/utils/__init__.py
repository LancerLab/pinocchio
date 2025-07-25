"""Test utilities for Pinocchio tests.

This package contains utilities specifically designed for testing,
including mock factories, test data factories, assertion helpers,
and other testing-specific utilities.

For module functionality utilities, see pinocchio.utils/.
"""

from .assertion_helpers import (
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
from .mock_factories import (
    create_async_mock_llm_client,
    create_mock_agent_response,
    create_mock_llm_client,
    create_task_planner_mock_response,
)
from .testing_data_factories import (
    create_completed_test_session,
    create_multi_task_plan,
    create_simple_task_plan,
    create_test_config,
    create_test_memory_data,
    create_test_session,
    create_test_session_data,
    create_test_session_metadata,
    create_test_session_with_interactions,
    create_test_task,
    create_test_task_dependency,
    create_test_task_plan,
)
from .testing_utils import assert_dict_structure as assert_dict_structure_test
from .testing_utils import (
    assert_json_file_structure,
    cleanup_test_files,
    create_test_directory_structure,
    create_test_json_file,
    create_test_logger,
    create_test_temp_dir,
    create_test_temp_file,
    load_test_json_file,
)

__all__ = [
    # Mock factories
    "create_mock_llm_client",
    "create_async_mock_llm_client",
    "create_mock_agent_response",
    "create_task_planner_mock_response",
    # Test data factories
    "create_test_task",
    "create_test_task_dependency",
    "create_test_task_plan",
    "create_multi_task_plan",
    "create_simple_task_plan",
    "create_test_session",
    "create_test_session_metadata",
    "create_test_session_with_interactions",
    "create_completed_test_session",
    "create_test_session_data",
    "create_test_memory_data",
    "create_test_config",
    # Test utilities
    "create_test_json_file",
    "load_test_json_file",
    "create_test_temp_dir",
    "create_test_temp_file",
    "create_test_logger",
    "cleanup_test_files",
    "create_test_directory_structure",
    "assert_json_file_structure",
    "assert_dict_structure_test",
    # Assertion helpers
    "assert_task_valid",
    "assert_task_plan_valid",
    "assert_session_valid",
    "assert_mock_called_with_args",
    "assert_dict_contains_keys",
    "assert_dict_has_values",
    "assert_list_contains_items",
    "assert_list_has_length",
    "assert_string_contains",
    "assert_string_starts_with",
    "assert_string_ends_with",
    "assert_task_status",
    "assert_session_status",
    "assert_task_priority",
    "assert_agent_type",
    "assert_task_dependencies",
    "assert_plan_completion_status",
    "assert_session_runtime",
]
