"""Tests for test data factories."""

import pytest

from pinocchio.data_models.task_planning import (
    AgentType,
    Task,
    TaskDependency,
    TaskPlan,
    TaskPriority,
    TaskStatus,
)
from pinocchio.session import Session, SessionStatus
from tests.utils.test_data_factories import (
    create_completed_test_session,
    create_multi_task_plan,
    create_simple_task_plan,
    create_test_session,
    create_test_session_with_interactions,
    create_test_task,
    create_test_task_dependency,
    create_test_task_plan,
)


class TestTestDataFactories:
    """Test test data factory functions."""

    def test_create_test_task(self):
        """Test creating a test task."""
        task = create_test_task()

        assert isinstance(task, Task)
        assert task.task_id == "test_task"
        assert task.agent_type == AgentType.GENERATOR
        assert task.task_description == "Test task description"
        assert task.priority == TaskPriority.MEDIUM
        assert len(task.dependencies) == 0

    def test_create_test_task_with_custom_params(self):
        """Test creating a test task with custom parameters."""
        task = create_test_task(
            task_id="custom_task",
            agent_type=AgentType.OPTIMIZER,
            description="Custom description",
            priority=TaskPriority.HIGH,
        )

        assert task.task_id == "custom_task"
        assert task.agent_type == AgentType.OPTIMIZER
        assert task.task_description == "Custom description"
        assert task.priority == TaskPriority.HIGH

    def test_create_test_task_with_dependencies(self):
        """Test creating a test task with dependencies."""
        dependency = create_test_task_dependency("dep_task", "required")
        task = create_test_task(dependencies=[dependency])

        assert len(task.dependencies) == 1
        assert task.dependencies[0].task_id == "dep_task"
        assert task.dependencies[0].dependency_type == "required"

    def test_create_test_task_plan(self):
        """Test creating a test task plan."""
        plan = create_test_task_plan()

        assert isinstance(plan, TaskPlan)
        assert plan.plan_id == "test_plan"
        assert plan.user_request == "Test user request"
        assert len(plan.tasks) == 1
        assert isinstance(plan.tasks[0], Task)

    def test_create_test_task_plan_with_custom_tasks(self):
        """Test creating a test task plan with custom tasks."""
        tasks = [
            create_test_task("task_1", AgentType.GENERATOR),
            create_test_task("task_2", AgentType.DEBUGGER),
        ]
        plan = create_test_task_plan(tasks=tasks)

        assert len(plan.tasks) == 2
        assert plan.tasks[0].task_id == "task_1"
        assert plan.tasks[1].task_id == "task_2"

    def test_create_test_session(self):
        """Test creating a test session."""
        session = create_test_session()

        assert isinstance(session, Session)
        assert session.session_id == "test_session"
        assert session.task_description == "Test task description"
        assert session.status == SessionStatus.ACTIVE
        assert session.creation_time is not None

    def test_create_test_task_dependency(self):
        """Test creating a test task dependency."""
        dependency = create_test_task_dependency("dep_task", "required")

        assert isinstance(dependency, TaskDependency)
        assert dependency.task_id == "dep_task"
        assert dependency.dependency_type == "required"

    def test_create_simple_task_plan(self):
        """Test creating a simple task plan."""
        plan = create_simple_task_plan()

        assert plan.plan_id == "simple_plan"
        assert plan.user_request == "Generate a simple function"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].agent_type == AgentType.GENERATOR
        assert plan.tasks[0].task_id == "task_1"

    def test_create_multi_task_plan(self):
        """Test creating a multi-task plan."""
        plan = create_multi_task_plan()

        assert plan.plan_id == "multi_plan"
        assert len(plan.tasks) == 3

        # Check task order and dependencies
        assert plan.tasks[0].agent_type == AgentType.GENERATOR
        assert plan.tasks[1].agent_type == AgentType.DEBUGGER
        assert plan.tasks[2].agent_type == AgentType.OPTIMIZER

        # Check dependencies
        assert len(plan.tasks[0].dependencies) == 0  # Generator has no dependencies
        assert len(plan.tasks[1].dependencies) == 1  # Debugger depends on generator
        assert len(plan.tasks[2].dependencies) == 1  # Optimizer depends on debugger

    def test_create_completed_test_session(self):
        """Test creating a completed test session."""
        session = create_completed_test_session()

        assert session.session_id == "completed_session"
        assert session.status == SessionStatus.COMPLETED
        assert session.end_time is not None

    def test_create_test_session_with_interactions(self):
        """Test creating a test session with interactions."""
        session = create_test_session_with_interactions()

        assert session.session_id == "session_with_interactions"
        assert session.status == SessionStatus.ACTIVE
        assert len(session.agent_interactions) == 1
        assert len(session.optimization_iterations) == 1
        assert session.agent_interactions[0]["agent_type"] == "generator"
        assert session.optimization_iterations[0]["iteration_number"] == 1


class TestTestDataFactoriesIntegration:
    """Integration tests for test data factories."""

    def test_factory_integration(self):
        """Test that factories work together properly."""
        # Create a dependency
        dependency = create_test_task_dependency("base_task", "required")

        # Create a task with the dependency
        task = create_test_task(task_id="dependent_task", dependencies=[dependency])

        # Create a plan with the task
        plan = create_test_task_plan(plan_id="integration_plan", tasks=[task])

        # Create a session
        session = create_test_session(
            session_id="integration_session", task_description="Integration test"
        )

        # Verify all objects are properly created
        assert isinstance(task, Task)
        assert isinstance(plan, TaskPlan)
        assert isinstance(session, Session)
        assert len(task.dependencies) == 1
        assert len(plan.tasks) == 1
        assert plan.tasks[0] == task
