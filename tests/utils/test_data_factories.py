"""Test data factories for creating test objects."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pinocchio.data_models.task_planning import (
    AgentType,
    Task,
    TaskDependency,
    TaskPlan,
    TaskPriority,
    TaskStatus,
)
from pinocchio.session import Session, SessionStatus


def create_test_task(
    task_id: str = "test_task",
    agent_type: AgentType = AgentType.GENERATOR,
    description: str = "Test task description",
    priority: TaskPriority = TaskPriority.CRITICAL,
    dependencies: Optional[List[TaskDependency]] = None,
    **kwargs,
) -> Task:
    """
    Create a test Task instance.

    Args:
        task_id: Task identifier
        agent_type: Type of agent for this task
        description: Task description
        priority: Task priority level
        dependencies: List of task dependencies
        **kwargs: Additional task parameters

    Returns:
        Task instance
    """
    if dependencies is None:
        dependencies = []

    return Task(
        task_id=task_id,
        agent_type=agent_type,
        task_description=description,
        priority=priority,
        dependencies=dependencies,
        **kwargs,
    )


def create_test_task_plan(
    plan_id: str = "test_plan",
    user_request: str = "Test request",
    tasks: Optional[List[Task]] = None,
    **kwargs,
) -> TaskPlan:
    """
    Create a test TaskPlan instance.

    Args:
        plan_id: Plan identifier
        user_request: User's original request
        tasks: List of tasks in the plan
        **kwargs: Additional plan parameters

    Returns:
        TaskPlan instance
    """
    if tasks is None:
        tasks = [create_test_task()]

    return TaskPlan(plan_id=plan_id, user_request=user_request, tasks=tasks, **kwargs)


def create_test_session(
    session_id: str = "test_session",
    task_description: str = "Test task",
    status: SessionStatus = SessionStatus.ACTIVE,
    **kwargs,
) -> Session:
    """
    Create a test Session instance.

    Args:
        session_id: Session identifier
        task_description: Description of the task
        status: Session status
        **kwargs: Additional session parameters

    Returns:
        Session instance
    """
    return Session(
        session_id=session_id,
        task_description=task_description,
        status=status,
        creation_time=datetime.now(),
        **kwargs,
    )


def create_test_task_dependency(
    task_id: str = "dependency_task", dependency_type: str = "required"
) -> TaskDependency:
    """
    Create a test TaskDependency instance.

    Args:
        task_id: ID of the task being depended on
        dependency_type: Type of dependency

    Returns:
        TaskDependency instance
    """
    return TaskDependency(task_id=task_id, dependency_type=dependency_type)


def create_simple_task_plan() -> TaskPlan:
    """
    Create a simple task plan with one generator task.

    Returns:
        Simple TaskPlan instance
    """
    task = create_test_task(
        task_id="task_1",
        agent_type=AgentType.GENERATOR,
        description="Generate test code",
        priority=TaskPriority.CRITICAL,
    )

    return create_test_task_plan(
        plan_id="simple_plan", user_request="Generate a simple function", tasks=[task]
    )


def create_multi_task_plan() -> TaskPlan:
    """
    Create a task plan with multiple tasks and dependencies.

    Returns:
        TaskPlan with generator -> debugger -> optimizer chain
    """
    # Create tasks
    generator_task = create_test_task(
        task_id="task_1",
        agent_type=AgentType.GENERATOR,
        description="Generate initial code",
        priority=TaskPriority.CRITICAL,
    )

    debugger_task = create_test_task(
        task_id="task_2",
        agent_type=AgentType.DEBUGGER,
        description="Debug generated code",
        priority=TaskPriority.CRITICAL,
        dependencies=[create_test_task_dependency("task_1", "required")],
    )

    optimizer_task = create_test_task(
        task_id="task_3",
        agent_type=AgentType.OPTIMIZER,
        description="Optimize code",
        priority=TaskPriority.HIGH,
        dependencies=[create_test_task_dependency("task_2", "required")],
    )

    return create_test_task_plan(
        plan_id="multi_task_plan",
        user_request="Generate, debug, and optimize code",
        tasks=[generator_task, debugger_task, optimizer_task],
    )


def create_failed_task_plan() -> TaskPlan:
    """
    Create a task plan with failed tasks for testing error scenarios.

    Returns:
        TaskPlan with failed tasks
    """
    failed_task = create_test_task(
        task_id="failed_task",
        agent_type=AgentType.GENERATOR,
        description="This task will fail",
        priority=TaskPriority.CRITICAL,
    )
    failed_task.mark_failed("Test error message")

    return create_test_task_plan(
        plan_id="failed_plan", user_request="Test failure scenario", tasks=[failed_task]
    )


def create_completed_session() -> Session:
    """
    Create a completed session for testing.

    Returns:
        Completed Session instance
    """
    session = create_test_session(
        session_id="completed_session",
        task_description="Completed test task",
        status=SessionStatus.COMPLETED,
    )
    session.complete_session()
    return session


def create_active_session() -> Session:
    """
    Create an active session for testing.

    Returns:
        Active Session instance
    """
    return create_test_session(
        session_id="active_session",
        task_description="Active test task",
        status=SessionStatus.ACTIVE,
    )
