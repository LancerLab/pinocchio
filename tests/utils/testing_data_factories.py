"""Test data factories for creating test objects."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pinocchio.data_models.task_planning import (
    AgentType,
    Task,
    TaskDependency,
    TaskPlan,
    TaskPriority,
    TaskResult,
    TaskStatus,
)
from pinocchio.session import Session, SessionStatus
from pinocchio.session.models.metadata import SessionMetadata


def create_test_task(
    task_id: str = "test_task",
    agent_type: AgentType = AgentType.GENERATOR,
    description: str = "Test task description",
    priority: TaskPriority = TaskPriority.MEDIUM,
    status: TaskStatus = TaskStatus.PENDING,
    input_data: Optional[Dict] = None,
    dependencies: Optional[List[TaskDependency]] = None,
    requirements: Optional[Dict] = None,
    result: Optional[TaskResult] = None,
    error_count: int = 0,
    max_retries: int = 3,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> Task:
    """Create a test Task instance with default values."""
    return Task(
        task_id=task_id,
        agent_type=agent_type,
        task_description=description,
        priority=priority,
        status=status,
        input_data=input_data or {},
        dependencies=dependencies or [],
        requirements=requirements or {},
        result=result,
        error_count=error_count,
        max_retries=max_retries,
        started_at=started_at,
        completed_at=completed_at,
    )


def create_test_task_dependency(
    task_id: str, dependency_type: str = "required"
) -> TaskDependency:
    """Create a test TaskDependency instance."""
    return TaskDependency(task_id=task_id, dependency_type=dependency_type)


def create_test_task_plan(
    plan_id: str = "test_plan",
    user_request: str = "Test user request",
    tasks: Optional[List[Task]] = None,
    status: TaskStatus = TaskStatus.PENDING,
    created_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> TaskPlan:
    """Create a test TaskPlan instance with default values."""
    if tasks is None:
        tasks = [
            create_test_task(
                task_id="task_1",
                agent_type=AgentType.GENERATOR,
                description="Generate code",
            )
        ]

    return TaskPlan(
        plan_id=plan_id,
        user_request=user_request,
        tasks=tasks,
        status=status,
        created_at=created_at or datetime.utcnow(),
        completed_at=completed_at,
    )


def create_test_session(
    session_id: str = "test_session",
    task_description: str = "Test task description",
    status: SessionStatus = SessionStatus.ACTIVE,
    agent_interactions: Optional[List[Dict]] = None,
    optimization_iterations: Optional[List[Dict]] = None,
    performance_trend: Optional[List[Dict]] = None,
    memory_versions: Optional[List[str]] = None,
    prompt_versions: Optional[List[str]] = None,
    knowledge_versions: Optional[List[str]] = None,
    code_version_ids: Optional[List[str]] = None,
    target_performance: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    creation_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    runtime_seconds: Optional[float] = None,
) -> Session:
    """Create a test Session instance with default values."""
    return Session(
        session_id=session_id,
        task_description=task_description,
        status=status,
        agent_interactions=agent_interactions or [],
        optimization_iterations=optimization_iterations or [],
        performance_trend=performance_trend or [],
        memory_versions=memory_versions or [],
        prompt_versions=prompt_versions or [],
        knowledge_versions=knowledge_versions or [],
        code_version_ids=code_version_ids or [],
        target_performance=target_performance or {},
        metadata=metadata or {},
        creation_time=creation_time or datetime.utcnow(),
        end_time=end_time,
        runtime_seconds=runtime_seconds,
    )


def create_simple_task_plan() -> TaskPlan:
    """Create a simple task plan with one generator task."""
    return create_test_task_plan(
        plan_id="simple_plan",
        user_request="Generate a simple function",
        tasks=[
            create_test_task(
                task_id="task_1",
                agent_type=AgentType.GENERATOR,
                description="Generate code",
            )
        ],
    )


def create_multi_task_plan() -> TaskPlan:
    """Create a multi-task plan with dependencies."""
    task1 = create_test_task(
        task_id="task_1",
        agent_type=AgentType.GENERATOR,
        description="Generate code",
    )

    task2 = create_test_task(
        task_id="task_2",
        agent_type=AgentType.DEBUGGER,
        description="Debug code",
        dependencies=[create_test_task_dependency("task_1", "required")],
    )

    task3 = create_test_task(
        task_id="task_3",
        agent_type=AgentType.OPTIMIZER,
        description="Optimize code",
        dependencies=[create_test_task_dependency("task_2", "required")],
    )

    return create_test_task_plan(
        plan_id="multi_plan",
        user_request="Generate, debug, and optimize code",
        tasks=[task1, task2, task3],
    )


def create_test_session_with_interactions() -> Session:
    """Create a test session with some agent interactions."""
    return create_test_session(
        session_id="session_with_interactions",
        task_description="Test task with interactions",
        agent_interactions=[
            {
                "agent_type": "generator",
                "data": {"input": "test", "output": "result"},
                "timestamp": datetime.utcnow(),
            }
        ],
        optimization_iterations=[
            {
                "iteration_number": 1,
                "data": {"optimization": "test"},
                "timestamp": datetime.utcnow(),
            }
        ],
    )


def create_completed_test_session() -> Session:
    """Create a completed test session."""
    return create_test_session(
        session_id="completed_session",
        task_description="Completed test task",
        status=SessionStatus.COMPLETED,
        end_time=datetime.utcnow(),
        runtime_seconds=120.5,
    )


def create_test_session_metadata(
    session_id: str = "test_session_metadata",
    name: str = "Test Session",
    task_description: str = "Test task description",
    status: str = "active",
    tags: Optional[List[str]] = None,
    user_inputs: Optional[Dict[str, Any]] = None,
    system_info: Optional[Dict[str, Any]] = None,
    creation_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    runtime_seconds: Optional[float] = None,
) -> SessionMetadata:
    """Create a test SessionMetadata instance with default values."""
    if creation_time is None:
        creation_time = datetime.utcnow()
    return SessionMetadata(
        session_id=session_id,
        name=name,
        task_description=task_description,
        status=status,
        tags=tags or [],
        user_inputs=user_inputs or {},
        system_info=system_info or {},
        creation_time=creation_time,
        end_time=end_time,
        runtime_seconds=runtime_seconds,
    )


def create_test_config(
    app_name: str = "pinocchio-test",
    version: str = "0.1.0",
    debug: bool = True,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
    api_key: str = "test-key",
    memory_path: str = "./data",
    max_items: int = 1000,
    session_timeout: int = 3600,
    auto_save: bool = True,
) -> Dict[str, Any]:
    """
    Create a test configuration dictionary.
    """
    return {
        "app": {
            "name": app_name,
            "version": version,
            "debug": debug,
        },
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
            "api_key": api_key,
        },
        "memory": {
            "storage_path": memory_path,
            "max_items": max_items,
        },
        "session": {
            "timeout": session_timeout,
            "auto_save": auto_save,
        },
    }


def create_test_session_data(
    session_id: str = "test_session",
    task_description: str = "Test task",
    status: str = "active",
    agent_interactions: Optional[List[Dict[str, Any]]] = None,
    optimization_iterations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create test session data (dict, not Session model).
    """
    if agent_interactions is None:
        agent_interactions = [
            {
                "agent_type": "generator",
                "data": {"input": "test", "output": "result"},
                "timestamp": "2023-01-01T00:00:00Z",
            }
        ]
    if optimization_iterations is None:
        optimization_iterations = [
            {
                "iteration_number": 1,
                "data": {"optimization": "test"},
                "timestamp": "2023-01-01T00:00:00Z",
            }
        ]
    return {
        "session_id": session_id,
        "task_description": task_description,
        "status": status,
        "agent_interactions": agent_interactions,
        "optimization_iterations": optimization_iterations,
        "creation_time": "2023-01-01T00:00:00Z",
    }


def create_test_memory_data(
    memory_id: str = "test_memory",
    agent_type: str = "generator",
    content: str = "Test memory content",
    created_at: str = "2023-01-01T00:00:00Z",
) -> Dict[str, Any]:
    """
    Create test memory data (dict, not model).
    """
    return {
        "memory_id": memory_id,
        "agent_type": agent_type,
        "content": content,
        "created_at": created_at,
    }
