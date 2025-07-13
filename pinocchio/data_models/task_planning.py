"""Task planning data models for Pinocchio multi-agent system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AgentType(str, Enum):
    """Available agent types."""

    GENERATOR = "generator"
    OPTIMIZER = "optimizer"
    DEBUGGER = "debugger"
    EVALUATOR = "evaluator"


@dataclass
class TaskDependency:
    """Task dependency information."""

    task_id: str
    dependency_type: str = "required"  # required, optional, conditional
    condition: Optional[str] = None  # condition for conditional dependencies


@dataclass
class TaskResult:
    """Result of task execution."""

    success: bool
    output: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task(BaseModel):
    """Individual task in the plan."""

    task_id: str = Field(..., description="Unique task identifier")
    agent_type: AgentType = Field(..., description="Agent type to execute this task")
    task_description: str = Field(..., description="Human-readable task description")
    requirements: Dict[str, Any] = Field(
        default_factory=dict, description="Task requirements"
    )
    optimization_goals: List[str] = Field(
        default_factory=list, description="Optimization goals"
    )

    # Execution control
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current task status"
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM, description="Task priority"
    )
    dependencies: List[TaskDependency] = Field(
        default_factory=list, description="Task dependencies"
    )

    # Execution metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Task creation time"
    )
    started_at: Optional[datetime] = Field(default=None, description="Task start time")
    completed_at: Optional[datetime] = Field(
        default=None, description="Task completion time"
    )

    # Results
    result: Optional[TaskResult] = Field(
        default=None, description="Task execution result"
    )

    # Retry and error handling
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_count: int = Field(default=0, description="Current retry count")
    error_count: int = Field(default=0, description="Number of errors encountered")

    # Context and data flow
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input data for the task"
    )
    output_data: Dict[str, Any] = Field(
        default_factory=dict, description="Output data from the task"
    )

    # Task-specific configuration
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Task-specific configuration"
    )

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    def can_execute(self, completed_tasks: List[str]) -> bool:
        """
        Check if task can be executed based on dependencies.

        Args:
            completed_tasks: List of completed task IDs

        Returns:
            True if task can be executed
        """
        if self.status != TaskStatus.PENDING:
            return False

        for dependency in self.dependencies:
            if dependency.dependency_type == "required":
                if dependency.task_id not in completed_tasks:
                    return False
            elif dependency.dependency_type == "conditional":
                if dependency.condition and dependency.task_id in completed_tasks:
                    # Check if condition is met
                    # This is a simplified implementation
                    pass

        return True

    def mark_started(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, result: TaskResult) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result

    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_count += 1
        self.result = TaskResult(success=False, output={}, error_message=error_message)

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.status == TaskStatus.FAILED and self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry count and reset status."""
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.result = None


class TaskPlan(BaseModel):
    """Complete task plan for a user request."""

    plan_id: str = Field(..., description="Unique plan identifier")
    user_request: str = Field(..., description="Original user request")

    # Tasks
    tasks: List[Task] = Field(
        default_factory=list, description="List of tasks in the plan"
    )

    # Plan metadata
    created_at: datetime = Field(
        default_factory=datetime.now, description="Plan creation time"
    )
    started_at: Optional[datetime] = Field(default=None, description="Plan start time")
    completed_at: Optional[datetime] = Field(
        default=None, description="Plan completion time"
    )

    # Plan status
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Overall plan status"
    )

    # Execution context
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Execution context"
    )
    session_id: Optional[str] = Field(default=None, description="Associated session ID")

    # Results and metrics
    final_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Final plan result"
    )
    execution_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Execution metrics"
    )

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    def get_ready_tasks(self) -> List[Task]:
        """
        Get tasks that are ready to execute.

        Returns:
            List of tasks ready for execution
        """
        completed_tasks = [
            task.task_id for task in self.tasks if task.status == TaskStatus.COMPLETED
        ]

        ready_tasks = []
        for task in self.tasks:
            if task.can_execute(completed_tasks):
                ready_tasks.append(task)

        return ready_tasks

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task identifier

        Returns:
            Task if found, None otherwise
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def is_completed(self) -> bool:
        """
        Check if all tasks are completed.

        Returns:
            True if all tasks are completed
        """
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks)

    def is_failed(self) -> bool:
        """
        Check if any critical task failed.

        Returns:
            True if any critical task failed
        """
        return any(
            task.status == TaskStatus.FAILED and task.priority == TaskPriority.CRITICAL
            for task in self.tasks
        )

    def get_progress(self) -> Dict[str, Any]:
        """
        Get plan execution progress.

        Returns:
            Progress information
        """
        total_tasks = len(self.tasks)
        completed_tasks = len(
            [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
        )
        failed_tasks = len([t for t in self.tasks if t.status == TaskStatus.FAILED])
        running_tasks = len([t for t in self.tasks if t.status == TaskStatus.RUNNING])

        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "running_tasks": running_tasks,
            "pending_tasks": total_tasks
            - completed_tasks
            - failed_tasks
            - running_tasks,
            "completion_percentage": (completed_tasks / total_tasks * 100)
            if total_tasks > 0
            else 0,
        }

    def mark_started(self) -> None:
        """Mark plan as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, final_result: Dict[str, Any]) -> None:
        """Mark plan as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.final_result = final_result

    def mark_failed(self) -> None:
        """Mark plan as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()


class TaskPlanningContext(BaseModel):
    """Context for task planning operations."""

    user_request: str = Field(..., description="Original user request")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

    # Extracted information
    requirements: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted requirements"
    )
    optimization_goals: List[str] = Field(
        default_factory=list, description="Optimization goals"
    )
    constraints: List[str] = Field(default_factory=list, description="Constraints")

    # Context data
    previous_results: Dict[str, Any] = Field(
        default_factory=dict, description="Previous execution results"
    )
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="User preferences"
    )

    # Planning metadata
    planning_strategy: str = Field(
        default="standard", description="Planning strategy to use"
    )
    max_tasks: int = Field(default=10, description="Maximum number of tasks in plan")

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)
