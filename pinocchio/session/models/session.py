"""
Session models for Pinocchio multi-agent system.

This module defines the models for managing session lifecycle, version tracking,
optimization iterations, and performance trends.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from pinocchio.memory.models.code import CodeMemory, CodeVersion
from pinocchio.session.context import set_current_session


class SessionStatus(str, Enum):
    """Session status enumeration."""

    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class Session(BaseModel):
    """
    Session model for tracking multi-agent collaboration lifecycle.

    Manages session lifecycle, version tracking, optimization iterations,
    and performance trends for high-performance code generation and optimization.
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str
    status: SessionStatus = SessionStatus.ACTIVE
    creation_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    runtime_seconds: Optional[float] = None

    # Version tracking for other modules
    memory_versions: List[str] = Field(default_factory=list)
    prompt_versions: List[str] = Field(default_factory=list)
    knowledge_versions: List[str] = Field(default_factory=list)

    # Optimization iteration history
    optimization_iterations: List[Dict[str, Any]] = Field(default_factory=list)
    # Structured performance metrics
    performance_trend: List[Dict[str, Any]] = Field(default_factory=list)
    # Current optimization targets
    target_performance: Optional[Dict[str, Any]] = None

    # Associated code versions (now managed directly)
    code_versions: Dict[str, CodeVersion] = Field(default_factory=dict)
    current_code_version_id: Optional[str] = None

    # Agent interaction history
    agent_interactions: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)

    def __init__(
        self, task_description: str, target_performance: Optional[Dict[str, Any]] = None
    ):
        """Initialize the session model."""
        self.task_description = task_description
        self.target_performance = target_performance

    def __del__(self):
        """Clean up resources when the session model is deleted."""
        pass  # No specific cleanup needed for this model

    def __enter__(self):
        """Enter the runtime context related to this object."""
        set_current_session(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        set_current_session(None)

    @classmethod
    def create_session(
        cls, task_description: str, target_performance: Optional[Dict[str, Any]] = None
    ) -> "Session":
        """
        Create a new session.

        Args:
            task_description: Description of the task
            target_performance: Target performance metrics

        Returns:
            New session instance
        """
        return cls(
            task_description=task_description, target_performance=target_performance
        )

    def add_agent_interaction(
        self, agent_type: str, interaction_data: Dict[str, Any]
    ) -> None:
        """
        Add an agent interaction to the session.

        Args:
            agent_type: Type of agent (generator/debugger/evaluator)
            interaction_data: Interaction data
        """
        interaction = {
            "agent_type": agent_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": interaction_data,
        }
        self.agent_interactions.append(interaction)

    def add_optimization_iteration(self, iteration_data: Dict[str, Any]) -> None:
        """
        Add an optimization iteration to the session.

        Args:
            iteration_data: Optimization iteration data
        """
        iteration = {
            "iteration_number": len(self.optimization_iterations) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "data": iteration_data,
        }
        self.optimization_iterations.append(iteration)

    def add_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add performance metrics to the trend.

        Args:
            metrics: Performance metrics data
        """
        trend_point = {"timestamp": datetime.utcnow().isoformat(), "metrics": metrics}
        self.performance_trend.append(trend_point)

    def add_version_reference(self, module: str, version_id: str) -> None:
        """
        Add a version reference for a module.

        Args:
            module: Module name (memory/prompt/knowledge)
            version_id: Version ID
        """
        if module == "memory":
            self.memory_versions.append(version_id)
        elif module == "prompt":
            self.prompt_versions.append(version_id)
        elif module == "knowledge":
            self.knowledge_versions.append(version_id)

    def add_code_version(self, code_version: CodeVersion) -> None:
        """Add a code version to the session."""
        self.code_versions[code_version.version_id] = code_version
        self.current_code_version_id = code_version.version_id

    def get_latest_code(self) -> str:
        """Get the latest code version."""
        if self.current_code_version_id:
            return self.code_versions[self.current_code_version_id].code
        return ""

    def get_code_version(self, version_id: str) -> Optional[CodeVersion]:
        """Get a specific code version."""
        return self.code_versions.get(version_id)

    def get_version_history(self) -> list:
        """Get the version history."""
        # Sort by time, with the latest first
        return sorted(
            self.code_versions.values(), key=lambda v: v.timestamp, reverse=True
        )

    def complete_session(self) -> None:
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.end_time = datetime.utcnow()
        if self.creation_time:
            self.runtime_seconds = (self.end_time - self.creation_time).total_seconds()

    def fail_session(self, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark session as failed, optionally with error details."""
        self.status = SessionStatus.FAILED
        self.end_time = datetime.utcnow()
        if self.creation_time:
            self.runtime_seconds = (self.end_time - self.creation_time).total_seconds()
        if error_details:
            self.metadata["error_details"] = error_details

    def pause_session(self) -> None:
        """Pause the session."""
        self.status = SessionStatus.PAUSED

    def resume_session(self) -> None:
        """Resume the session."""
        self.status = SessionStatus.ACTIVE

    def get_latest_optimization_iteration(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest optimization iteration.

        Returns:
            Latest optimization iteration or None
        """
        if self.optimization_iterations:
            return self.optimization_iterations[-1]
        return None

    def get_latest_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest performance metrics.

        Returns:
            Latest performance metrics or None
        """
        if self.performance_trend:
            return self.performance_trend[-1]
        return None

    def get_agent_interactions_by_type(self, agent_type: str) -> List[Dict[str, Any]]:
        """
        Get agent interactions by type.

        Args:
            agent_type: Type of agent

        Returns:
            List of interactions for the agent type
        """
        return [
            interaction
            for interaction in self.agent_interactions
            if interaction["agent_type"] == agent_type
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for serialization, converting datetime to ISO string.

        Returns:
            Dictionary representation of the session
        """
        import datetime
        from typing import Union

        def convert(
            obj: Union[datetime.datetime, list, dict, Any]
        ) -> Union[str, list, dict, Any]:
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj

        result = convert(self.model_dump())
        return result  # type: ignore

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """
        Create a Session from dictionary data.

        Args:
            data: Dictionary data

        Returns:
            Session instance
        """
        # Convert ISO string timestamps back to datetime objects
        if "creation_time" in data and isinstance(data["creation_time"], str):
            data["creation_time"] = datetime.fromisoformat(data["creation_time"])
        if "end_time" in data and isinstance(data["end_time"], str):
            data["end_time"] = datetime.fromisoformat(data["end_time"])

        return cls(**data)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Return optimization summary."""
        return {
            "total_iterations": len(self.optimization_iterations),
            "total_agent_interactions": len(self.agent_interactions),
            "performance_trend_length": len(self.performance_trend),
            "latest_iteration": self.get_latest_optimization_iteration(),
            "latest_performance": self.get_latest_performance_metrics(),
            "target_performance": self.target_performance,
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
        }


class SessionQuery(BaseModel):
    """
    Query model for searching sessions.
    """

    status: Optional[SessionStatus] = None
    agent_type: Optional[str] = None
    date_range: Optional[Dict[str, datetime]] = None
    limit: int = 10

    model_config = ConfigDict(use_enum_values=True)


class SessionExport(BaseModel):
    """Session export model."""

    session: Session
    memory_data: Optional[Dict[str, Any]] = None
    prompt_data: Optional[Dict[str, Any]] = None
    knowledge_data: Optional[Dict[str, Any]] = None
    code_versions: Optional[List[Dict[str, Any]]] = None
    export_timestamp: datetime = Field(default_factory=datetime.utcnow)
