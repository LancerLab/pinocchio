"""
Code version models for Pinocchio multi-agent system.

This module defines the models for tracking code versions and changes.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CodeVersion(BaseModel):
    """
    Code version record.

    Represents a specific version of code in the system, with metadata about
    its origin, parent version, and description.
    """

    version_id: str  # Version ID, automatically generated
    session_id: str
    code: str  # Code content
    language: str
    kernel_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_agent: str  # Agent that produced the code: generator, debugger, or optimizer
    parent_version_id: Optional[str] = None  # Parent version ID
    description: str = ""  # Version description
    optimization_techniques: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata

    def __init__(
        self,
        version_id: str,
        session_id: str,
        code: str,
        language: str,
        kernel_type: str,
        source_agent: str,
        description: str = "",
        parent_version_id: Optional[str] = None,
        optimization_techniques: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Create a new code version.

        Args:
            version_id: Version ID
            session_id: Session ID
            code: Code content
            language: Programming language
            kernel_type: Type of kernel (cpu, gpu, etc.)
            source_agent: Agent that created this version
            description: Version description
            parent_version_id: Parent version ID
            optimization_techniques: List of optimization techniques used
            hyperparameters: Optimization hyperparameters
            metadata: Additional metadata
        """
        super().__init__(
            version_id=version_id,
            session_id=session_id,
            code=code,
            language=language,
            kernel_type=kernel_type,
            source_agent=source_agent,
            description=description,
            parent_version_id=parent_version_id,
            optimization_techniques=optimization_techniques or [],
            hyperparameters=hyperparameters or {},
            metadata=metadata or {},
            **kwargs,
        )

    @classmethod
    def create_new_version(
        cls,
        session_id: str,
        code: str,
        language: str,
        kernel_type: str,
        source_agent: str,
        description: str = "",
        parent_version_id: Optional[str] = None,
        optimization_techniques: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CodeVersion":
        """Create a new code version."""
        return cls(
            version_id=str(uuid.uuid4()),
            session_id=session_id,
            code=code,
            language=language,
            kernel_type=kernel_type,
            source_agent=source_agent,
            description=description,
            parent_version_id=parent_version_id,
            optimization_techniques=optimization_techniques or [],
            hyperparameters=hyperparameters or {},
            metadata=metadata or {},
        )

    def get_diff(self, other: "CodeVersion") -> Dict[str, Any]:
        """
        Get the difference between this version and another version.

        Args:
            other: Another CodeVersion to compare with

        Returns:
            A dictionary containing the differences
        """
        # This is a simple implementation that just notes if they're different
        # A real implementation would use a diff algorithm
        return {
            "is_different": self.code != other.code,
            "this_version": self.version_id,
            "other_version": other.version_id,
            # Additional diff information would be added here
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "code_content": self.code,
            "commit_message": self.description,
            "created_at": self.timestamp.isoformat(),
            "created_by": self.source_agent,
            "optimization_techniques": self.optimization_techniques,
            "hyperparameters": self.hyperparameters,
            "performance_metrics": self.metadata,
            "change_summary": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeVersion":
        """Create from dictionary."""
        # Convert datetime string back to datetime object
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return cls(**data)


class CodeMemory(BaseModel):
    """
    Code version history.

    Maintains a history of code versions for a session, including the current version.
    """

    session_id: str  # Session ID
    versions: Dict[str, CodeVersion] = Field(
        default_factory=dict
    )  # version_id -> version object
    current_version_id: Optional[str] = None  # Current version ID

    def add_version(self, version: CodeVersion) -> str:
        """
        Add a new code version to the history.

        Args:
            version: The CodeVersion to add

        Returns:
            The version ID of the added version
        """
        self.versions[version.version_id] = version
        self.current_version_id = version.version_id
        return version.version_id

    def get_current_version(self) -> Optional[CodeVersion]:
        """
        Get the current code version.

        Returns:
            The current CodeVersion, or None if there is no current version
        """
        if self.current_version_id:
            return self.versions.get(self.current_version_id)
        return None

    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get version history as a list of dictionaries.

        Returns:
            A list of dictionaries containing version information, sorted by timestamp
        """
        # Sort by timestamp (newest first)
        sorted_versions = sorted(
            self.versions.values(), key=lambda v: v.timestamp, reverse=True
        )
        return [
            {
                "version_id": v.version_id,
                "source_agent": v.source_agent,
                "description": v.description,
                "timestamp": v.timestamp.isoformat(),
                "optimization_techniques": v.optimization_techniques,
                "is_current": v.version_id == self.current_version_id,
            }
            for v in sorted_versions
        ]

    def set_current_version(self, version_id: str) -> bool:
        """
        Set the current version.

        Args:
            version_id: The version ID to set as current

        Returns:
            True if successful, False if the version doesn't exist
        """
        if version_id in self.versions:
            self.current_version_id = version_id
            return True
        return False

    def get_version(self, version_id: str) -> Optional[CodeVersion]:
        """Get a code version by version_id."""
        return self.versions.get(version_id)
