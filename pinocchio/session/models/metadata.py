"""
Session models for Pinocchio.

This module defines the models for tracking session metadata.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SessionMetadata(BaseModel):
    """
    Session metadata.

    Contains metadata about a session, including its status, creation time,
    task description, and other relevant information.
    """

    session_id: str
    name: str
    creation_time: datetime = Field(default_factory=datetime.utcnow)
    task_description: str
    status: str = "active"  # active, completed, failed
    end_time: Optional[datetime] = None
    runtime_seconds: Optional[float] = None
    user_inputs: Dict[str, Any] = Field(default_factory=dict)
    system_info: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

    def mark_completed(self, status: str = "completed") -> None:
        """
        Mark the session as completed.

        Args:
            status: The final status of the session ("completed" or "failed")
        """
        self.status = status
        self.end_time = datetime.utcnow()
        if self.creation_time:
            self.runtime_seconds = (self.end_time - self.creation_time).total_seconds()

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the session.

        Args:
            tag: The tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> bool:
        """
        Remove a tag from the session.

        Args:
            tag: The tag to remove

        Returns:
            True if the tag was removed, False if it wasn't present
        """
        if tag in self.tags:
            self.tags.remove(tag)
            return True
        return False

    def update_user_input(self, key: str, value: Any) -> None:
        """
        Update a user input value.

        Args:
            key: The input key
            value: The input value
        """
        self.user_inputs[key] = value

    def update_system_info(self, key: str, value: Any) -> None:
        """
        Update a system info value.

        Args:
            key: The info key
            value: The info value
        """
        self.system_info[key] = value

    @classmethod
    def create_new_session(
        cls, task_description: str, name: str, tags: Optional[List[str]] = None
    ) -> "SessionMetadata":
        """
        Create a new session with a generated session ID.

        Args:
            task_description: Description of the session's task
            name: Name of the session
            tags: Optional list of tags for the session

        Returns:
            A new SessionMetadata instance
        """
        import platform
        import sys
        import uuid

        session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Create basic system info
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return cls(
            session_id=session_id,
            name=name,
            task_description=task_description,
            tags=tags or [],
            system_info=system_info,
        )
