"""
Base memory model for Pinocchio multi-agent system.

This module defines the base classes for all memory models used in the system.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseMemory(BaseModel):
    """
    Base class for all memory records.

    All memory records in the system inherit from this class, providing common
    fields like ID, timestamp, and session ID.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str  # Associated session ID
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_type: str  # generator, debugger, evaluator
    version_id: str  # Version ID for version management
    parent_version_id: Optional[str] = None  # Parent version for version tracking
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat(), uuid.UUID: lambda v: str(v)}
    )
