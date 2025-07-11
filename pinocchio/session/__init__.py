"""
Session module for Pinocchio multi-agent system.

This module provides session management functionality for tracking
multi-agent collaboration lifecycle, version tracking, optimization
iterations, and performance trends.
"""

from .manager import SessionManager
from .models.metadata import SessionMetadata
from .models.session import Session, SessionExport, SessionQuery, SessionStatus
from .utils import SessionUtils

__all__ = [
    "Session",
    "SessionQuery",
    "SessionExport",
    "SessionStatus",
    "SessionMetadata",
    "SessionManager",
    "SessionUtils",
]
