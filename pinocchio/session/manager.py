"""
Session manager for Pinocchio multi-agent system.

Manages session lifecycle, version tracking, optimization iterations,
and performance trends for high-performance code generation and optimization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models.session import Session, SessionExport, SessionQuery


class SessionManager:
    """
    Session manager for multi-agent collaboration lifecycle.

    Provides functionality for creating, managing, and tracking sessions
    with version tracking, optimization iterations, and performance trends.
    """

    def __init__(self, store_dir: str = "./session_store"):
        """
        Initialize the session manager.

        Args:
            store_dir: Directory for storing session data
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for active sessions
        self.active_sessions: Dict[str, Session] = {}

    def create_session(
        self, task_description: str, target_performance: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Create a new session.

        Args:
            task_description: Description of the task
            target_performance: Target performance metrics

        Returns:
            Created session
        """
        session = Session.create_session(task_description, target_performance)
        self.active_sessions[session.session_id] = session
        self._persist_session(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session or None if not found
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # Load from storage
        session = self._load_session(session_id)
        if session:
            self.active_sessions[session_id] = session
        return session

    def list_sessions(self, query: Optional[SessionQuery] = None) -> List[Session]:
        """
        List sessions based on query criteria.

        Args:
            query: Query criteria

        Returns:
            List of matching sessions
        """
        sessions = []

        # Load all session files
        for session_file in self.store_dir.glob("*.json"):
            try:
                session_id = session_file.stem
                session = self._load_session(session_id)
                if session:
                    sessions.append(session)
            except Exception as e:
                print(f"Error loading session {session_file}: {e}")

        # Apply filters
        if query:
            filtered_sessions = []
            for session in sessions:
                if self._matches_query(session, query):
                    filtered_sessions.append(session)
            sessions = filtered_sessions

        return sessions

    def complete_session(self, session_id: str) -> bool:
        """
        Complete a session.

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.complete_session()
            self._persist_session(session)
            return True
        return False

    def fail_session(
        self, session_id: str, error_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a session as failed.

        Args:
            session_id: Session ID
            error_details: Error details

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.fail_session(error_details)
            self._persist_session(session)
            return True
        return False

    def pause_session(self, session_id: str) -> bool:
        """
        Pause a session.

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.pause_session()
            self._persist_session(session)
            return True
        return False

    def resume_session(self, session_id: str) -> bool:
        """
        Resume a session.

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.resume_session()
            self._persist_session(session)
            return True
        return False

    def add_agent_interaction(
        self, session_id: str, agent_type: str, interaction_data: Dict[str, Any]
    ) -> bool:
        """
        Add an agent interaction to a session.

        Args:
            session_id: Session ID
            agent_type: Type of agent
            interaction_data: Interaction data

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.add_agent_interaction(agent_type, interaction_data)
            self._persist_session(session)
            return True
        return False

    def add_optimization_iteration(
        self, session_id: str, iteration_data: Dict[str, Any]
    ) -> bool:
        """
        Add an optimization iteration to a session.

        Args:
            session_id: Session ID
            iteration_data: Optimization iteration data

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.add_optimization_iteration(iteration_data)
            self._persist_session(session)
            return True
        return False

    def add_performance_metrics(self, session_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Add performance metrics to a session.

        Args:
            session_id: Session ID
            metrics: Performance metrics

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.add_performance_metrics(metrics)
            self._persist_session(session)
            return True
        return False

    def add_version_reference(
        self, session_id: str, module: str, version_id: str
    ) -> bool:
        """
        Add a version reference to a session.

        Args:
            session_id: Session ID
            module: Module name
            version_id: Version ID

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.add_version_reference(module, version_id)
            self._persist_session(session)
            return True
        return False

    def add_code_version(self, session_id: str, code_version_id: str) -> bool:
        """
        Add a code version to a session.

        Args:
            session_id: Session ID
            code_version_id: Code version ID

        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if session:
            session.add_code_version(code_version_id)
            self._persist_session(session)
            return True
        return False

    def get_optimization_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get optimization summary for a session.

        Args:
            session_id: Session ID

        Returns:
            Optimization summary or None
        """
        session = self.get_session(session_id)
        if session:
            return session.get_optimization_summary()
        return None

    def get_agent_interactions(
        self, session_id: str, agent_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get agent interactions for a session.

        Args:
            session_id: Session ID
            agent_type: Optional agent type filter

        Returns:
            List of agent interactions
        """
        session = self.get_session(session_id)
        if session:
            if agent_type:
                return session.get_agent_interactions_by_type(agent_type)
            return session.agent_interactions
        return []

    def get_performance_trend(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get performance trend for a session.

        Args:
            session_id: Session ID

        Returns:
            Performance trend data
        """
        session = self.get_session(session_id)
        if session:
            return session.performance_trend
        return []

    def export_session(
        self, session_id: str, include_module_data: bool = False
    ) -> Optional[SessionExport]:
        """
        Export a session with optional module data.

        Args:
            session_id: Session ID
            include_module_data: Whether to include module data

        Returns:
            Session export or None
        """
        session = self.get_session(session_id)
        if not session:
            return None

        export = SessionExport(session=session)

        if include_module_data:
            # This would integrate with memory, prompt, knowledge modules
            # For now, we'll leave these as None
            pass

        return export

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        # Remove from active sessions
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        # Delete session file
        session_file = self.store_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get session manager statistics.

        Returns:
            Statistics dictionary
        """
        sessions = self.list_sessions()

        # Calculate statistics
        status_counts: Dict[str, int] = {}
        total_runtime: float = 0.0
        valid_runtime_count: int = 0
        total_interactions = 0
        total_iterations = 0
        total_performance_points = 0

        for session in sessions:
            status = (
                session.status.value
                if hasattr(session.status, "value")
                else session.status
            )
            status_counts[status] = status_counts.get(status, 0) + 1

            if session.runtime_seconds:
                total_runtime += session.runtime_seconds
                valid_runtime_count += 1

            total_interactions += len(session.agent_interactions)
            total_iterations += len(session.optimization_iterations)
            total_performance_points += len(session.performance_trend)

        return {
            "total_sessions": len(sessions),
            "active_sessions": len(self.active_sessions),
            "status_distribution": status_counts,
            "average_runtime": float(
                total_runtime / valid_runtime_count if valid_runtime_count > 0 else 0
            ),
            "total_agent_interactions": total_interactions,
            "total_iterations": total_iterations,
            "total_optimization_iterations": total_iterations,
            "total_performance_points": total_performance_points,
        }

    def analyze_session_performance(self, session_id: str) -> dict:
        """
        Analyze session performance (stub for test compatibility).
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        # Calculate basic statistics
        total_interactions = len(session.agent_interactions)
        total_iterations = len(session.optimization_iterations)
        performance_points = len(session.performance_trend)

        # Count agent interactions by type
        agent_interaction_counts = {}
        for interaction in session.agent_interactions:
            agent_type = interaction.get("agent_type", "unknown")
            agent_interaction_counts[agent_type] = (
                agent_interaction_counts.get(agent_type, 0) + 1
            )

        return {
            "total_interactions": total_interactions,
            "total_iterations": total_iterations,
            "performance_points": performance_points,
            "agent_interaction_counts": agent_interaction_counts,
        }

    def generate_session_report(self, session) -> dict:
        """
        Generate a comprehensive session report.

        Args:
            session: Session object

        Returns:
            Session report dictionary
        """
        return {
            "session_id": session.session_id,
            "task_description": session.task_description,
            "status": session.status.value
            if hasattr(session.status, "value")
            else session.status,
            "optimization_summary": {
                "total_iterations": len(session.optimization_iterations),
                "total_agent_interactions": len(session.agent_interactions),
                "performance_trend_length": len(session.performance_trend),
            },
            "performance_analysis": {
                "total_performance_points": len(session.performance_trend),
                "latest_metrics": session.performance_trend[-1]
                if session.performance_trend
                else None,
            },
            "version_references": {
                "memory_versions": len(session.memory_versions),
                "knowledge_versions": len(session.knowledge_versions),
                "prompt_versions": len(session.prompt_versions),
            },
        }

    def _persist_session(self, session: Session) -> None:
        """Persist session to storage."""
        session_file = self.store_dir / f"{session.session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session.to_dict(), f, indent=2)

    def _load_session(self, session_id: str) -> Optional[Session]:
        """Load session from storage."""
        session_file = self.store_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r") as f:
                session_data = json.load(f)
            return Session.from_dict(session_data)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def _matches_query(self, session: Session, query: SessionQuery) -> bool:
        """Check if session matches query criteria."""
        # Status filter
        if query.status and session.status != query.status:
            return False

        # Date range filter
        if (
            query.date_range
            and "start" in query.date_range
            and session.creation_time < query.date_range["start"]
        ):
            return False
        if (
            query.date_range
            and "end" in query.date_range
            and session.creation_time > query.date_range["end"]
        ):
            return False

        # Agent type filter
        if query.agent_type:
            session_agent_types = {
                interaction["agent_type"] for interaction in session.agent_interactions
            }
            if query.agent_type not in session_agent_types:
                return False

        return True
