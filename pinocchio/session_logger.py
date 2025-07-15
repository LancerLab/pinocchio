"""Session logger for Pinocchio multi-agent system."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pinocchio.config.config_manager import ConfigManager
from pinocchio.memory.manager import MemoryManager

from .utils.file_utils import ensure_directory, safe_read_json, safe_write_json

logger = logging.getLogger(__name__)


class SessionLogger:
    """Session logger for managing session lifecycle and logging."""

    def __init__(
        self, user_prompt: str, sessions_dir: str = None, logs_dir: str = None
    ):
        """
        Initialize session logger.

        Args:
            user_prompt: Initial user prompt that started the session
            sessions_dir: Directory to store session files
        """
        config = ConfigManager()
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.user_prompt = user_prompt
        self.sessions_dir = Path(sessions_dir or config.config.storage.sessions_path)
        self.logs_dir = Path(logs_dir or config.get_logs_path())
        self.created_at = datetime.utcnow()
        self.completed_at: Optional[datetime] = None

        # Session data
        self.summary_logs: List[str] = []
        self.communication_logs: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

        # Ensure sessions directory exists
        ensure_directory(self.sessions_dir)

        logger.info(f"Session created: {self.session_id}")

    def log_summary(self, message: str) -> str:
        """
        Log a summary message.

        Args:
            message: Summary message to log

        Returns:
            Formatted log message with session ID
        """
        timestamp = datetime.utcnow().isoformat()
        formatted_message = f"[{self.session_id}] {message}"

        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "formatted": formatted_message,
        }

        self.summary_logs.append(log_entry)
        logger.info(formatted_message)

        return formatted_message

    def log_communication(
        self,
        step_id: str,
        agent_type: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
    ) -> None:
        """
        Log agent communication.

        Args:
            step_id: Workflow step ID
            agent_type: Type of agent
            request: Agent request data
            response: Agent response data
        """
        timestamp = datetime.utcnow().isoformat()

        communication_entry = {
            "timestamp": timestamp,
            "step_id": step_id,
            "agent_type": agent_type,
            "request": request,
            "response": response,
        }

        self.communication_logs.append(communication_entry)
        logger.debug(f"Communication logged for step {step_id} ({agent_type})")

    def update_context(self, key: str, value: Any) -> None:
        """
        Update session context.

        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value
        logger.debug(f"Context updated: {key}")

    def get_context(self) -> Dict[str, Any]:
        """
        Get current session context.

        Returns:
            Session context dictionary
        """
        return {
            "session_id": self.session_id,
            "user_prompt": self.user_prompt,
            "created_at": self.created_at.isoformat(),
            "summary_count": len(self.summary_logs),
            "communication_count": len(self.communication_logs),
            "custom_context": self.context,
        }

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to session.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        logger.debug(f"Metadata added: {key}")

    def complete_session(self, status: str = "completed") -> None:
        """
        Mark the session as completed.

        Args:
            status: The final status of the session ("completed" or "failed")
        """
        self.status = status
        self.completed_at = datetime.utcnow()
        if hasattr(self, "creation_time") and self.creation_time:
            self.runtime_seconds = self.completed_at - self.creation_time
        # === Added: Export all memories ===
        try:
            memory_manager = MemoryManager()
            agent_memories = memory_manager.get_agent_memories(self.session_id)
            mem_dir = Path("memories") / self.session_id
            mem_dir.mkdir(parents=True, exist_ok=True)
            mem_file = mem_dir / "agent_memories.json"
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(
                    [m.model_dump() for m in agent_memories],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            import logging

            logging.warning(
                f"Failed to export agent memories for session {self.session_id}: {e}"
            )

        self.add_metadata("status", status)
        self.add_metadata(
            "duration_seconds", (self.completed_at - self.created_at).total_seconds()
        )

        logger.info(f"Session completed: {self.session_id} ({status})")

    def save_to_file(self, filename: Optional[str] = None) -> Path:
        """
        Save session data to file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.session_id}_{timestamp}.json"

        file_path = self.sessions_dir / filename

        session_data = {
            "session_id": self.session_id,
            "user_prompt": self.user_prompt,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "summary_logs": self.summary_logs,
            "communication_logs": self.communication_logs,
            "context": self.context,
            "metadata": self.metadata,
        }

        if safe_write_json(session_data, file_path):
            logger.info(f"Session saved to: {file_path}")
            return file_path
        else:
            raise Exception(f"Failed to save session to: {file_path}")

    def get_latest_agent_output(
        self, agent_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest output from an agent.

        Args:
            agent_type: Optional agent type filter

        Returns:
            Latest agent output or None
        """
        for comm_log in reversed(self.communication_logs):
            if not agent_type or comm_log["agent_type"] == agent_type:
                return comm_log.get("response", {}).get("output")

        return None

    def get_agent_history(self, agent_type: str) -> List[Dict[str, Any]]:
        """
        Get communication history for specific agent type.

        Args:
            agent_type: Agent type to filter by

        Returns:
            List of communications for the agent type
        """
        return [
            comm_log
            for comm_log in self.communication_logs
            if comm_log["agent_type"] == agent_type
        ]

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get session summary.

        Returns:
            Session summary dictionary
        """
        agent_counts = {}
        for comm_log in self.communication_logs:
            agent_type = comm_log["agent_type"]
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1

        duration = None
        if self.completed_at:
            duration = (self.completed_at - self.created_at).total_seconds()

        return {
            "session_id": self.session_id,
            "user_prompt": self.user_prompt,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": duration,
            "summary_log_count": len(self.summary_logs),
            "communication_count": len(self.communication_logs),
            "agent_usage": agent_counts,
            "status": self.metadata.get("status", "active"),
            "metadata": self.metadata,
        }

    @classmethod
    def load_from_file(cls, file_path: str) -> "SessionLogger":
        """
        Load session from file.

        Args:
            file_path: Path to session file

        Returns:
            SessionLogger instance

        Raises:
            Exception: If file cannot be loaded
        """
        session_data = safe_read_json(file_path)
        if not session_data:
            raise Exception(f"Could not load session from: {file_path}")

        # Create session instance
        session = cls.__new__(cls)
        session.session_id = session_data["session_id"]
        session.user_prompt = session_data["user_prompt"]
        session.created_at = datetime.fromisoformat(session_data["created_at"])

        if session_data.get("completed_at"):
            session.completed_at = datetime.fromisoformat(session_data["completed_at"])
        else:
            session.completed_at = None

        session.summary_logs = session_data.get("summary_logs", [])
        session.communication_logs = session_data.get("communication_logs", [])
        session.context = session_data.get("context", {})
        session.metadata = session_data.get("metadata", {})

        # Set sessions directory based on file path
        session.sessions_dir = Path(file_path).parent

        logger.info(f"Session loaded: {session.session_id}")
        return session

    @classmethod
    def list_sessions(cls, sessions_dir: str = "./sessions") -> List[Dict[str, Any]]:
        """
        List all sessions in directory.

        Args:
            sessions_dir: Directory containing session files

        Returns:
            List of session summaries
        """
        sessions_dir = Path(sessions_dir)

        if not sessions_dir.exists():
            return []

        sessions = []
        for file_path in sessions_dir.glob("session_*.json"):
            try:
                session_data = safe_read_json(file_path)
                if session_data:
                    summary = {
                        "session_id": session_data["session_id"],
                        "user_prompt": session_data["user_prompt"],
                        "created_at": session_data["created_at"],
                        "completed_at": session_data.get("completed_at"),
                        "status": session_data.get("metadata", {}).get(
                            "status", "unknown"
                        ),
                        "file_path": str(file_path),
                    }
                    sessions.append(summary)
            except Exception as e:
                logger.warning(f"Could not read session file {file_path}: {e}")

        # Sort by creation time (newest first)
        sessions.sort(key=lambda s: s["created_at"], reverse=True)
        return sessions
