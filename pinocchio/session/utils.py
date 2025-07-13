"""
Session utilities for Pinocchio multi-agent system.

This module provides utility functions for session analysis, statistics,
and export functionality.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.file_utils import ensure_directory, safe_write_json
from .models.session import Session, SessionStatus


class SessionUtils:
    """Utility class for session analysis and management."""

    @staticmethod
    def analyze_session_performance(session: Session) -> Dict[str, Any]:
        """
        Analyze session performance metrics.

        Args:
            session: Session to analyze

        Returns:
            Performance analysis
        """
        analysis: Dict[str, Any] = {
            "total_runtime": session.runtime_seconds or 0,
            "total_interactions": len(session.agent_interactions),
            "total_iterations": len(session.optimization_iterations),
            "performance_points": len(session.performance_trend),
            "status": session.status,
        }

        # Analyze agent interactions
        agent_counts: Dict[str, int] = {}
        for interaction in session.agent_interactions:
            agent_type = interaction["agent_type"]
            agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
        analysis["agent_interaction_counts"] = agent_counts

        # Analyze performance trend
        if session.performance_trend:
            metrics = session.performance_trend[-1]["metrics"]
            analysis["latest_performance"] = metrics
            analysis[
                "performance_improvement"
            ] = SessionUtils._calculate_performance_improvement(session)

        return analysis

    @staticmethod
    def _calculate_performance_improvement(
        session: Session,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate performance improvement over iterations.

        Args:
            session: Session to analyze

        Returns:
            Performance improvement data
        """
        if len(session.performance_trend) < 2:
            return None

        first_metrics = session.performance_trend[0]["metrics"]
        latest_metrics = session.performance_trend[-1]["metrics"]

        improvement: Dict[str, Any] = {}
        for key in first_metrics:
            if key in latest_metrics and isinstance(first_metrics[key], (int, float)):
                first_val = first_metrics[key]
                latest_val = latest_metrics[key]
                if first_val != 0:
                    improvement[key] = {
                        "first": first_val,
                        "latest": latest_val,
                        "improvement_percent": ((latest_val - first_val) / first_val)
                        * 100,
                    }

        return improvement

    @staticmethod
    def generate_session_report(session: Session) -> Dict[str, Any]:
        """
        Generate a comprehensive session report.

        Args:
            session: Session to report on

        Returns:
            Session report
        """
        report = {
            "session_id": session.session_id,
            "task_description": session.task_description,
            "status": session.status,
            "creation_time": session.creation_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "runtime_seconds": session.runtime_seconds,
            "target_performance": session.target_performance,
            "optimization_summary": session.get_optimization_summary(),
            "performance_analysis": SessionUtils.analyze_session_performance(session),
            "version_references": {
                "memory_versions": len(session.memory_versions),
                "prompt_versions": len(session.prompt_versions),
                "knowledge_versions": len(session.knowledge_versions),
                "code_versions": len(session.code_version_ids),
            },
            "agent_interactions": session.agent_interactions,
            "optimization_iterations": session.optimization_iterations,
            "performance_trend": session.performance_trend,
        }

        return report

    @staticmethod
    def export_session_to_json(session: Session, output_path: str) -> str:
        """
        Export session to JSON file.

        Args:
            session: Session to export
            output_path: Output file path

        Returns:
            Path to exported file
        """
        report = SessionUtils.generate_session_report(session)

        output_file = Path(output_path)
        # Use utils function to ensure directory exists
        ensure_directory(output_file.parent)

        # Use utils function for safe JSON writing
        success = safe_write_json(report, output_file)
        if not success:
            raise RuntimeError(f"Failed to export session to {output_path}")

        return str(output_file)

    @staticmethod
    def compare_sessions(session1: Session, session2: Session) -> Dict[str, Any]:
        """
        Compare two sessions.

        Args:
            session1: First session
            session2: Second session

        Returns:
            Comparison results
        """
        comparison = {
            "runtime_comparison": {
                "session1_runtime": session1.runtime_seconds or 0,
                "session2_runtime": session2.runtime_seconds or 0,
                "difference": (session2.runtime_seconds or 0)
                - (session1.runtime_seconds or 0),
            },
            "interaction_comparison": {
                "session1_interactions": len(session1.agent_interactions),
                "session2_interactions": len(session2.agent_interactions),
                "difference": len(session2.agent_interactions)
                - len(session1.agent_interactions),
            },
            "iteration_comparison": {
                "session1_iterations": len(session1.optimization_iterations),
                "session2_iterations": len(session2.optimization_iterations),
                "difference": len(session2.optimization_iterations)
                - len(session1.optimization_iterations),
            },
            "performance_comparison": {
                "session1_performance_points": len(session1.performance_trend),
                "session2_performance_points": len(session2.performance_trend),
                "difference": len(session2.performance_trend)
                - len(session1.performance_trend),
            },
        }

        return comparison

    @staticmethod
    def validate_session_data(session: Session) -> Dict[str, Any]:
        """
        Validate session data integrity.

        Args:
            session: Session to validate

        Returns:
            Validation results
        """
        validation: Dict[str, Any] = {"is_valid": True, "errors": [], "warnings": []}

        # Check required fields
        if not session.task_description:
            validation["is_valid"] = False
            validation["errors"].append("Missing task description")

        if not session.session_id:
            validation["is_valid"] = False
            validation["errors"].append("Missing session ID")

        # Check data consistency
        if session.end_time and session.creation_time:
            if session.end_time < session.creation_time:
                validation["is_valid"] = False
                validation["errors"].append("End time is before creation time")

        # Check for orphaned references
        if session.memory_versions and not session.agent_interactions:
            validation["warnings"].append(
                "Memory versions exist but no agent interactions recorded"
            )

        if session.prompt_versions and not session.agent_interactions:
            validation["warnings"].append(
                "Prompt versions exist but no agent interactions recorded"
            )

        if session.knowledge_versions and not session.agent_interactions:
            validation["warnings"].append(
                "Knowledge versions exist but no agent interactions recorded"
            )

        return validation

    @staticmethod
    def get_session_statistics(sessions: List[Session]) -> Dict[str, Any]:
        """
        Get statistics for a list of sessions.

        Args:
            sessions: List of sessions

        Returns:
            Statistics
        """
        if not sessions:
            return {
                "total_sessions": 0,
                "status_distribution": {},
                "average_runtime": 0,
                "total_interactions": 0,
            }

        total_sessions = len(sessions)
        status_distribution: Dict[str, int] = {}
        total_runtime: float = 0.0
        total_interactions = 0

        for session in sessions:
            # Status distribution
            status = session.status
            status_distribution[status] = status_distribution.get(status, 0) + 1

            # Runtime
            if session.runtime_seconds:
                total_runtime += session.runtime_seconds

            # Interactions
            total_interactions += len(session.agent_interactions)

        return {
            "total_sessions": total_sessions,
            "status_distribution": status_distribution,
            "average_runtime": float(
                total_runtime / total_sessions if total_sessions > 0 else 0
            ),
            "total_interactions": total_interactions,
        }

    @staticmethod
    def filter_sessions_by_criteria(
        sessions: List[Session],
        status: Optional[SessionStatus] = None,
        agent_type: Optional[str] = None,
        min_runtime: Optional[float] = None,
        max_runtime: Optional[float] = None,
    ) -> List[Session]:
        """
        Filter sessions by criteria.

        Args:
            sessions: List of sessions
            status: Status filter
            agent_type: Agent type filter
            min_runtime: Minimum runtime filter
            max_runtime: Maximum runtime filter

        Returns:
            Filtered sessions
        """
        filtered_sessions = []

        for session in sessions:
            # Status filter
            if status and session.status != status:
                continue

            # Agent type filter
            if agent_type:
                session_agent_types = {
                    interaction["agent_type"]
                    for interaction in session.agent_interactions
                }
                if agent_type not in session_agent_types:
                    continue

            # Runtime filters
            if min_runtime and (session.runtime_seconds or 0) < min_runtime:
                continue
            if max_runtime and (session.runtime_seconds or 0) > max_runtime:
                continue

            filtered_sessions.append(session)

        return filtered_sessions
