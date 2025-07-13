"""
Tests for session manager functionality.
"""

import shutil
import tempfile

import pytest

from pinocchio.session import (
    Session,
    SessionManager,
    SessionQuery,
    SessionStatus,
    SessionUtils,
)
from tests.utils import create_test_session


class TestSession:
    """Test Session model functionality."""

    def test_create_session(self):
        """Test session creation."""
        session = create_test_session(task_description="Test task")
        assert session.task_description == "Test task"
        assert session.status == SessionStatus.ACTIVE
        assert session.session_id is not None
        assert session.creation_time is not None

    def test_add_agent_interaction(self):
        """Test adding agent interaction."""
        session = create_test_session(task_description="Test task")
        interaction_data = {"input": "test", "output": "result"}

        session.add_agent_interaction("generator", interaction_data)

        assert len(session.agent_interactions) == 1
        interaction = session.agent_interactions[0]
        assert interaction["agent_type"] == "generator"
        assert interaction["data"] == interaction_data
        assert "timestamp" in interaction

    def test_add_optimization_iteration(self):
        """Test adding optimization iteration."""
        session = create_test_session(task_description="Test task")
        iteration_data = {"optimization": "test"}

        session.add_optimization_iteration(iteration_data)

        assert len(session.optimization_iterations) == 1
        iteration = session.optimization_iterations[0]
        assert iteration["iteration_number"] == 1
        assert iteration["data"] == iteration_data
        assert "timestamp" in iteration

    def test_add_performance_metrics(self):
        """Test adding performance metrics."""
        session = create_test_session(task_description="Test task")
        metrics = {"speed": 100, "memory": 50}

        session.add_performance_metrics(metrics)

        assert len(session.performance_trend) == 1
        trend_point = session.performance_trend[0]
        assert trend_point["metrics"] == metrics
        assert "timestamp" in trend_point

    def test_add_version_reference(self):
        """Test adding version references."""
        session = create_test_session(task_description="Test task")

        session.add_version_reference("memory", "mem_v1")
        session.add_version_reference("prompt", "prompt_v1")
        session.add_version_reference("knowledge", "knowledge_v1")

        assert "mem_v1" in session.memory_versions
        assert "prompt_v1" in session.prompt_versions
        assert "knowledge_v1" in session.knowledge_versions

    def test_add_code_version(self):
        """Test adding code version."""
        session = create_test_session(task_description="Test task")

        session.add_code_version("code_v1")

        assert "code_v1" in session.code_version_ids

    def test_complete_session(self):
        """Test completing session."""
        session = create_test_session(task_description="Test task")

        session.complete_session()

        assert session.status == SessionStatus.COMPLETED
        assert session.end_time is not None
        assert session.runtime_seconds is not None

    def test_fail_session(self):
        """Test failing session."""
        session = create_test_session(task_description="Test task")
        error_details = {"error": "test error"}

        session.fail_session(error_details)

        assert session.status == SessionStatus.FAILED
        assert session.end_time is not None
        assert session.runtime_seconds is not None
        assert session.metadata["error_details"] == error_details

    def test_pause_resume_session(self):
        """Test pausing and resuming session."""
        session = create_test_session(task_description="Test task")

        session.pause_session()
        assert session.status == SessionStatus.PAUSED

        session.resume_session()
        assert session.status == SessionStatus.ACTIVE

    def test_get_latest_optimization_iteration(self):
        """Test getting latest optimization iteration."""
        session = create_test_session(task_description="Test task")

        # No iterations yet
        assert session.get_latest_optimization_iteration() is None

        # Add iterations
        session.add_optimization_iteration({"iter1": "data1"})
        session.add_optimization_iteration({"iter2": "data2"})

        latest = session.get_latest_optimization_iteration()
        assert latest["data"] == {"iter2": "data2"}
        assert latest["iteration_number"] == 2

    def test_get_latest_performance_metrics(self):
        """Test getting latest performance metrics."""
        session = create_test_session(task_description="Test task")

        # No metrics yet
        assert session.get_latest_performance_metrics() is None

        # Add metrics
        session.add_performance_metrics({"speed": 100})
        session.add_performance_metrics({"speed": 200})

        latest = session.get_latest_performance_metrics()
        assert latest["metrics"] == {"speed": 200}

    def test_get_agent_interactions_by_type(self):
        """Test getting agent interactions by type."""
        session = create_test_session(task_description="Test task")

        session.add_agent_interaction("generator", {"data": "gen1"})
        session.add_agent_interaction("debugger", {"data": "debug1"})
        session.add_agent_interaction("generator", {"data": "gen2"})

        generator_interactions = session.get_agent_interactions_by_type("generator")
        assert len(generator_interactions) == 2

        debugger_interactions = session.get_agent_interactions_by_type("debugger")
        assert len(debugger_interactions) == 1

    def test_get_optimization_summary(self):
        """Test getting optimization summary."""
        session = create_test_session(task_description="Test task")
        session.target_performance = {"speed": 1000}

        session.add_agent_interaction("generator", {"data": "test"})
        session.add_optimization_iteration({"optim": "test"})
        session.add_performance_metrics({"speed": 500})

        summary = session.get_optimization_summary()

        assert summary["total_iterations"] == 1
        assert summary["total_agent_interactions"] == 1
        assert summary["performance_trend_length"] == 1
        assert summary["target_performance"] == {"speed": 1000}
        assert summary["status"] == "active"


class TestSessionManager:
    """Test SessionManager functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def session_manager(self, temp_dir):
        """Create session manager for testing."""
        return SessionManager(store_dir=temp_dir)

    def test_create_session(self, session_manager):
        """Test creating session."""
        session = session_manager.create_session("Test task")

        assert session.task_description == "Test task"
        assert session.status == SessionStatus.ACTIVE
        assert session.session_id in session_manager.active_sessions

    def test_get_session(self, session_manager):
        """Test getting session."""
        session = session_manager.create_session("Test task")
        session_id = session.session_id

        retrieved_session = session_manager.get_session(session_id)
        assert retrieved_session.session_id == session_id
        assert retrieved_session.task_description == "Test task"

    def test_get_nonexistent_session(self, session_manager):
        """Test getting non-existent session."""
        session = session_manager.get_session("nonexistent")
        assert session is None

    def test_complete_session(self, session_manager):
        """Test completing session."""
        session = session_manager.create_session("Test task")

        result = session_manager.complete_session(session.session_id)
        assert result is True

        completed_session = session_manager.get_session(session.session_id)
        assert completed_session.status == SessionStatus.COMPLETED

    def test_fail_session(self, session_manager):
        """Test failing session."""
        session = session_manager.create_session("Test task")
        error_details = {"error": "test"}

        result = session_manager.fail_session(session.session_id, error_details)
        assert result is True

        failed_session = session_manager.get_session(session.session_id)
        assert failed_session.status == SessionStatus.FAILED
        assert failed_session.metadata["error_details"] == error_details

    def test_pause_resume_session(self, session_manager):
        """Test pausing and resuming session."""
        session = session_manager.create_session("Test task")

        # Pause
        result = session_manager.pause_session(session.session_id)
        assert result is True

        paused_session = session_manager.get_session(session.session_id)
        assert paused_session.status == SessionStatus.PAUSED

        # Resume
        result = session_manager.resume_session(session.session_id)
        assert result is True

        resumed_session = session_manager.get_session(session.session_id)
        assert resumed_session.status == SessionStatus.ACTIVE

    def test_add_agent_interaction(self, session_manager):
        """Test adding agent interaction."""
        session = session_manager.create_session("Test task")
        interaction_data = {"input": "test", "output": "result"}

        result = session_manager.add_agent_interaction(
            session.session_id, "generator", interaction_data
        )
        assert result is True

        updated_session = session_manager.get_session(session.session_id)
        assert len(updated_session.agent_interactions) == 1
        assert updated_session.agent_interactions[0]["agent_type"] == "generator"

    def test_add_optimization_iteration(self, session_manager):
        """Test adding optimization iteration."""
        session = session_manager.create_session("Test task")
        iteration_data = {"optimization": "test"}

        result = session_manager.add_optimization_iteration(
            session.session_id, iteration_data
        )
        assert result is True

        updated_session = session_manager.get_session(session.session_id)
        assert len(updated_session.optimization_iterations) == 1
        assert updated_session.optimization_iterations[0]["data"] == iteration_data

    def test_add_performance_metrics(self, session_manager):
        """Test adding performance metrics."""
        session = session_manager.create_session("Test task")
        metrics = {"speed": 100, "memory": 50}

        result = session_manager.add_performance_metrics(session.session_id, metrics)
        assert result is True

        updated_session = session_manager.get_session(session.session_id)
        assert len(updated_session.performance_trend) == 1
        assert updated_session.performance_trend[0]["metrics"] == metrics

    def test_add_version_reference(self, session_manager):
        """Test adding version reference."""
        session = session_manager.create_session("Test task")

        result = session_manager.add_version_reference(
            session.session_id, "memory", "mem_v1"
        )
        assert result is True

        updated_session = session_manager.get_session(session.session_id)
        assert "mem_v1" in updated_session.memory_versions

    def test_add_code_version(self, session_manager):
        """Test adding code version."""
        session = session_manager.create_session("Test task")

        result = session_manager.add_code_version(session.session_id, "code_v1")
        assert result is True

        updated_session = session_manager.get_session(session.session_id)
        assert "code_v1" in updated_session.code_version_ids

    def test_get_optimization_summary(self, session_manager):
        """Test getting optimization summary."""
        session = session_manager.create_session("Test task")
        session.target_performance = {"speed": 1000}

        session_manager.add_agent_interaction(
            session.session_id, "generator", {"data": "test"}
        )
        session_manager.add_optimization_iteration(
            session.session_id, {"optim": "test"}
        )
        session_manager.add_performance_metrics(session.session_id, {"speed": 500})

        summary = session_manager.get_optimization_summary(session.session_id)
        assert summary is not None
        assert summary["total_iterations"] == 1
        assert summary["total_agent_interactions"] == 1

    def test_get_agent_interactions(self, session_manager):
        """Test getting agent interactions."""
        session = session_manager.create_session("Test task")

        session_manager.add_agent_interaction(
            session.session_id, "generator", {"data": "gen1"}
        )
        session_manager.add_agent_interaction(
            session.session_id, "debugger", {"data": "debug1"}
        )
        session_manager.add_agent_interaction(
            session.session_id, "generator", {"data": "gen2"}
        )

        all_interactions = session_manager.get_agent_interactions(session.session_id)
        assert len(all_interactions) == 3

        generator_interactions = session_manager.get_agent_interactions(
            session.session_id, "generator"
        )
        assert len(generator_interactions) == 2

    def test_get_performance_trend(self, session_manager):
        """Test getting performance trend."""
        session = session_manager.create_session("Test task")

        session_manager.add_performance_metrics(session.session_id, {"speed": 100})
        session_manager.add_performance_metrics(session.session_id, {"speed": 200})

        trend = session_manager.get_performance_trend(session.session_id)
        assert len(trend) == 2
        assert trend[0]["metrics"]["speed"] == 100
        assert trend[1]["metrics"]["speed"] == 200

    def test_list_sessions(self, session_manager):
        """Test listing sessions."""
        session1 = session_manager.create_session("Task 1")
        session2 = session_manager.create_session("Task 2")

        sessions = session_manager.list_sessions()
        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    def test_list_sessions_with_query(self, session_manager):
        """Test listing sessions with query."""
        session1 = session_manager.create_session("Task 1")
        session2 = session_manager.create_session("Task 2")

        session_manager.complete_session(session1.session_id)

        # Query by status
        query = SessionQuery(status=SessionStatus.COMPLETED, limit=10)
        completed_sessions = session_manager.list_sessions(query)
        assert len(completed_sessions) == 1
        assert completed_sessions[0].session_id == session1.session_id

    def test_delete_session(self, session_manager):
        """Test deleting session."""
        session = session_manager.create_session("Test task")
        session_id = session.session_id

        result = session_manager.delete_session(session_id)
        assert result is True

        # Should not be in active sessions
        assert session_id not in session_manager.active_sessions

        # Should not be retrievable
        retrieved_session = session_manager.get_session(session_id)
        assert retrieved_session is None

    def test_get_statistics(self, session_manager):
        """Test getting statistics."""
        # Debug: Check temp directory contents before test
        temp_dir = session_manager.store_dir
        print(f"Debug: Temp directory: {temp_dir}")
        print(f"Debug: Files in temp directory before test:")
        for file_path in temp_dir.glob("*.json"):
            print(f"  {file_path.name}")

        # Debug: Check initial state
        print(
            f"Debug: Initial active_sessions count: {len(session_manager.active_sessions)}"
        )
        print(
            f"Debug: Initial active_sessions IDs: {list(session_manager.active_sessions.keys())}"
        )

        session1 = session_manager.create_session("Task 1")
        print(
            f"Debug: After creating session1, active_sessions count: {len(session_manager.active_sessions)}"
        )

        session2 = session_manager.create_session("Task 2")
        print(
            f"Debug: After creating session2, active_sessions count: {len(session_manager.active_sessions)}"
        )
        print(
            f"Debug: Active session IDs: {list(session_manager.active_sessions.keys())}"
        )

        session_manager.complete_session(session1.session_id)
        session_manager.add_agent_interaction(
            session1.session_id, "generator", {"data": "test"}
        )

        stats = session_manager.get_statistics()

        # Debug information
        print(f"Debug: total_sessions = {stats['total_sessions']}")
        print(f"Debug: active_sessions = {stats['active_sessions']}")
        print(f"Debug: status_distribution = {stats['status_distribution']}")

        # List all sessions to see what's there
        all_sessions = session_manager.list_sessions()
        print(f"Debug: All session IDs: {[s.session_id for s in all_sessions]}")
        print(
            f"Debug: All session descriptions: {[s.task_description for s in all_sessions]}"
        )

        # Check active sessions in memory
        print(
            f"Debug: Active sessions in memory: {len(session_manager.active_sessions)}"
        )
        for session_id, session in session_manager.active_sessions.items():
            print(f"Debug: Active session {session_id}: {session.task_description}")

        # Debug: Check temp directory contents after test
        print(f"Debug: Files in temp directory after test:")
        for file_path in temp_dir.glob("*.json"):
            print(f"  {file_path.name}")

        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == len(session_manager.active_sessions)
        assert "status_distribution" in stats
        assert stats["status_distribution"]["completed"] == 1
        assert stats["status_distribution"]["active"] == 1


class TestSessionUtils:
    """Test SessionUtils functionality."""

    def test_analyze_session_performance(self):
        """Test analyzing session performance."""
        session = create_test_session(task_description="Test task")
        session.add_agent_interaction("generator", {"data": "test"})
        session.add_agent_interaction("debugger", {"data": "test"})
        session.add_optimization_iteration({"optim": "test"})
        session.add_performance_metrics({"speed": 100})
        session.add_performance_metrics({"speed": 200})

        analysis = SessionUtils.analyze_session_performance(session)

        assert analysis["total_interactions"] == 2
        assert analysis["total_iterations"] == 1
        assert analysis["performance_points"] == 2
        assert "agent_interaction_counts" in analysis
        assert analysis["agent_interaction_counts"]["generator"] == 1
        assert analysis["agent_interaction_counts"]["debugger"] == 1

    def test_generate_session_report(self):
        """Test generating session report."""
        session = create_test_session(task_description="Test task")
        session.target_performance = {"speed": 1000}
        session.add_agent_interaction("generator", {"data": "test"})
        session.add_optimization_iteration({"optim": "test"})
        session.add_performance_metrics({"speed": 500})

        report = SessionUtils.generate_session_report(session)

        assert report["session_id"] == session.session_id
        assert report["task_description"] == "Test task"
        assert report["status"] == "active"
        assert "optimization_summary" in report
        assert "performance_analysis" in report
        assert "version_references" in report

    def test_compare_sessions(self):
        """Test comparing sessions."""
        session1 = create_test_session(task_description="Task 1")
        session2 = create_test_session(task_description="Task 2")

        session1.add_agent_interaction("generator", {"data": "test"})
        session2.add_agent_interaction("generator", {"data": "test"})
        session2.add_agent_interaction("debugger", {"data": "test"})

        comparison = SessionUtils.compare_sessions(session1, session2)

        assert "runtime_comparison" in comparison
        assert "interaction_comparison" in comparison
        assert "iteration_comparison" in comparison
        assert "performance_comparison" in comparison
        assert comparison["interaction_comparison"]["difference"] == 1

    def test_validate_session_data(self):
        """Test validating session data."""
        # Valid session
        session = create_test_session(task_description="Test task")
        validation = SessionUtils.validate_session_data(session)
        assert validation["is_valid"] is True
        assert len(validation["errors"]) == 0

        # Invalid session (missing task description)
        invalid_session = Session(session_id="test", task_description="")
        validation = SessionUtils.validate_session_data(invalid_session)
        assert validation["is_valid"] is False
        assert len(validation["errors"]) > 0

    def test_get_session_statistics(self):
        """Test getting session statistics."""
        session1 = create_test_session(task_description="Task 1")
        session2 = create_test_session(task_description="Task 2")

        session1.add_agent_interaction("generator", {"data": "test"})
        session2.add_agent_interaction("debugger", {"data": "test"})
        session2.add_agent_interaction("evaluator", {"data": "test"})

        stats = SessionUtils.get_session_statistics([session1, session2])

        assert stats["total_sessions"] == 2
        assert stats["total_interactions"] == 3
        assert "status_distribution" in stats
        assert stats["status_distribution"]["active"] == 2

    def test_filter_sessions_by_criteria(self):
        """Test filtering sessions by criteria."""
        session1 = create_test_session(task_description="Task 1")
        session2 = create_test_session(task_description="Task 2")

        session1.add_agent_interaction("generator", {"data": "test"})
        session2.add_agent_interaction("debugger", {"data": "test"})

        # Filter by agent type
        generator_sessions = SessionUtils.filter_sessions_by_criteria(
            [session1, session2], agent_type="generator"
        )
        assert len(generator_sessions) == 1
        assert generator_sessions[0].session_id == session1.session_id

        # Filter by status
        session1.complete_session()
        completed_sessions = SessionUtils.filter_sessions_by_criteria(
            [session1, session2], status=SessionStatus.COMPLETED
        )
        assert len(completed_sessions) == 1
        assert completed_sessions[0].session_id == session1.session_id
