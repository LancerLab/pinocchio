"""
Tests for the session models.
"""

from datetime import datetime

from pinocchio.session.models import SessionMetadata


class TestSessionMetadata:
    """Tests for the SessionMetadata class."""

    def test_session_metadata_creation(self):
        """Test creating a SessionMetadata instance."""
        session = SessionMetadata.create_new_session(
            task_description="Test task", name="Test Session", tags=["test", "example"]
        )

        assert session.task_description == "Test task"
        assert session.name == "Test Session"
        assert session.tags == ["test", "example"]
        assert session.status == "active"
        assert session.session_id.startswith("session_")
        assert session.system_info is not None

    def test_session_metadata_defaults(self):
        """Test SessionMetadata defaults."""
        session = SessionMetadata.create_new_session(
            task_description="Test task", name="Default Session"
        )

        assert session.name == "Default Session"
        assert session.tags == []
        assert session.end_time is None
        assert session.runtime_seconds is None

    def test_mark_completed(self):
        """Test marking a session as completed."""
        session = SessionMetadata.create_new_session(
            task_description="Test task", name="Test Session"
        )

        # Mark as completed
        session.mark_completed("completed")

        assert session.status == "completed"
        assert session.end_time is not None
        assert isinstance(session.end_time, datetime)
        assert session.runtime_seconds is not None
        assert session.runtime_seconds >= 0

        # Mark as failed
        session = SessionMetadata.create_new_session(
            task_description="Failed task", name="Failed Session"
        )
        session.mark_completed("failed")

        assert session.status == "failed"

    def test_session_metadata_serialization(self):
        """Test serializing and deserializing SessionMetadata."""
        session = SessionMetadata.create_new_session(
            task_description="Test task", name="Test Session", tags=["test"]
        )

        # Serialize to JSON
        json_str = session.model_dump_json()

        # Deserialize from JSON
        deserialized = SessionMetadata.model_validate_json(json_str)

        assert deserialized.session_id == session.session_id
        assert deserialized.name == session.name
        assert deserialized.task_description == session.task_description
        assert deserialized.tags == session.tags
        assert deserialized.status == session.status

    def test_tag_management(self):
        """Test tag management."""
        session = SessionMetadata.create_new_session(
            task_description="Test task", name="Tag Session", tags=["initial"]
        )

        # Add tags
        session.add_tag("tag1")
        session.add_tag("tag2")

        assert "initial" in session.tags
        assert "tag1" in session.tags
        assert "tag2" in session.tags

        # Remove tag
        session.remove_tag("initial")

        assert "initial" not in session.tags
        assert len(session.tags) == 2

        # Clear tags
        session.tags = []
        assert len(session.tags) == 0

        # Set new tags
        session.tags = ["new1", "new2"]
        assert session.tags == ["new1", "new2"]

    def test_input_and_info_updates(self):
        """Test updating user inputs and system info."""
        session = SessionMetadata.create_new_session(
            task_description="Test task", name="Info Session"
        )

        # Update user inputs
        session.update_user_input("query", "How to optimize matrix multiplication")
        session.update_user_input("parameters", {"optimize_for": "speed"})

        assert session.user_inputs["query"] == "How to optimize matrix multiplication"
        assert session.user_inputs["parameters"]["optimize_for"] == "speed"

        # Update system info
        session.update_system_info("memory_usage", "1.2GB")
        session.update_system_info("cpu_cores", 8)

        assert session.system_info["memory_usage"] == "1.2GB"
        assert session.system_info["cpu_cores"] == 8

        # Serialize and deserialize
        json_str = session.model_dump_json()
        deserialized = SessionMetadata.model_validate_json(json_str)

        assert deserialized.user_inputs == session.user_inputs
        assert deserialized.system_info == session.system_info
