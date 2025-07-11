"""
Tests for the memory models.
"""

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from pinocchio.memory import BaseMemory, CodeMemory, CodeVersion


class TestBaseMemory:
    """Tests for the BaseMemory class."""

    def test_base_memory_creation(self):
        """Test creating a BaseMemory instance."""
        memory = BaseMemory(
            session_id="test-session", agent_type="generator", version_id="test-version"
        )

        assert memory.session_id == "test-session"
        assert memory.agent_type == "generator"
        assert memory.version_id == "test-version"
        assert memory.id is not None
        assert isinstance(memory.timestamp, datetime)
        assert memory.metadata == {}

    def test_base_memory_with_metadata(self):
        """Test creating a BaseMemory instance with metadata."""
        metadata = {"key": "value", "nested": {"inner": 42}}
        memory = BaseMemory(
            session_id="test-session",
            agent_type="debugger",
            version_id="test-version",
            metadata=metadata,
        )

        assert memory.metadata == metadata
        assert memory.metadata["key"] == "value"
        assert memory.metadata["nested"]["inner"] == 42

    def test_base_memory_serialization(self):
        """Test serializing a BaseMemory instance to JSON."""
        memory = BaseMemory(
            session_id="test-session",
            agent_type="evaluator",
            version_id="test-version",
            metadata={"key": "value"},
        )

        # Convert to JSON
        json_str = memory.model_dump_json()
        json_data = json.loads(json_str)

        assert json_data["session_id"] == "test-session"
        assert json_data["agent_type"] == "evaluator"
        assert json_data["version_id"] == "test-version"
        assert json_data["metadata"]["key"] == "value"
        assert "id" in json_data
        assert "timestamp" in json_data

    def test_base_memory_deserialization(self):
        """Test deserializing JSON to a BaseMemory instance."""
        json_data = {
            "id": "test-id",
            "session_id": "test-session",
            "agent_type": "generator",
            "version_id": "test-version",
            "timestamp": "2023-01-01T12:00:00",
            "metadata": {"key": "value"},
        }

        memory = BaseMemory.model_validate(json_data)

        assert memory.id == "test-id"
        assert memory.session_id == "test-session"
        assert memory.agent_type == "generator"
        assert memory.version_id == "test-version"
        assert memory.timestamp.isoformat() == "2023-01-01T12:00:00"
        assert memory.metadata["key"] == "value"

    def test_base_memory_validation(self):
        """Test validation of BaseMemory fields."""
        # Missing required field
        with pytest.raises(ValidationError):
            BaseMemory()  # Missing required fields

        # Missing agent_type
        with pytest.raises(ValidationError):
            BaseMemory(session_id="test-session", version_id="test-version")

        # Missing version_id
        with pytest.raises(ValidationError):
            BaseMemory(session_id="test-session", agent_type="generator")

        # Invalid timestamp format
        with pytest.raises(ValidationError):
            BaseMemory.model_validate(
                {
                    "session_id": "test-session",
                    "agent_type": "generator",
                    "version_id": "test-version",
                    "timestamp": "invalid-timestamp",
                }
            )


class TestCodeVersion:
    """Tests for the CodeVersion class."""

    def test_code_version_creation(self):
        """Test creating a CodeVersion instance."""
        code = "def example(): return 42"
        version = CodeVersion.create_new_version(
            session_id="test-session",
            code=code,
            language="python",
            kernel_type="cpu",
            source_agent="generator",
            description="Test version",
        )

        assert version.code == code
        assert version.session_id == "test-session"
        assert version.language == "python"
        assert version.kernel_type == "cpu"
        assert version.source_agent == "generator"
        assert version.description == "Test version"
        assert version.version_id is not None
        assert version.parent_version_id is None

    def test_code_version_with_parent(self):
        """Test creating a CodeVersion with a parent version."""
        parent = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): pass",
            language="python",
            kernel_type="cpu",
            source_agent="generator",
        )

        child = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="cpu",
            source_agent="debugger",
            parent_version_id=parent.version_id,
            description="Fixed version",
        )

        assert child.parent_version_id == parent.version_id
        assert child.source_agent == "debugger"
        assert child.description == "Fixed version"

    def test_code_version_diff(self):
        """Test the diff functionality."""
        version1 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="cpu",
            source_agent="generator",
        )

        version2 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 43",
            language="python",
            kernel_type="cpu",
            source_agent="debugger",
            parent_version_id=version1.version_id,
        )

        diff = version1.get_diff(version2)
        assert diff["is_different"] is True
        assert diff["this_version"] == version1.version_id
        assert diff["other_version"] == version2.version_id

        # Test with identical code
        version3 = CodeVersion.create_new_version(
            session_id="test-session",
            code=version1.code,
            language="python",
            kernel_type="cpu",
            source_agent="optimizer",
        )

        diff = version1.get_diff(version3)
        assert diff["is_different"] is False

    def test_code_version_serialization(self):
        """Test serializing and deserializing a CodeVersion."""
        version = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="cpu",
            source_agent="generator",
            description="Test version",
        )

        json_str = version.model_dump_json()
        deserialized = CodeVersion.model_validate_json(json_str)

        assert deserialized.code == version.code
        assert deserialized.version_id == version.version_id
        assert deserialized.source_agent == version.source_agent
        assert deserialized.description == version.description


class TestCodeMemory:
    """Tests for the CodeMemory class."""

    def test_code_memory_operations(self):
        """Test CodeMemory operations."""
        memory = CodeMemory(session_id="test-session")

        # Add a version
        version1 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="cpu",
            source_agent="generator",
            description="Initial version",
        )

        version_id = memory.add_version(version1)
        assert version_id == version1.version_id
        assert memory.current_version_id == version1.version_id

        # Get the version
        retrieved = memory.get_version(version1.version_id)
        assert retrieved.code == version1.code

        # Get current version
        current = memory.get_current_version()
        assert current.code == "def example(): return 42"

        # Add another version
        version2 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 43",
            language="python",
            kernel_type="cpu",
            source_agent="debugger",
            parent_version_id=version1.version_id,
            description="Fixed version",
        )

        memory.add_version(version2)

        # Get version history
        history = memory.get_version_history()
        assert len(history) == 2

        # Check that the newest version is first
        assert history[0]["version_id"] == version2.version_id
        assert history[0]["is_current"] is True
        assert history[1]["is_current"] is False

    def test_get_nonexistent_version(self):
        """Test getting a version that doesn't exist."""
        memory = CodeMemory(session_id="test-session")

        # Get a nonexistent version
        retrieved = memory.get_version("nonexistent-id")
        assert retrieved is None

        # Get current version when there isn't one
        retrieved = memory.get_current_version()
        assert retrieved is None

    def test_set_nonexistent_version(self):
        """Test setting a current version that doesn't exist."""
        memory = CodeMemory(session_id="test-session")

        # Try to set a nonexistent version as current
        result = memory.set_current_version("nonexistent-id")
        assert result is False

    def test_code_memory_serialization(self):
        """Test serializing and deserializing CodeMemory."""
        memory = CodeMemory(session_id="test-session")

        # Add versions
        version1 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="cpu",
            source_agent="generator",
        )

        version2 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 43",
            language="python",
            kernel_type="cpu",
            source_agent="debugger",
            parent_version_id=version1.version_id,
        )

        memory.add_version(version1)
        memory.add_version(version2)

        # Serialize and deserialize
        json_str = memory.model_dump_json()
        deserialized = CodeMemory.model_validate_json(json_str)

        assert deserialized.session_id == memory.session_id
        assert deserialized.current_version_id == memory.current_version_id
        assert len(deserialized.versions) == len(memory.versions)
