"""
Tests for the memory models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from pinocchio.memory import BaseMemory, CodeMemory, CodeVersion
from pinocchio.utils import assert_dict_structure
from tests.utils import (
    assert_session_valid,
    assert_task_plan_valid,
    assert_task_valid,
    create_test_json_file,
    create_test_session,
    create_test_task,
    create_test_task_plan,
    load_test_json_file,
)


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
        json_data = load_test_json_file(create_test_json_file(memory.model_dump()))

        assert json_data is not None
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

        # Create test JSON file
        json_file = create_test_json_file(json_data)
        loaded_data = load_test_json_file(json_file)

        assert loaded_data is not None
        memory = BaseMemory.model_validate(loaded_data)

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
        """Test serializing a CodeVersion instance."""
        version = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="cpu",
            source_agent="generator",
            description="Test version",
        )

        # Test model_dump
        data = version.model_dump()
        assert_dict_structure(
            data,
            [
                "version_id",
                "session_id",
                "code",
                "language",
                "kernel_type",
                "source_agent",
                "timestamp",
                "description",
                "optimization_techniques",
                "hyperparameters",
                "metadata",
            ],
            optional_keys=["parent_version_id"],
        )
        assert data["session_id"] == "test-session"
        assert data["code"] == "def example(): return 42"
        assert data["language"] == "python"
        assert data["kernel_type"] == "cpu"
        assert data["source_agent"] == "generator"
        assert data["description"] == "Test version"


class TestCodeMemory:
    """Tests for the CodeMemory class."""

    def test_code_memory_operations(self):
        """Test basic CodeMemory operations."""
        memory = CodeMemory(session_id="test-session")

        # Create test versions
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

        # Add versions to memory
        memory.add_version(version1)
        memory.add_version(version2)

        # Test getting versions
        retrieved_version1 = memory.get_version(version1.version_id)
        assert retrieved_version1 is not None
        assert retrieved_version1.code == version1.code
        assert retrieved_version1.source_agent == "generator"

        retrieved_version2 = memory.get_version(version2.version_id)
        assert retrieved_version2 is not None
        assert retrieved_version2.code == version2.code
        assert retrieved_version2.source_agent == "debugger"
        assert retrieved_version2.parent_version_id == version1.version_id

        # Test getting current version
        current = memory.get_current_version()
        assert current is not None
        assert current.version_id == version2.version_id

        # Test getting version history
        history = memory.get_version_history()
        assert len(history) == 2
        # Check that the newest version is first
        assert history[0]["version_id"] == version2.version_id
        assert history[0]["is_current"] is True
        assert history[1]["is_current"] is False

    def test_get_nonexistent_version(self):
        """Test getting a version that doesn't exist."""
        memory = CodeMemory(session_id="test-session")
        version = memory.get_version("nonexistent-id")
        assert version is None

    def test_set_nonexistent_version(self):
        """Test setting a version that doesn't exist."""
        memory = CodeMemory(session_id="test-session")
        # This should not raise an exception
        result = memory.set_current_version("nonexistent-id")
        assert result is False

    def test_code_memory_serialization(self):
        """Test serializing a CodeMemory instance."""
        memory = CodeMemory(session_id="test-session")

        version = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="cpu",
            source_agent="generator",
        )

        memory.add_version(version)

        # Test model_dump
        data = memory.model_dump()
        assert_dict_structure(data, ["session_id", "versions", "current_version_id"])
        assert data["session_id"] == "test-session"
        assert len(data["versions"]) == 1
        assert version.version_id in data["versions"]
