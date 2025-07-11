"""
Tests for knowledge manager.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from pinocchio.knowledge.manager import KnowledgeManager
from pinocchio.knowledge.models.knowledge import (
    KnowledgeCategory,
    KnowledgeContentType,
    KnowledgeFragment,
    KnowledgeQuery,
)


class TestKnowledgeManager:
    """Test KnowledgeManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a knowledge manager instance."""
        return KnowledgeManager(storage_path=temp_dir)

    def test_add_fragment(self, manager):
        """Test adding a knowledge fragment."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling Optimization",
            content={"technique": "tiling", "description": "Cache optimization"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment_id = manager.add_fragment(fragment)

        assert fragment_id == fragment.fragment_id
        assert manager.get_fragment(fragment_id) == fragment
        assert len(manager.get_session_fragments("session_123")) == 1

    def test_get_fragment(self, manager):
        """Test getting a knowledge fragment."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Test Fragment",
            content={"data": "test"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(fragment)

        retrieved = manager.get_fragment(fragment.fragment_id)
        assert retrieved == fragment

        not_found = manager.get_fragment("nonexistent")
        assert not_found is None

    def test_get_session_fragments(self, manager):
        """Test getting fragments for a session."""
        fragment1 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Fragment 1",
            content={"data": "1"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment2 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="debugger",
            category=KnowledgeCategory.DOMAIN,
            title="Fragment 2",
            content={"data": "2"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment3 = KnowledgeFragment.create_fragment(
            session_id="session_456",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Fragment 3",
            content={"data": "3"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(fragment1)
        manager.add_fragment(fragment2)
        manager.add_fragment(fragment3)

        session_fragments = manager.get_session_fragments("session_123")
        assert len(session_fragments) == 2
        assert fragment1 in session_fragments
        assert fragment2 in session_fragments
        assert fragment3 not in session_fragments

    def test_search_fragments(self, manager):
        """Test searching for fragments."""
        fragment1 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling Optimization",
            content={"technique": "tiling"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment2 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="debugger",
            category=KnowledgeCategory.DOMAIN,
            title="Domain Knowledge",
            content={"domain": "matrix"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(fragment1)
        manager.add_fragment(fragment2)

        # Search by session
        query = KnowledgeQuery(session_id="session_123")
        result = manager.search_fragments(query)
        assert result.total_count == 2

        # Search by agent type
        query = KnowledgeQuery(session_id="session_123", agent_type="generator")
        result = manager.search_fragments(query)
        assert result.total_count == 1
        assert result.fragments[0] == fragment1

        # Search by category
        query = KnowledgeQuery(
            session_id="session_123", category=KnowledgeCategory.OPTIMIZATION
        )
        result = manager.search_fragments(query)
        assert result.total_count == 1
        assert result.fragments[0] == fragment1

        # Search by keywords
        query = KnowledgeQuery(session_id="session_123", keywords=["tiling"])
        result = manager.search_fragments(query)
        assert result.total_count == 1
        assert result.fragments[0] == fragment1

    def test_extract_agent_knowledge(self, manager):
        """Test extracting knowledge for a specific agent."""
        fragment1 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Generator Knowledge",
            content={"data": "generator"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment2 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="debugger",
            category=KnowledgeCategory.DOMAIN,
            title="Debugger Knowledge",
            content={"data": "debugger"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(fragment1)
        manager.add_fragment(fragment2)

        generator_result = manager.extract_agent_knowledge("session_123", "generator")
        assert generator_result.total_count == 1
        assert generator_result.fragments[0] == fragment1

        debugger_result = manager.extract_agent_knowledge("session_123", "debugger")
        assert debugger_result.total_count == 1
        assert debugger_result.fragments[0] == fragment2

    def test_extract_optimization_knowledge(self, manager):
        """Test extracting optimization knowledge."""
        opt_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content={"technique": "tiling"},
            content_type=KnowledgeContentType.JSON,
        )

        domain_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.DOMAIN,
            title="Domain",
            content={"domain": "matrix"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(opt_fragment)
        manager.add_fragment(domain_fragment)

        opt_fragments = manager.extract_optimization_knowledge("session_123")
        assert len(opt_fragments) == 1
        assert opt_fragments[0] == opt_fragment

    def test_extract_domain_knowledge(self, manager):
        """Test extracting domain knowledge."""
        domain_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.DOMAIN,
            title="Matrix Operations",
            content={"domain": "matrix"},
            content_type=KnowledgeContentType.JSON,
        )

        opt_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content={"technique": "tiling"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(domain_fragment)
        manager.add_fragment(opt_fragment)

        domain_fragments = manager.extract_domain_knowledge("session_123")
        assert len(domain_fragments) == 1
        assert domain_fragments[0] == domain_fragment

    def test_extract_dsl_knowledge(self, manager):
        """Test extracting DSL knowledge."""
        dsl_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.DSL,
            title="CUDA Syntax",
            content={"language": "cuda", "syntax": {}},
            content_type=KnowledgeContentType.JSON,
        )

        opt_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content={"technique": "tiling"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(dsl_fragment)
        manager.add_fragment(opt_fragment)

        dsl_fragments = manager.extract_dsl_knowledge("session_123")
        assert len(dsl_fragments) == 1
        assert dsl_fragments[0] == dsl_fragment

    def test_get_version_history(self, manager):
        """Test getting version history."""
        fragment1 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Version 1",
            content={"data": "v1"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment2 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Version 2",
            content={"data": "v2"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(fragment1)
        manager.add_fragment(fragment2)

        history = manager.get_version_history("session_123")
        assert history is not None
        assert len(history.fragments) == 2
        assert history.fragments[0] == fragment1
        assert history.fragments[1] == fragment2

    def test_create_fragment_from_content(self, manager):
        """Test creating fragment from content."""
        fragment = manager.create_fragment_from_content(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Test Fragment",
            content={"data": "test"},
            content_type=KnowledgeContentType.JSON,
        )

        assert fragment.session_id == "session_123"
        assert fragment.agent_type == "generator"
        assert fragment.category == KnowledgeCategory.OPTIMIZATION
        assert fragment.title == "Test Fragment"
        assert fragment.content_type == KnowledgeContentType.JSON

        # Check that fragment was added to manager
        retrieved = manager.get_fragment(fragment.fragment_id)
        assert retrieved == fragment

    def test_clear_session(self, manager):
        """Test clearing fragments for a session."""
        fragment1 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Fragment 1",
            content={"data": "1"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment2 = KnowledgeFragment.create_fragment(
            session_id="session_456",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Fragment 2",
            content={"data": "2"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(fragment1)
        manager.add_fragment(fragment2)

        # Clear session 123
        manager.clear_session("session_123")

        # Check that session 123 fragments are removed
        session_fragments = manager.get_session_fragments("session_123")
        assert len(session_fragments) == 0

        # Check that session 456 fragments remain
        session_fragments = manager.get_session_fragments("session_456")
        assert len(session_fragments) == 1
        assert session_fragments[0] == fragment2

    def test_get_statistics(self, manager):
        """Test getting manager statistics."""
        fragment1 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Fragment 1",
            content={"data": "1"},
            content_type=KnowledgeContentType.JSON,
        )

        fragment2 = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="debugger",
            category=KnowledgeCategory.DOMAIN,
            title="Fragment 2",
            content={"data": "2"},
            content_type=KnowledgeContentType.MARKDOWN,
        )

        manager.add_fragment(fragment1)
        manager.add_fragment(fragment2)

        stats = manager.get_statistics()

        assert stats["total_fragments"] == 2
        assert stats["total_sessions"] == 1
        assert stats["total_versions"] == 1
        assert stats["category_counts"]["optimization"] == 1
        assert stats["category_counts"]["domain"] == 1
        assert stats["content_type_counts"]["json"] == 1
        assert stats["content_type_counts"]["markdown"] == 1

    def test_persistence(self, manager, temp_dir):
        """Test fragment persistence to storage."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Test Fragment",
            content={"data": "test"},
            content_type=KnowledgeContentType.JSON,
        )

        manager.add_fragment(fragment)

        # Check that file was created
        fragment_file = Path(temp_dir) / f"{fragment.fragment_id}.json"
        assert fragment_file.exists()

        # Create new manager and load from storage
        new_manager = KnowledgeManager(storage_path=temp_dir)
        new_manager.load_from_storage()

        # Check that fragment was loaded
        loaded_fragment = new_manager.get_fragment(fragment.fragment_id)
        assert loaded_fragment is not None
        assert loaded_fragment.fragment_id == fragment.fragment_id
        assert loaded_fragment.title == fragment.title
