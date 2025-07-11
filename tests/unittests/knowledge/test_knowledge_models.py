"""
Tests for knowledge models.
"""

from pinocchio.knowledge.models.knowledge import (
    KnowledgeCategory,
    KnowledgeContentType,
    KnowledgeExtractionResult,
    KnowledgeFragment,
    KnowledgeQuery,
    KnowledgeVersionHistory,
)


class TestKnowledgeFragment:
    """Test KnowledgeFragment model."""

    def test_create_fragment(self):
        """Test creating a knowledge fragment."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling Optimization",
            content={"technique": "tiling", "description": "Cache optimization"},
            content_type=KnowledgeContentType.JSON,
        )

        assert fragment.session_id == "session_123"
        assert fragment.agent_type == "generator"
        assert fragment.category == KnowledgeCategory.OPTIMIZATION
        assert fragment.title == "Tiling Optimization"
        assert fragment.content_type == KnowledgeContentType.JSON
        assert fragment.version.startswith("v")
        assert fragment.fragment_id is not None

    def test_extract_structured_data_json(self):
        """Test extracting structured data from JSON content."""
        content = {"technique": "tiling", "parameters": {"tile_size": 32}}
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content=content,
            content_type=KnowledgeContentType.JSON,
        )

        structured = fragment.extract_structured_data()
        assert structured == content

    def test_extract_structured_data_markdown(self):
        """Test extracting structured data from Markdown content."""
        content = "# Tiling Optimization\n\nThis is a technique for cache optimization."
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content=content,
            content_type=KnowledgeContentType.MARKDOWN,
        )

        structured = fragment.extract_structured_data()
        assert "markdown_content" in structured
        assert structured["markdown_content"] == content

    def test_extract_structured_data_text(self):
        """Test extracting structured data from text content."""
        content = "Simple text content"
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OTHER,
            title="Text Fragment",
            content=content,
            content_type=KnowledgeContentType.TEXT,
        )

        structured = fragment.extract_structured_data()
        assert "text_content" in structured
        assert structured["text_content"] == content


class TestKnowledgeVersionHistory:
    """Test KnowledgeVersionHistory model."""

    def test_add_fragment(self):
        """Test adding fragments to version history."""
        history = KnowledgeVersionHistory(session_id="session_123")

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

        history.add_fragment(fragment1)
        history.add_fragment(fragment2)

        assert len(history.fragments) == 2
        assert len(history.version_chain) == 2
        assert history.fragments[0] == fragment1
        assert history.fragments[1] == fragment2

    def test_get_latest_fragment(self):
        """Test getting the latest fragment."""
        history = KnowledgeVersionHistory(session_id="session_123")

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

        history.add_fragment(fragment1)
        history.add_fragment(fragment2)

        latest = history.get_latest_fragment()
        assert latest == fragment2

    def test_get_fragment_by_version(self):
        """Test getting fragment by version."""
        history = KnowledgeVersionHistory(session_id="session_123")

        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Test Fragment",
            content={"data": "test"},
            content_type=KnowledgeContentType.JSON,
        )

        history.add_fragment(fragment)

        found = history.get_fragment_by_version(fragment.version)
        assert found == fragment

        not_found = history.get_fragment_by_version("nonexistent")
        assert not_found is None


class TestKnowledgeQuery:
    """Test KnowledgeQuery model."""

    def test_create_query(self):
        """Test creating a knowledge query."""
        query = KnowledgeQuery(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            keywords=["tiling", "cache"],
            content_type=KnowledgeContentType.JSON,
            limit=5,
        )

        assert query.session_id == "session_123"
        assert query.agent_type == "generator"
        assert query.category == KnowledgeCategory.OPTIMIZATION
        assert query.keywords == ["tiling", "cache"]
        assert query.content_type == KnowledgeContentType.JSON
        assert query.limit == 5


class TestKnowledgeExtractionResult:
    """Test KnowledgeExtractionResult model."""

    def test_add_fragment(self):
        """Test adding fragments to extraction result."""
        result = KnowledgeExtractionResult(
            session_id="session_123", agent_type="generator"
        )

        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Test Fragment",
            content={"data": "test"},
            content_type=KnowledgeContentType.JSON,
        )

        result.add_fragment(fragment)

        assert len(result.fragments) == 1
        assert result.total_count == 1
        assert result.fragments[0] == fragment

    def test_get_fragments_by_category(self):
        """Test filtering fragments by category."""
        result = KnowledgeExtractionResult(
            session_id="session_123", agent_type="generator"
        )

        opt_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Optimization Fragment",
            content={"data": "opt"},
            content_type=KnowledgeContentType.JSON,
        )

        domain_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.DOMAIN,
            title="Domain Fragment",
            content={"data": "domain"},
            content_type=KnowledgeContentType.JSON,
        )

        result.add_fragment(opt_fragment)
        result.add_fragment(domain_fragment)

        opt_fragments = result.get_fragments_by_category(KnowledgeCategory.OPTIMIZATION)
        assert len(opt_fragments) == 1
        assert opt_fragments[0] == opt_fragment

        domain_fragments = result.get_fragments_by_category(KnowledgeCategory.DOMAIN)
        assert len(domain_fragments) == 1
        assert domain_fragments[0] == domain_fragment

    def test_get_structured_knowledge(self):
        """Test getting structured knowledge."""
        result = KnowledgeExtractionResult(
            session_id="session_123", agent_type="generator"
        )

        opt_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Optimization Fragment",
            content={"data": "opt"},
            content_type=KnowledgeContentType.JSON,
        )

        domain_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.DOMAIN,
            title="Domain Fragment",
            content={"data": "domain"},
            content_type=KnowledgeContentType.JSON,
        )

        result.add_fragment(opt_fragment)
        result.add_fragment(domain_fragment)

        structured = result.get_structured_knowledge()

        assert "optimization" in structured
        assert "domain" in structured
        assert len(structured["optimization"]) == 1
        assert len(structured["domain"]) == 1
        assert structured["optimization"][0]["title"] == "Optimization Fragment"
        assert structured["domain"][0]["title"] == "Domain Fragment"


class TestKnowledgeCategories:
    """Test knowledge categories."""

    def test_category_values(self):
        """Test category enum values."""
        assert KnowledgeCategory.DOMAIN.value == "domain"
        assert KnowledgeCategory.OPTIMIZATION.value == "optimization"
        assert KnowledgeCategory.DSL.value == "dsl"
        assert KnowledgeCategory.OTHER.value == "other"


class TestKnowledgeContentTypes:
    """Test knowledge content types."""

    def test_content_type_values(self):
        """Test content type enum values."""
        assert KnowledgeContentType.JSON.value == "json"
        assert KnowledgeContentType.MARKDOWN.value == "markdown"
        assert KnowledgeContentType.TEXT.value == "text"
