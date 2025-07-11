"""
Tests for knowledge utilities.
"""

from pinocchio.knowledge.models.knowledge import (
    KnowledgeCategory,
    KnowledgeContentType,
    KnowledgeFragment,
)
from pinocchio.knowledge.utils import KnowledgeUtils


class TestKnowledgeUtils:
    """Test KnowledgeUtils class."""

    def test_parse_markdown_content(self):
        """Test parsing markdown content."""
        content = """# Tiling Optimization

This is a technique for cache optimization.

## Code Example

```cuda
__global__ void tiled_matrix_multiply(float* A, float* B, float* C, int N) {
    // Implementation
}
```

## Benefits

- Improves cache hit rate
- Reduces memory bandwidth
- Better performance
"""

        parsed = KnowledgeUtils.parse_markdown_content(content)
        assert len(parsed["headers"]) == 3
        assert parsed["headers"][0]["text"] == "Tiling Optimization"
        assert parsed["headers"][1]["text"] == "Code Example"

        assert len(parsed["code_blocks"]) == 1
        assert "__global__ void tiled_matrix_multiply" in parsed["code_blocks"][0]

        assert len(parsed["lists"]) == 1
        assert "Improves cache hit rate" in parsed["lists"][0][0]

        assert "This is a technique" in parsed["text_content"]

    def test_extract_code_from_markdown(self):
        """Test extracting code blocks from markdown."""
        content = """# Example

```cuda
__global__ void kernel() {
    // Code here
}
```

More text

```python
def function():
    pass
```
"""

        code_blocks = KnowledgeUtils.extract_code_from_markdown(content)

        assert len(code_blocks) == 2
        assert "__global__ void kernel()" in code_blocks[0]
        assert "def function():" in code_blocks[1]

    def test_convert_to_structured_json_json(self):
        """Test converting JSON fragment to structured format."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content={"technique": "tiling", "parameters": {"tile_size": 32}},
            content_type=KnowledgeContentType.JSON,
        )

        structured = KnowledgeUtils.convert_to_structured_json(fragment)

        assert structured["technique"] == "tiling"
        assert structured["parameters"]["tile_size"] == 32

    def test_convert_to_structured_json_markdown(self):
        """Test converting Markdown fragment to structured format."""
        content = "# Tiling Optimization\n\nCache optimization technique."
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content=content,
            content_type=KnowledgeContentType.MARKDOWN,
        )

        structured = KnowledgeUtils.convert_to_structured_json(fragment)

        assert structured["title"] == "Tiling"
        assert structured["category"] == "optimization"
        assert "parsed_content" in structured
        assert (
            "Tiling Optimization" in structured["parsed_content"]["headers"][0]["text"]
        )

    def test_convert_to_structured_json_text(self):
        """Test converting text fragment to structured format."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OTHER,
            title="Text Fragment",
            content="Simple text content",
            content_type=KnowledgeContentType.TEXT,
        )

        structured = KnowledgeUtils.convert_to_structured_json(fragment)

        assert structured["title"] == "Text Fragment"
        assert structured["category"] == "other"
        assert structured["content"] == "Simple text content"

    def test_merge_fragments(self):
        """Test merging multiple fragments."""
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
            title="Matrix Operations",
            content={"domain": "matrix"},
            content_type=KnowledgeContentType.JSON,
        )

        dsl_fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.DSL,
            title="CUDA Syntax",
            content={"language": "cuda"},
            content_type=KnowledgeContentType.JSON,
        )

        merged = KnowledgeUtils.merge_fragments(
            [opt_fragment, domain_fragment, dsl_fragment]
        )

        assert len(merged["optimization_techniques"]) == 1
        assert len(merged["domain_knowledge"]) == 1
        assert len(merged["dsl_syntax"]) == 1
        assert len(merged["other"]) == 0

        assert merged["optimization_techniques"][0]["title"] == "Tiling"
        assert merged["domain_knowledge"][0]["title"] == "Matrix Operations"
        assert merged["dsl_syntax"][0]["title"] == "CUDA Syntax"

    def test_create_optimization_knowledge(self):
        """Test creating optimization knowledge fragment."""
        fragment = KnowledgeUtils.create_optimization_knowledge(
            technique="tiling",
            description="Cache optimization technique",
            parameters={"tile_size": 32, "tile_factor": 2},
            examples=["Example 1", "Example 2"],
            session_id="session_123",
            agent_type="generator",
        )

        assert fragment.category == KnowledgeCategory.OPTIMIZATION
        assert fragment.title == "Optimization Technique: tiling"
        assert fragment.content_type == KnowledgeContentType.JSON
        assert fragment.session_id == "session_123"
        assert fragment.agent_type == "generator"

        content = fragment.content
        assert content["technique"] == "tiling"
        assert content["description"] == "Cache optimization technique"
        assert content["parameters"]["tile_size"] == 32
        assert content["examples"] == ["Example 1", "Example 2"]

    def test_create_domain_knowledge(self):
        """Test creating domain knowledge fragment."""
        fragment = KnowledgeUtils.create_domain_knowledge(
            domain="matrix_operations",
            concepts=["matrix multiplication", "eigenvalues"],
            patterns=["block decomposition", "iterative methods"],
            session_id="session_123",
            agent_type="generator",
        )

        assert fragment.category == KnowledgeCategory.DOMAIN
        assert fragment.title == "Domain Knowledge: matrix_operations"
        assert fragment.content_type == KnowledgeContentType.JSON
        assert fragment.session_id == "session_123"
        assert fragment.agent_type == "generator"

        content = fragment.content
        assert content["domain"] == "matrix_operations"
        assert content["concepts"] == ["matrix multiplication", "eigenvalues"]
        assert content["patterns"] == ["block decomposition", "iterative methods"]

    def test_create_dsl_knowledge(self):
        """Test creating DSL knowledge fragment."""
        fragment = KnowledgeUtils.create_dsl_knowledge(
            language="cuda",
            syntax={"kernel": "__global__", "threads": "threadIdx"},
            templates=["template1", "template2"],
            session_id="session_123",
            agent_type="generator",
        )

        assert fragment.category == KnowledgeCategory.DSL
        assert fragment.title == "DSL Knowledge: cuda"
        assert fragment.content_type == KnowledgeContentType.JSON
        assert fragment.session_id == "session_123"
        assert fragment.agent_type == "generator"

        content = fragment.content
        assert content["language"] == "cuda"
        assert content["syntax"]["kernel"] == "__global__"
        assert content["templates"] == ["template1", "template2"]

    def test_validate_fragment_content_valid_json(self):
        """Test validating valid JSON fragment content."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content={"technique": "tiling", "description": "Cache optimization"},
            content_type=KnowledgeContentType.JSON,
        )

        assert KnowledgeUtils.validate_fragment_content(fragment) is True

    def test_validate_fragment_content_invalid_json(self):
        """Test validating invalid JSON fragment content."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content="Not a dict",  # Invalid for JSON content type
            content_type=KnowledgeContentType.JSON,
        )

        assert KnowledgeUtils.validate_fragment_content(fragment) is False

    def test_validate_fragment_content_missing_required_fields(self):
        """Test validating fragment with missing required fields."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content={"technique": "tiling"},  # Missing description
            content_type=KnowledgeContentType.JSON,
        )

        assert KnowledgeUtils.validate_fragment_content(fragment) is False

    def test_validate_fragment_content_valid_markdown(self):
        """Test validating valid Markdown fragment content."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content="# Tiling Optimization\n\nDescription here.",
            content_type=KnowledgeContentType.MARKDOWN,
        )

        assert KnowledgeUtils.validate_fragment_content(fragment) is True

    def test_validate_fragment_content_invalid_markdown(self):
        """Test validating invalid Markdown fragment content."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling",
            content="",  # Empty content
            content_type=KnowledgeContentType.MARKDOWN,
        )

        assert KnowledgeUtils.validate_fragment_content(fragment) is False

    def test_extract_keywords(self):
        """Test extracting keywords from fragment."""
        fragment = KnowledgeFragment.create_fragment(
            session_id="session_123",
            agent_type="generator",
            category=KnowledgeCategory.OPTIMIZATION,
            title="Tiling Matrix Optimization",
            content={
                "technique": "tiling",
                "description": "Cache optimization for matrix operations",
                "parameters": ["tile_size", "block_size"],
            },
            content_type=KnowledgeContentType.JSON,
        )

        keywords = KnowledgeUtils.extract_keywords(fragment)

        # Check that important keywords are extracted
        assert "tiling" in keywords
        assert "matrix" in keywords
        assert "optimization" in keywords
        assert "cache" in keywords
        assert "operations" in keywords

        # Check that common words are filtered out
        assert "the" not in keywords
        assert "for" not in keywords
        assert "and" not in keywords

        # Check that short words are filtered out
        assert "a" not in keywords
        assert "an" not in keywords
