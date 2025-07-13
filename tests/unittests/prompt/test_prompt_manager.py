"""
Tests for the prompt manager.
"""

import shutil
import tempfile

import pytest

from pinocchio.prompt import (
    AgentType,
    PromptManager,
    PromptType,
    StructuredInput,
    StructuredOutput,
)
from tests.utils import (
    assert_session_valid,
    assert_task_valid,
    create_mock_llm_client,
    create_test_session,
    create_test_task,
)


class TestPromptManager:
    """Tests for PromptManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create a PromptManager instance."""
        return PromptManager(storage_path=temp_dir)

    def test_create_template(self, manager):
        """Test template creation."""
        template = manager.create_template(
            template_name="generator",
            content="Generate {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="Test template",
            tags=["test", "generation"],
        )

        assert template.template_name == "generator"
        assert template.content == "Generate {{task}}"
        assert template.agent_type == AgentType.GENERATOR
        assert template.prompt_type == PromptType.CODE_GENERATION
        assert template.description == "Test template"
        assert "test" in template.tags

    def test_create_template_with_schemas(self, manager):
        """Test template creation with structured schemas."""
        input_schema = StructuredInput(
            code_snippet="def test(): pass", requirements={"language": "python"}
        )
        output_schema = StructuredOutput(
            generated_code="def optimized(): pass", confidence_score=0.9
        )

        template = manager.create_template(
            template_name="optimizer",
            content="Optimize: {{code_snippet}}",
            agent_type=AgentType.OPTIMIZER,
            prompt_type=PromptType.CODE_OPTIMIZATION,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        assert template.input_schema.code_snippet == "def test(): pass"
        assert template.output_schema.generated_code == "def optimized(): pass"

    def test_get_template(self, manager):
        """Test template retrieval."""
        # Create template
        template = manager.create_template(template_name="test", content="Test content")

        # Get template
        retrieved = manager.get_template("test")
        assert retrieved is not None
        assert retrieved.content == "Test content"

        # Get specific version
        retrieved_version = manager.get_template("test", template.version_id)
        assert retrieved_version.version_id == template.version_id

    def test_get_template_by_agent(self, manager):
        """Test template retrieval by agent type."""
        # Create templates for different agents
        manager.create_template(
            template_name="code_gen",
            content="Generate code",
            agent_type=AgentType.GENERATOR,
        )

        manager.create_template(
            template_name="code_debug",
            content="Debug code",
            agent_type=AgentType.DEBUGGER,
        )

        # Get by agent type
        gen_retrieved = manager.get_template("code_gen", agent_type=AgentType.GENERATOR)
        debug_retrieved = manager.get_template(
            "code_debug", agent_type=AgentType.DEBUGGER
        )

        assert gen_retrieved.agent_type == AgentType.GENERATOR
        assert debug_retrieved.agent_type == AgentType.DEBUGGER

    def test_format_template(self, manager):
        """Test template formatting."""
        template = manager.create_template(
            template_name="test", content="Hello {{name}}, generate {{task}}"
        )

        variables = {"name": "Alice", "task": "a function"}
        formatted = manager.format_template("test", variables)

        assert formatted == "Hello Alice, generate a function"

    def test_format_structured_prompt(self, manager):
        """Test structured prompt formatting."""
        template = manager.create_template(
            template_name="test",
            content="Optimize this code: {{code_snippet}} with requirements: {{requirements}}",
        )

        structured_input = StructuredInput(
            code_snippet="def slow_function(): pass",
            requirements={"language": "python", "performance": "high"},
        )

        formatted = manager.format_structured_prompt("test", structured_input)

        assert "def slow_function(): pass" in formatted
        assert "python" in formatted
        assert "high" in formatted

    def test_list_templates(self, manager):
        """Test template listing."""
        # Create multiple templates
        manager.create_template("template1", "Content 1")
        manager.create_template("template2", "Content 2")
        manager.create_template("template3", "Content 3", agent_type=AgentType.DEBUGGER)

        # List all templates
        all_templates = manager.list_templates()
        assert len(all_templates) == 3
        assert "template1" in all_templates
        assert "template2" in all_templates
        assert "template3" in all_templates

        # List by agent type
        debugger_templates = manager.list_templates(agent_type=AgentType.DEBUGGER)
        assert len(debugger_templates) == 1
        assert "template3" in debugger_templates

    def test_list_template_versions(self, manager):
        """Test template version listing."""
        # Create multiple versions
        manager.create_template("test", "Version 1")
        manager.create_template("test", "Version 2")

        versions = manager.list_template_versions("test")
        assert len(versions) == 2

    def test_set_current_version(self, manager):
        """Test setting current version."""
        template1 = manager.create_template("test", "Version 1")
        manager.create_template("test", "Version 2")

        # Set first version as current
        success = manager.set_current_version("test", template1.version_id)
        assert success is True

        # Verify current version
        current = manager.get_template("test")
        assert current.version_id == template1.version_id

    def test_remove_template(self, manager):
        """Test template removal."""
        template = manager.create_template("test", "Test content")

        # Remove specific version
        success = manager.remove_template("test", template.version_id)
        assert success is True

        # Verify template is removed
        retrieved = manager.get_template("test")
        assert retrieved is None

    def test_search_templates(self, manager):
        """Test template search."""
        manager.create_template(
            "python_generator",
            "Generate Python code for {{task}}",
            description="Python code generation",
            tags=["python", "generation"],
        )

        manager.create_template(
            "debugger",
            "Debug Python code",
            description="Python debugging",
            tags=["python", "debugging"],
        )

        # Search by content
        results = manager.search_templates("Python")
        assert len(results) == 2

        # Search by agent type
        results = manager.search_templates("generation", AgentType.GENERATOR)
        assert len(results) == 1
        assert results[0].template_name == "python_generator"

    def test_update_template_stats(self, manager):
        """Test template statistics update."""
        manager.create_template("test", "Test content")

        # Update stats
        success = manager.update_template_stats("test", True, 0.1)
        assert success is True

        # Verify stats are updated
        retrieved = manager.get_template("test")
        assert retrieved.usage_count == 1
        assert retrieved.success_rate == 1.0
        assert retrieved.average_response_time == 0.1

    def test_get_performance_stats(self, manager):
        """Test performance statistics."""
        manager.create_template("test1", "Content 1")
        manager.create_template("test2", "Content 2")

        # Update stats
        manager.update_template_stats("test1", True, 0.1)
        manager.update_template_stats("test1", True, 0.2)
        manager.update_template_stats("test2", False, 0.3)

        stats = manager.get_performance_stats()
        assert stats["total_templates"] == 2
        assert stats["total_usage"] == 3
        assert stats["overall_success_rate"] == 2 / 3

    def test_export_template(self, manager):
        """Test template export."""
        manager.create_template("test", "Test content", description="Test description")

        # Export as JSON
        exported = manager.export_template("test", format="json")
        assert exported is not None
        assert "test" in exported
        assert "Test content" in exported

        # Export as YAML
        exported_yaml = manager.export_template("test", format="yaml")
        assert exported_yaml is not None
        assert "template_name: test" in exported_yaml

    def test_import_template(self, manager):
        """Test template import."""
        template_data = {
            "template_id": "imported-id",
            "template_name": "imported",
            "version_id": "imported-version",
            "agent_type": "generator",
            "prompt_type": "code_generation",
            "content": "Imported content",
            "description": "Imported description",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "tags": ["imported"],
            "priority": 1,
            "usage_count": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0,
            "parent_version_id": None,
            "change_log": [],
            "input_schema": None,
            "output_schema": None,
        }

        imported = manager.import_template(template_data)
        assert imported.template_name == "imported"
        assert imported.content == "Imported content"

        # Verify template is available
        retrieved = manager.get_template("imported")
        assert retrieved is not None
        assert retrieved.content == "Imported content"

    def test_persistence(self, temp_dir):
        """Test template persistence across manager instances."""
        # Create manager and template
        manager1 = PromptManager(storage_path=temp_dir)
        template = manager1.create_template("test", "Persistent content")

        # Create new manager instance
        manager2 = PromptManager(storage_path=temp_dir)

        # Verify template is loaded
        retrieved = manager2.get_template("test")
        assert retrieved is not None
        assert retrieved.content == "Persistent content"

    def test_edge_cases(self, manager):
        """Test edge cases."""
        # Get nonexistent template
        template = manager.get_template("nonexistent")
        assert template is None

        # Format nonexistent template
        formatted = manager.format_template("nonexistent", {"test": "value"})
        assert formatted is None

        # Update stats for nonexistent template
        success = manager.update_template_stats("nonexistent", True, 0.1)
        assert success is False

        # Export nonexistent template
        exported = manager.export_template("nonexistent")
        assert exported is None

        # Set current version for nonexistent template
        success = manager.set_current_version("nonexistent", "nonexistent-version")
        assert success is False
