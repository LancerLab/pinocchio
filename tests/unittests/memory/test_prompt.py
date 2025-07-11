"""
Tests for the prompt template models.
"""

from pinocchio.prompt.models import AgentType, PromptMemory, PromptTemplate, PromptType


class TestPromptModels:
    """Tests for the prompt template models."""

    def test_prompt_template(self):
        """Test PromptTemplate basic creation and serialization."""
        template = PromptTemplate.create_new_version(
            template_name="generator",
            content="Generate a function that {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="Basic generator template",
        )

        assert template.template_name == "generator"
        assert template.content == "Generate a function that {{task}}"
        assert template.description == "Basic generator template"
        assert template.version_id is not None
        assert template.agent_type == AgentType.GENERATOR
        assert template.prompt_type == PromptType.CODE_GENERATION

        # Test to_dict/from_dict
        d = template.to_dict()
        t2 = PromptTemplate.from_dict(d)
        assert t2.template_name == template.template_name
        assert t2.agent_type == template.agent_type
        assert t2.prompt_type == template.prompt_type
        assert t2.content == template.content
        assert t2.version_id == template.version_id

    def test_prompt_memory(self):
        """Test PromptMemory operations."""
        memory = PromptMemory()

        # Add a template
        template1 = PromptTemplate.create_new_version(
            template_name="generator",
            content="Generate a function that {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="V1",
        )
        version_id = memory.add_template(template1)
        assert version_id == template1.version_id

        # Get the template
        retrieved = memory.get_template("generator")
        assert retrieved.content == template1.content

        # Add another version
        template2 = PromptTemplate.create_new_version(
            template_name="generator",
            content="Create a function to {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="V2",
        )
        memory.add_template(template2)

        # List templates
        templates = memory.list_templates()
        assert "generator" in templates
        assert templates["generator"] == template2.version_id

        # List template versions
        versions = memory.list_template_versions("generator")
        assert len(versions) == 2
        assert template1.version_id in versions
        assert template2.version_id in versions

        # Set current version
        result = memory.set_current_version("generator", template1.version_id)
        assert result is True

        # Get current template
        current = memory.get_template("generator")
        assert current.version_id == template1.version_id

        # Test get_template_by_agent
        by_agent = memory.get_template_by_agent(AgentType.GENERATOR, "generator")
        assert by_agent is not None
        assert by_agent.template_name == "generator"

        # Test list_templates_by_agent
        agent_templates = memory.list_templates_by_agent(AgentType.GENERATOR)
        assert "generator" in agent_templates

        # Test search_templates
        results = memory.search_templates("function", agent_type=AgentType.GENERATOR)
        assert any(t.template_name == "generator" for t in results)

        # Test record_usage and get_performance_stats
        memory.record_usage("generator", success=True)
        stats = memory.get_performance_stats()
        assert stats["total_templates"] >= 1
        assert stats["total_usage"] >= 1
        assert stats["overall_success_rate"] >= 0.0

    def test_prompt_memory_edge_cases(self):
        """Test edge cases for PromptMemory."""
        memory = PromptMemory()

        # Get nonexistent template
        template = memory.get_template("nonexistent")
        assert template is None

        # Get nonexistent version
        template = memory.get_template("nonexistent", "nonexistent-version")
        assert template is None

        # Set nonexistent template version
        result = memory.set_current_version("nonexistent", "nonexistent-version")
        assert result is False

        # List versions of nonexistent template
        versions = memory.list_template_versions("nonexistent")
        assert versions == {}

        # Add template with same name but different content
        template1 = PromptTemplate.create_new_version(
            template_name="test",
            content="Version 1",
            agent_type=AgentType.OPTIMIZER,
            prompt_type=PromptType.CODE_OPTIMIZATION,
            description="V1",
        )
        memory.add_template(template1)

        template2 = PromptTemplate.create_new_version(
            template_name="test",
            content="Version 2",
            agent_type=AgentType.OPTIMIZER,
            prompt_type=PromptType.CODE_OPTIMIZATION,
            description="V2",
        )
        memory.add_template(template2)

        # Verify that both versions exist
        versions = memory.list_template_versions("test")
        assert len(versions) == 2

        # Test getting a template that exists but has no current version set
        # First create a situation where this can happen
        test_memory = PromptMemory()
        test_memory.templates = {"test": {template1.version_id: template1}}
        # Note: we don't set current_versions["test"]

        # Now try to get the template
        retrieved = test_memory.get_template("test")
        assert retrieved is None
