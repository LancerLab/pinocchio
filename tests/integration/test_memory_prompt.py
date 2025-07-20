"""
Integration tests between Memory and Prompt modules.
"""

import pytest

from pinocchio.memory.models import CodeMemory, CodeVersion
from pinocchio.prompt.models import PromptMemory, PromptTemplate


@pytest.fixture
def prompt_memory():
    """Create a prompt memory instance."""
    return PromptMemory()


@pytest.fixture
def code_memory():
    """Create a code memory instance."""
    return CodeMemory(session_id="test-session")


@pytest.fixture
def sample_code_version(code_memory):
    """Create a sample code version."""
    code = "def example(): return 42"
    version = CodeVersion.create_new_version(
        session_id="test-session",
        code=code,
        language="python",
        kernel_type="example_function",
        source_agent="generator",
        description="Test version",
    )
    code_memory.add_version(version)
    return version


@pytest.fixture
def memory_prompt_bridge(prompt_memory, code_memory):
    """Create a memory-prompt bridge."""
    from pinocchio.memory.bridge import MemoryPromptBridge

    return MemoryPromptBridge(prompt_memory=prompt_memory, code_memory=code_memory)


class TestMemoryPromptIntegration:
    """Integration tests between Memory and Prompt modules."""

    def test_prompt_template_versioning(self, prompt_memory):
        """Test versioning of prompt templates in memory."""
        # Create initial version
        template1 = PromptTemplate.create_new_version(
            template_name="generator",
            content="Generate a function that {{task}}",
            description="V1",
        )

        version_id1 = prompt_memory.add_template(template1)

        # Create updated version
        template2 = PromptTemplate.create_new_version(
            template_name="generator",
            content="Create a function to {{task}} with {{language}}",
            description="V2",
        )

        version_id2 = prompt_memory.add_template(template2)

        # Verify both versions exist
        versions = prompt_memory.list_template_versions("generator")
        assert version_id1 in versions
        assert version_id2 in versions

        # Verify current version is the latest
        current = prompt_memory.get_template("generator")
        assert current.version_id == version_id2

        # Switch to previous version
        prompt_memory.set_current_version("generator", version_id1)
        current = prompt_memory.get_template("generator")
        assert current.version_id == version_id1

    def test_prompt_formatting_with_code(
        self, memory_prompt_bridge, sample_code_version
    ):
        """Test formatting prompts with code from memory."""
        # Create a template that references code
        template = PromptTemplate.create_new_version(
            template_name="debugger",
            content="Debug the following code:\n\n```\n{{code}}\n```\n\nFix any issues you find.",
        )

        memory_prompt_bridge.prompt_memory.add_template(template)

        # Create a prompt with code reference
        prompt = memory_prompt_bridge.create_debugger_prompt(
            sample_code_version.version_id
        )

        # Verify prompt contains the code
        assert sample_code_version.code in prompt["content"]
        assert "Debug the following code" in prompt["content"]

    def test_prompt_with_agent_history(self, memory_prompt_bridge, code_memory):
        """Test creating prompts with agent interaction history."""
        # Set up mock memory manager with interaction history
        from unittest.mock import MagicMock

        from pinocchio.memory.models import DebuggerMemory, GeneratorMemory

        memory_manager = MagicMock()
        memory_prompt_bridge.memory_manager = memory_manager

        # Create code versions
        v1 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): pass",
            language="python",
            kernel_type="example_function",
            source_agent="generator",
            description="Initial version",
        )
        code_memory.add_version(v1)

        v2 = CodeVersion.create_new_version(
            session_id="test-session",
            code="def example(): return 42",
            language="python",
            kernel_type="example_function",
            source_agent="debugger",
            parent_version_id=v1.version_id,
            description="Fixed version",
        )
        code_memory.add_version(v2)

        # Mock agent memories
        generator_memory = GeneratorMemory(
            session_id="test-session",
            version_id=v1.version_id,
            input_data={"prompt": "Generate a function"},
            output_data={"code": v1.code},
            processing_time_ms=100,
            generation_strategy="basic",
            code_version_id=v1.version_id,
        )

        debugger_memory = DebuggerMemory(
            session_id="test-session",
            version_id=v2.version_id,
            input_data={"code": v1.code},
            output_data={"fixed_code": v2.code},
            processing_time_ms=150,
            errors=["Function doesn't return a value"],
            warnings=[],
            compilation_status="success",
            code_version_id=v2.version_id,
        )

        # Set up mock to return these memories
        memory_manager.query_agent_memories.return_value = [
            debugger_memory,
            generator_memory,
        ]

        # Create optimizer template
        template = PromptTemplate.create_new_version(
            template_name="optimizer",
            content="Optimize the following code:\n\n```\n{{code}}\n```\n\nPrevious issues:\n{{previous_issues}}",
        )
        memory_prompt_bridge.prompt_memory.add_template(template)

        # Create optimizer prompt
        prompt = memory_prompt_bridge.create_optimizer_prompt(v2.version_id)

        # Verify prompt contains code and history
        assert v2.code in prompt["content"]
        assert "Function doesn't return a value" in prompt["previous_issues"]

    def test_template_inheritance(self, prompt_memory):
        """Test template inheritance and composition."""
        # Create base template
        base_template = PromptTemplate.create_new_version(
            template_name="base",
            content="# {{title}}\n\n{{content}}\n\n## Instructions\n{{instructions}}",
        )
        prompt_memory.add_template(base_template)

        # Create specialized template that uses base
        specialized_template = PromptTemplate.create_new_version(
            template_name="specialized",
            content="{% include 'base' %}\n\n## Additional Notes\n{{notes}}",
        )
        prompt_memory.add_template(specialized_template)

        # Set up template formatter
        from pinocchio.prompt.formatter import TemplateFormatter

        formatter = TemplateFormatter()

        # Format the specialized template
        variables = {
            "title": "Test Template",
            "content": "This is the main content.",
            "instructions": "Follow these instructions.",
            "notes": "These are additional notes.",
        }

        # This would normally be handled by the bridge, but we're testing directly
        base_content = prompt_memory.get_template("base").content
        specialized_content = prompt_memory.get_template("specialized").content

        # Replace include tag with base content
        content = specialized_content.replace("{% include 'base' %}", base_content)

        # Format the content
        formatted = formatter.format_string(content, variables)

        # Verify all variables were substituted
        assert "# Test Template" in formatted
        assert "This is the main content." in formatted
        assert "Follow these instructions." in formatted
        assert "These are additional notes." in formatted
