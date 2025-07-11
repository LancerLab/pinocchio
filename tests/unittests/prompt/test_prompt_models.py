"""
Tests for the prompt template models.
"""

from pinocchio.prompt.models import (
    AgentType,
    PromptMemory,
    PromptTemplate,
    PromptType,
    StructuredInput,
    StructuredOutput,
)


class TestStructuredInput:
    """Tests for StructuredInput model."""

    def test_structured_input_creation(self):
        """Test StructuredInput creation."""
        input_data = StructuredInput(
            code_snippet="def test_function(): pass",
            requirements={"language": "python", "framework": "pytest"},
            context={"project": "test_project"},
            constraints=["no external dependencies"],
            performance_metrics={"execution_time": 0.1},
            optimization_targets=["speed", "memory"],
            debug_info={"error_count": 0},
            evaluation_criteria=["correctness", "performance"],
        )

        assert input_data.code_snippet == "def test_function(): pass"
        assert input_data.requirements["language"] == "python"
        assert input_data.context["project"] == "test_project"
        assert "no external dependencies" in input_data.constraints
        assert input_data.performance_metrics["execution_time"] == 0.1
        assert "speed" in input_data.optimization_targets
        assert input_data.debug_info["error_count"] == 0
        assert "correctness" in input_data.evaluation_criteria

    def test_structured_input_to_dict(self):
        """Test StructuredInput to_dict conversion."""
        input_data = StructuredInput(
            code_snippet="test code", requirements={"test": "value"}
        )

        result = input_data.to_dict()
        assert result["code_snippet"] == "test code"
        assert result["requirements"]["test"] == "value"
        assert result["context"] is None

    def test_structured_input_from_dict(self):
        """Test StructuredInput from_dict creation."""
        data = {
            "code_snippet": "test code",
            "requirements": {"test": "value"},
            "constraints": ["constraint1"],
        }

        input_data = StructuredInput.from_dict(data)
        assert input_data.code_snippet == "test code"
        assert input_data.requirements["test"] == "value"
        assert input_data.constraints == ["constraint1"]


class TestStructuredOutput:
    """Tests for StructuredOutput model."""

    def test_structured_output_creation(self):
        """Test StructuredOutput creation."""
        output_data = StructuredOutput(
            generated_code="def optimized_function(): pass",
            debug_suggestions=["Add error handling"],
            evaluation_results={"score": 0.95},
            optimization_suggestions=["Use vectorization"],
            performance_improvements={"speed": 2.5},
            knowledge_fragments=[{"type": "pattern", "content": "optimization"}],
            confidence_score=0.9,
            execution_time=0.05,
        )

        assert output_data.generated_code == "def optimized_function(): pass"
        assert "Add error handling" in output_data.debug_suggestions
        assert output_data.evaluation_results["score"] == 0.95
        assert "Use vectorization" in output_data.optimization_suggestions
        assert output_data.performance_improvements["speed"] == 2.5
        assert output_data.knowledge_fragments[0]["type"] == "pattern"
        assert output_data.confidence_score == 0.9
        assert output_data.execution_time == 0.05

    def test_structured_output_to_dict(self):
        """Test StructuredOutput to_dict conversion."""
        output_data = StructuredOutput(generated_code="test code", confidence_score=0.8)

        result = output_data.to_dict()
        assert result["generated_code"] == "test code"
        assert result["confidence_score"] == 0.8
        assert result["debug_suggestions"] is None

    def test_structured_output_from_dict(self):
        """Test StructuredOutput from_dict creation."""
        data = {
            "generated_code": "test code",
            "confidence_score": 0.8,
            "debug_suggestions": ["suggestion1"],
        }

        output_data = StructuredOutput.from_dict(data)
        assert output_data.generated_code == "test code"
        assert output_data.confidence_score == 0.8
        assert output_data.debug_suggestions == ["suggestion1"]


class TestPromptTemplate:
    """Tests for PromptTemplate model."""

    def test_prompt_template_creation(self):
        """Test PromptTemplate creation."""
        template = PromptTemplate.create_new_version(
            template_name="generator",
            content="Generate a function that {{task}}",
            agent_type=AgentType.GENERATOR,
            prompt_type=PromptType.CODE_GENERATION,
            description="Basic generator template",
            tags=["code-generation", "python"],
        )

        assert template.template_name == "generator"
        assert template.content == "Generate a function that {{task}}"
        assert template.agent_type == AgentType.GENERATOR
        assert template.prompt_type == PromptType.CODE_GENERATION
        assert template.description == "Basic generator template"
        assert "code-generation" in template.tags
        assert template.version_id is not None
        assert template.template_id is not None

    def test_prompt_template_with_schemas(self):
        """Test PromptTemplate with structured schemas."""
        input_schema = StructuredInput(
            code_snippet="def test(): pass", requirements={"language": "python"}
        )
        output_schema = StructuredOutput(
            generated_code="def optimized(): pass", confidence_score=0.9
        )

        template = PromptTemplate.create_new_version(
            template_name="optimizer",
            content="Optimize this code: {{code_snippet}}",
            agent_type=AgentType.OPTIMIZER,
            prompt_type=PromptType.CODE_OPTIMIZATION,
            input_schema=input_schema,
            output_schema=output_schema,
        )

        assert template.input_schema.code_snippet == "def test(): pass"
        assert template.output_schema.generated_code == "def optimized(): pass"
        assert template.agent_type == AgentType.OPTIMIZER
        assert template.prompt_type == PromptType.CODE_OPTIMIZATION

    def test_prompt_template_to_dict(self):
        """Test PromptTemplate to_dict conversion."""
        template = PromptTemplate.create_new_version(
            template_name="test",
            content="Test template",
            agent_type=AgentType.DEBUGGER,
            prompt_type=PromptType.CODE_DEBUGGING,
        )

        result = template.to_dict()
        assert result["template_name"] == "test"
        assert result["content"] == "Test template"
        assert result["agent_type"] == "debugger"
        assert result["prompt_type"] == "code_debugging"
        assert "template_id" in result
        assert "version_id" in result

    def test_prompt_template_from_dict(self):
        """Test PromptTemplate from_dict creation."""
        data = {
            "template_id": "test-id",
            "template_name": "test",
            "version_id": "test-version",
            "agent_type": "evaluator",
            "prompt_type": "code_evaluation",
            "content": "Test template",
            "description": "Test description",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "tags": ["test"],
            "priority": 1,
            "usage_count": 0,
            "success_rate": 0.0,
            "average_response_time": 0.0,
            "parent_version_id": None,
            "change_log": [],
            "input_schema": None,
            "output_schema": None,
        }

        template = PromptTemplate.from_dict(data)
        assert template.template_name == "test"
        assert template.agent_type == AgentType.EVALUATOR
        assert template.prompt_type == PromptType.CODE_EVALUATION
        assert template.content == "Test template"

    def test_prompt_template_usage_stats(self):
        """Test PromptTemplate usage statistics update."""
        template = PromptTemplate.create_new_version(
            template_name="test", content="Test template"
        )

        # First usage - successful
        template.update_usage_stats(True, 0.1)
        assert template.usage_count == 1
        assert template.success_rate == 1.0
        assert template.average_response_time == 0.1

        # Second usage - failed
        template.update_usage_stats(False, 0.2)
        assert template.usage_count == 2
        assert template.success_rate == 0.5
        assert abs(template.average_response_time - 0.15) < 1e-10


class TestPromptMemory:
    """Tests for PromptMemory model."""

    def test_prompt_memory_add_template(self):
        """Test adding templates to memory."""
        memory = PromptMemory()

        template1 = PromptTemplate.create_new_version(
            template_name="generator",
            content="Generate {{task}}",
            agent_type=AgentType.GENERATOR,
        )

        version_id = memory.add_template(template1)
        assert version_id == template1.version_id

        # Verify template is stored
        retrieved = memory.get_template("generator")
        assert retrieved.content == template1.content
        assert retrieved.agent_type == AgentType.GENERATOR

    def test_prompt_memory_multi_agent_support(self):
        """Test multi-agent template support."""
        memory = PromptMemory()

        # Add templates for different agents
        generator_template = PromptTemplate.create_new_version(
            template_name="code_gen",
            content="Generate code for {{task}}",
            agent_type=AgentType.GENERATOR,
        )

        debugger_template = PromptTemplate.create_new_version(
            template_name="code_debug",
            content="Debug this code: {{code}}",
            agent_type=AgentType.DEBUGGER,
        )

        memory.add_template(generator_template)
        memory.add_template(debugger_template)

        # Test agent-specific retrieval
        gen_retrieved = memory.get_template_by_agent(AgentType.GENERATOR, "code_gen")
        debug_retrieved = memory.get_template_by_agent(AgentType.DEBUGGER, "code_debug")

        assert gen_retrieved.agent_type == AgentType.GENERATOR
        assert debug_retrieved.agent_type == AgentType.DEBUGGER

    def test_prompt_memory_version_management(self):
        """Test version management in memory."""
        memory = PromptMemory()

        # Add first version
        template1 = PromptTemplate.create_new_version(
            template_name="test", content="Version 1", description="V1"
        )
        memory.add_template(template1)

        # Add second version
        template2 = PromptTemplate.create_new_version(
            template_name="test", content="Version 2", description="V2"
        )
        memory.add_template(template2)

        # List versions
        versions = memory.list_template_versions("test")
        assert len(versions) == 2
        assert template1.version_id in versions
        assert template2.version_id in versions

        # Set current version
        memory.set_current_version("test", template1.version_id)
        current = memory.get_template("test")
        assert current.version_id == template1.version_id

    def test_prompt_memory_search(self):
        """Test template search functionality."""
        memory = PromptMemory()

        template1 = PromptTemplate.create_new_version(
            template_name="python_generator",
            content="Generate Python code for {{task}}",
            description="Python code generation",
            tags=["python", "generation"],
        )

        template2 = PromptTemplate.create_new_version(
            template_name="debugger",
            content="Debug Python code",
            description="Python debugging",
            tags=["python", "debugging"],
        )

        memory.add_template(template1)
        memory.add_template(template2)

        # Search by content
        results = memory.search_templates("Python")
        assert len(results) == 2

        # Search by agent type
        results = memory.search_templates("generation", AgentType.GENERATOR)
        assert len(results) == 1
        assert results[0].template_name == "python_generator"

    def test_prompt_memory_remove_template(self):
        """Test template removal."""
        memory = PromptMemory()

        template = PromptTemplate.create_new_version(
            template_name="test", content="Test template"
        )
        memory.add_template(template)

        # Remove specific version
        success = memory.remove_template("test", template.version_id)
        assert success is True

        # Verify template is removed
        retrieved = memory.get_template("test")
        assert retrieved is None

    def test_prompt_memory_performance_stats(self):
        """Test performance statistics."""
        memory = PromptMemory()

        template1 = PromptTemplate.create_new_version(
            template_name="test1", content="Template 1"
        )
        template1.update_usage_stats(True, 0.1)
        template1.update_usage_stats(True, 0.2)

        template2 = PromptTemplate.create_new_version(
            template_name="test2", content="Template 2"
        )
        template2.update_usage_stats(False, 0.3)

        memory.add_template(template1)
        memory.add_template(template2)

        stats = memory.get_performance_stats()
        assert stats["total_templates"] == 2
        assert stats["total_usage"] == 3
        assert stats["overall_success_rate"] == 2 / 3  # 2 successes out of 3 total

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

        # Search with no results
        results = memory.search_templates("nonexistent")
        assert len(results) == 0
