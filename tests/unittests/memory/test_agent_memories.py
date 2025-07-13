"""
Tests for the agent memory models.
"""

from pinocchio.memory import (
    DebuggerMemory,
    EvaluatorMemory,
    GeneratorMemory,
    OptimizerMemory,
)
from tests.utils import (
    assert_session_valid,
    assert_task_valid,
    create_mock_llm_client,
    create_test_session,
    create_test_task,
)


class TestAgentMemories:
    """Tests for the agent memory classes."""

    def test_generator_memory(self):
        """Test GeneratorMemory."""
        memory = GeneratorMemory(
            session_id="test-session",
            version_id="test-version",
            input_data={"prompt": "Generate a function"},
            output_data={"code": "def example(): return 42"},
            processing_time_ms=100,
            generation_strategy="basic",
        )

        assert memory.agent_type == "generator"
        assert memory.session_id == "test-session"
        assert memory.version_id == "test-version"
        assert memory.input_data["prompt"] == "Generate a function"
        assert memory.output_data["code"] == "def example(): return 42"
        assert memory.processing_time_ms == 100
        assert memory.generation_strategy == "basic"

    def test_debugger_memory(self):
        """Test DebuggerMemory."""
        memory = DebuggerMemory(
            session_id="test-session",
            version_id="test-version",
            input_data={"code": "def example(): return 42"},
            output_data={"fixed_code": "def example(): return 42"},
            processing_time_ms=150,
            errors=["No issues found"],
            warnings=[],
            compilation_status="success",
        )

        assert memory.agent_type == "debugger"
        assert memory.version_id == "test-version"
        assert memory.errors[0] == "No issues found"
        assert memory.compilation_status == "success"
        assert memory.warnings == []

    def test_optimizer_memory(self):
        """Test OptimizerMemory."""
        memory = OptimizerMemory(
            session_id="test-session",
            version_id="test-version",
            input_data={"code": "def example(): return 42"},
            output_data={"optimized_code": "def example(): return 42"},
            processing_time_ms=200,
            optimization_patterns=[{"name": "No optimization needed"}],
            recommendation="No changes needed",
        )

        assert memory.agent_type == "optimizer"
        assert memory.version_id == "test-version"
        assert memory.optimization_patterns[0]["name"] == "No optimization needed"
        assert memory.recommendation == "No changes needed"

    def test_evaluator_memory(self):
        """Test EvaluatorMemory."""
        memory = EvaluatorMemory(
            session_id="test-session",
            version_id="test-version",
            input_data={"code": "def example(): return 42"},
            output_data={"evaluation": "Good"},
            processing_time_ms=120,
            current_optimization_techniques=["loop_unrolling"],
            current_hyperparameters={"optimization_level": 2},
            performance_analysis={"execution_time": 0.001},
            bottlenecks=["memory_access"],
            target_performance={"target_time": 0.0005},
        )

        assert memory.agent_type == "evaluator"
        assert memory.version_id == "test-version"
        assert memory.current_optimization_techniques == ["loop_unrolling"]
        assert memory.current_hyperparameters["optimization_level"] == 2
        assert memory.performance_analysis["execution_time"] == 0.001
        assert memory.bottlenecks == ["memory_access"]

    def test_agent_memory_with_error(self):
        """Test agent memory with error details."""
        memory = GeneratorMemory(
            session_id="test-session",
            version_id="test-version",
            input_data={"prompt": "Generate a function"},
            output_data={},
            processing_time_ms=50,
            generation_strategy="basic",
            status="error",
            error_details={"type": "ApiError", "message": "API timeout"},
        )

        assert memory.status == "error"
        assert memory.error_details["type"] == "ApiError"
        assert memory.error_details["message"] == "API timeout"

    def test_agent_memory_with_code_version(self):
        """Test agent memory with code version reference."""
        memory = DebuggerMemory(
            session_id="test-session",
            version_id="test-version",
            input_data={"code": "def example(): return 42"},
            output_data={"fixed_code": "def example(): return 42"},
            processing_time_ms=150,
            identified_issues=[],
            fixed_issues=[],
            compilation_success=True,
            code_version_id="test-version-id",
        )

        assert memory.code_version_id == "test-version-id"

    def test_agent_memory_serialization(self):
        """Test serializing and deserializing agent memories."""
        memory = GeneratorMemory(
            session_id="test-session",
            version_id="test-version",
            input_data={"prompt": "Generate a function"},
            output_data={"code": "def example(): return 42"},
            processing_time_ms=100,
            generation_strategy="basic",
        )

        json_str = memory.model_dump_json()
        deserialized = GeneratorMemory.model_validate_json(json_str)

        assert deserialized.session_id == memory.session_id
        assert deserialized.version_id == memory.version_id
        assert deserialized.input_data == memory.input_data
        assert deserialized.output_data == memory.output_data
        assert deserialized.generation_strategy == memory.generation_strategy
