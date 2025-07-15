"""
Test module demonstrating usage of Agent Initial Prompts functionality.

This test module serves as both validation and documentation, showing developers
how the agent prompt system works and how to use it effectively.

Key Features Demonstrated:
1. Agent-specific CUDA expertise integration
2. Context-aware prompt generation
3. Cross-agent prompt consistency
4. Real-world usage patterns
"""

import os
import sys
from typing import Any, Dict, Optional

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pinocchio.agents import (
    Agent,
    DebuggerAgent,
    EvaluatorAgent,
    GeneratorAgent,
    OptimizerAgent,
)
from pinocchio.llm import BaseLLMClient


class MockBaseLLMClient(BaseLLMClient):
    """Mock LLM interface for testing purposes."""

    def __init__(self):
        super().__init__()
        self.last_request = None
        self.response_template = "Mock response based on: {prompt_preview}"

    async def complete(self, prompt: str, agent_type: Optional[str] = None) -> str:
        """Store request for inspection and return mock response."""
        self.last_request = {
            "prompt": prompt,
            "agent_type": agent_type,
            "prompt_length": len(prompt),
            "contains_cuda_context": "CUDA" in prompt,
        }

        # Return contextual mock response
        preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        return self.response_template.format(prompt_preview=preview)

    async def complete_structured(
        self, prompt: str, agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return structured mock response."""
        await self.complete(prompt, agent_type)  # Store request

        return {
            "code": "// Mock CUDA code for testing",
            "language": "cuda",
            "explanation": "Mock explanation",
            "optimization_techniques": ["shared_memory", "coalescing"],
            "performance_notes": "Mock performance analysis",
        }

    def send_request(self, prompt: str, context: dict = None) -> str:
        """Backward compatibility method."""
        import asyncio

        return asyncio.create_task(self.complete(prompt)).result()


class TestAgentPromptsUsage:
    """
    Test suite demonstrating usage patterns for agent initial prompts.

    These tests show how developers can:
    1. Create agents with CUDA expertise
    2. Access and modify default prompts
    3. Understand prompt content and structure
    4. Validate agent-specific behaviors
    """

    def setup_method(self):
        """Set up test environment with mock LLM interface."""
        self.mock_llm = MockBaseLLMClient()

    def test_base_agent_cuda_context_usage(self):
        """
        Demonstrates how base Agent class provides CUDA context.

        Usage Pattern:
        - Any agent can access default CUDA expertise
        - Context includes optimization principles, memory management, etc.
        - Provides foundation for specialized agents
        """
        # Create concrete agent instance (using GeneratorAgent as example)
        agent = GeneratorAgent(self.mock_llm)

        # Get default CUDA context - this is how developers access it
        cuda_context = agent._get_default_cuda_context()

        # Validate structure and content
        assert isinstance(cuda_context, str)
        assert len(cuda_context) > 500  # Substantial content

        # Check for essential CUDA concepts
        essential_concepts = [
            "cuda",
            "memory coalescing",
            "shared memory",
            "performance",
            "optimization",
            "gpu",
        ]

        cuda_lower = cuda_context.lower()
        for concept in essential_concepts:
            assert concept.lower() in cuda_lower, f"Missing concept: {concept}"

        # Demonstrate customization - developers can extend this
        custom_context = (
            cuda_context + "\n\nCustom project guidelines: Use compute capability 7.5+"
        )
        assert "compute capability 7.5+" in custom_context

        print(f"✓ Base CUDA context length: {len(cuda_context)} characters")
        print(f"✓ Contains {len(essential_concepts)} essential concepts")

    def test_generator_agent_prompt_usage(self):
        """
        Demonstrates GeneratorAgent prompt construction and usage.

        Usage Pattern:
        - Generator builds prompts with CUDA generation expertise
        - Includes specific generation guidelines and best practices
        - Shows how to create code generation requests
        """
        generator = GeneratorAgent(self.mock_llm)

        # Demonstrate typical usage - generate CUDA kernel
        user_request = "Create a matrix multiplication CUDA kernel"

        # This is how developers would use the generator
        result = generator.generate_simple_code(user_request)

        # Validate the result structure instead of inspecting prompt
        assert isinstance(result, dict)
        assert "code" in result
        assert "language" in result
        assert "explanation" in result

        # Test that the result contains meaningful content
        assert len(result["code"]) > 50  # Substantial code
        assert result["language"] in ["cuda", "choreo_dsl"]

        # Since generate_simple_code doesn't call LLM, test prompt building separately
        test_request = {
            "task_description": user_request,
            "requirements": {"performance": "high"},
            "optimization_goals": ["memory_efficiency"],
        }
        prompt = generator._build_generation_prompt(test_request)

        # Validate prompt contains expected sections
        expected_sections = [
            "CUDA programming expert",
            "code generation",
            "optimization guidelines",
            "memory management",
            "performance considerations",
        ]

        prompt_lower = prompt.lower()
        for section in expected_sections:
            assert section.lower() in prompt_lower, f"Missing section: {section}"

        # Check that user request is properly integrated
        assert user_request in prompt

        # Demonstrate prompt length (should be substantial for comprehensive guidance)
        assert len(prompt) > 1000, "Prompt should be comprehensive"

        print(f"✓ Generator prompt length: {len(prompt)} characters")
        print(f"✓ User request properly integrated: '{user_request}'")
        print(f"✓ Contains {len(expected_sections)} expected sections")

    def test_optimizer_agent_prompt_usage(self):
        """
        Demonstrates OptimizerAgent prompt construction and optimization focus.

        Usage Pattern:
        - Optimizer specializes in performance optimization
        - Includes specific optimization techniques and metrics
        - Shows how to request code optimization
        """
        optimizer = OptimizerAgent(self.mock_llm)

        # Demonstrate optimization request
        code_to_optimize = """
        __global__ void simple_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = data[idx] * 2.0f;
            }
        }
        """

        # This is how developers would use the optimizer
        optimizer.optimize_code(code_to_optimize)

        # Inspect optimization-specific prompt content
        last_request = self.mock_llm.last_request
        prompt = last_request["prompt"]

        # Check for optimization-specific guidance
        optimization_concepts = [
            "performance optimization",
            "memory bandwidth",
            "occupancy optimization",
            "instruction throughput",
            "memory coalescing",
            "bank conflicts",
            "warp utilization",
        ]

        prompt_lower = prompt.lower()
        found_concepts = [
            concept
            for concept in optimization_concepts
            if concept.lower() in prompt_lower
        ]

        assert (
            len(found_concepts) >= 5
        ), f"Found only {len(found_concepts)} optimization concepts"

        # Ensure code is included for optimization
        assert "simple_kernel" in prompt

        print(
            f"✓ Optimizer prompt includes {len(found_concepts)} optimization concepts"
        )
        print(f"✓ Code properly included for optimization")

    def test_debugger_agent_prompt_usage(self):
        """
        Demonstrates DebuggerAgent prompt construction and debugging capabilities.

        Usage Pattern:
        - Debugger focuses on error detection and correction
        - Includes debugging tools and techniques
        - Shows how to request code debugging
        """
        debugger = DebuggerAgent(self.mock_llm)

        # Demonstrate debugging request with problematic code
        buggy_code = """
        __global__ void buggy_kernel(float* input, float* output, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            // Bug: no bounds checking
            output[idx] = input[idx] + 1.0f;
        }
        """

        # This is how developers would use the debugger
        debugger.debug_code(buggy_code)

        # Inspect debugging-specific prompt content
        last_request = self.mock_llm.last_request
        prompt = last_request["prompt"]

        # Check for debugging-specific guidance
        debugging_concepts = [
            "debugging",
            "error detection",
            "bounds checking",
            "memory errors",
            "race conditions",
            "synchronization",
            "validation",
        ]

        prompt_lower = prompt.lower()
        found_concepts = [
            concept for concept in debugging_concepts if concept.lower() in prompt_lower
        ]

        assert (
            len(found_concepts) >= 4
        ), f"Found only {len(found_concepts)} debugging concepts"

        # Ensure code is included for debugging
        assert "buggy_kernel" in prompt

        print(f"✓ Debugger prompt includes {len(found_concepts)} debugging concepts")
        print(f"✓ Buggy code properly included for analysis")

    def test_evaluator_agent_prompt_usage(self):
        """
        Demonstrates EvaluatorAgent prompt construction and evaluation metrics.

        Usage Pattern:
        - Evaluator focuses on performance measurement and analysis
        - Includes specific metrics and benchmarking approaches
        - Shows how to request performance evaluation
        """
        evaluator = EvaluatorAgent(self.mock_llm)

        # Demonstrate evaluation request
        code_to_evaluate = """
        __global__ void optimized_kernel(float* data, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                data[idx] = __fmaf_rn(data[idx], 2.0f, 0.0f);
            }
        }
        """

        # This is how developers would use the evaluator
        evaluator.evaluate_code(code_to_evaluate)

        # Inspect evaluation-specific prompt content
        last_request = self.mock_llm.last_request
        prompt = last_request["prompt"]

        # Check for evaluation-specific guidance
        evaluation_concepts = [
            "performance evaluation",
            "benchmarking",
            "throughput",
            "latency",
            "memory bandwidth",
            "occupancy",
            "efficiency metrics",
            "profiling",
        ]

        prompt_lower = prompt.lower()
        found_concepts = [
            concept
            for concept in evaluation_concepts
            if concept.lower() in prompt_lower
        ]

        assert (
            len(found_concepts) >= 5
        ), f"Found only {len(found_concepts)} evaluation concepts"

        # Ensure code is included for evaluation
        assert "optimized_kernel" in prompt

        print(f"✓ Evaluator prompt includes {len(found_concepts)} evaluation concepts")
        print(f"✓ Code properly included for evaluation")

    def test_cross_agent_prompt_consistency(self):
        """
        Demonstrates consistency across different agent prompts.

        Usage Pattern:
        - All agents share core CUDA expertise
        - Each adds specialized knowledge
        - Shows how to ensure consistent behavior
        """
        # Create all agent types
        agents = {
            "generator": GeneratorAgent(self.mock_llm),
            "optimizer": OptimizerAgent(self.mock_llm),
            "debugger": DebuggerAgent(self.mock_llm),
            "evaluator": EvaluatorAgent(self.mock_llm),
        }

        # Get base CUDA context from each
        base_contexts = {}
        for name, agent in agents.items():
            base_contexts[name] = agent._get_default_cuda_context()

        # Verify all agents share the same base context
        first_context = list(base_contexts.values())[0]
        for name, context in base_contexts.items():
            assert context == first_context, f"Agent {name} has different base context"

        print("✓ All agents share consistent base CUDA context")

        # Test specialized behavior by sending same request to all
        test_request = "Analyze this CUDA kernel performance"
        responses = {}

        for name, agent in agents.items():
            # Use appropriate method for each agent type
            if name == "generator":
                agent.generate_code(test_request)
            elif name == "optimizer":
                agent.optimize_code(test_request)
            elif name == "debugger":
                agent.debug_code(test_request)
            elif name == "evaluator":
                agent.evaluate_code(test_request)

            responses[name] = self.mock_llm.last_request["prompt"]

        # Verify each agent adds specialized context while maintaining base
        specializations = {
            "generator": ["generation", "create", "implement"],
            "optimizer": ["optimize", "performance", "efficiency"],
            "debugger": ["debug", "error", "validate"],
            "evaluator": ["evaluate", "benchmark", "measure"],
        }

        for name, keywords in specializations.items():
            prompt_lower = responses[name].lower()
            found_keywords = [kw for kw in keywords if kw in prompt_lower]
            assert (
                len(found_keywords) > 0
            ), f"Agent {name} missing specialization keywords"

        print("✓ Each agent maintains specialization while sharing base context")

    def test_prompt_customization_patterns(self):
        """
        Demonstrates how developers can customize agent prompts.

        Usage Pattern:
        - Shows how to extend base functionality
        - Demonstrates custom prompt injection
        - Provides patterns for domain-specific customization
        """

        class CustomizedGeneratorAgent(GeneratorAgent):
            """Example of how developers can customize agent prompts."""

            def _get_default_cuda_context(self):
                """Override to add custom domain knowledge."""
                base_context = super()._get_default_cuda_context()

                custom_addition = """

                CUSTOM PROJECT REQUIREMENTS:
                - Target Compute Capability: 7.5 (RTX 2080 Ti)
                - Memory Constraints: 11GB GDDR6
                - Performance Target: >500 GFLOPS for GEMM operations
                - Code Style: Follow Google CUDA Style Guide
                - Error Handling: Always include CUDA error checking
                """

                return base_context + custom_addition

            def generate_code(self, user_request: str, custom_context: str = None):
                """Enhanced generation with custom context support."""
                if custom_context:
                    enhanced_request = (
                        f"{user_request}\n\nAdditional Context: {custom_context}"
                    )
                    return super().generate_code(enhanced_request)
                return super().generate_code(user_request)

        # Demonstrate usage of customized agent
        custom_agent = CustomizedGeneratorAgent(self.mock_llm)

        # Test that custom context is included
        custom_agent.generate_code("Create a GEMM kernel")

        last_request = self.mock_llm.last_request
        prompt = last_request["prompt"]

        # Verify custom additions are present
        assert "Compute Capability: 7.5" in prompt
        assert "RTX 2080 Ti" in prompt
        assert "500 GFLOPS" in prompt
        assert "Google CUDA Style Guide" in prompt

        print("✓ Custom agent successfully adds domain-specific requirements")

        # Test enhanced generation with additional context
        custom_context = "This kernel will be used in a machine learning training loop"
        custom_agent.generate_code("Optimize for training workload", custom_context)

        last_request = self.mock_llm.last_request
        prompt = last_request["prompt"]

        assert "machine learning training loop" in prompt
        assert "training workload" in prompt

        print("✓ Custom context properly integrated into prompts")


if __name__ == "__main__":
    """
    Run usage tests and display results.

    This section shows developers how to run tests and interpret results.
    """
    print("Running Agent Prompts Usage Tests...")
    print("=" * 50)

    # Create test instance
    test_instance = TestAgentPromptsUsage()
    test_instance.setup_method()

    # Run each test with clear output
    tests = [
        ("Base Agent CUDA Context", test_instance.test_base_agent_cuda_context_usage),
        ("Generator Agent Prompts", test_instance.test_generator_agent_prompt_usage),
        ("Optimizer Agent Prompts", test_instance.test_optimizer_agent_prompt_usage),
        ("Debugger Agent Prompts", test_instance.test_debugger_agent_prompt_usage),
        ("Evaluator Agent Prompts", test_instance.test_evaluator_agent_prompt_usage),
        ("Cross-Agent Consistency", test_instance.test_cross_agent_prompt_consistency),
        ("Prompt Customization", test_instance.test_prompt_customization_patterns),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        try:
            print(f"\n{name}:")
            test_func()
            print(f"✅ {name} - PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {name} - FAILED: {e}")

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All agent prompt usage tests passed!")
        print("\nDevelopers can now:")
        print("- Understand how agent prompts work")
        print("- Create customized agents with domain expertise")
        print("- Integrate CUDA knowledge consistently")
        print("- Extend functionality for specific projects")
    else:
        print("⚠️ Some tests failed. Check implementation.")
