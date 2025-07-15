"""
Test module demonstrating usage of Prompt Manager Integration functionality.

This test module serves as both validation and documentation, showing developers
how to use the integrated prompt management system that combines memory, knowledge,
and context-aware prompt generation capabilities.

Key Features Demonstrated:
1. Context-aware prompt generation with memory and knowledge integration
2. Template-based prompt construction with dynamic content
3. Keyword extraction and intelligent content retrieval
4. Multi-component prompt enhancement
5. Prompt optimization for different agent types and scenarios
"""

import os
import sys
import tempfile
from unittest.mock import Mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pinocchio.knowledge import KnowledgeManager
from pinocchio.memory import MemoryManager
from pinocchio.prompt import PromptManager


class TestPromptManagerIntegrationUsage:
    """
    Test suite demonstrating usage patterns for integrated prompt management.

    These tests show how developers can:
    1. Integrate memory, knowledge, and prompt systems
    2. Generate context-aware prompts automatically
    3. Extract keywords and retrieve relevant content
    4. Create enhanced prompts for different scenarios
    5. Optimize prompt performance and quality
    """

    def setup_method(self):
        """Set up test environment with integrated prompt management system."""
        self.temp_dir = tempfile.mkdtemp()

        # Initialize core components
        self.memory_manager = MemoryManager(self.temp_dir)
        self.knowledge_manager = KnowledgeManager(self.temp_dir)
        self.prompt_manager = PromptManager()

        # Initialize knowledge base and sample memories
        self.knowledge_manager.add_cuda_knowledge_base()
        self._populate_sample_memories()

        # Integrate systems
        self.prompt_manager.integrate_memory_and_knowledge(
            self.memory_manager, self.knowledge_manager
        )

    def _populate_sample_memories(self):
        """Populate memory manager with sample development memories."""
        sample_memories = [
            {
                "content": "Successfully optimized matrix multiplication kernel using shared memory",
                "context": {"task_type": "optimization", "agent": "optimizer"},
                "metadata": {"performance_gain": "3x speedup"},
                "tags": ["matrix_multiplication", "shared_memory", "optimization"],
            },
            {
                "content": "Debugged memory coalescing issues in vector addition kernel",
                "context": {"task_type": "debugging", "agent": "debugger"},
                "metadata": {
                    "issue": "uncoalesced_access",
                    "solution": "data_restructuring",
                },
                "tags": ["debugging", "memory_coalescing", "vector_addition"],
            },
            {
                "content": "Generated efficient GEMM kernel with tensor core utilization",
                "context": {"task_type": "generation", "agent": "generator"},
                "metadata": {"performance": "12 TFLOPS", "features": "tensor_cores"},
                "tags": ["GEMM", "tensor_cores", "generation", "high_performance"],
            },
        ]

        for memory_data in sample_memories:
            self.memory_manager.store_memory(
                content=memory_data["content"],
                context=memory_data["context"],
                metadata=memory_data["metadata"],
                tags=memory_data["tags"],
            )

    def test_basic_prompt_integration_usage(self):
        """
        Demonstrates basic integration of memory and knowledge in prompt generation.

        Usage Pattern:
        - Generate prompts with automatic memory and knowledge enhancement
        - Understand component integration and content selection
        - Validate integrated prompt quality and relevance
        """

        # Test basic integration
        task_description = "Optimize CUDA kernel for better memory performance"

        # This is how developers use integrated prompt generation
        enhanced_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Help me with this task: {task_description}",
            context={"task_type": "optimization", "domain": "CUDA"},
            include_memory=True,
            include_knowledge=True,
        )

        # Validate prompt enhancement
        assert task_description in enhanced_prompt
        assert (
            len(enhanced_prompt) > len(task_description) * 3
        )  # Should be significantly enhanced

        # Check for memory integration
        assert (
            "RELEVANT EXPERIENCE:" in enhanced_prompt
            or "PAST EXPERIENCE:" in enhanced_prompt
        )

        # Check for knowledge integration
        assert "EXPERT KNOWLEDGE:" in enhanced_prompt or "KNOWLEDGE:" in enhanced_prompt

        print(f"✓ Basic prompt integration successful")
        print(f"✓ Enhanced prompt length: {len(enhanced_prompt)} characters")
        print(f"✓ Includes both memory and knowledge content")

        # Test component isolation
        memory_only_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Task: {task_description}",
            context={"task_type": "optimization"},
            include_memory=True,
            include_knowledge=False,
        )

        knowledge_only_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Task: {task_description}",
            context={"task_type": "optimization"},
            include_memory=False,
            include_knowledge=True,
        )

        # Validate selective integration
        assert len(memory_only_prompt) < len(enhanced_prompt)
        assert len(knowledge_only_prompt) < len(enhanced_prompt)

        print("✓ Selective component integration works correctly")

    def test_keyword_extraction_and_content_retrieval_usage(self):
        """
        Demonstrates automatic keyword extraction and intelligent content retrieval.

        Usage Pattern:
        - Extract keywords from user requests automatically
        - Retrieve relevant memory and knowledge based on keywords
        - Understand keyword weighting and relevance scoring
        """

        # Test keyword extraction
        complex_request = """
        I need to optimize a CUDA kernel for matrix multiplication that's currently
        running slowly due to memory coalescing issues. The kernel should work well
        on RTX 3080 hardware and achieve high occupancy. I've had problems with
        shared memory bank conflicts before.
        """

        # Extract keywords automatically
        extracted_keywords = self.prompt_manager._extract_keywords(complex_request)

        expected_keywords = [
            "optimize",
            "CUDA",
            "kernel",
            "matrix",
            "multiplication",
            "memory",
            "coalescing",
            "RTX",
            "occupancy",
            "shared",
            "bank",
            "conflicts",
        ]

        # Validate keyword extraction
        found_keywords = [
            kw
            for kw in expected_keywords
            if kw.lower() in [k.lower() for k in extracted_keywords]
        ]
        assert (
            len(found_keywords) >= 6
        ), f"Should extract key terms, found: {found_keywords}"

        print(f"✓ Extracted {len(extracted_keywords)} keywords from complex request")
        print(f"✓ Key terms found: {found_keywords[:6]}")

        # Test intelligent content retrieval
        relevant_content = self.prompt_manager._retrieve_relevant_content(
            keywords=extracted_keywords,
            context={"task_type": "optimization"},
            max_memory_items=2,
            max_knowledge_items=2,
        )

        # Validate content retrieval
        assert "memories" in relevant_content
        assert "knowledge" in relevant_content
        assert len(relevant_content["memories"]) <= 2
        assert len(relevant_content["knowledge"]) <= 2

        # Check relevance of retrieved content
        memory_content = " ".join([m["content"] for m in relevant_content["memories"]])
        knowledge_content = " ".join(
            [k["content"] for k in relevant_content["knowledge"]]
        )

        # Should find content related to optimization and memory
        assert any(
            kw.lower() in memory_content.lower() for kw in ["optim", "memory", "matrix"]
        )
        assert any(
            kw.lower() in knowledge_content.lower()
            for kw in ["memory", "optim", "coalescing"]
        )

        print(f"✓ Retrieved {len(relevant_content['memories'])} relevant memories")
        print(
            f"✓ Retrieved {len(relevant_content['knowledge'])} relevant knowledge fragments"
        )

        # Test keyword weighting for different domains
        debugging_request = (
            "Debug race condition in CUDA reduction kernel with synchronization issues"
        )
        debug_keywords = self.prompt_manager._extract_keywords(debugging_request)

        debug_content = self.prompt_manager._retrieve_relevant_content(
            keywords=debug_keywords,
            context={"task_type": "debugging"},
            max_memory_items=1,
            max_knowledge_items=1,
        )

        # Should prioritize debugging-related content
        if debug_content["memories"]:
            debug_memory = debug_content["memories"][0]
            assert "debug" in debug_memory["content"].lower() or "debug" in [
                tag.lower() for tag in debug_memory["tags"]
            ]

        print("✓ Keyword extraction and content retrieval adapts to different domains")

    def test_template_based_prompt_enhancement_usage(self):
        """
        Demonstrates template-based prompt construction with dynamic content integration.

        Usage Pattern:
        - Use predefined templates for consistent prompt structure
        - Integrate dynamic content into template placeholders
        - Customize templates for different agent types and scenarios
        """

        # Define template for optimization tasks
        optimization_template = """
You are a CUDA optimization expert with extensive experience in high-performance computing.

OPTIMIZATION TASK:
{task_description}

{memory_section}

{knowledge_section}

OPTIMIZATION APPROACH:
Based on the above experience and expert knowledge, please:
1. Analyze the current implementation challenges
2. Identify specific optimization opportunities
3. Provide detailed implementation recommendations
4. Estimate expected performance improvements

Please focus on practical, implementable solutions.
"""

        # Test template-based prompt generation
        task_context = {
            "task_description": "Optimize convolution kernel for better memory bandwidth utilization",
            "agent_type": "optimizer",
            "domain": "CUDA",
        }

        # Generate prompt using template
        templated_prompt = self.prompt_manager.format_template_with_context(
            template=optimization_template,
            context=task_context,
            include_memory=True,
            include_knowledge=True,
        )

        # Validate template integration
        assert task_context["task_description"] in templated_prompt
        assert "OPTIMIZATION TASK:" in templated_prompt
        assert "OPTIMIZATION APPROACH:" in templated_prompt

        # Check dynamic content integration
        assert (
            "RELEVANT EXPERIENCE:" in templated_prompt
            or "PAST EXPERIENCE:" in templated_prompt
        )
        assert (
            "EXPERT KNOWLEDGE:" in templated_prompt or "KNOWLEDGE:" in templated_prompt
        )

        print("✓ Template-based prompt generation successful")
        print(f"✓ Template properly filled with dynamic content")

        # Test different template for debugging
        debugging_template = """
You are a CUDA debugging specialist with deep expertise in identifying and resolving GPU computing issues.

DEBUGGING TASK:
{task_description}

{memory_section}

{knowledge_section}

DEBUGGING METHODOLOGY:
1. Analyze the problem symptoms and context
2. Apply relevant debugging techniques from experience
3. Provide step-by-step diagnostic approach
4. Suggest specific tools and validation methods

Focus on systematic problem-solving and verification.
"""

        debug_context = {
            "task_description": "Debug memory access violation in reduction kernel",
            "agent_type": "debugger",
            "domain": "CUDA",
        }

        debug_prompt = self.prompt_manager.format_template_with_context(
            template=debugging_template,
            context=debug_context,
            include_memory=True,
            include_knowledge=True,
        )

        # Validate debugging template
        assert debug_context["task_description"] in debug_prompt
        assert "DEBUGGING TASK:" in debug_prompt
        assert "DEBUGGING METHODOLOGY:" in debug_prompt

        print("✓ Multiple template types work correctly")
        print("✓ Templates adapt to different agent specializations")

    def test_agent_specific_prompt_customization_usage(self):
        """
        Demonstrates agent-specific prompt customization and enhancement patterns.

        Usage Pattern:
        - Customize prompts for different agent types
        - Apply agent-specific memory and knowledge filtering
        - Optimize prompt content for agent capabilities
        """

        base_task = "Create efficient CUDA kernel for matrix operations"

        # Generator agent prompt
        generator_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Code generation task: {base_task}",
            context={
                "agent_type": "generator",
                "task_type": "generation",
                "domain": "CUDA",
            },
            include_memory=True,
            include_knowledge=True,
        )

        # Optimizer agent prompt
        optimizer_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Optimization task: {base_task}",
            context={
                "agent_type": "optimizer",
                "task_type": "optimization",
                "domain": "CUDA",
            },
            include_memory=True,
            include_knowledge=True,
        )

        # Debugger agent prompt
        debugger_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Debug task: {base_task}",
            context={
                "agent_type": "debugger",
                "task_type": "debugging",
                "domain": "CUDA",
            },
            include_memory=True,
            include_knowledge=True,
        )

        # Evaluator agent prompt
        evaluator_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Evaluation task: {base_task}",
            context={
                "agent_type": "evaluator",
                "task_type": "evaluation",
                "domain": "CUDA",
            },
            include_memory=True,
            include_knowledge=True,
        )

        # Validate agent-specific customization
        agent_prompts = {
            "generator": generator_prompt,
            "optimizer": optimizer_prompt,
            "debugger": debugger_prompt,
            "evaluator": evaluator_prompt,
        }

        for agent_type, prompt in agent_prompts.items():
            assert base_task in prompt
            assert len(prompt) > 500  # Should be substantial
            print(f"✓ {agent_type} prompt: {len(prompt)} characters")

        # Analyze prompt differences
        unique_content = {}
        for agent_type, prompt in agent_prompts.items():
            # Extract unique phrases that indicate agent specialization
            if agent_type == "generator":
                unique_indicators = ["generation", "create", "implement", "code"]
            elif agent_type == "optimizer":
                unique_indicators = ["optimize", "performance", "efficiency", "improve"]
            elif agent_type == "debugger":
                unique_indicators = ["debug", "error", "validate", "diagnose"]
            elif agent_type == "evaluator":
                unique_indicators = ["evaluate", "benchmark", "measure", "assess"]

            found_indicators = [
                ind for ind in unique_indicators if ind.lower() in prompt.lower()
            ]
            unique_content[agent_type] = found_indicators

        # Validate specialization
        for agent_type, indicators in unique_content.items():
            assert len(indicators) >= 2, f"{agent_type} should have specialized content"
            print(f"✓ {agent_type} specialization indicators: {indicators[:3]}")

        print("✓ Agent-specific prompt customization works correctly")

    def test_multi_scenario_prompt_generation_usage(self):
        """
        Demonstrates prompt generation for various real-world scenarios.

        Usage Pattern:
        - Handle different types of CUDA development tasks
        - Adapt prompts to specific hardware targets and constraints
        - Integrate relevant context from multiple sources
        """

        # Scenario 1: Performance optimization for specific hardware
        perf_scenario = {
            "request": "Optimize GEMM kernel for RTX 4090 with 24GB memory",
            "context": {
                "hardware": "RTX_4090",
                "memory_constraint": "24GB",
                "task_type": "optimization",
                "target_operation": "GEMM",
            },
        }

        perf_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Hardware-specific optimization: {perf_scenario['request']}",
            context=perf_scenario["context"],
            include_memory=True,
            include_knowledge=True,
        )

        # Should include hardware and memory considerations
        assert "RTX" in perf_prompt or "4090" in perf_prompt
        assert "GEMM" in perf_prompt
        print(f"✓ Hardware-specific optimization prompt: {len(perf_prompt)} chars")

        # Scenario 2: Debugging complex race conditions
        debug_scenario = {
            "request": "Debug intermittent race condition in multi-GPU reduction",
            "context": {
                "issue_type": "race_condition",
                "complexity": "multi_gpu",
                "task_type": "debugging",
                "algorithm": "reduction",
            },
        }

        debug_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Complex debugging task: {debug_scenario['request']}",
            context=debug_scenario["context"],
            include_memory=True,
            include_knowledge=True,
        )

        # Should include debugging and synchronization guidance
        assert (
            "race" in debug_prompt.lower() or "synchronization" in debug_prompt.lower()
        )
        assert "reduction" in debug_prompt.lower()
        print(f"✓ Complex debugging prompt: {len(debug_prompt)} chars")

        # Scenario 3: Machine learning kernel generation
        ml_scenario = {
            "request": "Generate custom attention mechanism kernel for transformer training",
            "context": {
                "domain": "machine_learning",
                "algorithm": "attention",
                "use_case": "transformer_training",
                "task_type": "generation",
            },
        }

        ml_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"ML kernel generation: {ml_scenario['request']}",
            context=ml_scenario["context"],
            include_memory=True,
            include_knowledge=True,
        )

        # Should include ML and attention-specific guidance
        assert "attention" in ml_prompt.lower()
        assert "transformer" in ml_prompt.lower() or "training" in ml_prompt.lower()
        print(f"✓ ML-specific generation prompt: {len(ml_prompt)} chars")

        # Scenario 4: Performance evaluation and benchmarking
        eval_scenario = {
            "request": "Evaluate and benchmark convolution kernels across different architectures",
            "context": {
                "task_type": "evaluation",
                "comparison_type": "multi_architecture",
                "operation": "convolution",
                "metrics": ["throughput", "energy_efficiency", "accuracy"],
            },
        }

        eval_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Comprehensive evaluation: {eval_scenario['request']}",
            context=eval_scenario["context"],
            include_memory=True,
            include_knowledge=True,
        )

        # Should include evaluation methodology and metrics
        assert "evaluate" in eval_prompt.lower() or "benchmark" in eval_prompt.lower()
        assert "convolution" in eval_prompt.lower()
        print(f"✓ Evaluation and benchmarking prompt: {len(eval_prompt)} chars")

        print("\n✓ Multi-scenario prompt generation handles diverse real-world tasks")

    def test_prompt_quality_and_optimization_usage(self):
        """
        Demonstrates prompt quality assessment and optimization techniques.

        Usage Pattern:
        - Assess prompt quality and relevance
        - Optimize prompt length and content density
        - Balance comprehensiveness with clarity
        """

        task = "Optimize memory access patterns in CUDA matrix multiplication"

        # Generate prompt with different settings
        concise_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Task: {task}",
            context={"task_type": "optimization"},
            include_memory=True,
            include_knowledge=True,
            max_memory_items=1,
            max_knowledge_items=1,
        )

        comprehensive_prompt = self.prompt_manager.create_context_aware_prompt(
            base_prompt=f"Task: {task}",
            context={"task_type": "optimization"},
            include_memory=True,
            include_knowledge=True,
            max_memory_items=3,
            max_knowledge_items=3,
        )

        # Analyze prompt characteristics
        concise_stats = {
            "length": len(concise_prompt),
            "word_count": len(concise_prompt.split()),
            "sections": concise_prompt.count("---")
            + concise_prompt.count("EXPERIENCE:")
            + concise_prompt.count("KNOWLEDGE:"),
        }

        comprehensive_stats = {
            "length": len(comprehensive_prompt),
            "word_count": len(comprehensive_prompt.split()),
            "sections": comprehensive_prompt.count("---")
            + comprehensive_prompt.count("EXPERIENCE:")
            + comprehensive_prompt.count("KNOWLEDGE:"),
        }

        # Validate optimization effects
        assert comprehensive_stats["length"] > concise_stats["length"]
        assert comprehensive_stats["word_count"] > concise_stats["word_count"]

        print("✓ Prompt optimization controls:")
        print(
            f"  Concise: {concise_stats['length']} chars, {concise_stats['word_count']} words"
        )
        print(
            f"  Comprehensive: {comprehensive_stats['length']} chars, {comprehensive_stats['word_count']} words"
        )

        # Test content quality metrics
        quality_metrics = {
            "task_clarity": task in comprehensive_prompt,
            "context_richness": comprehensive_stats["sections"] >= 2,
            "actionable_guidance": any(
                word in comprehensive_prompt.lower()
                for word in ["analyze", "implement", "optimize", "consider"]
            ),
            "domain_expertise": any(
                term in comprehensive_prompt.lower()
                for term in ["cuda", "memory", "kernel", "gpu"]
            ),
            "structured_format": "1." in comprehensive_prompt
            or "2." in comprehensive_prompt
            or "-" in comprehensive_prompt,
        }

        quality_score = sum(quality_metrics.values()) / len(quality_metrics)

        print(f"\n✓ Prompt quality assessment:")
        for metric, passed in quality_metrics.items():
            print(f"  {metric}: {'✓' if passed else '✗'}")
        print(f"  Overall quality score: {quality_score:.1%}")

        assert quality_score >= 0.8, "Prompt should meet high quality standards"

        # Test prompt adaptability
        scenarios = [
            {"domain": "CUDA", "task_type": "optimization"},
            {"domain": "CUDA", "task_type": "debugging"},
            {"domain": "CUDA", "task_type": "generation"},
            {"domain": "CUDA", "task_type": "evaluation"},
        ]

        adaptability_scores = []
        for scenario in scenarios:
            scenario_prompt = self.prompt_manager.create_context_aware_prompt(
                base_prompt=f"Task: {task}",
                context=scenario,
                include_memory=True,
                include_knowledge=True,
            )

            # Check scenario-specific adaptation
            task_type = scenario["task_type"]
            relevant_terms = {
                "optimization": ["optimize", "performance", "efficiency"],
                "debugging": ["debug", "error", "validate"],
                "generation": ["generate", "create", "implement"],
                "evaluation": ["evaluate", "benchmark", "measure"],
            }

            found_terms = sum(
                1
                for term in relevant_terms[task_type]
                if term in scenario_prompt.lower()
            )
            adaptability_scores.append(found_terms / len(relevant_terms[task_type]))

        avg_adaptability = sum(adaptability_scores) / len(adaptability_scores)
        print(f"\n✓ Prompt adaptability score: {avg_adaptability:.1%}")

        assert avg_adaptability >= 0.5, "Prompts should adapt to different scenarios"

        print("✓ Prompt quality and optimization controls work effectively")


if __name__ == "__main__":
    """
    Run prompt manager integration usage tests and display results.

    This section demonstrates comprehensive integrated prompt management testing.
    """
    print("Running Prompt Manager Integration Usage Tests...")
    print("=" * 60)

    # Create test instance
    test_instance = TestPromptManagerIntegrationUsage()
    test_instance.setup_method()

    # Run each test with clear output
    tests = [
        ("Basic Prompt Integration", test_instance.test_basic_prompt_integration_usage),
        (
            "Keyword Extraction and Retrieval",
            test_instance.test_keyword_extraction_and_content_retrieval_usage,
        ),
        (
            "Template-Based Enhancement",
            test_instance.test_template_based_prompt_enhancement_usage,
        ),
        (
            "Agent-Specific Customization",
            test_instance.test_agent_specific_prompt_customization_usage,
        ),
        (
            "Multi-Scenario Generation",
            test_instance.test_multi_scenario_prompt_generation_usage,
        ),
        (
            "Quality and Optimization",
            test_instance.test_prompt_quality_and_optimization_usage,
        ),
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
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All prompt manager integration tests passed!")
        print("\nKey Capabilities Demonstrated:")
        print("- Seamless integration of memory, knowledge, and prompt systems")
        print("- Intelligent keyword extraction and content retrieval")
        print("- Template-based prompt construction with dynamic content")
        print("- Agent-specific prompt customization and optimization")
        print("- Multi-scenario prompt generation for diverse tasks")
        print("- Quality assessment and optimization controls")
        print("\nDevelopers can now:")
        print("- Create sophisticated context-aware prompt systems")
        print("- Integrate multiple AI system components seamlessly")
        print("- Generate high-quality, relevant prompts automatically")
        print("- Customize prompts for different agent types and scenarios")
        print("- Optimize prompt performance and content quality")
    else:
        print("⚠️ Some tests failed. Check prompt manager integration implementation.")
