"""
Test module demonstrating usage of Knowledge System functionality.

This test module serves as both validation and documentation, showing developers
how to use the knowledge management system for CUDA expertise, domain knowledge,
and intelligent knowledge retrieval.

Key Features Demonstrated:
1. CUDA knowledge base with comprehensive domain expertise
2. Keyword-based knowledge queries with relevance scoring
3. Knowledge fragment organization and categorization
4. Integration with prompt generation for expert guidance
5. Dynamic knowledge base expansion and maintenance
"""

import json
import os
import sys
import tempfile
from unittest.mock import Mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pinocchio.knowledge import KnowledgeManager


class TestKnowledgeSystemUsage:
    """
    Test suite demonstrating usage patterns for the knowledge system.

    These tests show how developers can:
    1. Access comprehensive CUDA knowledge base
    2. Query knowledge using keywords and context
    3. Integrate knowledge with prompt generation
    4. Expand knowledge base with custom domains
    5. Maintain and organize knowledge fragments
    """

    def setup_method(self):
        """Set up test environment with knowledge manager and CUDA knowledge base."""
        self.temp_dir = tempfile.mkdtemp()
        self.knowledge_manager = KnowledgeManager(self.temp_dir)

        # Initialize with CUDA knowledge base
        self.knowledge_manager.add_cuda_knowledge_base()

    def test_cuda_knowledge_base_access_usage(self):
        """
        Demonstrates access to the comprehensive CUDA knowledge base.

        Usage Pattern:
        - Access built-in CUDA expertise and best practices
        - Understand knowledge organization by categories
        - Validate knowledge content and structure
        """

        # Get all available knowledge fragments
        all_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=[""], min_score=0.0
        )

        assert len(all_knowledge) >= 10, "Should have comprehensive CUDA knowledge base"
        print(
            f"✓ CUDA knowledge base contains {len(all_knowledge)} knowledge fragments"
        )

        # Validate knowledge structure
        required_fields = ["title", "content", "category", "tags", "difficulty_level"]

        for knowledge in all_knowledge[:3]:  # Check first 3 fragments
            for field in required_fields:
                assert field in knowledge, f"Missing field: {field}"

            # Validate content quality
            assert (
                len(knowledge["content"]) > 100
            ), "Knowledge content should be substantial"
            assert len(knowledge["tags"]) >= 2, "Should have multiple relevant tags"
            assert knowledge["difficulty_level"] in [
                "beginner",
                "intermediate",
                "advanced",
            ], "Valid difficulty level"

        print("✓ Knowledge fragments have proper structure and substantial content")

        # Check knowledge categories
        categories = set(knowledge["category"] for knowledge in all_knowledge)
        expected_categories = [
            "memory_management",
            "performance_optimization",
            "debugging",
            "parallel_algorithms",
            "hardware_architecture",
        ]

        for expected_cat in expected_categories:
            assert (
                expected_cat in categories
            ), f"Missing knowledge category: {expected_cat}"

        print(
            f"✓ Knowledge base covers {len(categories)} categories: {sorted(categories)}"
        )

        # Validate CUDA-specific content
        cuda_concepts_found = set()
        for knowledge in all_knowledge:
            content_lower = knowledge["content"].lower()
            tags_lower = [tag.lower() for tag in knowledge["tags"]]

            # Check for essential CUDA concepts
            cuda_concepts = [
                "memory coalescing",
                "occupancy",
                "warp",
                "shared memory",
                "global memory",
                "registers",
                "streaming multiprocessor",
            ]

            for concept in cuda_concepts:
                if concept in content_lower or any(
                    concept in tag for tag in tags_lower
                ):
                    cuda_concepts_found.add(concept)

        assert (
            len(cuda_concepts_found) >= 5
        ), f"Should cover major CUDA concepts, found: {cuda_concepts_found}"
        print(
            f"✓ Knowledge base covers {len(cuda_concepts_found)} essential CUDA concepts"
        )

    def test_keyword_based_knowledge_queries_usage(self):
        """
        Demonstrates keyword-based knowledge search with relevance scoring.

        Usage Pattern:
        - Search knowledge using domain-specific keywords
        - Understand relevance scoring and ranking
        - Filter knowledge by categories and difficulty levels
        """

        # Test single keyword search
        print("--- Memory Optimization Knowledge ---")
        memory_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["memory", "optimization"], min_score=0.2
        )

        assert len(memory_knowledge) >= 2, "Should find memory optimization knowledge"

        for knowledge in memory_knowledge:
            print(f"Score: {knowledge['relevance_score']:.2f} - {knowledge['title']}")
            print(
                f"  Category: {knowledge['category']} | Difficulty: {knowledge['difficulty_level']}"
            )

            # Validate relevance
            content_and_tags = (
                knowledge["content"].lower() + " ".join(knowledge["tags"]).lower()
            )
            assert "memory" in content_and_tags or "optim" in content_and_tags

        print(
            f"✓ Found {len(memory_knowledge)} relevant memory optimization knowledge fragments"
        )

        # Test specific CUDA concept search
        print("\n--- Occupancy Optimization Knowledge ---")
        occupancy_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["occupancy", "optimization"], min_score=0.3
        )

        for knowledge in occupancy_knowledge:
            print(f"Score: {knowledge['relevance_score']:.2f} - {knowledge['title']}")

            # Validate occupancy-specific content
            content_lower = knowledge["content"].lower()
            assert "occupancy" in content_lower or "occupancy" in [
                tag.lower() for tag in knowledge["tags"]
            ]

        print(
            f"✓ Found {len(occupancy_knowledge)} occupancy optimization knowledge fragments"
        )

        # Test debugging knowledge search
        print("\n--- Debugging Knowledge ---")
        debug_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["debug", "error", "validation"],
            category_filter="debugging",
            min_score=0.2,
        )

        for knowledge in debug_knowledge:
            assert knowledge["category"] == "debugging"
            print(f"Debug knowledge: {knowledge['title']}")

        print(f"✓ Found {len(debug_knowledge)} debugging-specific knowledge fragments")

        # Test difficulty-based filtering
        print("\n--- Advanced Knowledge Only ---")
        advanced_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["optimization"], difficulty_filter="advanced", min_score=0.1
        )

        for knowledge in advanced_knowledge:
            assert knowledge["difficulty_level"] == "advanced"
            print(f"Advanced: {knowledge['title']}")

        print(f"✓ Found {len(advanced_knowledge)} advanced-level knowledge fragments")

        # Test ranking verification
        all_optimization = self.knowledge_manager.query_by_keywords(
            keywords=["optimization"], min_score=0.0
        )

        # Verify proper ranking by relevance score
        for i in range(len(all_optimization) - 1):
            assert (
                all_optimization[i]["relevance_score"]
                >= all_optimization[i + 1]["relevance_score"]
            )

        print("✓ Knowledge fragments properly ranked by relevance score")

    def test_knowledge_integration_with_prompts_usage(self):
        """
        Demonstrates integration of knowledge system with prompt generation.

        Usage Pattern:
        - Format knowledge for prompt inclusion
        - Generate expert-level prompts using relevant knowledge
        - Show knowledge-enhanced prompt examples for different scenarios
        """

        # Scenario 1: Memory optimization guidance
        print("--- Memory Optimization Prompt Enhancement ---")
        task_description = (
            "Optimize CUDA kernel for better memory bandwidth utilization"
        )

        relevant_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["memory", "bandwidth", "coalescing", "optimization"],
            min_score=0.3,
            max_results=2,
        )

        assert len(relevant_knowledge) >= 1, "Should find relevant memory knowledge"

        # Format knowledge for prompt inclusion
        formatted_knowledge = []
        for knowledge in relevant_knowledge:
            formatted = self.knowledge_manager._format_fragment_for_prompt(knowledge)
            formatted_knowledge.append(formatted)

            # Validate formatting
            assert "TITLE:" in formatted
            assert "CATEGORY:" in formatted
            assert "CONTENT:" in formatted
            assert knowledge["title"] in formatted
            assert knowledge["content"] in formatted

        # Generate expert prompt with knowledge
        expert_prompt = f"""
You are a CUDA optimization expert with deep knowledge of memory systems.

TASK: {task_description}

EXPERT KNOWLEDGE:
"""

        for i, formatted in enumerate(formatted_knowledge, 1):
            expert_prompt += f"\n--- Knowledge Fragment {i} ---\n{formatted}\n"

        expert_prompt += """
Based on the expert knowledge above, provide detailed optimization recommendations.
"""

        # Validate prompt construction
        assert task_description in expert_prompt
        assert "EXPERT KNOWLEDGE:" in expert_prompt
        assert "memory" in expert_prompt.lower()

        print(
            f"✓ Generated expert prompt with {len(formatted_knowledge)} knowledge fragments"
        )
        print(f"✓ Prompt length: {len(expert_prompt)} characters")

        # Scenario 2: Debugging guidance
        print("\n--- Debugging Guidance Prompt Enhancement ---")
        debug_task = "Debug race condition in CUDA reduction kernel"

        debug_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["race", "condition", "synchronization", "reduction"],
            category_filter="debugging",
            min_score=0.2,
            max_results=2,
        )

        if debug_knowledge:
            debug_prompt = (
                f"Debug Task: {debug_task}\n\nRelevant Debugging Knowledge:\n"
            )
            for knowledge in debug_knowledge:
                formatted = self.knowledge_manager._format_fragment_for_prompt(
                    knowledge
                )
                debug_prompt += f"\n{formatted}\n"

            assert debug_task in debug_prompt
            print(
                f"✓ Generated debugging prompt with {len(debug_knowledge)} knowledge fragments"
            )

        # Scenario 3: Architecture-specific optimization
        print("\n--- Architecture-Specific Optimization ---")
        arch_task = "Optimize kernel for Ampere architecture (RTX 3080)"

        arch_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["architecture", "Ampere", "optimization", "hardware"],
            min_score=0.2,
            max_results=2,
        )

        if arch_knowledge:
            arch_prompt = f"Architecture Task: {arch_task}\n\nArchitecture Knowledge:\n"
            for knowledge in arch_knowledge:
                formatted = self.knowledge_manager._format_fragment_for_prompt(
                    knowledge
                )
                arch_prompt += f"\n{formatted}\n"

            print(
                f"✓ Generated architecture-specific prompt with {len(arch_knowledge)} knowledge fragments"
            )

        print("\n✓ Knowledge-enhanced prompt generation works for various scenarios")

    def test_custom_knowledge_expansion_usage(self):
        """
        Demonstrates how to expand the knowledge base with custom domain knowledge.

        Usage Pattern:
        - Add custom knowledge fragments to existing base
        - Organize custom knowledge by categories and tags
        - Integrate custom knowledge with existing CUDA knowledge
        """

        # Add custom machine learning optimization knowledge
        custom_ml_knowledge = [
            {
                "title": "CUDA Optimizations for Deep Learning Training",
                "content": """
Deep learning training workloads have specific CUDA optimization patterns:

1. **Batch Size Optimization**: Larger batch sizes improve GPU utilization but require more memory.
   - Use gradient accumulation for effective large batches
   - Monitor memory usage vs. throughput trade-offs

2. **Mixed Precision Training**: Use FP16 operations where possible
   - Enables 2x memory savings and faster computation on modern GPUs
   - Requires careful loss scaling to prevent gradient underflow

3. **Tensor Core Utilization**: Ensure tensor dimensions are multiples of 8/16
   - Matrix dimensions should align with Tensor Core requirements
   - Use appropriate data types (FP16, BF16, INT8)

4. **Data Loading Pipeline**: Overlap data loading with computation
   - Use CUDA streams for concurrent data transfer
   - Implement efficient data preprocessing on GPU
""",
                "category": "machine_learning",
                "tags": [
                    "deep_learning",
                    "training",
                    "batch_optimization",
                    "mixed_precision",
                    "tensor_cores",
                ],
                "difficulty_level": "advanced",
                "domain": "machine_learning",
            },
            {
                "title": "CUDA Memory Management for Large Models",
                "content": """
Large machine learning models require sophisticated memory management:

1. **Gradient Checkpointing**: Trade computation for memory
   - Recompute intermediate activations instead of storing them
   - Significantly reduces memory footprint for deep networks

2. **Model Parallelism**: Split models across multiple GPUs
   - Partition layers across devices for models too large for single GPU
   - Implement efficient inter-GPU communication

3. **Dynamic Memory Allocation**: Use memory pools and caching
   - Pre-allocate memory pools to reduce allocation overhead
   - Implement smart caching strategies for frequent allocations

4. **Memory-Efficient Optimizers**: Use optimizers with reduced memory footprint
   - Implement 8-bit optimizers where applicable
   - Use gradient compression techniques
""",
                "category": "memory_management",
                "tags": [
                    "large_models",
                    "gradient_checkpointing",
                    "model_parallelism",
                    "memory_pools",
                ],
                "difficulty_level": "advanced",
                "domain": "machine_learning",
            },
            {
                "title": "CUDA Profiling for ML Workloads",
                "content": """
Profiling machine learning workloads requires specific considerations:

1. **Training vs. Inference Profiling**: Different optimization targets
   - Training: Focus on throughput and memory efficiency
   - Inference: Focus on latency and batch processing

2. **Layer-wise Analysis**: Profile individual neural network layers
   - Identify computational bottlenecks in specific layers
   - Analyze memory access patterns for each operation type

3. **End-to-End Pipeline Profiling**: Include data loading and preprocessing
   - Profile complete training pipeline, not just kernel execution
   - Identify data pipeline bottlenecks and optimization opportunities
""",
                "category": "debugging",
                "tags": [
                    "profiling",
                    "machine_learning",
                    "training",
                    "inference",
                    "pipeline",
                ],
                "difficulty_level": "intermediate",
                "domain": "machine_learning",
            },
        ]

        # Add custom knowledge to the knowledge base
        for knowledge_data in custom_ml_knowledge:
            knowledge_id = self.knowledge_manager.add_knowledge_fragment(
                title=knowledge_data["title"],
                content=knowledge_data["content"],
                category=knowledge_data["category"],
                tags=knowledge_data["tags"],
                difficulty_level=knowledge_data["difficulty_level"],
                metadata={"domain": knowledge_data["domain"]},
            )
            assert knowledge_id is not None

        print(
            f"✓ Added {len(custom_ml_knowledge)} custom machine learning knowledge fragments"
        )

        # Verify custom knowledge integration
        ml_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["machine_learning", "deep_learning"], min_score=0.1
        )

        assert len(ml_knowledge) >= len(custom_ml_knowledge)

        for knowledge in ml_knowledge:
            print(f"ML Knowledge: {knowledge['title']}")
            assert (
                "machine_learning" in knowledge.get("tags", [])
                or knowledge.get("metadata", {}).get("domain") == "machine_learning"
            )

        print(f"✓ Custom knowledge properly integrated with existing knowledge base")

        # Test cross-domain knowledge queries
        print("\n--- Cross-Domain Knowledge Query ---")
        optimization_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["optimization", "memory"], min_score=0.2
        )

        # Should find both CUDA optimization and ML-specific optimization knowledge
        categories_found = set(
            knowledge["category"] for knowledge in optimization_knowledge
        )

        print(
            f"✓ Cross-domain query found knowledge from categories: {categories_found}"
        )
        print(
            f"✓ Total optimization knowledge fragments: {len(optimization_knowledge)}"
        )

        # Test domain-specific filtering
        ml_specific = self.knowledge_manager.query_by_keywords(
            keywords=[""], min_score=0.0
        )

        ml_count = sum(
            1
            for k in ml_specific
            if k.get("metadata", {}).get("domain") == "machine_learning"
        )
        cuda_count = len(ml_specific) - ml_count

        print(
            f"✓ Knowledge base composition: {cuda_count} CUDA knowledge, {ml_count} ML knowledge"
        )

    def test_knowledge_categorization_and_organization_usage(self):
        """
        Demonstrates knowledge categorization and organizational patterns.

        Usage Pattern:
        - Organize knowledge by categories and difficulty levels
        - Use tags for cross-cutting concerns
        - Implement knowledge hierarchies and relationships
        """

        # Analyze knowledge organization
        all_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=[""], min_score=0.0
        )

        # Category analysis
        category_distribution = {}
        for knowledge in all_knowledge:
            category = knowledge["category"]
            category_distribution[category] = category_distribution.get(category, 0) + 1

        print("✓ Knowledge distribution by category:")
        for category, count in sorted(category_distribution.items()):
            print(f"  {category}: {count} fragments")

        # Difficulty level analysis
        difficulty_distribution = {}
        for knowledge in all_knowledge:
            difficulty = knowledge["difficulty_level"]
            difficulty_distribution[difficulty] = (
                difficulty_distribution.get(difficulty, 0) + 1
            )

        print("\n✓ Knowledge distribution by difficulty:")
        for difficulty, count in sorted(difficulty_distribution.items()):
            print(f"  {difficulty}: {count} fragments")

        # Tag analysis
        tag_frequency = {}
        for knowledge in all_knowledge:
            for tag in knowledge["tags"]:
                tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

        # Most common tags
        common_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        print("\n✓ Most common knowledge tags:")
        for tag, freq in common_tags:
            print(f"  {tag}: {freq} occurrences")

        # Test category-specific queries
        print("\n--- Category-Specific Knowledge Access ---")
        categories_to_test = [
            "memory_management",
            "performance_optimization",
            "debugging",
        ]

        for category in categories_to_test:
            category_knowledge = self.knowledge_manager.query_by_keywords(
                keywords=[""], category_filter=category, min_score=0.0
            )

            print(f"{category}: {len(category_knowledge)} knowledge fragments")

            # Verify all fragments belong to the category
            for knowledge in category_knowledge:
                assert knowledge["category"] == category

        # Test difficulty progression
        print("\n--- Difficulty Level Progression ---")
        difficulties = ["beginner", "intermediate", "advanced"]

        for difficulty in difficulties:
            difficulty_knowledge = self.knowledge_manager.query_by_keywords(
                keywords=[""], difficulty_filter=difficulty, min_score=0.0
            )

            print(f"{difficulty}: {len(difficulty_knowledge)} knowledge fragments")

            # Sample knowledge titles for each difficulty
            if difficulty_knowledge:
                sample_titles = [k["title"] for k in difficulty_knowledge[:2]]
                for title in sample_titles:
                    print(f"  Sample: {title}")

        print("\n✓ Knowledge organization and categorization works correctly")

    def test_knowledge_maintenance_and_updates_usage(self):
        """
        Demonstrates knowledge maintenance and update operations.

        Usage Pattern:
        - Update existing knowledge fragments
        - Add new knowledge based on emerging patterns
        - Maintain knowledge quality and relevance
        """

        # Find a knowledge fragment to update
        optimization_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=["optimization"], min_score=0.3, max_results=1
        )

        if optimization_knowledge:
            knowledge_to_update = optimization_knowledge[0]
            knowledge_id = knowledge_to_update["id"]

            print(f"✓ Found knowledge to update: {knowledge_to_update['title']}")

            # Update with additional information
            updated_content = (
                knowledge_to_update["content"]
                + """

RECENT UPDATES:
- New optimization techniques for Ampere architecture
- Updated best practices based on CUDA 11.8 features
- Performance improvements with cooperative groups
"""
            )

            update_success = self.knowledge_manager.update_knowledge_fragment(
                knowledge_id,
                content=updated_content,
                metadata={"last_updated": "2024-01-01", "version": "2.0"},
            )

            assert update_success == True

            # Verify update
            updated_knowledge = self.knowledge_manager.get_knowledge_fragment(
                knowledge_id
            )
            assert updated_knowledge is not None
            assert "RECENT UPDATES:" in updated_knowledge["content"]
            assert updated_knowledge["metadata"]["version"] == "2.0"

            print("✓ Knowledge fragment updated successfully")

        # Test adding contemporary knowledge
        contemporary_knowledge = {
            "title": "CUDA Optimization for H100 Architecture",
            "content": """
Latest optimizations for NVIDIA H100 (Hopper architecture):

1. **Fourth-Generation Tensor Cores**: Enhanced mixed-precision capabilities
   - Support for FP8 data types for training acceleration
   - Improved sparsity support for transformer models

2. **Thread Block Clusters**: New execution model for better scaling
   - Coordinate multiple thread blocks for improved locality
   - Enhanced cooperative kernel execution patterns

3. **Distributed Shared Memory**: Hardware-managed cache coherence
   - Automatic data movement between SM clusters
   - Reduced programmer complexity for multi-SM algorithms

4. **Enhanced Memory Subsystem**: Improved bandwidth and latency
   - HBM3 memory with higher bandwidth
   - Optimized memory controllers for AI workloads
""",
            "category": "hardware_architecture",
            "tags": [
                "H100",
                "Hopper",
                "tensor_cores",
                "thread_block_clusters",
                "modern_gpu",
            ],
            "difficulty_level": "advanced",
        }

        new_knowledge_id = self.knowledge_manager.add_knowledge_fragment(
            title=contemporary_knowledge["title"],
            content=contemporary_knowledge["content"],
            category=contemporary_knowledge["category"],
            tags=contemporary_knowledge["tags"],
            difficulty_level=contemporary_knowledge["difficulty_level"],
            metadata={"architecture": "H100", "year": "2024"},
        )

        assert new_knowledge_id is not None
        print("✓ Added contemporary H100 architecture knowledge")

        # Test knowledge quality validation
        all_knowledge = self.knowledge_manager.query_by_keywords(
            keywords=[""], min_score=0.0
        )

        quality_metrics = {
            "total_fragments": len(all_knowledge),
            "substantial_content": sum(
                1 for k in all_knowledge if len(k["content"]) > 200
            ),
            "well_tagged": sum(1 for k in all_knowledge if len(k["tags"]) >= 3),
            "categorized": sum(1 for k in all_knowledge if k["category"]),
            "difficulty_assigned": sum(
                1 for k in all_knowledge if k["difficulty_level"]
            ),
        }

        print("\n✓ Knowledge base quality metrics:")
        for metric, value in quality_metrics.items():
            percentage = (
                (value / quality_metrics["total_fragments"]) * 100
                if quality_metrics["total_fragments"] > 0
                else 0
            )
            print(
                f"  {metric}: {value}/{quality_metrics['total_fragments']} ({percentage:.1f}%)"
            )

        # Validate high quality standards
        assert (
            quality_metrics["substantial_content"] / quality_metrics["total_fragments"]
            > 0.8
        )
        assert quality_metrics["well_tagged"] / quality_metrics["total_fragments"] > 0.7

        print("✓ Knowledge base maintains high quality standards")


if __name__ == "__main__":
    """
    Run knowledge system usage tests and display results.

    This section demonstrates comprehensive knowledge system testing.
    """
    print("Running Knowledge System Usage Tests...")
    print("=" * 50)

    # Create test instance
    test_instance = TestKnowledgeSystemUsage()
    test_instance.setup_method()

    # Run each test with clear output
    tests = [
        (
            "CUDA Knowledge Base Access",
            test_instance.test_cuda_knowledge_base_access_usage,
        ),
        (
            "Keyword-Based Knowledge Queries",
            test_instance.test_keyword_based_knowledge_queries_usage,
        ),
        (
            "Knowledge Integration with Prompts",
            test_instance.test_knowledge_integration_with_prompts_usage,
        ),
        (
            "Custom Knowledge Expansion",
            test_instance.test_custom_knowledge_expansion_usage,
        ),
        (
            "Knowledge Categorization and Organization",
            test_instance.test_knowledge_categorization_and_organization_usage,
        ),
        (
            "Knowledge Maintenance and Updates",
            test_instance.test_knowledge_maintenance_and_updates_usage,
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

    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All knowledge system tests passed!")
        print("\nKey Capabilities Demonstrated:")
        print("- Comprehensive CUDA knowledge base with expert-level content")
        print("- Intelligent keyword-based knowledge retrieval")
        print("- Seamless integration with prompt generation systems")
        print("- Flexible knowledge expansion and customization")
        print("- Robust knowledge organization and categorization")
        print("- Effective knowledge maintenance and quality control")
        print("\nDevelopers can now:")
        print("- Leverage extensive CUDA expertise in their applications")
        print("- Build intelligent knowledge-driven AI systems")
        print("- Expand knowledge bases with domain-specific content")
        print("- Create expert-level prompt generation with knowledge context")
        print("- Maintain high-quality, organized knowledge repositories")
    else:
        print("⚠️ Some tests failed. Check knowledge system implementation.")
